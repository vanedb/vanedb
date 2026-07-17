use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

use ::vanedb::distance::DistanceMetric;
use ::vanedb::hnsw::HnswIndex;
use ::vanedb::store::VectorStore;
use ::vanedb::VaneError;

fn to_pyerr(e: VaneError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Extract a single vector. Fast path: any 1-D float32 buffer (numpy array,
/// array.array, memoryview) copied wholesale; fallback: generic sequence
/// extraction (lists, float64 arrays), matching the pre-buffer behavior.
fn vec_f32(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f32>> {
    if let Ok(buf) = PyBuffer::<f32>::get(obj) {
        if buf.dimensions() != 1 {
            return Err(PyValueError::new_err(format!(
                "expected a 1-D vector, got a {}-D buffer",
                buf.dimensions()
            )));
        }
        return buf.to_vec(obj.py());
    }
    obj.extract()
}

/// Extract a batch of vectors as (row_count, flat row-major f32).
/// Fast path: a 2-D float32 buffer of shape (n, dim); fallback: a sequence of
/// float sequences. Row width is validated here because the core's flat-length
/// check alone cannot catch ragged rows whose total happens to match.
fn batch_f32(obj: &Bound<'_, PyAny>, dim: usize) -> PyResult<(usize, Vec<f32>)> {
    if let Ok(buf) = PyBuffer::<f32>::get(obj) {
        if buf.dimensions() != 2 {
            return Err(PyValueError::new_err(format!(
                "expected a 2-D array of shape (n, {dim}), got a {}-D buffer",
                buf.dimensions()
            )));
        }
        let shape = buf.shape();
        if shape[1] != dim {
            return Err(PyValueError::new_err(format!(
                "dimension mismatch: expected vectors of dimension {dim}, got {}",
                shape[1]
            )));
        }
        return Ok((shape[0], buf.to_vec(obj.py())?));
    }
    let rows: Vec<Vec<f32>> = obj.extract().map_err(|_| {
        PyTypeError::new_err(
            "vectors must be a 2-D float32 buffer (e.g. numpy array) or a sequence of float sequences",
        )
    })?;
    let mut flat = Vec::with_capacity(rows.len() * dim);
    for row in &rows {
        if row.len() != dim {
            return Err(PyValueError::new_err(format!(
                "dimension mismatch: expected vectors of dimension {dim}, got {}",
                row.len()
            )));
        }
        flat.extend_from_slice(row);
    }
    Ok((rows.len(), flat))
}

/// Extract ids. Fast paths: 1-D uint64 or int64 buffers (int64 is numpy's
/// default integer dtype; negative values are rejected); fallback: sequence.
fn ids_u64(obj: &Bound<'_, PyAny>) -> PyResult<Vec<u64>> {
    if let Ok(buf) = PyBuffer::<u64>::get(obj) {
        if buf.dimensions() != 1 {
            return Err(PyValueError::new_err("ids must be 1-D"));
        }
        return buf.to_vec(obj.py());
    }
    if let Ok(buf) = PyBuffer::<i64>::get(obj) {
        if buf.dimensions() != 1 {
            return Err(PyValueError::new_err("ids must be 1-D"));
        }
        return buf
            .to_vec(obj.py())?
            .into_iter()
            .map(|x| {
                u64::try_from(x).map_err(|_| PyValueError::new_err(format!("negative id: {x}")))
            })
            .collect();
    }
    obj.extract()
}

fn check_batch_len(ids: &[u64], rows: usize) -> PyResult<()> {
    if ids.len() != rows {
        return Err(PyValueError::new_err(format!(
            "ids length {} does not match number of vectors {rows}",
            ids.len()
        )));
    }
    Ok(())
}

/// Distance metric enum.
#[pyclass(eq, eq_int, from_py_object)]
#[derive(Clone, Copy, PartialEq)]
enum PyDistanceMetric {
    L2 = 0,
    Cosine = 1,
    Dot = 2,
}

impl From<PyDistanceMetric> for DistanceMetric {
    fn from(m: PyDistanceMetric) -> Self {
        match m {
            PyDistanceMetric::L2 => DistanceMetric::L2,
            PyDistanceMetric::Cosine => DistanceMetric::Cosine,
            PyDistanceMetric::Dot => DistanceMetric::Dot,
        }
    }
}

/// Brute-force vector store with thread-safe k-NN search.
#[pyclass]
struct PyVectorStore {
    inner: VectorStore,
}

#[pymethods]
impl PyVectorStore {
    #[new]
    #[pyo3(signature = (dim, metric=PyDistanceMetric::L2))]
    fn new(dim: usize, metric: PyDistanceMetric) -> PyResult<Self> {
        let inner = VectorStore::new(dim, metric.into()).map_err(to_pyerr)?;
        Ok(Self { inner })
    }

    /// Add one vector. Accepts a 1-D float32 buffer (numpy) or any float sequence.
    fn add(&self, id: u64, vector: &Bound<'_, PyAny>) -> PyResult<()> {
        let v = vec_f32(vector)?;
        self.inner.add(id, &v).map_err(to_pyerr)
    }

    /// Bulk insert. `ids`: 1-D uint64/int64 buffer or int sequence; `vectors`:
    /// 2-D float32 buffer of shape (n, dim) or sequence of float sequences.
    /// All-or-nothing; the GIL is released while inserting.
    fn add_batch(
        &self,
        py: Python<'_>,
        ids: &Bound<'_, PyAny>,
        vectors: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let ids = ids_u64(ids)?;
        let (rows, flat) = batch_f32(vectors, self.inner.dimension())?;
        check_batch_len(&ids, rows)?;
        py.detach(|| self.inner.add_batch(&ids, &flat))
            .map_err(to_pyerr)
    }

    /// k-NN search. Accepts a 1-D float32 buffer (numpy) or any float sequence.
    fn search(
        &self,
        py: Python<'_>,
        query: &Bound<'_, PyAny>,
        k: usize,
    ) -> PyResult<Vec<(u64, f32)>> {
        let q = vec_f32(query)?;
        let results = py.detach(|| self.inner.search(&q, k)).map_err(to_pyerr)?;
        Ok(results.into_iter().map(|r| (r.id, r.distance)).collect())
    }

    fn get(&self, id: u64) -> PyResult<Vec<f32>> {
        self.inner.get(id).map_err(to_pyerr)
    }

    fn remove(&self, id: u64) -> PyResult<()> {
        self.inner.remove(id).map_err(to_pyerr)
    }

    fn contains(&self, id: u64) -> bool {
        self.inner.contains(id)
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    #[getter]
    fn dimension(&self) -> usize {
        self.inner.dimension()
    }
}

/// HNSW approximate nearest-neighbor index.
#[pyclass]
struct PyHnswIndex {
    inner: HnswIndex,
}

#[pymethods]
impl PyHnswIndex {
    #[new]
    #[pyo3(signature = (dim, metric=PyDistanceMetric::L2, capacity=100000, m=16, ef_construction=200, seed=42))]
    fn new(
        dim: usize,
        metric: PyDistanceMetric,
        capacity: usize,
        m: usize,
        ef_construction: usize,
        seed: u64,
    ) -> PyResult<Self> {
        let inner = HnswIndex::builder(dim, metric.into())
            .capacity(capacity)
            .m(m)
            .ef_construction(ef_construction)
            .seed(seed)
            .build()
            .map_err(to_pyerr)?;
        Ok(Self { inner })
    }

    /// Add one vector. Accepts a 1-D float32 buffer (numpy) or any float sequence.
    fn add(&self, py: Python<'_>, id: u64, vector: &Bound<'_, PyAny>) -> PyResult<()> {
        let v = vec_f32(vector)?;
        py.detach(|| self.inner.add(id, &v)).map_err(to_pyerr)
    }

    /// Bulk insert. `ids`: 1-D uint64/int64 buffer or int sequence; `vectors`:
    /// 2-D float32 buffer of shape (n, dim) or sequence of float sequences.
    /// All-or-nothing; the GIL is released while the graph is built.
    fn add_batch(
        &self,
        py: Python<'_>,
        ids: &Bound<'_, PyAny>,
        vectors: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let ids = ids_u64(ids)?;
        let (rows, flat) = batch_f32(vectors, self.inner.dimension())?;
        check_batch_len(&ids, rows)?;
        py.detach(|| self.inner.add_batch(&ids, &flat))
            .map_err(to_pyerr)
    }

    /// k-NN search. Accepts a 1-D float32 buffer (numpy) or any float sequence.
    fn search(
        &self,
        py: Python<'_>,
        query: &Bound<'_, PyAny>,
        k: usize,
    ) -> PyResult<Vec<(u64, f32)>> {
        let q = vec_f32(query)?;
        let results = py.detach(|| self.inner.search(&q, k)).map_err(to_pyerr)?;
        Ok(results.into_iter().map(|r| (r.id, r.distance)).collect())
    }

    fn get_vector(&self, id: u64) -> PyResult<Vec<f32>> {
        self.inner.get_vector(id).map_err(to_pyerr)
    }

    fn contains(&self, id: u64) -> bool {
        self.inner.contains(id)
    }

    fn save(&self, path: &str) -> PyResult<()> {
        self.inner.save(path).map_err(to_pyerr)
    }

    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let inner = HnswIndex::load(path).map_err(to_pyerr)?;
        Ok(Self { inner })
    }

    #[getter]
    fn ef_search(&self) -> usize {
        self.inner.get_ef_search()
    }

    #[setter]
    fn set_ef_search(&self, ef: usize) {
        self.inner.set_ef_search(ef);
    }

    fn __len__(&self) -> usize {
        self.inner.size()
    }

    #[getter]
    fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    #[getter]
    fn capacity(&self) -> usize {
        self.inner.capacity()
    }
}

#[pymodule]
fn vanedb(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", "0.1.0")?;
    m.add_class::<PyDistanceMetric>()?;
    m.add_class::<PyVectorStore>()?;
    m.add_class::<PyHnswIndex>()?;
    Ok(())
}
