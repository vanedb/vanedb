use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyFileNotFoundError, PyOSError, PyTypeError, PyValueError};
use pyo3::prelude::*;

use ::vanedb::approx::ApproxIndex;
use ::vanedb::distance::Metric;
use ::vanedb::flat::FlatIndex;
use ::vanedb::VaneError;
use ::vanedb::{DiskIndex, DiskIndexBuilder};

fn to_pyerr(e: VaneError) -> PyErr {
    // A missing file and a corrupt one call for different handling, so they
    // must not share an exception class. FileNotFoundError is checked first
    // because it is a subclass of OSError.
    match &e {
        VaneError::FileNotFound { .. } => PyFileNotFoundError::new_err(e.to_string()),
        VaneError::Io { .. } => PyOSError::new_err(e.to_string()),
        // Corrupt data and every validation failure stay ValueError.
        _ => PyValueError::new_err(e.to_string()),
    }
}

/// Converts a Python int to an id.
///
/// PyO3's own `u64` conversion raises `OverflowError` for a negative value,
/// and `OverflowError` is not a `ValueError` subclass -- so `except ValueError`
/// silently missed negative ids. Every method taking an id goes through this.
fn one_id(obj: &Bound<'_, PyAny>) -> PyResult<u64> {
    if let Ok(id) = obj.extract::<u64>() {
        return Ok(id);
    }
    let signed: i64 = obj.extract()?;
    u64::try_from(signed).map_err(|_| PyValueError::new_err(format!("negative id: {signed}")))
}

/// Extract a single vector. Fast paths: any 1-D float32 or float64 buffer
/// (numpy array, array.array, memoryview) copied wholesale; fallback: generic
/// sequence extraction (lists), matching the pre-buffer behavior. Rank is
/// validated in every buffer branch so shape errors do not depend on dtype.
fn vec_f32(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f32>> {
    if let Ok(buf) = PyBuffer::<f32>::get(obj) {
        check_rank(buf.dimensions(), 1, "expected a 1-D vector")?;
        return buf.to_vec(obj.py());
    }
    if let Ok(buf) = PyBuffer::<f64>::get(obj) {
        check_rank(buf.dimensions(), 1, "expected a 1-D vector")?;
        return Ok(buf.to_vec(obj.py())?.iter().map(|&x| x as f32).collect());
    }
    obj.extract()
}

fn check_rank(got: usize, expected: usize, what: &str) -> PyResult<()> {
    if got != expected {
        return Err(PyValueError::new_err(format!(
            "{what}, got a {got}-D buffer"
        )));
    }
    Ok(())
}

/// Extract a batch of vectors as (row_count, flat row-major f32).
/// Fast paths: a 2-D float32 or float64 buffer of shape (n, dim); fallback: a
/// sequence of float sequences. Row width is validated here because the core's
/// flat-length check alone cannot catch ragged rows whose total happens to
/// match. Rank and width are validated in every buffer branch so shape errors
/// do not depend on dtype.
fn batch_f32(obj: &Bound<'_, PyAny>, dim: usize) -> PyResult<(usize, Vec<f32>)> {
    fn check_shape(rank: usize, shape: &[usize], dim: usize) -> PyResult<()> {
        check_rank(
            rank,
            2,
            &format!("expected a 2-D array of shape (n, {dim})"),
        )?;
        if shape[1] != dim {
            return Err(PyValueError::new_err(format!(
                "dimension mismatch: expected vectors of dimension {dim}, got {}",
                shape[1]
            )));
        }
        Ok(())
    }
    if let Ok(buf) = PyBuffer::<f32>::get(obj) {
        check_shape(buf.dimensions(), buf.shape(), dim)?;
        return Ok((buf.shape()[0], buf.to_vec(obj.py())?));
    }
    if let Ok(buf) = PyBuffer::<f64>::get(obj) {
        check_shape(buf.dimensions(), buf.shape(), dim)?;
        let flat = buf.to_vec(obj.py())?.iter().map(|&x| x as f32).collect();
        return Ok((buf.shape()[0], flat));
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
    match obj.extract::<Vec<u64>>() {
        Ok(ids) => Ok(ids),
        Err(_) => {
            // Retry as signed so a negative id in a plain list reports as
            // ValueError, matching the buffer path above.
            let signed: Vec<i64> = obj.extract()?;
            signed
                .into_iter()
                .map(|x| {
                    u64::try_from(x).map_err(|_| PyValueError::new_err(format!("negative id: {x}")))
                })
                .collect()
        }
    }
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
///
/// The `Py` prefix disambiguates these wrappers from the core types they hold,
/// which are imported in this file. It is an implementation detail: the names
/// exported to Python match `vanedb_cpp` exactly, so swapping engines is an
/// import-line change.
#[pyclass(name = "Metric", eq, eq_int, from_py_object)]
#[derive(Clone, Copy, PartialEq)]
enum PyMetric {
    L2 = 0,
    #[pyo3(name = "COSINE")]
    Cosine = 1,
    #[pyo3(name = "DOT")]
    Dot = 2,
}

impl From<PyMetric> for Metric {
    fn from(m: PyMetric) -> Self {
        match m {
            PyMetric::L2 => Metric::L2,
            PyMetric::Cosine => Metric::Cosine,
            PyMetric::Dot => Metric::Dot,
        }
    }
}

/// Brute-force vector store with thread-safe k-NN search.
#[pyclass(name = "FlatIndex")]
struct PyStore {
    inner: FlatIndex,
}

#[pymethods]
impl PyStore {
    #[new]
    #[pyo3(signature = (dim, metric=PyMetric::L2))]
    fn new(dim: usize, metric: PyMetric) -> PyResult<Self> {
        let inner = FlatIndex::new(dim, metric.into()).map_err(to_pyerr)?;
        Ok(Self { inner })
    }

    /// Add one vector. Accepts a 1-D float32 buffer (numpy) or any float sequence.
    fn add(
        &self,
        #[pyo3(from_py_with = one_id)] id: u64,
        vector: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
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

    fn get(&self, #[pyo3(from_py_with = one_id)] id: u64) -> PyResult<Vec<f32>> {
        self.inner.get(id).map_err(to_pyerr)
    }

    fn remove(&self, #[pyo3(from_py_with = one_id)] id: u64) -> PyResult<()> {
        self.inner.remove(id).map_err(to_pyerr)
    }

    fn contains(&self, #[pyo3(from_py_with = one_id)] id: u64) -> bool {
        self.inner.contains(id)
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Number of vectors stored.
    ///
    /// `len(store)` is the Pythonic spelling; `size()` is what vanedb_cpp and
    /// the wasm bindings expose. Both work here so neither spelling ties a
    /// program to one engine (#85).
    fn size(&self) -> usize {
        self.inner.len()
    }

    #[getter]
    fn dimension(&self) -> usize {
        self.inner.dimension()
    }
}

/// HNSW approximate nearest-neighbor index.
#[pyclass(name = "ApproxIndex")]
struct PyIndex {
    inner: ApproxIndex,
}

#[pymethods]
impl PyIndex {
    #[new]
    #[pyo3(signature = (dim, metric=PyMetric::L2, capacity=100000, m=16, ef_construction=200, seed=42))]
    fn new(
        dim: usize,
        metric: PyMetric,
        capacity: usize,
        m: usize,
        ef_construction: usize,
        seed: u64,
    ) -> PyResult<Self> {
        let inner = ApproxIndex::builder(dim, metric.into())
            .capacity(capacity)
            .m(m)
            .ef_construction(ef_construction)
            .seed(seed)
            .build()
            .map_err(to_pyerr)?;
        Ok(Self { inner })
    }

    /// Add one vector. Accepts a 1-D float32 buffer (numpy) or any float sequence.
    fn add(
        &self,
        py: Python<'_>,
        #[pyo3(from_py_with = one_id)] id: u64,
        vector: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
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

    fn get_vector(&self, #[pyo3(from_py_with = one_id)] id: u64) -> PyResult<Vec<f32>> {
        self.inner.get_vector(id).map_err(to_pyerr)
    }

    fn contains(&self, #[pyo3(from_py_with = one_id)] id: u64) -> bool {
        self.inner.contains(id)
    }

    fn save(&self, path: &str) -> PyResult<()> {
        self.inner.save(path).map_err(to_pyerr)
    }

    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let inner = ApproxIndex::load(path).map_err(to_pyerr)?;
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

    /// Inserts `vector` under `id`, replacing any existing entry.
    ///
    /// Both halves happen under one write lock, so a concurrent reader never
    /// observes the id missing. The old slot is tombstoned, so a long upsert
    /// loop still needs `compact()`.
    fn upsert(
        &self,
        #[pyo3(from_py_with = one_id)] id: u64,
        vector: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let v = vec_f32(vector)?;
        self.inner.upsert(id, &v).map_err(to_pyerr)
    }

    /// Number of tombstoned slots: removed vectors whose space is not yet
    /// reclaimed. Re-adding a removed id allocates a fresh slot, so an upsert
    /// loop grows this even at constant length.
    fn tombstones(&self) -> usize {
        self.inner.tombstones()
    }

    /// Rebuilds the graph without tombstoned slots, reclaiming their space.
    ///
    /// A full rebuild, holding the write lock throughout, so concurrent
    /// searches block. Ids and vectors are preserved.
    fn compact(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.inner.compact()).map_err(to_pyerr)
    }

    /// Removes the vector stored under `id`.
    ///
    /// Tombstoned: the node keeps its graph links, which may be the only
    /// route between live neighbourhoods, and simply stops appearing in
    /// results. The id becomes free for reuse. Space is not reclaimed.
    fn remove(&self, #[pyo3(from_py_with = one_id)] id: u64) -> PyResult<()> {
        self.inner.remove(id).map_err(to_pyerr)
    }

    /// Number of vectors in the graph. See `FlatIndex.size`.
    fn size(&self) -> usize {
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

/// Builds a `DiskIndex` file. Vectors are held in memory until `save`; the
/// memory saving is on the reading side.
#[pyclass(name = "DiskIndexBuilder")]
struct PyDiskStoreBuilder {
    inner: DiskIndexBuilder,
}

#[pymethods]
impl PyDiskStoreBuilder {
    #[new]
    #[pyo3(signature = (dim, metric=PyMetric::L2))]
    fn new(dim: usize, metric: PyMetric) -> PyResult<Self> {
        Ok(Self {
            inner: DiskIndexBuilder::new(dim, metric.into()).map_err(to_pyerr)?,
        })
    }

    fn add(
        &mut self,
        #[pyo3(from_py_with = one_id)] id: u64,
        vector: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let v = vec_f32(vector)?;
        self.inner.add(id, &v).map_err(to_pyerr)
    }

    /// Writes the store to `path`, atomically: built beside the destination
    /// and renamed in after an fsync.
    fn save(&self, py: Python<'_>, path: &str) -> PyResult<()> {
        py.detach(|| self.inner.save(path)).map_err(to_pyerr)
    }

    fn __len__(&self) -> usize {
        self.inner.size()
    }

    fn size(&self) -> usize {
        self.inner.size()
    }

    #[getter]
    fn dimension(&self) -> usize {
        self.inner.dimension()
    }
}

/// Exact search over a memory-mapped file. Read-only; build one with
/// `DiskIndexBuilder`.
#[pyclass(name = "DiskIndex")]
struct PyDiskStore {
    inner: DiskIndex,
}

#[pymethods]
impl PyDiskStore {
    /// Maps the store at `path`.
    ///
    /// Validates the header and every stored value, so this is linear in the
    /// corpus rather than a constant-cost mapping.
    #[staticmethod]
    fn open(py: Python<'_>, path: &str) -> PyResult<Self> {
        let inner = py.detach(|| DiskIndex::open(path)).map_err(to_pyerr)?;
        Ok(Self { inner })
    }

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

    fn get(&self, #[pyo3(from_py_with = one_id)] id: u64) -> PyResult<Vec<f32>> {
        self.inner.get(id).map(<[f32]>::to_vec).map_err(to_pyerr)
    }

    fn contains(&self, #[pyo3(from_py_with = one_id)] id: u64) -> bool {
        self.inner.contains(id)
    }

    fn __len__(&self) -> usize {
        self.inner.size()
    }

    fn size(&self) -> usize {
        self.inner.size()
    }

    #[getter]
    fn dimension(&self) -> usize {
        self.inner.dimension()
    }
}

#[pymodule]
fn vanedb(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // From Cargo.toml, never a literal: a hardcoded string silently
    // disagreed with the wheel's own metadata the first time the
    // version moved.
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<PyMetric>()?;
    m.add_class::<PyStore>()?;
    m.add_class::<PyIndex>()?;
    m.add_class::<PyDiskStore>()?;
    m.add_class::<PyDiskStoreBuilder>()?;
    // maturin's generated __init__ does `from .vanedb import *` and copies
    // __all__ verbatim, so this list is the package's entire public surface --
    // omitting __version__ here removes it from the package altogether.
    m.add(
        "__all__",
        vec![
            "Metric",
            "FlatIndex",
            "ApproxIndex",
            "DiskIndex",
            "DiskIndexBuilder",
            "__version__",
        ],
    )?;
    Ok(())
}
