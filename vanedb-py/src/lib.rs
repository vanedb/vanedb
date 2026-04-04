use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use ::vanedb::distance::DistanceMetric;
use ::vanedb::hnsw::HnswIndex;
use ::vanedb::store::VectorStore;
use ::vanedb::VaneError;

fn to_pyerr(e: VaneError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Distance metric enum.
#[pyclass(eq, eq_int)]
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

    fn add(&self, id: u64, vector: Vec<f32>) -> PyResult<()> {
        self.inner.add(id, &vector).map_err(to_pyerr)
    }

    fn search(&self, query: Vec<f32>, k: usize) -> PyResult<Vec<(u64, f32)>> {
        let results = self.inner.search(&query, k).map_err(to_pyerr)?;
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

    fn add(&self, id: u64, vector: Vec<f32>) -> PyResult<()> {
        self.inner.add(id, &vector).map_err(to_pyerr)
    }

    fn search(&self, query: Vec<f32>, k: usize) -> PyResult<Vec<(u64, f32)>> {
        let results = self.inner.search(&query, k).map_err(to_pyerr)?;
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
