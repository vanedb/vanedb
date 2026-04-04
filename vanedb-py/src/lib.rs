use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use ::vanedb::distance::DistanceMetric;
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

#[pymodule]
fn vanedb(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", "0.1.0")?;
    m.add_class::<PyDistanceMetric>()?;
    m.add_class::<PyVectorStore>()?;
    Ok(())
}
