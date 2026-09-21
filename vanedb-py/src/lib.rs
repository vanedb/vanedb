use std::borrow::Cow;
use std::cell::RefCell;
use std::path::PathBuf;
use std::sync::Arc;

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{
    PyFileNotFoundError, PyOSError, PyOverflowError, PyRuntimeError, PyTypeError, PyValueError,
};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::PyInt;
use pyo3::Borrowed;

use ::vanedb::approx::{ApproxIndex, Filter, SearchParams};
use ::vanedb::distance::Metric;
use ::vanedb::flat::{FlatIndex, SearchResult};
use ::vanedb::VaneError;
use ::vanedb::{DiskIndex, DiskIndexBuilder};

fn to_pyerr(e: VaneError) -> PyErr {
    // A missing file and a corrupt one call for different handling, so they
    // must not share an exception class. FileNotFoundError is checked first
    // because it is a subclass of OSError.
    match &e {
        VaneError::FileNotFound { .. } => PyFileNotFoundError::new_err(e.to_string()),
        VaneError::Io { .. } => PyOSError::new_err(e.to_string()),
        // A lookup miss is a value, not an error (RFC 0011): `get` returns
        // `None`, so `NotFound` reaches Python only from `remove`, where a
        // caller removing what is not there has a bug. That is the
        // validation bucket, and `KeyError` is no longer raised by any method.
        //
        // Corrupt data and every validation failure are ValueError.
        //
        // The catch-all is right for validation, but `VaneError` is
        // `#[non_exhaustive]` and not every future variant is a validation
        // failure. `Backend` is the one already written: it means a compute
        // backend is unavailable, which is an environment condition a caller
        // should fall back from, not a bad argument they should fix. It cannot
        // reach here today — it is constructed only in `gpu/metal.rs`, behind
        // the `gpu-metal` feature, which Python does not build — but it must
        // get its own class (`RuntimeError`) before that feature is exposed,
        // rather than inheriting `ValueError` by falling through.
        _ => PyValueError::new_err(e.to_string()),
    }
}

/// Converts a Python int to an unsigned value, preserving type errors.
///
/// PyO3's own `u64` conversion raises `OverflowError` for a negative value,
/// and `OverflowError` is not a `ValueError` subclass, so `except ValueError`
/// would miss it. Extract only once so user-defined `__index__` methods are
/// not called again after an error.
fn unsigned_value(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<u64> {
    obj.extract::<u64>().map_err(|e| {
        if e.is_instance_of::<PyOverflowError>(obj.py()) {
            PyValueError::new_err(format!("{name} out of range for an unsigned integer"))
        } else {
            e
        }
    })
}

fn one_id(obj: &Bound<'_, PyAny>) -> PyResult<u64> {
    unsigned_value(obj, "id")
}

fn one_seed(obj: &Bound<'_, PyAny>) -> PyResult<u64> {
    unsigned_value(obj, "seed")
}

/// Converts a Python int to a count, dimension or other size.
///
/// The additional checked conversion preserves the platform's size limit.
fn one_usize(obj: &Bound<'_, PyAny>) -> PyResult<usize> {
    usize::try_from(unsigned_value(obj, "size")?)
        .map_err(|_| PyValueError::new_err("size out of range for this platform"))
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
    for row in &rows {
        if row.len() != dim {
            return Err(PyValueError::new_err(format!(
                "dimension mismatch: expected vectors of dimension {dim}, got {}",
                row.len()
            )));
        }
    }
    let count = rows.len();
    let capacity = count
        .checked_mul(dim)
        .filter(|n| *n <= isize::MAX as usize / std::mem::size_of::<f32>())
        .ok_or_else(|| PyValueError::new_err("batch is too large"))?;
    let mut flat = Vec::new();
    flat.try_reserve_exact(capacity)
        .map_err(|_| PyValueError::new_err("not enough memory for this batch"))?;
    flat.extend(rows.into_iter().flatten());
    Ok((count, flat))
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
    // Fall back element-wise through the same converter the single-id
    // methods use, so a list and a numpy array report identically: negative
    // and out-of-range ids are ValueError, never OverflowError.
    let items = obj.try_iter()?;
    let mut ids = Vec::new();
    for item in items {
        ids.push(one_id(&item?)?);
    }
    Ok(ids)
}

thread_local! {
    /// Indexes whose predicate search is running on this thread, by address.
    ///
    /// A predicate runs under the searched index's read lock. Calling back
    /// into that index from the callback would block on the same lock --
    /// immediately for a write, or for a read once a writer is queued on
    /// another thread, because parking_lot read locks are not recursive --
    /// and the process would hang with no diagnostic. The list is per thread
    /// because the predicate runs on the searching thread; other threads may
    /// still queue on the index as usual. A nested search on a different
    /// index is supported, hence a list rather than a flag.
    static IN_PREDICATE_SEARCH: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
}

/// Marks `index` as searching with a predicate on this thread until dropped.
struct PredicateScope(usize);

impl PredicateScope {
    fn enter<T>(index: &T) -> Self {
        let addr = index as *const T as usize;
        IN_PREDICATE_SEARCH.with(|s| s.borrow_mut().push(addr));
        Self(addr)
    }
}

impl Drop for PredicateScope {
    fn drop(&mut self) {
        IN_PREDICATE_SEARCH.with(|s| {
            let mut s = s.borrow_mut();
            if let Some(pos) = s.iter().rposition(|&addr| addr == self.0) {
                s.remove(pos);
            }
        });
    }
}

/// Raises instead of deadlocking when a predicate calls back into the index
/// it is filtering. Every method that takes the index's lock checks first.
fn reject_reentry<T>(index: &T) -> PyResult<()> {
    let addr = index as *const T as usize;
    if IN_PREDICATE_SEARCH.with(|s| s.borrow().contains(&addr)) {
        return Err(PyRuntimeError::new_err(
            "a filter predicate must not call methods on the index being searched: \
             the search holds its read lock; consult external metadata or search \
             a different index instead",
        ));
    }
    Ok(())
}

/// Extracted filter representation from Python keyword arguments.
enum PyFilterHolder<'a> {
    None,
    Allow(Vec<u64>),
    Deny(Vec<u64>),
    Predicate {
        predicate: Box<dyn Fn(u64) -> bool + Sync + 'a>,
        error: Arc<parking_lot::Mutex<Option<PyErr>>>,
    },
}

fn extract_py_filter<'a>(
    _py: Python<'a>,
    filter: Option<&Bound<'a, PyAny>>,
    allow_ids: Option<&Bound<'a, PyAny>>,
    deny_ids: Option<&Bound<'a, PyAny>>,
) -> PyResult<PyFilterHolder<'a>> {
    let mut count = 0;
    if filter.is_some() {
        count += 1;
    }
    if allow_ids.is_some() {
        count += 1;
    }
    if deny_ids.is_some() {
        count += 1;
    }
    if count > 1 {
        return Err(PyValueError::new_err(
            "specify at most one of filter, allow_ids, or deny_ids",
        ));
    }

    if let Some(allow_obj) = allow_ids {
        let ids = ids_u64(allow_obj)?;
        return Ok(PyFilterHolder::Allow(ids));
    }
    if let Some(deny_obj) = deny_ids {
        let ids = ids_u64(deny_obj)?;
        return Ok(PyFilterHolder::Deny(ids));
    }
    if let Some(filter_obj) = filter {
        if !filter_obj.is_callable() {
            return Err(PyTypeError::new_err("filter must be a callable"));
        }
        let callable = filter_obj.clone().unbind();
        let error = Arc::new(parking_lot::Mutex::new(None));
        let callback_error = Arc::clone(&error);
        let pred = move |id: u64| {
            // Preserve the first exception, and never invoke user code again
            // after it fails. The core predicate API returns only bool.
            if callback_error.lock().is_some() {
                return false;
            }
            Python::attach(|py| {
                let res = callable.call1(py, (id,)).and_then(|val| val.is_truthy(py));
                match res {
                    Ok(accepted) => accepted,
                    Err(err) => {
                        *callback_error.lock() = Some(err);
                        false
                    }
                }
            })
        };
        return Ok(PyFilterHolder::Predicate {
            predicate: Box::new(pred),
            error,
        });
    }

    Ok(PyFilterHolder::None)
}

fn run_search_with_filter<T>(
    py: Python<'_>,
    index: &T,
    py_filter: &PyFilterHolder<'_>,
    base_params: SearchParams<'_>,
    search_fn: impl Fn(&SearchParams<'_>) -> ::vanedb::Result<Vec<SearchResult>> + Send + Sync,
) -> PyResult<Vec<(u64, f32)>> {
    reject_reentry(index)?;
    let results = match py_filter {
        PyFilterHolder::None => py.detach(|| search_fn(&base_params)).map_err(to_pyerr)?,
        PyFilterHolder::Allow(ids) => {
            let params = base_params.clone().filter(Filter::Allow(ids));
            py.detach(|| search_fn(&params)).map_err(to_pyerr)?
        }
        PyFilterHolder::Deny(ids) => {
            let params = base_params.clone().filter(Filter::Deny(ids));
            py.detach(|| search_fn(&params)).map_err(to_pyerr)?
        }
        PyFilterHolder::Predicate { predicate, error } => {
            let params = base_params.filter(Filter::Predicate(predicate.as_ref()));
            let _scope = PredicateScope::enter(index);
            // As with id lists, do not hold the GIL while acquiring core
            // locks. The callback acquires it only while running Python.
            let result = py.detach(|| search_fn(&params));
            if let Some(err) = error.lock().take() {
                return Err(err);
            }
            result.map_err(to_pyerr)?
        }
    };
    Ok(results.into_iter().map(|r| (r.id, r.distance)).collect())
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

/// The Python `Metric` is a real `enum.IntEnum`, defined in
/// `python/vanedb/__init__.py`, so it has `.name`, `.value`, iteration,
/// hashing and pickling the way every other Python enum does (RFC 0011). A
/// PyO3 class cannot be one: `iter(Metric)` needs `__iter__` on the
/// *metaclass*, which `#[pyclass]` cannot supply. This side therefore holds
/// only the wire values, which are also the on-disk `metric` field and the C
/// ABI's `uint32_t`, and converts at the boundary in both directions.
#[derive(Clone, Copy, PartialEq)]
struct PyMetric(Metric);

impl PyMetric {
    const L2: u8 = 0;
    const COSINE: u8 = 1;
    const DOT: u8 = 2;

    fn wire(self) -> u8 {
        match self.0 {
            Metric::Cosine => Self::COSINE,
            Metric::Dot => Self::DOT,
            // `Metric` is #[non_exhaustive]; a metric this binding does not
            // know cannot be constructed through it, so L2 is unreachable-but-
            // total rather than a silent substitution.
            _ => Self::L2,
        }
    }
}

/// `vanedb.Metric`, looked up once. The package's `__init__` defines the
/// enum after importing this extension, so the lookup happens at first use
/// rather than at module initialisation, when the name does not exist yet.
static METRIC_ENUM: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

fn metric_enum(py: Python<'_>) -> PyResult<&Bound<'_, PyAny>> {
    METRIC_ENUM
        .get_or_try_init(py, || {
            py.import("vanedb")?.getattr("Metric").map(|m| m.unbind())
        })
        .map(|m| m.bind(py))
}

impl<'py> IntoPyObject<'py> for PyMetric {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    /// The `Metric` member for this value: `Metric(0)` is `Metric.L2`, so a
    /// reported metric is the same singleton a caller passed in.
    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        metric_enum(py)?.call1((self.wire(),))
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for PyMetric {
    type Error = PyErr;

    /// A `Metric` member, or any integer holding one of its values: `IntEnum`
    /// members are ints, so `Metric.COSINE` and `1` name the same metric and
    /// `Metric(1)` is how the enum itself spells that. "Integer" is decided
    /// by `__index__`, the way `Metric(...)` itself decides it, so a NumPy
    /// integer is accepted and so is `bool` (`Metric(True)` is
    /// `Metric.COSINE`). Anything else is a `TypeError`; an integer naming
    /// no metric is a `ValueError`, as `Metric(7)` would be.
    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let py = obj.py();
        let enum_class = metric_enum(py)?;
        if !obj.is_instance(enum_class)? && !obj.hasattr("__index__")? {
            return Err(PyTypeError::new_err(format!(
                "metric must be a vanedb.Metric, got {}",
                obj.get_type().name()?
            )));
        }
        // `__index__` turns a member, a NumPy integer or a bool into the plain
        // int the enum looks up by value; routing through the enum makes an
        // unknown value fail the way the enum says:
        // `ValueError: 7 is not a valid Metric`.
        let value = obj.call_method0("__index__")?;
        let member = enum_class.call1((value,))?;
        let metric = match member.getattr("value")?.extract::<u8>()? {
            Self::L2 => Metric::L2,
            Self::COSINE => Metric::Cosine,
            Self::DOT => Metric::Dot,
            other => {
                return Err(PyValueError::new_err(format!(
                    "{other} is not a valid Metric"
                )))
            }
        };
        Ok(PyMetric(metric))
    }
}

impl From<Metric> for PyMetric {
    fn from(m: Metric) -> Self {
        PyMetric(m)
    }
}

impl From<PyMetric> for Metric {
    fn from(m: PyMetric) -> Self {
        m.0
    }
}

/// Brute-force vector store with thread-safe k-NN search.
#[pyclass(module = "vanedb", name = "FlatIndex")]
struct PyStore {
    inner: FlatIndex,
}

#[pymethods]
impl PyStore {
    #[new]
    #[pyo3(signature = (dim, metric=PyMetric(Metric::L2)))]
    fn new(#[pyo3(from_py_with = one_usize)] dim: usize, metric: PyMetric) -> PyResult<Self> {
        let inner = FlatIndex::new(dim, metric.into()).map_err(to_pyerr)?;
        Ok(Self { inner })
    }

    /// Add one vector. Accepts a 1-D float32 buffer (numpy) or any float sequence.
    fn add(
        &self,
        py: Python<'_>,
        #[pyo3(from_py_with = one_id)] id: u64,
        vector: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        reject_reentry(self)?;
        let v = vec_f32(vector)?;
        py.detach(|| self.inner.add(id, &v)).map_err(to_pyerr)
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
        reject_reentry(self)?;
        let ids = ids_u64(ids)?;
        let (rows, flat) = batch_f32(vectors, self.inner.dimension())?;
        check_batch_len(&ids, rows)?;
        py.detach(|| self.inner.add_batch(&ids, &flat))
            .map_err(to_pyerr)
    }

    /// k-NN search. Accepts a 1-D float32 buffer (numpy) or any float sequence.
    ///
    /// Choose one of filter, allow_ids, or deny_ids. ID lists must be sorted
    /// and unique and avoid a Python call per candidate. Predicates must be
    /// stable and synchronous, and must not access this index while search
    /// holds its read lock. Callback exceptions propagate to the caller.
    #[pyo3(signature = (query, k, *, filter=None, allow_ids=None, deny_ids=None))]
    fn search(
        &self,
        py: Python<'_>,
        query: &Bound<'_, PyAny>,
        #[pyo3(from_py_with = one_usize)] k: usize,
        filter: Option<&Bound<'_, PyAny>>,
        allow_ids: Option<&Bound<'_, PyAny>>,
        deny_ids: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Vec<(u64, f32)>> {
        let q = vec_f32(query)?;
        let py_filter = extract_py_filter(py, filter, allow_ids, deny_ids)?;
        run_search_with_filter(py, self, &py_filter, SearchParams::new(), |p| {
            self.inner.search_with(&q, k, p)
        })
    }

    /// The vector stored under `id`, or `None` when no vector is stored under
    /// it. A miss is a value, not an error (RFC 0011); `contains` is the
    /// cheaper probe when the vector is not needed.
    fn get(
        &self,
        py: Python<'_>,
        #[pyo3(from_py_with = one_id)] id: u64,
    ) -> PyResult<Option<Vec<f32>>> {
        reject_reentry(self)?;
        py.detach(|| self.inner.get(id)).map_err(to_pyerr)
    }

    /// The same operation as `get`, under the spelling `ApproxIndex` uses.
    /// Both exist so a program is not tied to one index type (#85).
    fn get_vector(
        &self,
        py: Python<'_>,
        #[pyo3(from_py_with = one_id)] id: u64,
    ) -> PyResult<Option<Vec<f32>>> {
        self.get(py, id)
    }

    /// Removes the vector stored under `id`. Raises `ValueError` when none is:
    /// a caller that removes what is not there has a bug, and `remove` is not
    /// named `get`.
    fn remove(&self, py: Python<'_>, #[pyo3(from_py_with = one_id)] id: u64) -> PyResult<()> {
        reject_reentry(self)?;
        py.detach(|| self.inner.remove(id)).map_err(to_pyerr)
    }

    fn contains(&self, py: Python<'_>, #[pyo3(from_py_with = one_id)] id: u64) -> PyResult<bool> {
        reject_reentry(self)?;
        Ok(py.detach(|| self.inner.contains(id)))
    }

    /// `len(index)` is the count and `not index` the emptiness test, as for
    /// any Python container.
    fn __len__(&self, py: Python<'_>) -> PyResult<usize> {
        reject_reentry(self)?;
        Ok(py.detach(|| self.inner.len()))
    }

    /// Number of vectors stored: an alias of `len(index)`, kept because it is
    /// the spelling C++ and JavaScript users type first (RFC 0011).
    fn size(&self, py: Python<'_>) -> PyResult<usize> {
        reject_reentry(self)?;
        Ok(py.detach(|| self.inner.len()))
    }

    /// The metric this index was built with.
    ///
    /// Use this to confirm the distance convention expected by queries.
    #[getter]
    fn metric(&self) -> PyMetric {
        self.inner.metric().into()
    }

    #[getter]
    fn dimension(&self) -> usize {
        self.inner.dimension()
    }
}

/// HNSW approximate nearest-neighbor index.
#[pyclass(module = "vanedb", name = "ApproxIndex")]
struct PyIndex {
    inner: ApproxIndex,
}

#[pymethods]
impl PyIndex {
    #[new]
    #[pyo3(signature = (dim, metric=PyMetric(Metric::L2), capacity=100000, m=16, ef_construction=200, seed=42))]
    fn new(
        #[pyo3(from_py_with = one_usize)] dim: usize,
        metric: PyMetric,
        #[pyo3(from_py_with = one_usize)] capacity: usize,
        #[pyo3(from_py_with = one_usize)] m: usize,
        #[pyo3(from_py_with = one_usize)] ef_construction: usize,
        #[pyo3(from_py_with = one_seed)] seed: u64,
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
        reject_reentry(self)?;
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
        reject_reentry(self)?;
        let ids = ids_u64(ids)?;
        let (rows, flat) = batch_f32(vectors, self.inner.dimension())?;
        check_batch_len(&ids, rows)?;
        py.detach(|| self.inner.add_batch(&ids, &flat))
            .map_err(to_pyerr)
    }

    /// k-NN search. Accepts a 1-D float32 buffer (numpy) or any float sequence.
    ///
    /// `ef_search` overrides the beam width for this query alone. Without it,
    /// raising recall for one query means assigning to the shared `ef_search`
    /// property -- and because every search releases the GIL, a concurrent
    /// thread can observe that mutation. The C ABI takes the same parameter
    /// per call for the same reason.
    ///
    /// Choose one of filter, allow_ids, or deny_ids. ID lists must be sorted
    /// and unique and avoid a Python call per candidate. Predicates must be
    /// stable and synchronous, and must not access this index while search
    /// holds its read lock. Callback exceptions propagate to the caller.
    /// max_ef_search limits beam widening, not the number of nodes visited.
    #[pyo3(signature = (query, k, *, ef_search=None, max_ef_search=None, filter=None, allow_ids=None, deny_ids=None))]
    #[allow(clippy::too_many_arguments)] // Public keyword-only search options.
    fn search(
        &self,
        py: Python<'_>,
        query: &Bound<'_, PyAny>,
        #[pyo3(from_py_with = one_usize)] k: usize,
        ef_search: Option<&Bound<'_, PyAny>>,
        max_ef_search: Option<&Bound<'_, PyAny>>,
        filter: Option<&Bound<'_, PyAny>>,
        allow_ids: Option<&Bound<'_, PyAny>>,
        deny_ids: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Vec<(u64, f32)>> {
        let q = vec_f32(query)?;
        let ef = ef_search.map(one_usize).transpose()?;
        let max_ef = max_ef_search.map(one_usize).transpose()?;
        let py_filter = extract_py_filter(py, filter, allow_ids, deny_ids)?;

        let mut base_params = ::vanedb::SearchParams::new();
        if let Some(e) = ef {
            base_params = base_params.ef_search(e);
        }
        if let Some(m) = max_ef {
            base_params = base_params.max_ef_search(m);
        }

        run_search_with_filter(py, self, &py_filter, base_params, |p| {
            self.inner.search_with(&q, k, p)
        })
    }

    /// The vector stored under `id`, or `None` when no vector is stored under
    /// it. A miss is a value, not an error (RFC 0011); `contains` is the
    /// cheaper probe when the vector is not needed.
    fn get_vector(
        &self,
        py: Python<'_>,
        #[pyo3(from_py_with = one_id)] id: u64,
    ) -> PyResult<Option<Vec<f32>>> {
        reject_reentry(self)?;
        py.detach(|| self.inner.get_vector(id)).map_err(to_pyerr)
    }

    /// The same operation as `get_vector`, under the spelling `FlatIndex` and
    /// `DiskIndex` use. Both exist so a program that outgrows an exact index
    /// does not have to rename every call site (#85).
    fn get(
        &self,
        py: Python<'_>,
        #[pyo3(from_py_with = one_id)] id: u64,
    ) -> PyResult<Option<Vec<f32>>> {
        self.get_vector(py, id)
    }

    fn contains(&self, py: Python<'_>, #[pyo3(from_py_with = one_id)] id: u64) -> PyResult<bool> {
        reject_reentry(self)?;
        Ok(py.detach(|| self.inner.contains(id)))
    }

    /// Writes the graph to `path`, which may be a `str` or any `os.PathLike`
    /// — `pathlib.Path` included.
    fn save(&self, py: Python<'_>, path: PathBuf) -> PyResult<()> {
        reject_reentry(self)?;
        py.detach(|| self.inner.save(&path)).map_err(to_pyerr)
    }

    /// Reads a graph written by `save`. Accepts the same path types.
    #[staticmethod]
    fn load(py: Python<'_>, path: PathBuf) -> PyResult<Self> {
        let inner = py.detach(|| ApproxIndex::load(&path)).map_err(to_pyerr)?;
        Ok(Self { inner })
    }

    /// Serializes the graph as a VNDB file. Compact first if tombstones
    /// should not be included.
    fn to_bytes(&self, py: Python<'_>) -> PyResult<Vec<u8>> {
        reject_reentry(self)?;
        py.detach(|| self.inner.to_bytes()).map_err(to_pyerr)
    }

    /// Reads a graph from a VNDB file (or a legacy Rust file) in memory.
    #[staticmethod]
    fn from_bytes(py: Python<'_>, data: Vec<u8>) -> PyResult<Self> {
        let inner = py
            .detach(move || ApproxIndex::from_bytes(&data))
            .map_err(to_pyerr)?;
        Ok(Self { inner })
    }

    /// The index's own search beam width: the default a `search` without
    /// `ef_search=` uses, and the value `save` writes into the file. Default
    /// 50.
    #[getter]
    fn ef_search(&self) -> usize {
        self.inner.ef_search()
    }

    #[setter]
    fn set_ef_search(&self, #[pyo3(from_py_with = one_usize)] ef: usize) {
        self.inner.set_ef_search(ef);
    }

    /// `len(index)` is the live count and `not index` the emptiness test, as
    /// for any Python container.
    fn __len__(&self, py: Python<'_>) -> PyResult<usize> {
        reject_reentry(self)?;
        Ok(py.detach(|| self.inner.len()))
    }

    /// Inserts `vector` under `id`, replacing any existing entry.
    ///
    /// Both halves happen under one write lock, so a concurrent reader never
    /// observes the id missing. The old slot is tombstoned, so a long upsert
    /// loop still needs `compact()`.
    fn upsert(
        &self,
        py: Python<'_>,
        #[pyo3(from_py_with = one_id)] id: u64,
        vector: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        reject_reentry(self)?;
        let v = vec_f32(vector)?;
        py.detach(|| self.inner.upsert(id, &v)).map_err(to_pyerr)
    }

    /// Number of tombstoned slots: removed vectors whose space is not yet
    /// reclaimed. Re-adding a removed id allocates a fresh slot, so an upsert
    /// loop grows this even at constant length.
    #[getter]
    fn tombstones(&self, py: Python<'_>) -> PyResult<usize> {
        reject_reentry(self)?;
        Ok(py.detach(|| self.inner.tombstones()))
    }

    /// Rebuilds the graph without tombstoned slots, reclaiming their space.
    ///
    /// A full rebuild, holding the write lock throughout, so concurrent
    /// searches block. Ids and vectors are preserved.
    fn compact(&self, py: Python<'_>) -> PyResult<()> {
        reject_reentry(self)?;
        py.detach(|| self.inner.compact()).map_err(to_pyerr)
    }

    /// Removes the vector stored under `id`. Raises `ValueError` when none is:
    /// a caller that removes what is not there has a bug, and `remove` is not
    /// named `get`.
    ///
    /// Tombstoned: the node keeps its graph links, which may be the only
    /// route between live neighbourhoods, and simply stops appearing in
    /// results. The id becomes free for reuse. Space is not reclaimed.
    fn remove(&self, py: Python<'_>, #[pyo3(from_py_with = one_id)] id: u64) -> PyResult<()> {
        reject_reentry(self)?;
        py.detach(|| self.inner.remove(id)).map_err(to_pyerr)
    }

    /// Number of vectors in the graph: an alias of `len(index)`. See
    /// `FlatIndex.size`.
    fn size(&self, py: Python<'_>) -> PyResult<usize> {
        reject_reentry(self)?;
        Ok(py.detach(|| self.inner.len()))
    }

    /// The metric this index was built with.
    ///
    /// Worth having on a loaded index: `ApproxIndex.load` reads the metric out
    /// of the file, and without this the caller cannot check that their query
    /// convention matches.
    #[getter]
    fn metric(&self) -> PyMetric {
        self.inner.metric().into()
    }

    #[getter]
    fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    #[getter]
    fn capacity(&self, py: Python<'_>) -> PyResult<usize> {
        reject_reentry(self)?;
        Ok(py.detach(|| self.inner.capacity()))
    }

    /// The graph's `M`.
    ///
    /// Worth having for the same reason as `metric`: `ApproxIndex.load` reads
    /// this from the file, and a caller who did not build the graph has no
    /// other way to know what they are searching.
    #[getter]
    fn m(&self) -> usize {
        self.inner.m()
    }

    /// The `ef_construction` the graph was built with.
    #[getter]
    fn ef_construction(&self) -> usize {
        self.inner.ef_construction()
    }

    /// The seed the graph was built with. Two indexes built from the same
    /// vectors with the same seed have the same topology.
    #[getter]
    fn seed(&self) -> u64 {
        self.inner.seed()
    }
}

/// Builds a `DiskIndex` file. Vectors are held in memory until `save`; the
/// memory saving is on the reading side.
#[pyclass(module = "vanedb", name = "DiskIndexBuilder")]
struct PyDiskStoreBuilder {
    // The core builder takes `&mut self`, which PyO3 turns into a runtime
    // borrow. That was safe only because the GIL serialised every call —
    // releasing it around `add` below is exactly what would let two borrows
    // overlap and raise `RuntimeError: Already borrowed`. The lock is what
    // makes the GIL release safe, and it also makes this the last of the five
    // classes to take `&self`, matching how the index types synchronise.
    //
    // An RwLock rather than a Mutex because only `add` needs `&mut` in the
    // core; `save`, `size` and `dimension` take `&self` there, so they have no
    // reason to exclude each other.
    inner: parking_lot::RwLock<DiskIndexBuilder>,
}

#[pymethods]
impl PyDiskStoreBuilder {
    #[new]
    #[pyo3(signature = (dim, metric=PyMetric(Metric::L2)))]
    fn new(#[pyo3(from_py_with = one_usize)] dim: usize, metric: PyMetric) -> PyResult<Self> {
        Ok(Self {
            inner: parking_lot::RwLock::new(
                DiskIndexBuilder::new(dim, metric.into()).map_err(to_pyerr)?,
            ),
        })
    }

    fn add(
        &self,
        py: Python<'_>,
        #[pyo3(from_py_with = one_id)] id: u64,
        vector: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let v = vec_f32(vector)?;
        py.detach(|| self.inner.write().add(id, &v))
            .map_err(to_pyerr)
    }

    /// Writes the store to `path`, atomically: built beside the destination
    /// and renamed in after an fsync. `path` may be a `str` or any
    /// `os.PathLike` — `pathlib.Path` included.
    fn save(&self, py: Python<'_>, path: PathBuf) -> PyResult<()> {
        py.detach(|| self.inner.read().save(&path))
            .map_err(to_pyerr)
    }

    fn __len__(&self, py: Python<'_>) -> usize {
        py.detach(|| self.inner.read().len())
    }

    /// Number of vectors collected so far: an alias of `len(builder)`. See
    /// `FlatIndex.size`.
    fn size(&self, py: Python<'_>) -> usize {
        py.detach(|| self.inner.read().len())
    }

    #[getter]
    fn dimension(&self, py: Python<'_>) -> usize {
        py.detach(|| self.inner.read().dimension())
    }
}

/// Exact search over a memory-mapped file. Read-only; build one with
/// `DiskIndexBuilder`.
#[pyclass(module = "vanedb", name = "DiskIndex")]
struct PyDiskStore {
    inner: DiskIndex,
}

#[pymethods]
impl PyDiskStore {
    /// Maps the store at `path`.
    ///
    /// Validates the header and every stored value, so this is linear in the
    /// corpus rather than a constant-cost mapping.
    ///
    /// `path` may be a `str` or any `os.PathLike` — `pathlib.Path` included.
    ///
    /// The caller must keep the underlying file's contents and length unchanged,
    /// in every process, from before this call until the index is released.
    /// In-place writes or truncation can cause undefined behavior or crash the
    /// process. `DiskIndexBuilder.save` safely replaces the file atomically,
    /// leaving existing mappings intact.
    #[staticmethod]
    fn open(py: Python<'_>, path: PathBuf) -> PyResult<Self> {
        // SAFETY: the Python caller must uphold the external file immutability
        // requirement documented above; the binding cannot enforce it.
        let inner = py
            .detach(|| unsafe { DiskIndex::open(&path) })
            .map_err(to_pyerr)?;
        Ok(Self { inner })
    }

    /// Exact search with one of filter, allow_ids, or deny_ids. ID lists must
    /// be sorted and unique. A Python predicate is called per candidate;
    /// prefer lists when possible. Callback exceptions propagate to the caller.
    /// Predicates must be stable and synchronous; consult external metadata
    /// instead of accessing this same index from the callback.
    #[pyo3(signature = (query, k, *, filter=None, allow_ids=None, deny_ids=None))]
    fn search(
        &self,
        py: Python<'_>,
        query: &Bound<'_, PyAny>,
        #[pyo3(from_py_with = one_usize)] k: usize,
        filter: Option<&Bound<'_, PyAny>>,
        allow_ids: Option<&Bound<'_, PyAny>>,
        deny_ids: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Vec<(u64, f32)>> {
        let q = vec_f32(query)?;
        let py_filter = extract_py_filter(py, filter, allow_ids, deny_ids)?;
        run_search_with_filter(py, self, &py_filter, SearchParams::new(), |p| {
            self.inner.search_with(&q, k, p)
        })
    }

    /// The vector stored under `id`, or `None` when no vector is stored under
    /// it. A miss is a value, not an error (RFC 0011); `contains` is the
    /// cheaper probe when the vector is not needed.
    fn get(
        &self,
        py: Python<'_>,
        #[pyo3(from_py_with = one_id)] id: u64,
    ) -> PyResult<Option<Vec<f32>>> {
        reject_reentry(self)?;
        // Reads through the mapping, which can take a major page fault on a
        // cold file — the same reason `search` detaches.
        py.detach(|| self.inner.get(id).map(|v| v.map(Cow::into_owned)))
            .map_err(to_pyerr)
    }

    /// The same operation as `get`, under the spelling `ApproxIndex` uses.
    /// Both exist so a program is not tied to one index type (#85).
    fn get_vector(
        &self,
        py: Python<'_>,
        #[pyo3(from_py_with = one_id)] id: u64,
    ) -> PyResult<Option<Vec<f32>>> {
        self.get(py, id)
    }

    fn contains(&self, py: Python<'_>, #[pyo3(from_py_with = one_id)] id: u64) -> PyResult<bool> {
        reject_reentry(self)?;
        Ok(py.detach(|| self.inner.contains(id)))
    }

    fn __len__(&self) -> PyResult<usize> {
        reject_reentry(self)?;
        Ok(self.inner.len())
    }

    /// Number of vectors in the mapped file: an alias of `len(index)`. See
    /// `FlatIndex.size`.
    fn size(&self) -> PyResult<usize> {
        reject_reentry(self)?;
        Ok(self.inner.len())
    }

    /// The metric this index was built with.
    ///
    /// Worth having on a loaded index: `open` reads the metric out of the
    /// file, and without this the caller cannot check that their query
    /// convention matches.
    #[getter]
    fn metric(&self) -> PyMetric {
        self.inner.metric().into()
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
    // `Metric` is defined in `python/vanedb/__init__.py` as an `enum.IntEnum`
    // (see `PyMetric`), so it is not a class of this module. `__all__` below
    // is what `__init__` star-imports, and it adds `Metric` itself.
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
            "FlatIndex",
            "ApproxIndex",
            "DiskIndex",
            "DiskIndexBuilder",
            "__version__",
        ],
    )?;
    Ok(())
}
