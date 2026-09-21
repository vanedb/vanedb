use js_sys::{BigInt, Function, Reflect};
use std::cell::RefCell;
use std::sync::{Arc, Mutex};
use wasm_bindgen::prelude::*;

use vanedb::approx::{ApproxIndex, Filter, SearchParams};
use vanedb::distance::Metric;
use vanedb::flat::{FlatIndex, SearchResult};

/// Search results with lossless 64-bit ids.
// The Wasm prefix disambiguates these wrappers from the core types
// imported above; js_name keeps it out of the public JS API, which uses
// the same three names as the Python packages (#83).
#[wasm_bindgen(js_name = SearchResults)]
pub struct WasmSearchResults {
    ids: Vec<u64>,
    distances: Vec<f32>,
}

#[wasm_bindgen(js_class = SearchResults)]
impl WasmSearchResults {
    /// Matched ids, in rank order, as a `BigUint64Array`.
    #[wasm_bindgen(getter)]
    pub fn ids(&self) -> Vec<u64> {
        self.ids.clone()
    }

    /// Distances, parallel to `ids`, as a `Float32Array`.
    #[wasm_bindgen(getter)]
    pub fn distances(&self) -> Vec<f32> {
        self.distances.clone()
    }

    /// Number of matches returned.
    #[wasm_bindgen(getter)]
    pub fn length(&self) -> usize {
        self.ids.len()
    }
}

impl From<Vec<SearchResult>> for WasmSearchResults {
    fn from(results: Vec<SearchResult>) -> Self {
        let mut ids = Vec::with_capacity(results.len());
        let mut distances = Vec::with_capacity(results.len());
        for result in results {
            ids.push(result.id);
            distances.push(result.distance);
        }
        Self { ids, distances }
    }
}

fn to_jserr(e: vanedb::VaneError) -> JsError {
    JsError::new(&e.to_string())
}

// Accept the JavaScript bigint before the Wasm i64 boundary can wrap it.
fn one_id(id: BigInt) -> Result<u64, JsError> {
    u64::try_from(id).map_err(|_| JsError::new("id must be between 0 and 2**64 - 1"))
}

fn extract_id_list(val: &JsValue, name: &str) -> Result<Vec<u64>, JsValue> {
    if !js_sys::Array::is_array(val) && !val.is_instance_of::<js_sys::BigUint64Array>() {
        return Err(JsError::new(&format!(
            "{name} must be an array of IDs or a BigUint64Array"
        ))
        .into());
    }
    let iter =
        js_sys::try_iter(val)?.ok_or_else(|| JsError::new(&format!("{name} must be iterable")))?;
    let mut ids = Vec::new();
    for item in iter {
        let item = item?;
        let id_val = if item.is_bigint() {
            one_id(BigInt::from(item))?
        } else if let Some(n) = item.as_f64() {
            // IDs are u64, not Wasm usize counts. Number inputs are lossless
            // only through MAX_SAFE_INTEGER; use BigInt for larger IDs.
            if !n.is_finite() || n.fract() != 0.0 || !(0.0..=9_007_199_254_740_991.0).contains(&n) {
                return Err(JsError::new(
                    "id must be a nonnegative safe integer or a uint64 BigInt",
                )
                .into());
            }
            n as u64
        } else {
            return Err(
                JsError::new(&format!("{name} elements must be Numbers or BigInts")).into(),
            );
        };
        ids.push(id_val);
    }
    Ok(ids)
}

thread_local! {
    /// Indexes whose predicate search is running, by address.
    ///
    /// A predicate runs under the searched index's read lock. Calling back
    /// into that index from the callback would block on that lock, and in a
    /// single-threaded wasm module nothing can ever release it: the page
    /// hangs with no diagnostic. Every method that takes the lock checks this
    /// list first and throws instead. A nested search on a different index is
    /// supported, hence a list rather than a flag.
    static IN_PREDICATE_SEARCH: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
}

/// Marks `index` as searching with a predicate until dropped.
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

/// The `code` property on the error thrown for a re-entrant call, so callers
/// can branch on it without matching the message.
pub const REENTRANT_SEARCH_CODE: &str = "ERR_REENTRANT_SEARCH";

/// Throws instead of deadlocking when a predicate calls back into the index
/// it is filtering. The thrown value is an `Error` whose `code` is
/// [`REENTRANT_SEARCH_CODE`].
fn reject_reentry<T>(index: &T) -> Result<(), JsValue> {
    let addr = index as *const T as usize;
    if IN_PREDICATE_SEARCH.with(|s| s.borrow().contains(&addr)) {
        let error = js_sys::Error::new(
            "a filter predicate must not call methods on the index being searched: \
             the search holds its read lock; consult external metadata or search \
             a different index instead",
        );
        // Reflect::set fails only on a frozen or exotic target; a fresh Error
        // is neither, and the message alone still identifies the failure.
        let _ = Reflect::set(
            &error,
            &JsValue::from_str("code"),
            &JsValue::from_str(REENTRANT_SEARCH_CODE),
        );
        return Err(error.into());
    }
    Ok(())
}

struct ParsedFilter {
    allow: Option<Vec<u64>>,
    deny: Option<Vec<u64>>,
    predicate: Option<Function>,
}

fn filter_from_parsed<'a>(
    parsed: Option<&'a ParsedFilter>,
    pred_holder: &'a mut Option<Box<dyn Fn(u64) -> bool + Sync>>,
    error: &Option<Arc<Mutex<Option<JsValue>>>>,
) -> Option<Filter<'a>> {
    if let Some(p) = parsed {
        if let Some(ref allow) = p.allow {
            return Some(Filter::Allow(allow.as_slice()));
        } else if let Some(ref deny) = p.deny {
            return Some(Filter::Deny(deny.as_slice()));
        } else if let Some(ref func) = p.predicate {
            let f_clone = func.clone();
            let error = Arc::clone(error.as_ref().expect("predicate error slot"));
            *pred_holder = Some(Box::new(move |id: u64| {
                if error.lock().unwrap().is_some() {
                    return false;
                }
                let js_id = JsValue::from(BigInt::from(id));
                let res = f_clone.call1(&JsValue::NULL, &js_id);
                match res {
                    Ok(v) => v.is_truthy(),
                    Err(err) => {
                        *error.lock().unwrap() = Some(err);
                        false
                    }
                }
            }));
            return Some(Filter::Predicate(pred_holder.as_ref().unwrap().as_ref()));
        }
    }
    None
}

fn parse_filter_options(options: &JsValue) -> Result<Option<ParsedFilter>, JsValue> {
    if options.is_undefined() || options.is_null() {
        return Ok(None);
    }
    if !options.is_object() {
        return Err(JsError::new("search options must be an object").into());
    }

    let allow_val = Reflect::get(options, &JsValue::from_str("allow"))?;
    let allow = if !allow_val.is_undefined() && !allow_val.is_null() {
        Some(extract_id_list(&allow_val, "allow")?)
    } else {
        None
    };

    let deny_val = Reflect::get(options, &JsValue::from_str("deny"))?;
    let deny = if !deny_val.is_undefined() && !deny_val.is_null() {
        Some(extract_id_list(&deny_val, "deny")?)
    } else {
        None
    };

    let pred_val = Reflect::get(options, &JsValue::from_str("predicate"))?;
    let predicate = if !pred_val.is_undefined() && !pred_val.is_null() {
        if pred_val.is_function() {
            Some(Function::from(pred_val))
        } else {
            return Err(JsError::new("predicate must be a function").into());
        }
    } else {
        None
    };

    if usize::from(allow.is_some()) + usize::from(deny.is_some()) + usize::from(predicate.is_some())
        > 1
    {
        return Err(JsError::new("specify at most one of allow, deny, or predicate").into());
    }

    if allow.is_none() && deny.is_none() && predicate.is_none() {
        return Ok(None);
    }

    Ok(Some(ParsedFilter {
        allow,
        deny,
        predicate,
    }))
}

fn optional_count(options: &JsValue, name: &str) -> Result<Option<usize>, JsValue> {
    let value = Reflect::get(options, &JsValue::from_str(name))?;
    if value.is_undefined() || value.is_null() {
        return Ok(None);
    }
    let number = value
        .as_f64()
        .ok_or_else(|| JsError::new(&format!("{name} must be a number")))?;
    Ok(Some(count(number, name)?))
}

#[wasm_bindgen(typescript_custom_section)]
const SEARCH_OPTIONS: &'static str = r#"
export interface SearchFilterOptions {
    /** Sorted, unique IDs; Number IDs must be nonnegative safe integers. */
    allow?: (number | bigint)[] | BigUint64Array;
    /** Mutually exclusive with allow and predicate. */
    deny?: (number | bigint)[] | BigUint64Array;
    /** Stable synchronous predicate, called per candidate (possibly repeatedly).
     * Slower than ID lists; must not access, mutate, or free this same index.
     * Exceptions propagate from search with the original thrown value. */
    predicate?: (id: bigint) => boolean;
}
export interface ApproxSearchOptions extends SearchFilterOptions {
    efSearch?: number;
    maxEfSearch?: number;
}
"#;

// Preserve the number until validation: Wasm's i32 boundary would silently
// truncate fractions and wrap negative or oversized JavaScript numbers.
/// The seed this crate has always used when none is supplied. Named so the
/// default is one value rather than a literal repeated in code and docs.
const DEFAULT_SEED: u64 = 42;

fn count(value: f64, name: &str) -> Result<usize, JsError> {
    if !value.is_finite() || value.fract() != 0.0 || !(0.0..=u32::MAX as f64).contains(&value) {
        return Err(JsError::new(&format!(
            "{name} must be an integer between 0 and 4294967295"
        )));
    }
    Ok(value as usize)
}

/// The spelling `parse_metric` accepts, so a reported metric can be fed
/// straight back into a constructor.
fn metric_name(m: Metric) -> &'static str {
    match m {
        Metric::Cosine => "cosine",
        Metric::Dot => "dot",
        // Metric is #[non_exhaustive]; a variant this binding cannot parse
        // cannot reach an index built through it.
        _ => "l2",
    }
}

/// A non-string `metric` used to trap inside wasm-bindgen's string
/// marshalling with `RuntimeError: memory access out of bounds` -- an opaque VM
/// fault where every other bad argument to these constructors throws a
/// descriptive `Error`. Taking a `JsValue` and checking it here keeps the
/// failure in JavaScript.
fn metric_from_value(metric: &JsValue) -> Result<Metric, JsError> {
    match metric.as_string() {
        Some(name) => parse_metric(&name),
        None => Err(JsError::new(
            "metric must be a string: 'l2', 'cosine', or 'dot'",
        )),
    }
}

fn parse_metric(metric: &str) -> Result<Metric, JsError> {
    match metric {
        "l2" | "L2" => Ok(Metric::L2),
        "cosine" | "Cosine" => Ok(Metric::Cosine),
        "dot" | "Dot" => Ok(Metric::Dot),
        _ => Err(JsError::new(&format!(
            "unknown metric: {metric}. Use 'l2', 'cosine', or 'dot'"
        ))),
    }
}

#[wasm_bindgen]
pub fn version() -> String {
    // From Cargo.toml, never a literal: a hardcoded string drifts silently,
    // and so does a test that pins the same literal.
    env!("CARGO_PKG_VERSION").to_string()
}

/// Brute-force vector store for the browser.
#[wasm_bindgen(js_name = FlatIndex)]
pub struct WasmStore {
    inner: FlatIndex,
}

#[wasm_bindgen(js_class = FlatIndex)]
impl WasmStore {
    #[wasm_bindgen(constructor)]
    pub fn new(
        dim: f64,
        #[wasm_bindgen(unchecked_param_type = "string")] metric: &JsValue,
    ) -> Result<WasmStore, JsError> {
        let m = metric_from_value(metric)?;
        let inner = FlatIndex::new(count(dim, "dimension")?, m).map_err(to_jserr)?;
        Ok(Self { inner })
    }

    pub fn add(&self, id: BigInt, vector: &[f32]) -> Result<(), JsValue> {
        reject_reentry(self)?;
        Ok(self.inner.add(one_id(id)?, vector).map_err(to_jserr)?)
    }

    /// Bulk insert in one wasm call: `ids` is a BigUint64Array of n ids and
    /// `vectors` a Float32Array of n × dim values (row-major). All-or-nothing:
    /// on error the store is unchanged.
    pub fn add_batch(&self, ids: &[u64], vectors: &[f32]) -> Result<(), JsValue> {
        reject_reentry(self)?;
        Ok(self.inner.add_batch(ids, vectors).map_err(to_jserr)?)
    }

    /// Search for k nearest neighbors, with optional filter options.
    ///
    /// Ids come back as a `BigUint64Array` and distances as a `Float32Array`,
    /// parallel by index.
    /// Choose exactly one filter. ID lists must be sorted and unique.
    /// Predicates must be synchronous and stable and must not access this
    /// same index while search holds its read lock. Exceptions propagate.
    pub fn search(
        &self,
        query: &[f32],
        k: f64,
        #[wasm_bindgen(unchecked_optional_param_type = "SearchFilterOptions | null")]
        options: Option<JsValue>,
    ) -> Result<WasmSearchResults, JsValue> {
        let k = count(k, "k")?;
        let parsed = match options {
            Some(ref opt) => parse_filter_options(opt)?,
            None => None,
        };

        let callback_error = parsed
            .as_ref()
            .filter(|p| p.predicate.is_some())
            .map(|_| Arc::new(Mutex::new(None)));
        let mut pred_closure = None;
        let filter = filter_from_parsed(parsed.as_ref(), &mut pred_closure, &callback_error);

        let mut params = SearchParams::new();
        if let Some(f) = filter {
            params = params.filter(f);
        }

        reject_reentry(self)?;
        let _scope = callback_error
            .is_some()
            .then(|| PredicateScope::enter(self));
        let results = self.inner.search_with(query, k, &params);
        if let Some(error) = callback_error
            .as_ref()
            .and_then(|e| e.lock().unwrap().take())
        {
            return Err(error);
        }
        Ok(WasmSearchResults::from(results.map_err(to_jserr)?))
    }

    /// The vector stored under `id`, as a `Float32Array`, or `undefined` when
    /// no vector is stored under it. A miss is a value, not an error (RFC
    /// 0011); an id outside the unsigned 64-bit range still throws.
    pub fn get(&self, id: BigInt) -> Result<Option<Vec<f32>>, JsValue> {
        reject_reentry(self)?;
        Ok(self.inner.get(one_id(id)?).map_err(to_jserr)?)
    }

    /// The same operation as `get`, under the spelling `ApproxIndex` also
    /// accepts. Both exist on both index types so a program is not tied to one
    /// (#85) — `ApproxIndex` had the pair and `FlatIndex` only `get`, so the
    /// one swap the pair exists for was the one that broke.
    pub fn get_vector(&self, id: BigInt) -> Result<Option<Vec<f32>>, JsValue> {
        reject_reentry(self)?;
        self.get(id)
    }

    pub fn remove(&self, id: BigInt) -> Result<(), JsValue> {
        reject_reentry(self)?;
        Ok(self.inner.remove(one_id(id)?).map_err(to_jserr)?)
    }

    pub fn contains(&self, id: BigInt) -> Result<bool, JsValue> {
        reject_reentry(self)?;
        Ok(self.inner.contains(one_id(id)?))
    }

    /// Number of vectors. `size()` is the Map/Set spelling JavaScript
    /// callers expect (RFC 0011); `size() === 0` is the emptiness test.
    pub fn size(&self) -> Result<usize, JsValue> {
        reject_reentry(self)?;
        Ok(self.inner.len())
    }

    /// The metric this index was built with, in the spelling the constructor
    /// accepts.
    pub fn metric(&self) -> String {
        metric_name(self.inner.metric()).to_string()
    }

    pub fn dimension(&self) -> usize {
        self.inner.dimension()
    }
}

/// HNSW approximate nearest-neighbor index for the browser.
#[wasm_bindgen(js_name = ApproxIndex)]
pub struct WasmIndex {
    inner: ApproxIndex,
}

#[wasm_bindgen(js_class = ApproxIndex)]
impl WasmIndex {
    /// Removes the vector stored under `id`. Tombstoned: the node keeps its
    /// graph links, which may be the only route between live neighbourhoods,
    /// and simply stops appearing in results. `tombstones` counts them and
    /// `compact` reclaims them.
    ///
    /// Takes `&self` like every other mutator on this type. With `&mut self`,
    /// wasm-bindgen gives a JS caller holding any other borrow of the object
    /// "recursive use of an object detected" rather than a deletion.
    pub fn remove(&self, id: BigInt) -> Result<(), JsValue> {
        reject_reentry(self)?;
        Ok(self.inner.remove(one_id(id)?).map_err(to_jserr)?)
    }

    /// Replaces the vector stored under `id`, inserting it if absent.
    ///
    /// One locked operation rather than `remove` then `add`. Those two can
    /// fail between the halves and leave the id deleted, and any reader
    /// running against the same memory sees a window where it is missing;
    /// this has neither. A replaced slot is tombstoned like any other
    /// removal, so `tombstones` counts it and `compact` reclaims it.
    pub fn upsert(&self, id: BigInt, vector: &[f32]) -> Result<(), JsValue> {
        reject_reentry(self)?;
        Ok(self.inner.upsert(one_id(id)?, vector).map_err(to_jserr)?)
    }

    /// How many removed slots the graph still carries.
    ///
    /// A browser is the most memory-constrained runtime this crate targets, and
    /// a tombstone holds its vector and links until compaction. Without this a
    /// caller could delete but could not tell what deleting had cost.
    pub fn tombstones(&self) -> Result<usize, JsValue> {
        reject_reentry(self)?;
        Ok(self.inner.tombstones())
    }

    /// Rebuilds the graph without its tombstoned slots, reclaiming their
    /// memory. Live ids and their vectors are preserved; only the removed
    /// slots go. Cost is a full rebuild, so call it when churn has accumulated
    /// rather than after each removal.
    pub fn compact(&self) -> Result<(), JsValue> {
        reject_reentry(self)?;
        Ok(self.inner.compact().map_err(to_jserr)?)
    }

    /// `seed` is optional and defaults to 42, the value this constructor used
    /// to hardcode. Supplying it makes construction reproducible: two indexes
    /// built from the same vectors with the same seed have the same topology.
    #[wasm_bindgen(constructor)]
    pub fn new(
        dim: f64,
        #[wasm_bindgen(unchecked_param_type = "string")] metric: &JsValue,
        capacity: f64,
        m: f64,
        ef_construction: f64,
        seed: Option<f64>,
    ) -> Result<WasmIndex, JsError> {
        let met = metric_from_value(metric)?;
        // Ids beyond 2^53 are not exactly representable as f64, so a seed
        // arrives through the same numeric gate as every other count rather
        // than being cast silently.
        let seed = match seed {
            Some(value) => count(value, "seed")? as u64,
            None => DEFAULT_SEED,
        };
        let inner = ApproxIndex::builder(count(dim, "dimension")?, met)
            .capacity(count(capacity, "capacity")?)
            .m(count(m, "m")?)
            .ef_construction(count(ef_construction, "ef_construction")?)
            .seed(seed)
            .build()
            .map_err(to_jserr)?;
        Ok(Self { inner })
    }

    pub fn add(&self, id: BigInt, vector: &[f32]) -> Result<(), JsValue> {
        reject_reentry(self)?;
        Ok(self.inner.add(one_id(id)?, vector).map_err(to_jserr)?)
    }

    /// Bulk insert in one wasm call: `ids` is a BigUint64Array of n ids and
    /// `vectors` a Float32Array of n × dim values (row-major). All-or-nothing:
    /// on error the index is unchanged.
    pub fn add_batch(&self, ids: &[u64], vectors: &[f32]) -> Result<(), JsValue> {
        reject_reentry(self)?;
        Ok(self.inner.add_batch(ids, vectors).map_err(to_jserr)?)
    }

    /// Search for k nearest neighbors.
    ///
    /// Ids come back as a `BigUint64Array` and distances as a `Float32Array`,
    /// parallel by index. Ids are never narrowed to `f32`: values at or above
    /// 2^24 are not exactly representable, so distinct records collided and
    /// callers could act on the wrong record (#39).
    ///
    /// A numeric third argument widens the beam for this query alone and
    /// leaves the index's own `efSearch` untouched. The property is shared
    /// state, so raising it to rescue one hard query silently pays for it on
    /// every later one; this is the way to spend that cost once. Measure
    /// recall and latency on your own data when choosing one.
    ///
    /// `options` may be either a number (`efSearch`), or an options object containing
    /// `{ efSearch?: number, maxEfSearch?: number, allow?: number[] | bigint[], deny?: number[] | bigint[], predicate?: (id: bigint) => boolean }`.
    /// Choose exactly one filter; ID lists must be sorted and unique.
    /// Predicates run per candidate and may run more than once for an ID.
    /// They must be stable and synchronous and must not access this same
    /// index while search holds its read lock. Exceptions propagate.
    /// `maxEfSearch` caps beam widening, not the number of visited nodes.
    // The parameter name reaches the TypeScript declarations verbatim, so it
    // is spelled the way RFC 0004 and the JavaScript surface spell it.
    #[allow(non_snake_case)]
    pub fn search(
        &self,
        query: &[f32],
        k: f64,
        #[wasm_bindgen(unchecked_optional_param_type = "number | ApproxSearchOptions | null")]
        efSearchOrOptions: Option<JsValue>,
    ) -> Result<WasmSearchResults, JsValue> {
        let k = count(k, "k")?;
        let mut params = SearchParams::new();

        let parsed = match efSearchOrOptions {
            Some(ref val) if val.is_object() => {
                if let Some(ef) = optional_count(val, "efSearch")? {
                    params = params.ef_search(ef);
                }
                if let Some(max_ef) = optional_count(val, "maxEfSearch")? {
                    params = params.max_ef_search(max_ef);
                }

                parse_filter_options(val)?
            }
            Some(ref val) => {
                if let Some(num) = val.as_f64() {
                    params = params.ef_search(count(num, "efSearch")?);
                } else if !val.is_null() && !val.is_undefined() {
                    return Err(JsError::new("search options must be a number or an object").into());
                }
                None
            }
            None => None,
        };

        let callback_error = parsed
            .as_ref()
            .filter(|p| p.predicate.is_some())
            .map(|_| Arc::new(Mutex::new(None)));
        let mut pred_closure = None;
        let filter = filter_from_parsed(parsed.as_ref(), &mut pred_closure, &callback_error);

        if let Some(f) = filter {
            params = params.filter(f);
        }

        reject_reentry(self)?;
        let _scope = callback_error
            .is_some()
            .then(|| PredicateScope::enter(self));
        let results = self.inner.search_with(query, k, &params);
        if let Some(error) = callback_error
            .as_ref()
            .and_then(|e| e.lock().unwrap().take())
        {
            return Err(error);
        }
        Ok(WasmSearchResults::from(results.map_err(to_jserr)?))
    }

    pub fn contains(&self, id: BigInt) -> Result<bool, JsValue> {
        reject_reentry(self)?;
        Ok(self.inner.contains(one_id(id)?))
    }

    /// The vector stored under `id`, as a `Float32Array`, or `undefined` when
    /// no vector is stored under it. A miss is a value, not an error (RFC
    /// 0011); an id outside the unsigned 64-bit range still throws.
    pub fn get_vector(&self, id: BigInt) -> Result<Option<Vec<f32>>, JsValue> {
        reject_reentry(self)?;
        Ok(self.inner.get_vector(one_id(id)?).map_err(to_jserr)?)
    }

    /// The same operation as `get_vector`, under the spelling `FlatIndex` uses.
    /// Both exist so a program is not tied to one index type (#85).
    pub fn get(&self, id: BigInt) -> Result<Option<Vec<f32>>, JsValue> {
        reject_reentry(self)?;
        self.get_vector(id)
    }

    /// Number of live vectors. `size()` is the Map/Set spelling JavaScript
    /// callers expect (RFC 0011); `size() === 0` is the emptiness test.
    pub fn size(&self) -> Result<usize, JsValue> {
        reject_reentry(self)?;
        Ok(self.inner.len())
    }

    /// The metric this index was built with, in the spelling the constructor
    /// accepts.
    pub fn metric(&self) -> String {
        metric_name(self.inner.metric()).to_string()
    }

    pub fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    /// The graph's `M`.
    pub fn m(&self) -> usize {
        self.inner.m()
    }

    /// The `ef_construction` the graph was built with.
    pub fn ef_construction(&self) -> usize {
        self.inner.ef_construction()
    }

    /// The seed the graph was built with.
    pub fn seed(&self) -> u64 {
        self.inner.seed()
    }

    /// The capacity hint the graph was built with. Not a limit: the index
    /// grows past it, so this may be smaller than `size`.
    pub fn capacity(&self) -> Result<usize, JsValue> {
        reject_reentry(self)?;
        Ok(self.inner.capacity())
    }

    /// The index's own search beam width: the default a `search` without a
    /// per-query width uses, and the value `toBytes` writes into the file.
    /// Default 50. Exposed as the `efSearch` property, in the camelCase the
    /// search options object already uses (RFC 0011).
    #[wasm_bindgen(getter, js_name = efSearch)]
    pub fn ef_search(&self) -> usize {
        self.inner.ef_search()
    }

    /// Sets `efSearch`. Rejects anything but an integer between 0 and
    /// 4294967295, leaving the current value in place.
    #[wasm_bindgen(setter, js_name = efSearch)]
    pub fn set_ef_search(&self, ef: f64) -> Result<(), JsError> {
        self.inner.set_ef_search(count(ef, "efSearch")?);
        Ok(())
    }

    /// Serializes the graph as a VNDB file. The returned `Uint8Array` is a
    /// copy out of wasm memory; its length equals the file size. Compact
    /// first if tombstones should not be written.
    #[wasm_bindgen(js_name = toBytes)]
    pub fn to_bytes(&self) -> Result<Vec<u8>, JsValue> {
        reject_reentry(self)?;
        Ok(self.inner.to_bytes().map_err(to_jserr)?)
    }

    /// Reads a VNDB graph — or a legacy Rust file — from `bytes`.
    #[wasm_bindgen(js_name = fromBytes)]
    pub fn from_bytes(bytes: &[u8]) -> Result<WasmIndex, JsError> {
        Ok(Self {
            inner: ApproxIndex::from_bytes(bytes).map_err(to_jserr)?,
        })
    }
}
