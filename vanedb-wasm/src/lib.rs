use js_sys::BigInt;
use wasm_bindgen::prelude::*;

use vanedb::approx::ApproxIndex;
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
    pub fn new(dim: f64, metric: &str) -> Result<WasmStore, JsError> {
        let m = parse_metric(metric)?;
        let inner = FlatIndex::new(count(dim, "dimension")?, m).map_err(to_jserr)?;
        Ok(Self { inner })
    }

    pub fn add(&self, id: BigInt, vector: &[f32]) -> Result<(), JsError> {
        self.inner.add(one_id(id)?, vector).map_err(to_jserr)
    }

    /// Bulk insert in one wasm call: `ids` is a BigUint64Array of n ids and
    /// `vectors` a Float32Array of n × dim values (row-major). All-or-nothing:
    /// on error the store is unchanged.
    pub fn add_batch(&self, ids: &[u64], vectors: &[f32]) -> Result<(), JsError> {
        self.inner.add_batch(ids, vectors).map_err(to_jserr)
    }

    /// Search for k nearest neighbors.
    ///
    /// Ids come back as a `BigUint64Array` and distances as a `Float32Array`,
    /// parallel by index. Ids are never narrowed to `f32`: values at or above
    /// 2^24 are not exactly representable, so distinct records collided and
    /// callers could act on the wrong record (#39).
    pub fn search(&self, query: &[f32], k: f64) -> Result<WasmSearchResults, JsError> {
        let results = self.inner.search(query, count(k, "k")?).map_err(to_jserr)?;
        Ok(WasmSearchResults::from(results))
    }

    pub fn get(&self, id: BigInt) -> Result<Vec<f32>, JsError> {
        self.inner.get(one_id(id)?).map_err(to_jserr)
    }

    pub fn remove(&self, id: BigInt) -> Result<(), JsError> {
        self.inner.remove(one_id(id)?).map_err(to_jserr)
    }

    pub fn contains(&self, id: BigInt) -> Result<bool, JsError> {
        Ok(self.inner.contains(one_id(id)?))
    }

    pub fn size(&self) -> usize {
        self.inner.len()
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
    pub fn remove(&self, id: BigInt) -> Result<(), JsError> {
        self.inner.remove(one_id(id)?).map_err(to_jserr)
    }

    /// How many removed slots the graph still carries.
    ///
    /// A browser is the most memory-constrained runtime this crate targets, and
    /// a tombstone holds its vector and links until compaction. Without this a
    /// caller could delete but could not tell what deleting had cost.
    pub fn tombstones(&self) -> usize {
        self.inner.tombstones()
    }

    /// Rebuilds the graph without its tombstoned slots, reclaiming their
    /// memory. Live ids and their vectors are preserved; only the removed
    /// slots go. Cost is a full rebuild, so call it when churn has accumulated
    /// rather than after each removal.
    pub fn compact(&self) -> Result<(), JsError> {
        self.inner.compact().map_err(to_jserr)
    }

    /// `seed` is optional and defaults to 42, the value this constructor used
    /// to hardcode. Supplying it makes construction reproducible: two indexes
    /// built from the same vectors with the same seed have the same topology.
    #[wasm_bindgen(constructor)]
    pub fn new(
        dim: f64,
        metric: &str,
        capacity: f64,
        m: f64,
        ef_construction: f64,
        seed: Option<f64>,
    ) -> Result<WasmIndex, JsError> {
        let met = parse_metric(metric)?;
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

    pub fn add(&self, id: BigInt, vector: &[f32]) -> Result<(), JsError> {
        self.inner.add(one_id(id)?, vector).map_err(to_jserr)
    }

    /// Bulk insert in one wasm call: `ids` is a BigUint64Array of n ids and
    /// `vectors` a Float32Array of n × dim values (row-major). All-or-nothing:
    /// on error the index is unchanged.
    pub fn add_batch(&self, ids: &[u64], vectors: &[f32]) -> Result<(), JsError> {
        self.inner.add_batch(ids, vectors).map_err(to_jserr)
    }

    /// Search for k nearest neighbors.
    ///
    /// Ids come back as a `BigUint64Array` and distances as a `Float32Array`,
    /// parallel by index. Ids are never narrowed to `f32`: values at or above
    /// 2^24 are not exactly representable, so distinct records collided and
    /// callers could act on the wrong record (#39).
    pub fn search(&self, query: &[f32], k: f64) -> Result<WasmSearchResults, JsError> {
        let results = self.inner.search(query, count(k, "k")?).map_err(to_jserr)?;
        Ok(WasmSearchResults::from(results))
    }

    pub fn contains(&self, id: BigInt) -> Result<bool, JsError> {
        Ok(self.inner.contains(one_id(id)?))
    }

    /// The vector stored under `id`, as a `Float32Array`.
    pub fn get_vector(&self, id: BigInt) -> Result<Vec<f32>, JsError> {
        self.inner.get_vector(one_id(id)?).map_err(to_jserr)
    }

    /// The same operation as `get_vector`, under the spelling `FlatIndex` uses.
    /// Both exist so a program is not tied to one index type (#85).
    pub fn get(&self, id: BigInt) -> Result<Vec<f32>, JsError> {
        self.get_vector(id)
    }

    pub fn size(&self) -> usize {
        self.inner.size()
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
    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    #[wasm_bindgen(getter)]
    pub fn ef_search(&self) -> usize {
        self.inner.get_ef_search()
    }

    #[wasm_bindgen(setter)]
    pub fn set_ef_search(&self, ef: f64) -> Result<(), JsError> {
        self.inner.set_ef_search(count(ef, "ef_search")?);
        Ok(())
    }
}
