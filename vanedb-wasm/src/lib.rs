use wasm_bindgen::prelude::*;

use vanedb::distance::DistanceMetric;
use vanedb::hnsw::HnswIndex;
use vanedb::store::{SearchResult, VectorStore};

/// Search results with lossless 64-bit ids.
#[wasm_bindgen]
pub struct WasmSearchResults {
    ids: Vec<u64>,
    distances: Vec<f32>,
}

#[wasm_bindgen]
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

fn parse_metric(metric: &str) -> Result<DistanceMetric, JsError> {
    match metric {
        "l2" | "L2" => Ok(DistanceMetric::L2),
        "cosine" | "Cosine" => Ok(DistanceMetric::Cosine),
        "dot" | "Dot" => Ok(DistanceMetric::Dot),
        _ => Err(JsError::new(&format!(
            "unknown metric: {metric}. Use 'l2', 'cosine', or 'dot'"
        ))),
    }
}

#[wasm_bindgen]
pub fn version() -> String {
    "0.1.0".to_string()
}

/// Brute-force vector store for the browser.
#[wasm_bindgen]
pub struct WasmVectorStore {
    inner: VectorStore,
}

#[wasm_bindgen]
impl WasmVectorStore {
    #[wasm_bindgen(constructor)]
    pub fn new(dim: usize, metric: &str) -> Result<WasmVectorStore, JsError> {
        let m = parse_metric(metric)?;
        let inner = VectorStore::new(dim, m).map_err(to_jserr)?;
        Ok(Self { inner })
    }

    pub fn add(&self, id: u64, vector: &[f32]) -> Result<(), JsError> {
        self.inner.add(id, vector).map_err(to_jserr)
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
    pub fn search(&self, query: &[f32], k: usize) -> Result<WasmSearchResults, JsError> {
        let results = self.inner.search(query, k).map_err(to_jserr)?;
        Ok(WasmSearchResults::from(results))
    }

    pub fn get(&self, id: u64) -> Result<Vec<f32>, JsError> {
        self.inner.get(id).map_err(to_jserr)
    }

    pub fn remove(&self, id: u64) -> Result<(), JsError> {
        self.inner.remove(id).map_err(to_jserr)
    }

    pub fn contains(&self, id: u64) -> bool {
        self.inner.contains(id)
    }

    pub fn size(&self) -> usize {
        self.inner.len()
    }

    pub fn dimension(&self) -> usize {
        self.inner.dimension()
    }
}

/// HNSW approximate nearest-neighbor index for the browser.
#[wasm_bindgen]
pub struct WasmHnswIndex {
    inner: HnswIndex,
}

#[wasm_bindgen]
impl WasmHnswIndex {
    #[wasm_bindgen(constructor)]
    pub fn new(
        dim: usize,
        metric: &str,
        capacity: usize,
        m: usize,
        ef_construction: usize,
    ) -> Result<WasmHnswIndex, JsError> {
        let met = parse_metric(metric)?;
        let inner = HnswIndex::builder(dim, met)
            .capacity(capacity)
            .m(m)
            .ef_construction(ef_construction)
            .seed(42)
            .build()
            .map_err(to_jserr)?;
        Ok(Self { inner })
    }

    pub fn add(&self, id: u64, vector: &[f32]) -> Result<(), JsError> {
        self.inner.add(id, vector).map_err(to_jserr)
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
    pub fn search(&self, query: &[f32], k: usize) -> Result<WasmSearchResults, JsError> {
        let results = self.inner.search(query, k).map_err(to_jserr)?;
        Ok(WasmSearchResults::from(results))
    }

    pub fn contains(&self, id: u64) -> bool {
        self.inner.contains(id)
    }

    pub fn size(&self) -> usize {
        self.inner.size()
    }

    pub fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    #[wasm_bindgen(getter)]
    pub fn ef_search(&self) -> usize {
        self.inner.get_ef_search()
    }

    #[wasm_bindgen(setter)]
    pub fn set_ef_search(&self, ef: usize) {
        self.inner.set_ef_search(ef);
    }
}
