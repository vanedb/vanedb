use wasm_bindgen::prelude::*;

use vanedb::distance::DistanceMetric;
use vanedb::hnsw::HnswIndex;
use vanedb::store::VectorStore;

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

    /// Search for k nearest neighbors. Returns flat array: [id0, dist0, id1, dist1, ...].
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<f32>, JsError> {
        let results = self.inner.search(query, k).map_err(to_jserr)?;
        let mut flat = Vec::with_capacity(results.len() * 2);
        for r in results {
            flat.push(r.id as f32);
            flat.push(r.distance);
        }
        Ok(flat)
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

    /// Search for k nearest neighbors. Returns flat array: [id0, dist0, id1, dist1, ...].
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<f32>, JsError> {
        let results = self.inner.search(query, k).map_err(to_jserr)?;
        let mut flat = Vec::with_capacity(results.len() * 2);
        for r in results {
            flat.push(r.id as f32);
            flat.push(r.distance);
        }
        Ok(flat)
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
