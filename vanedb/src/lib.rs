pub mod distance;
pub mod error;
#[cfg(any(feature = "gpu-metal", feature = "gpu-cuda"))]
pub mod gpu;
pub mod hnsw;
#[cfg(feature = "mmap")]
pub mod mmap;
pub mod store;

pub use distance::DistanceMetric;
pub use error::{Result, VaneError};
pub use hnsw::HnswIndex;
#[cfg(feature = "mmap")]
pub use mmap::{MmapVectorStore, MmapVectorStoreBuilder};
pub use store::{SearchResult, VectorStore};
