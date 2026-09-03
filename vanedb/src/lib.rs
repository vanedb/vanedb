//! Embeddable vector database for edge AI.
//!
//! Three ways to hold vectors, all searchable by k nearest neighbours:
//!
//! - [`VectorStore`] — exact brute-force scan, held in memory.
//! - [`HnswIndex`] — approximate graph index: sub-linear search, recall traded
//!   against speed through `ef_search`.
//! - [`MmapVectorStore`] — exact scan over a memory-mapped file, so a corpus
//!   larger than RAM stays searchable (feature `mmap`).
//!
//! Each takes a [`DistanceMetric`] and returns [`SearchResult`]s nearest first.
//!
//! ```
//! use vanedb::{DistanceMetric, HnswIndex};
//!
//! let index = HnswIndex::builder(3, DistanceMetric::Cosine)
//!     .capacity(1_000)
//!     .build()?;
//! index.add(1, &[1.0, 0.0, 0.0])?;
//!
//! let hits = index.search(&[0.9, 0.1, 0.0], 1)?;
//! assert_eq!(hits[0].id, 1);
//! # Ok::<(), vanedb::VaneError>(())
//! ```
//!
//! Distance kernels dispatch to NEON or AVX2 at runtime and fall back to a
//! portable scalar path, which is the reference the others must agree with.
//!
//! A header-only C++ implementation sharing these file formats and graph
//! construction is maintained alongside this crate.

#![warn(missing_docs)]

mod atomic_write;
pub mod distance;
pub mod error;
#[cfg(any(feature = "gpu-metal", feature = "gpu-cuda"))]
pub mod gpu;
pub mod hnsw;
#[cfg(feature = "mmap")]
pub mod mmap;
pub mod store;
mod validation;

pub use distance::DistanceMetric;
pub use error::{Result, VaneError};
pub use hnsw::HnswIndex;
#[cfg(feature = "mmap")]
pub use mmap::{MmapVectorStore, MmapVectorStoreBuilder};
pub use store::{SearchResult, VectorStore};
