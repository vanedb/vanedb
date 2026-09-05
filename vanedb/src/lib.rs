//! Embeddable vector database for edge AI.
//!
//! Three ways to hold vectors, all searchable by k nearest neighbours:
//!
//! - [`Store`] — exact brute-force scan, held in memory.
//! - [`Index`] — approximate graph index: sub-linear search, recall traded
//!   against speed through `ef_search`.
//! - [`DiskStore`] — exact scan over a memory-mapped file, so a corpus
//!   larger than RAM stays searchable (feature `disk`).
//!
//! Each takes a [`Metric`] and returns [`SearchResult`]s nearest first.
//!
//! ```
//! use vanedb::{Metric, Index};
//!
//! let index = Index::builder(3, Metric::Cosine)
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
#[cfg(feature = "disk")]
pub mod disk;
pub mod distance;
pub mod error;
#[cfg(any(feature = "gpu-metal", feature = "gpu-cuda"))]
pub mod gpu;
pub mod index;
pub mod store;
mod validation;

#[cfg(feature = "disk")]
pub use disk::{DiskStore, DiskStoreBuilder};
pub use distance::Metric;
pub use error::{Result, VaneError};
pub use index::Index;
pub use store::{SearchResult, Store};
