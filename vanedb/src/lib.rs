//! Embeddable vector database for edge AI.
//!
//! Three ways to hold vectors, all searchable by k nearest neighbours:
//!
//! - [`FlatIndex`] — exact brute-force scan, held in memory.
//! - [`ApproxIndex`] — approximate graph index: sub-linear search, recall traded
//!   against speed through `ef_search`.
//! - [`DiskIndex`] — exact scan over a memory-mapped file, so a corpus
//!   larger than RAM stays searchable (feature `disk`).
//!
//! Each takes a [`Metric`] and returns [`SearchResult`]s nearest first.
//!
//! ```
//! use vanedb::{Metric, ApproxIndex};
//!
//! let index = ApproxIndex::builder(3, Metric::Cosine)
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
//! Both formats are little-endian. `DiskIndex` writes `VNDB` v1, a specified,
//! versioned format anchored to shared fixtures; either engine reads the
//! other's file, checked by a cross-load test. [`ApproxIndex::save`] is
//! engine-specific and not yet a stable public format — treat a saved graph as
//! a way to avoid rebuilding, not as a system of record.
//!
//! A header-only C++ implementation is maintained alongside this crate; the two
//! share graph construction as well as the `DiskIndex` format.

#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

pub mod approx;
mod atomic_write;
#[cfg(feature = "disk")]
#[cfg_attr(docsrs, doc(cfg(feature = "disk")))]
pub mod disk;
pub mod distance;
pub mod error;
pub mod flat;
/// GPU backends, behind `gpu-metal` or `gpu-cuda`.
#[cfg(any(feature = "gpu-metal", feature = "gpu-cuda"))]
#[cfg_attr(docsrs, doc(cfg(any(feature = "gpu-metal", feature = "gpu-cuda"))))]
pub mod gpu;
mod validation;

pub use approx::ApproxIndex;
#[cfg(feature = "disk")]
#[cfg_attr(docsrs, doc(cfg(feature = "disk")))]
pub use disk::{DiskIndex, DiskIndexBuilder};
pub use distance::Metric;
pub use error::{Result, VaneError};
pub use flat::{FlatIndex, SearchResult};
