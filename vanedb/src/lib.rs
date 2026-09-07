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
//! Both formats have fixed-width little-endian fields. `DiskIndex` writes
//! `VNDB` v1; [`ApproxIndex::save`] writes the shared `VNDB` v2 graph format.
//! Rust and the supplementary C++ engine preserve vectors, links, IDs and
//! tombstones across graph load/save. Further insertions may differ across
//! engines. The Rust loader also reads legacy Rust v1/v2 graphs; save to a new
//! path to migrate, retaining the original and source vectors for verification.
//! Older readers cannot open VNDB v2. The Rust engine in VaneDB 1.x will
//! continue to read valid VNDB v1 disk and VNDB v2 graph files written by
//! VaneDB 1.0.0, within documented
//! resource limits. Identical topology after future insertions is not promised.

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
#[cfg(feature = "gpu-metal")]
#[cfg_attr(docsrs, doc(cfg(feature = "gpu-metal")))]
pub mod gpu;
mod validation;

pub use approx::{ApproxIndex, SearchParams};
#[cfg(feature = "disk")]
#[cfg_attr(docsrs, doc(cfg(feature = "disk")))]
pub use disk::{DiskIndex, DiskIndexBuilder};
pub use distance::Metric;
pub use error::{Result, VaneError};
pub use flat::{FlatIndex, SearchResult};
