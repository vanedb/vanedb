//! Chunked vector storage for the graph index.
//!
//! Vectors live in fixed-size chunks rather than one flat allocation, so the
//! index grows as vectors arrive instead of reserving its whole capacity up
//! front. A chunk holds a power-of-two count of *whole* vectors, which keeps
//! `get` returning one contiguous slice — the SIMD kernels read a vector at a
//! time, so nothing is lost by splitting between vectors.

/// Chunk size target. Big enough that the chunk list stays short for large
/// corpora, small enough that a mostly-empty index costs little.
const TARGET_CHUNK_BYTES: usize = 1 << 20;

/// Vectors of a fixed dimension, stored in chunks and grown on demand.
pub(super) struct ChunkedVectors {
    chunks: Vec<Vec<f32>>,
    dim: usize,
    /// Vectors per chunk; always a power of two so indexing is shift + mask.
    per_chunk: usize,
    shift: u32,
    mask: usize,
    len: usize,
}

/// Largest power of two vectors of `dim` floats that fits the chunk target,
/// and at least one — a single vector always gets its own chunk if it must.
///
/// A power of two is what makes indexing a shift and a mask rather than a
/// division, so this rounds *down*: rounding up would overshoot the target.
fn vectors_per_chunk(dim: usize) -> usize {
    let fits = (TARGET_CHUNK_BYTES / (dim * std::mem::size_of::<f32>())).max(1);
    1usize << (usize::BITS - 1 - fits.leading_zeros())
}

impl ChunkedVectors {
    pub(super) fn new(dim: usize) -> Self {
        debug_assert!(dim > 0, "dimension is validated before construction");
        let per_chunk = vectors_per_chunk(dim);
        Self {
            chunks: Vec::new(),
            dim,
            per_chunk,
            shift: per_chunk.trailing_zeros(),
            mask: per_chunk - 1,
            len: 0,
        }
    }

    /// Pre-allocates room for `n` vectors. A hint only: pushing past `n` grows.
    pub(super) fn with_capacity(dim: usize, n: usize) -> Self {
        let mut s = Self::new(dim);
        s.chunks.reserve(n.div_ceil(s.per_chunk));
        s
    }

    pub(super) fn len(&self) -> usize {
        self.len
    }

    #[inline]
    fn locate(&self, iid: usize) -> (usize, usize) {
        (iid >> self.shift, (iid & self.mask) * self.dim)
    }

    /// The vector at `iid`, as one contiguous slice.
    #[inline]
    pub(super) fn get(&self, iid: usize) -> &[f32] {
        let (chunk, offset) = self.locate(iid);
        &self.chunks[chunk][offset..offset + self.dim]
    }

    /// Appends a vector, allocating a chunk when the current one is full.
    pub(super) fn push(&mut self, vector: &[f32]) {
        debug_assert_eq!(vector.len(), self.dim);
        if self.len & self.mask == 0 {
            let mut chunk = Vec::new();
            chunk.reserve_exact(self.per_chunk * self.dim);
            self.chunks.push(chunk);
        }
        let last = self.chunks.last_mut().expect("just ensured a chunk exists");
        last.extend_from_slice(vector);
        self.len += 1;
    }

    /// Copies every stored vector into one flat row-major buffer, for saving.
    pub(super) fn to_flat(&self, count: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(count * self.dim);
        for iid in 0..count {
            out.extend_from_slice(self.get(iid));
        }
        out
    }

    /// Rebuilds storage from one flat row-major buffer, as written by
    /// [`to_flat`](Self::to_flat).
    pub(super) fn from_flat(dim: usize, flat: &[f32]) -> Self {
        let mut s = Self::with_capacity(dim, flat.len() / dim);
        for vector in flat.chunks_exact(dim) {
            s.push(vector);
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_hold_whole_vectors_near_the_size_target() {
        // A vector must never straddle a boundary, so the count is a power of
        // two and the chunk lands at or under the target.
        for (dim, expected) in [(768, 256), (384, 512), (128, 2048), (8, 32768)] {
            let n = vectors_per_chunk(dim);
            assert_eq!(n, expected, "dim {dim}");
            assert!(n.is_power_of_two(), "dim {dim}: {n} is not a power of two");
            assert!(
                n * dim * 4 <= TARGET_CHUNK_BYTES,
                "dim {dim}: chunk exceeds the target"
            );
        }
    }

    #[test]
    fn a_dimension_larger_than_the_target_still_stores_one_vector_per_chunk() {
        let huge = TARGET_CHUNK_BYTES; // 4 MiB per vector
        assert_eq!(vectors_per_chunk(huge), 1);
        let mut v = ChunkedVectors::new(huge);
        v.push(&vec![1.0; huge]);
        assert_eq!(v.get(0)[0], 1.0);
    }

    #[test]
    fn an_empty_store_allocates_nothing() {
        let v = ChunkedVectors::new(768);
        assert_eq!(v.len(), 0);
        assert!(v.chunks.is_empty(), "an empty index must not hold chunks");
    }

    #[test]
    fn capacity_is_a_hint_that_does_not_add_vectors() {
        let v = ChunkedVectors::with_capacity(128, 100_000);
        assert_eq!(v.len(), 0);
        assert!(v.chunks.is_empty(), "reserving must not materialise chunks");
    }

    #[test]
    fn vectors_round_trip_across_chunk_boundaries() {
        let dim = 4;
        let per_chunk = vectors_per_chunk(dim);
        let n = per_chunk * 2 + 3; // spans three chunks
        let mut v = ChunkedVectors::new(dim);
        for i in 0..n {
            v.push(&[i as f32, 1.0, 2.0, 3.0]);
        }
        assert_eq!(v.len(), n);
        for i in 0..n {
            assert_eq!(v.get(i), &[i as f32, 1.0, 2.0, 3.0], "vector {i}");
        }
    }

    #[test]
    fn growth_is_not_capped_by_the_reserve_hint() {
        let mut v = ChunkedVectors::with_capacity(2, 1);
        for i in 0..1000 {
            v.push(&[i as f32, 0.0]);
        }
        assert_eq!(v.len(), 1000);
        assert_eq!(v.get(999), &[999.0, 0.0]);
    }

    #[test]
    fn flat_round_trip_preserves_every_vector() {
        let dim = 3;
        let mut v = ChunkedVectors::new(dim);
        for i in 0..70 {
            v.push(&[i as f32, i as f32 + 0.5, -(i as f32)]);
        }
        let flat = v.to_flat(v.len());
        assert_eq!(flat.len(), 70 * dim);
        let back = ChunkedVectors::from_flat(dim, &flat);
        assert_eq!(back.len(), 70);
        for i in 0..70 {
            assert_eq!(back.get(i), v.get(i), "vector {i}");
        }
    }
}
