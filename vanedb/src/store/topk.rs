//! Bounded top-k selection for brute-force scans.
//!
//! The scan used to materialise one `SearchResult` per stored vector and then
//! `select_nth_unstable` over the whole array. That is quickselect: about `2n`
//! comparisons plus roughly `n` swaps of 16-byte structs, over a buffer that
//! outgrows L1 as the corpus grows — which is why the gap against
//! `std::partial_sort` widened with `n`.
//!
//! Keeping a `k`-element max-heap instead never allocates the `n`-element
//! buffer at all: `k` is tiny and stays in L1, and most candidates are
//! rejected by one comparison against the current worst.

use std::collections::BinaryHeap;

use super::SearchResult;

/// Collects the `k` nearest of a candidate stream, nearest first.
///
/// Ordering is `SearchResult`'s, so ties break on id exactly as a full sort
/// would — the heap holds the *worst* of the current best, and a candidate
/// only displaces it when it compares strictly less.
pub(crate) fn select<I>(candidates: I, k: usize) -> Vec<SearchResult>
where
    I: Iterator<Item = SearchResult>,
{
    if k == 0 {
        return Vec::new();
    }
    let mut heap: BinaryHeap<SearchResult> = BinaryHeap::with_capacity(k);
    for candidate in candidates {
        if heap.len() < k {
            heap.push(candidate);
        } else if let Some(worst) = heap.peek() {
            // One comparison rejects the common case.
            if candidate < *worst {
                heap.pop();
                heap.push(candidate);
            }
        }
    }
    let mut out = heap.into_vec();
    out.sort_unstable();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn results(pairs: &[(u64, f32)]) -> Vec<SearchResult> {
        pairs
            .iter()
            .map(|&(id, d)| SearchResult::new(id, d))
            .collect()
    }

    /// The property that matters: identical to sorting everything and taking k.
    fn reference(mut all: Vec<SearchResult>, k: usize) -> Vec<SearchResult> {
        all.sort_unstable();
        all.truncate(k);
        all
    }

    #[test]
    fn matches_a_full_sort_for_every_k() {
        let all = results(&[(5, 3.0), (1, 1.0), (9, 2.0), (2, 5.0), (7, 4.0)]);
        for k in 0..=7 {
            assert_eq!(
                select(all.clone().into_iter(), k),
                reference(all.clone(), k),
                "k = {k}"
            );
        }
    }

    #[test]
    fn ties_break_on_id_like_a_full_sort() {
        // Every distance equal: the k lowest ids must win, regardless of the
        // order they arrive in.
        let ascending = results(&(0..40).map(|i| (i, 1.0)).collect::<Vec<_>>());
        let mut descending = ascending.clone();
        descending.reverse();
        let expected: Vec<u64> = (0..5).collect();
        for order in [ascending, descending] {
            let ids: Vec<u64> = select(order.into_iter(), 5)
                .into_iter()
                .map(|r| r.id)
                .collect();
            assert_eq!(ids, expected);
        }
    }

    #[test]
    fn k_larger_than_the_stream_returns_everything_sorted() {
        let all = results(&[(3, 9.0), (1, 1.0), (2, 5.0)]);
        let got = select(all.clone().into_iter(), 100);
        assert_eq!(got, reference(all, 100));
        assert_eq!(got.len(), 3);
    }

    #[test]
    fn an_empty_stream_yields_nothing() {
        assert!(select(std::iter::empty(), 5).is_empty());
    }
}
