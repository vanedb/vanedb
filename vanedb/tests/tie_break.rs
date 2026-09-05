//! `SearchResult`'s `Ord` tie-breaks on id so that equal distances produce a
//! deterministic order. That only holds if the tie-break is applied before the
//! candidate set is cut down to k.

use vanedb::{Index, Metric, Store};

const N: u64 = 40;
const K: usize = 5;
const V: [f32; 2] = [1.0, 0.0];

fn ids(results: Vec<vanedb::SearchResult>) -> Vec<u64> {
    results.into_iter().map(|r| r.id).collect()
}

#[test]
fn equal_distances_break_on_id_in_both_backends() {
    let store = Store::new(2, Metric::L2).unwrap();
    let index = Index::builder(2, Metric::L2)
        .capacity(N as usize)
        .seed(7)
        .build()
        .unwrap();
    for id in 0..N {
        store.add(id, &V).unwrap();
        index.add(id, &V).unwrap();
    }
    index.set_ef_search(50); // wider than the corpus: every node is a candidate

    let from_store = ids(store.search(&V, K).unwrap());
    let from_index = ids(index.search(&V, K).unwrap());
    assert_eq!(
        from_store, from_index,
        "Store and Index disagree on which of {N} equidistant vectors are the \
         top {K}; the id tie-break is applied after truncation"
    );
    assert_eq!(
        from_store,
        vec![0, 1, 2, 3, 4],
        "ties must resolve to lowest ids"
    );
}

#[test]
fn insertion_order_does_not_change_the_tie_break() {
    let forward = Index::builder(2, Metric::L2)
        .capacity(N as usize)
        .seed(7)
        .build()
        .unwrap();
    let reverse = Index::builder(2, Metric::L2)
        .capacity(N as usize)
        .seed(7)
        .build()
        .unwrap();
    for id in 0..N {
        forward.add(id, &V).unwrap();
    }
    for id in (0..N).rev() {
        reverse.add(id, &V).unwrap();
    }
    forward.set_ef_search(50);
    reverse.set_ef_search(50);
    assert_eq!(
        ids(forward.search(&V, K).unwrap()),
        ids(reverse.search(&V, K).unwrap()),
        "the same equidistant set returns different ids depending on insert order"
    );
}
