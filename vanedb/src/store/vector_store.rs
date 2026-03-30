use std::collections::HashMap;

use parking_lot::RwLock;

use crate::distance::{distance_fn, DistanceFn, DistanceMetric};
use crate::error::{Result, VaneError};
use crate::store::SearchResult;

pub struct VectorStore {
    dim: usize,
    metric: DistanceMetric,
    dist_fn: DistanceFn,
    inner: RwLock<Inner>,
}

struct Inner {
    ids: Vec<u64>,
    data: Vec<f32>,
    id_to_index: HashMap<u64, usize>,
}

impl VectorStore {
    pub fn new(dim: usize, metric: DistanceMetric) -> Result<Self> {
        if dim == 0 {
            return Err(VaneError::EmptyVector);
        }
        Ok(Self {
            dim,
            metric,
            dist_fn: distance_fn(metric),
            inner: RwLock::new(Inner {
                ids: Vec::new(),
                data: Vec::new(),
                id_to_index: HashMap::new(),
            }),
        })
    }

    pub fn add(&self, id: u64, vector: &[f32]) -> Result<()> {
        if vector.len() != self.dim {
            return Err(VaneError::DimensionMismatch {
                expected: self.dim,
                got: vector.len(),
            });
        }
        let mut inner = self.inner.write();
        if inner.id_to_index.contains_key(&id) {
            return Err(VaneError::DuplicateId { id });
        }
        let index = inner.ids.len();
        inner.ids.push(id);
        inner.data.extend_from_slice(vector);
        inner.id_to_index.insert(id, index);
        Ok(())
    }

    pub fn get(&self, id: u64) -> Result<Vec<f32>> {
        let inner = self.inner.read();
        let &index = inner
            .id_to_index
            .get(&id)
            .ok_or(VaneError::NotFound { id })?;
        let start = index * self.dim;
        Ok(inner.data[start..start + self.dim].to_vec())
    }

    pub fn remove(&self, id: u64) -> Result<()> {
        let mut inner = self.inner.write();
        let index = inner
            .id_to_index
            .remove(&id)
            .ok_or(VaneError::NotFound { id })?;
        let last = inner.ids.len() - 1;
        if index != last {
            // Swap-remove: move last vector into the removed slot
            let last_id = inner.ids[last];
            inner.ids[index] = last_id;
            let src_start = last * self.dim;
            let dst_start = index * self.dim;
            for j in 0..self.dim {
                inner.data[dst_start + j] = inner.data[src_start + j];
            }
            inner.id_to_index.insert(last_id, index);
        }
        inner.ids.pop();
        let new_len = inner.ids.len() * self.dim;
        inner.data.truncate(new_len);
        Ok(())
    }

    pub fn contains(&self, id: u64) -> bool {
        self.inner.read().id_to_index.contains_key(&id)
    }

    pub fn len(&self) -> usize {
        self.inner.read().ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn dimension(&self) -> usize {
        self.dim
    }

    pub fn metric(&self) -> DistanceMetric {
        self.metric
    }

    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        if query.len() != self.dim {
            return Err(VaneError::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }
        if k == 0 {
            return Err(VaneError::InvalidK);
        }
        let inner = self.inner.read();
        let n = inner.ids.len();
        if n == 0 {
            return Ok(Vec::new());
        }

        let mut results: Vec<SearchResult> = (0..n)
            .map(|i| {
                let start = i * self.dim;
                let vec = &inner.data[start..start + self.dim];
                SearchResult::new(inner.ids[i], (self.dist_fn)(query, vec))
            })
            .collect();

        results.sort();
        results.truncate(k);
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_rejects_zero_dimension() {
        assert!(VectorStore::new(0, DistanceMetric::L2).is_err());
    }

    #[test]
    fn add_and_get() {
        let store = VectorStore::new(3, DistanceMetric::L2).unwrap();
        let vec = vec![1.0, 2.0, 3.0];
        store.add(1, &vec).unwrap();
        assert_eq!(store.get(1).unwrap(), vec);
    }

    #[test]
    fn add_wrong_dimension() {
        let store = VectorStore::new(3, DistanceMetric::L2).unwrap();
        let result = store.add(1, &[1.0, 2.0]);
        assert!(matches!(
            result,
            Err(VaneError::DimensionMismatch {
                expected: 3,
                got: 2
            })
        ));
    }

    #[test]
    fn add_duplicate_id() {
        let store = VectorStore::new(3, DistanceMetric::L2).unwrap();
        store.add(1, &[1.0, 2.0, 3.0]).unwrap();
        assert!(matches!(
            store.add(1, &[4.0, 5.0, 6.0]),
            Err(VaneError::DuplicateId { id: 1 })
        ));
    }

    #[test]
    fn get_missing_id() {
        let store = VectorStore::new(3, DistanceMetric::L2).unwrap();
        assert!(matches!(
            store.get(42),
            Err(VaneError::NotFound { id: 42 })
        ));
    }

    #[test]
    fn remove_vector() {
        let store = VectorStore::new(3, DistanceMetric::L2).unwrap();
        store.add(1, &[1.0, 2.0, 3.0]).unwrap();
        store.add(2, &[4.0, 5.0, 6.0]).unwrap();
        store.remove(1).unwrap();
        assert!(!store.contains(1));
        assert!(store.contains(2));
        assert_eq!(store.len(), 1);
        assert_eq!(store.get(2).unwrap(), vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn remove_missing_id() {
        let store = VectorStore::new(3, DistanceMetric::L2).unwrap();
        assert!(matches!(
            store.remove(42),
            Err(VaneError::NotFound { id: 42 })
        ));
    }

    #[test]
    fn len_and_is_empty() {
        let store = VectorStore::new(3, DistanceMetric::L2).unwrap();
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
        store.add(1, &[1.0, 2.0, 3.0]).unwrap();
        assert!(!store.is_empty());
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn search_l2_finds_nearest() {
        let store = VectorStore::new(2, DistanceMetric::L2).unwrap();
        store.add(1, &[0.0, 0.0]).unwrap();
        store.add(2, &[1.0, 0.0]).unwrap();
        store.add(3, &[10.0, 10.0]).unwrap();

        let results = store.search(&[0.0, 0.1], 2).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, 1);
        assert_eq!(results[1].id, 2);
    }

    #[test]
    fn search_cosine_finds_similar() {
        let store = VectorStore::new(2, DistanceMetric::Cosine).unwrap();
        store.add(1, &[1.0, 0.0]).unwrap();
        store.add(2, &[0.0, 1.0]).unwrap();
        store.add(3, &[-1.0, 0.0]).unwrap();

        let results = store.search(&[0.9, 0.1], 1).unwrap();
        assert_eq!(results[0].id, 1);
    }

    #[test]
    fn search_k_larger_than_store() {
        let store = VectorStore::new(2, DistanceMetric::L2).unwrap();
        store.add(1, &[0.0, 0.0]).unwrap();
        let results = store.search(&[1.0, 1.0], 10).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn search_empty_store() {
        let store = VectorStore::new(2, DistanceMetric::L2).unwrap();
        let results = store.search(&[1.0, 1.0], 5).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn search_wrong_dimension() {
        let store = VectorStore::new(3, DistanceMetric::L2).unwrap();
        assert!(matches!(
            store.search(&[1.0, 2.0], 5),
            Err(VaneError::DimensionMismatch {
                expected: 3,
                got: 2
            })
        ));
    }

    #[test]
    fn search_k_zero() {
        let store = VectorStore::new(2, DistanceMetric::L2).unwrap();
        assert!(matches!(
            store.search(&[1.0, 2.0], 0),
            Err(VaneError::InvalidK)
        ));
    }

    #[test]
    fn concurrent_add_and_search() {
        use std::sync::Arc;
        use std::thread;

        let store = Arc::new(VectorStore::new(3, DistanceMetric::L2).unwrap());
        let mut handles = vec![];

        // 10 writer threads
        for i in 0..10u64 {
            let store = Arc::clone(&store);
            handles.push(thread::spawn(move || {
                let v = vec![i as f32; 3];
                store.add(i, &v).unwrap();
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(store.len(), 10);

        // 10 reader threads searching concurrently
        let mut handles = vec![];
        for _ in 0..10 {
            let store = Arc::clone(&store);
            handles.push(thread::spawn(move || {
                let results = store.search(&[5.0, 5.0, 5.0], 3).unwrap();
                assert_eq!(results.len(), 3);
            }));
        }

        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn store_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<VectorStore>();
    }
}
