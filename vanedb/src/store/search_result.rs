/// A single result from a vector search.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    /// The ID of the matched vector.
    pub id: u64,
    /// The distance from the query vector (lower = closer).
    pub distance: f32,
}

impl SearchResult {
    pub fn new(id: u64, distance: f32) -> Self {
        Self { id, distance }
    }
}

impl Eq for SearchResult {}

impl PartialOrd for SearchResult {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SearchResult {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_results_sort_by_distance() {
        let mut results = [
            SearchResult::new(1, 5.0),
            SearchResult::new(2, 1.0),
            SearchResult::new(3, 3.0),
        ];
        results.sort();
        assert_eq!(results[0].id, 2);
        assert_eq!(results[1].id, 3);
        assert_eq!(results[2].id, 1);
    }

    #[test]
    fn search_result_equality() {
        let a = SearchResult::new(1, 2.5);
        let b = SearchResult::new(1, 2.5);
        assert_eq!(a, b);
    }
}
