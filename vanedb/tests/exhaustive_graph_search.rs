use vanedb::{ApproxIndex, Filter, FlatIndex, Metric, SearchParams};

// An independently encoded, valid VNDB v2 file with three disconnected
// layer-zero nodes. Loading and searching must not rewrite its topology.
fn disconnected(metric: u32) -> Vec<u8> {
    let mut b = b"VNDB".to_vec();
    for value in [2u32, 1, metric] {
        b.extend(value.to_le_bytes());
    }
    for value in [2u64, 3, 3, 2, 10, 1, 42, 0] {
        b.extend(value.to_le_bytes());
    }
    b.extend(0i32.to_le_bytes());
    b.extend(1u32.to_le_bytes());
    b.extend(0u64.to_le_bytes());
    for (id, vector) in [(90u64, [1.0f32, 0.0]), (10, [0.0, 1.0]), (50, [1.0, 1.0])] {
        b.extend(id.to_le_bytes());
        b.extend(0u32.to_le_bytes());
        b.extend(0u32.to_le_bytes());
        for coordinate in vector {
            b.extend(coordinate.to_le_bytes());
        }
        b.extend(0u64.to_le_bytes());
    }
    b
}

#[test]
fn full_width_recovers_disconnected_legacy_graphs_without_rewriting_them() {
    for (code, metric) in [(0, Metric::L2), (1, Metric::Cosine), (2, Metric::Dot)] {
        let bytes = disconnected(code);
        let index = ApproxIndex::from_bytes(&bytes).unwrap();
        let flat = FlatIndex::new(2, metric).unwrap();
        for (id, vector) in [(90, [1., 0.]), (10, [0., 1.]), (50, [1., 1.])] {
            flat.add(id, &vector).unwrap();
        }
        let q = [1., 1.];
        assert_eq!(index.search(&q, 3).unwrap(), flat.search(&q, 3).unwrap());
        assert_eq!(
            index
                .search_with(&q, 1, &SearchParams::new().ef_search(3))
                .unwrap(),
            flat.search(&q, 1).unwrap()
        );
        for id in [10, 50, 90] {
            let allow = [id];
            let params = SearchParams::new()
                .ef_search(1)
                .max_ef_search(3)
                .filter(Filter::Allow(&allow));
            assert_eq!(index.search_with(&q, 1, &params).unwrap()[0].id, id);
        }
        assert_eq!(index.to_bytes().unwrap(), bytes);
        index.remove(90).unwrap(); // Tombstoned entry still excluded by exact scan.
        let results = index
            .search_with(&q, usize::MAX, &SearchParams::new().ef_search(3))
            .unwrap();
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.id != 90));
        index.compact().unwrap();
        assert_eq!(index.search(&q, 3).unwrap(), results);
    }
}

#[test]
fn full_width_clustered_graphs_match_exact_search_over_many_seeds() {
    for metric in [Metric::Dot, Metric::Cosine] {
        for seed in 0..20 {
            let index = ApproxIndex::builder(2, metric)
                .m(16)
                .ef_construction(100)
                .seed(seed)
                .build()
                .unwrap();
            let flat = FlatIndex::new(2, metric).unwrap();
            let mut state = seed + 1;
            for id in 0..400 {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let cluster = (state >> 62) as f32;
                let noise = ((state >> 32) as u32) as f32 / u32::MAX as f32 * 0.05;
                let v = [cluster * 5.0 + noise, (3.0 - cluster) * 5.0 - noise];
                index.add(id, &v).unwrap();
                flat.add(id, &v).unwrap();
            }
            let params = SearchParams::new().ef_search(400);
            assert_eq!(
                index.search_with(&[1., 1.], 400, &params).unwrap(),
                flat.search(&[1., 1.], 400).unwrap()
            );
            let allow = [17, 137, 399];
            let params = params.filter(Filter::Allow(&allow));
            assert_eq!(
                index.search_with(&[1., 1.], 3, &params).unwrap(),
                flat.search_with(&[1., 1.], 3, &params).unwrap()
            );
        }
    }
}
