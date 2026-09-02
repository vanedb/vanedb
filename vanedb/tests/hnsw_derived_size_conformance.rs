//! Cross-engine HNSW size cases from `conformance/hnsw_derived_sizes.tsv`.

use vanedb::{DistanceMetric, HnswIndex, VaneError};

#[derive(Debug)]
struct Case<'a> {
    name: &'a str,
    dimension: usize,
    max_elements: usize,
    m: usize,
    overflow: &'a str,
}

fn parse_size(value: &str) -> usize {
    match value {
        "SIZE_MAX" => usize::MAX,
        "HALF_SIZE_MAX_PLUS_ONE" => usize::MAX / 2 + 1,
        value => value.parse().expect("valid symbolic size"),
    }
}

fn cases() -> Vec<Case<'static>> {
    include_str!("../../conformance/hnsw_derived_sizes.tsv")
        .lines()
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(|line| {
            let fields: Vec<_> = line.split('\t').collect();
            assert_eq!(fields.len(), 5, "valid HNSW derived-size fixture row");
            Case {
                name: fields[0],
                dimension: parse_size(fields[1]),
                max_elements: parse_size(fields[2]),
                m: parse_size(fields[3]),
                overflow: fields[4],
            }
        })
        .collect()
}

#[test]
fn builder_rejects_shared_derived_size_overflows() {
    let cases = cases();
    assert_eq!(cases.len(), 2);

    for case in cases {
        let result = HnswIndex::builder(case.dimension, DistanceMetric::L2)
            .capacity(case.max_elements)
            .m(case.m)
            .build();
        let Err(error) = result else {
            panic!("{}: expected derived-size overflow", case.name);
        };
        assert!(
            matches!(error, VaneError::InvalidParameter(_)),
            "{}: expected InvalidParameter, got {error:?}",
            case.name
        );
        let message = error.to_string();
        match case.overflow {
            "capacity_times_dimension" => {
                assert!(message.contains("capacity * dim overflows usize"));
            }
            "m_times_two" => assert!(message.contains("M * 2 overflows usize")),
            other => panic!("{}: unknown overflow kind {other}", case.name),
        }
    }
}
