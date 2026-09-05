//! Cross-engine HNSW id_map cases from `conformance/index_id_map_consistency.tsv`.
//!
//! The loader validated only `id_map.len() == count` and that each internal
//! index was in range, so a file could map an external id onto another slot.
//! Reads then returned the right bytes under the wrong identity (#42).

use std::collections::HashMap;
use std::fs;
use std::io::Write;

use vanedb::Index;

const HNSW_MAGIC: u32 = u32::from_le_bytes(*b"HNSW");
const HNSW_VERSION: u32 = 2;
const DIM: usize = 2;

/// Field-order mirror of the private `HnswData` in `src/hnsw/persistence.rs`.
/// bincode encodes by field order, so this serializes identically.
#[derive(serde::Serialize)]
struct HnswDataMirror {
    dim: usize,
    metric: u32,
    max_elements: usize,
    m: usize,
    m_max: usize,
    m_max0: usize,
    ef_construction: usize,
    ef_search: usize,
    mult: f64,
    seed: u64,
    count: usize,
    entry_point: Option<usize>,
    max_level: i32,
    vectors: Vec<f32>,
    ext_ids: Vec<u64>,
    levels: Vec<i32>,
    neighbors: Vec<Vec<Vec<usize>>>,
    id_map: HashMap<u64, usize>,
}

struct Case {
    name: String,
    count: usize,
    ext_ids: Vec<u64>,
    id_map: HashMap<u64, usize>,
    accept: bool,
}

fn cases() -> Vec<Case> {
    include_str!("../../conformance/index_id_map_consistency.tsv")
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(|line| {
            let f: Vec<&str> = line.split('\t').collect();
            assert_eq!(f.len(), 5, "valid id_map fixture row: {line}");
            let id_map = if f[3] == "-" {
                HashMap::new()
            } else {
                f[3].split(',')
                    .map(|pair| {
                        let (k, v) = pair.split_once(':').expect("key:value pair");
                        (k.parse().unwrap(), v.parse().unwrap())
                    })
                    .collect()
            };
            Case {
                name: f[0].to_string(),
                count: f[1].parse().unwrap(),
                ext_ids: f[2].split(',').map(|v| v.parse().unwrap()).collect(),
                id_map,
                accept: match f[4] {
                    "accept" => true,
                    "reject" => false,
                    other => panic!("unknown expectation: {other}"),
                },
            }
        })
        .collect()
}

fn payload(case: &Case) -> HnswDataMirror {
    HnswDataMirror {
        dim: DIM,
        metric: 0,
        max_elements: case.count.max(1),
        m: 2,
        m_max: 2,
        m_max0: 4,
        ef_construction: 10,
        ef_search: 10,
        mult: 1.0,
        seed: 7,
        count: case.count,
        entry_point: Some(0),
        max_level: 0,
        // Distinguishable per slot, so a mis-mapped id returns detectably
        // wrong data rather than the same bytes.
        vectors: (0..case.count).flat_map(|i| [i as f32, 0.0]).collect(),
        ext_ids: case.ext_ids.clone(),
        levels: vec![0; case.count],
        neighbors: vec![vec![vec![]]; case.count],
        id_map: case.id_map.clone(),
    }
}

fn write_case(dir: &std::path::Path, case: &Case) -> std::path::PathBuf {
    let mut bytes = HNSW_MAGIC.to_le_bytes().to_vec();
    bytes.extend_from_slice(&HNSW_VERSION.to_le_bytes());
    bytes.extend_from_slice(
        &bincode::serde::encode_to_vec(payload(case), bincode::config::legacy()).unwrap(),
    );
    let path = dir.join(format!("{}.idx", case.name));
    let mut f = fs::File::create(&path).unwrap();
    f.write_all(&bytes).unwrap();
    path
}

#[test]
fn loader_enforces_the_shared_id_map_contract() {
    let dir = std::env::temp_dir().join(format!("vanedb-issue-42-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();

    let all = cases();
    assert!(all.len() >= 6, "fixture should cover every invalid shape");

    for case in &all {
        let path = write_case(&dir, case);
        let result = Index::load(&path);
        if case.accept {
            let index =
                result.unwrap_or_else(|e| panic!("{}: expected load to succeed: {e}", case.name));
            assert_eq!(index.size(), case.count, "{}", case.name);
            for (i, &ext_id) in case.ext_ids.iter().enumerate() {
                assert!(index.contains(ext_id), "{}: {ext_id} not found", case.name);
                assert_eq!(
                    index.get_vector(ext_id).unwrap(),
                    vec![i as f32, 0.0],
                    "{}: external id {ext_id} resolved to the wrong slot",
                    case.name
                );
            }
        } else {
            assert!(
                result.is_err(),
                "{}: expected the loader to reject this file",
                case.name
            );
        }
    }

    let _ = fs::remove_dir_all(&dir);
}
