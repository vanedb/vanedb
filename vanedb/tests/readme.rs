//! The crate README is the crates.io front page and nothing compiles it, so
//! this pins it to `examples/quickstart.rs`, which the build does compile.

fn readmes() -> Vec<String> {
    let mut readmes = vec![include_str!("../README.md").to_owned()];
    let repository = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap();
    // The repository guide is checked in the monorepo. It is not part of a
    // standalone crate archive, whose own README is always checked above.
    if repository.join("vanedb-py/Cargo.toml").is_file() {
        readmes.push(std::fs::read_to_string(repository.join("README.md")).unwrap());
    }
    readmes
}

#[test]
fn readme_example_matches_the_compiled_one() {
    let example = include_str!("../examples/quickstart.rs");
    let compiled: Vec<_> = example
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    for readme in readmes() {
        let block: Vec<_> = readme
            .lines()
            .skip_while(|l| !l.starts_with("```rust"))
            .skip(1)
            .take_while(|l| !l.starts_with("```"))
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .collect();
        assert_eq!(
            block, compiled,
            "README Rust example must match examples/quickstart.rs"
        );
    }
}

/// Prose drifts more easily than code blocks: the root README claimed `FlatIndex`
/// persisted with `save`/`load`, which it never has. A sentence that pairs a
/// backticked type with a backticked method is checked against that type's
/// methods, listed below and held to the real API by
/// `every_listed_method_still_exists`.
///
/// Two limits are worth stating rather than discovering. A sentence naming no
/// type in backticks is not checked at all, so most of the prose about
/// `remove` and `compact` passes through untouched. And a list that is missing
/// a real method rejects a true sentence: every list here was short of the
/// shipping API, which made `ApproxIndex` plus `remove` — or `DiskIndex` plus
/// `len` — a documentable fact this test refused.
#[test]
fn documented_methods_exist() {
    let store: &[&str] = &[
        "new",
        "add",
        "add_batch",
        "get",
        "get_vector",
        "remove",
        "contains",
        "len",
        "size",
        "is_empty",
        "dimension",
        "metric",
        "search",
    ];
    let index: &[&str] = &[
        "builder",
        "add",
        "add_batch",
        "search",
        "search_with",
        "save",
        "load",
        "get",
        "get_vector",
        "remove",
        "upsert",
        "compact",
        "tombstones",
        "len",
        "size",
        "is_empty",
        "capacity",
        "dimension",
        "metric",
        "contains",
        "m",
        "ef_construction",
        "seed",
        "set_ef_search",
        "get_ef_search",
    ];
    let disk: &[&str] = &[
        "open",
        "search",
        "get",
        "get_vector",
        "contains",
        "len",
        "size",
        "is_empty",
        "dimension",
        "metric",
    ];
    let disk_builder: &[&str] = &["new", "add", "save", "len", "size", "is_empty", "dimension"];
    let all_types = ["FlatIndex", "ApproxIndex", "DiskIndex", "DiskIndexBuilder"];

    for readme in readmes() {
        // Markdown wraps sentences across lines, so judge sentences, not lines:
        // a type and a method sharing a line may belong to different claims.
        //
        // Strip \r before \n: a CRLF checkout leaves ".\r " where the split
        // expects ". ", so two paragraphs merge into one apparent sentence and
        // a method gets attributed to a type from the next paragraph. That
        // failed on Windows only.
        let flat = readme.replace('\r', "").replace('\n', " ");
        for sentence in flat.split(". ") {
            for (ty, methods) in [
                ("DiskIndexBuilder", disk_builder),
                ("DiskIndex", disk),
                ("FlatIndex", store),
                ("ApproxIndex", index),
            ] {
                if !sentence.contains(&format!("`{ty}`")) {
                    continue;
                }
                // A sentence naming several types cannot be attributed to one.
                if all_types
                    .iter()
                    .any(|o| *o != ty && sentence.contains(&format!("`{o}`")))
                {
                    continue;
                }
                for word in sentence.split('`').skip(1).step_by(2) {
                    for claimed in word.split('/') {
                        let claimed =
                            claimed.trim_matches(|c: char| !c.is_alphanumeric() && c != '_');
                        let names_a_method = store.contains(&claimed)
                            || index.contains(&claimed)
                            || disk.contains(&claimed);
                        if names_a_method {
                            assert!(
                                methods.contains(&claimed),
                                "a README sentence attributes `{claimed}` to `{ty}`, which has no \
                                 such method:\n  {sentence}"
                            );
                        }
                    }
                }
            }
        }
    }
}

/// The lists in `documented_methods_exist` are hand-written, and a hand-written
/// mirror of an API rots in the direction that matters: the bug this file exists
/// to catch is prose naming a method a type does not have, and a list still
/// carrying a deleted name cannot catch it. Naming each method here as a value
/// costs nothing at runtime — this never runs — and makes removing or renaming
/// one a compile error in the test that documents it.
#[test]
fn every_listed_method_still_exists() {
    #[allow(unused)]
    fn flat_index() {
        use vanedb::FlatIndex;
        let _ = (
            FlatIndex::new,
            FlatIndex::add,
            FlatIndex::add_batch,
            FlatIndex::get,
            FlatIndex::get_vector,
            FlatIndex::remove,
            FlatIndex::contains,
            FlatIndex::len,
            FlatIndex::size,
            FlatIndex::is_empty,
            FlatIndex::dimension,
            FlatIndex::metric,
            FlatIndex::search,
        );
    }

    #[allow(unused)]
    fn approx_index() {
        use vanedb::ApproxIndex;
        let _ = (
            ApproxIndex::builder,
            ApproxIndex::add,
            ApproxIndex::add_batch,
            ApproxIndex::search,
            ApproxIndex::search_with,
            ApproxIndex::get,
            ApproxIndex::get_vector,
            ApproxIndex::remove,
            ApproxIndex::upsert,
            ApproxIndex::compact,
            ApproxIndex::tombstones,
            ApproxIndex::len,
            ApproxIndex::size,
            ApproxIndex::is_empty,
            ApproxIndex::capacity,
            ApproxIndex::dimension,
            ApproxIndex::metric,
            ApproxIndex::contains,
        );
        // `save`/`load` take `impl AsRef<Path>`, which a turbofish cannot name.
        // Naming them at a call site instead needs a receiver, and a parameter
        // supplies one without constructing anything: this is never called.
        fn paths(index: &ApproxIndex) {
            let _ = index.save(std::path::Path::new(""));
            let _ = ApproxIndex::load(std::path::Path::new(""));
        }
        let _ = (
            ApproxIndex::m,
            ApproxIndex::ef_construction,
            ApproxIndex::seed,
            ApproxIndex::set_ef_search,
            ApproxIndex::get_ef_search,
        );
    }

    #[cfg(feature = "disk")]
    #[allow(unused)]
    fn disk_index() {
        use vanedb::{DiskIndex, DiskIndexBuilder};
        // Same `impl AsRef<Path>` reason as above; never called.
        fn paths(builder: &DiskIndexBuilder) {
            let _ = builder.save(std::path::Path::new(""));
            // SAFETY: never called — this body is type-checked, never run.
            let _ = unsafe { DiskIndex::open(std::path::Path::new("")) };
        }
        let _ = (
            DiskIndex::search,
            DiskIndex::get,
            DiskIndex::get_vector,
            DiskIndex::contains,
            DiskIndex::len,
            DiskIndex::size,
            DiskIndex::is_empty,
            DiskIndex::dimension,
            DiskIndex::metric,
        );
        let _ = (
            DiskIndexBuilder::new,
            DiskIndexBuilder::add,
            DiskIndexBuilder::len,
            DiskIndexBuilder::size,
            DiskIndexBuilder::is_empty,
            DiskIndexBuilder::dimension,
        );
    }
}
