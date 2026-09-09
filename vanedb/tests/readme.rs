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
/// persisted with `save`/`load`, which it never has. Every sentence pairing a
/// type with a method name is checked against that type's real methods.
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
        "save",
        "load",
        "size",
        "is_empty",
        "capacity",
        "dimension",
        "metric",
        "contains",
        "get_vector",
        "set_ef_search",
        "get_ef_search",
    ];
    let disk: &[&str] = &[
        "open",
        "search",
        "get",
        "get_vector",
        "contains",
        "size",
        "dimension",
        "metric",
    ];
    let disk_builder: &[&str] = &["new", "add", "save", "size"];
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
