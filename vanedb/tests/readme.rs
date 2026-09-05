//! The crate README is the crates.io front page and nothing compiles it, so
//! this pins it to `examples/quickstart.rs`, which the build does compile.

/// Statements between the markers in the example, normalised for comparison.
fn marked_body(src: &str, begin: &str, end: &str) -> Vec<String> {
    src.lines()
        .skip_while(|l| !l.contains(begin))
        .skip(1)
        .take_while(|l| !l.contains(end))
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect()
}

#[test]
fn readme_example_matches_the_compiled_one() {
    let readme = include_str!("../README.md");
    let example = include_str!("../examples/quickstart.rs");

    let block: Vec<String> = readme
        .lines()
        .skip_while(|l| !l.starts_with("```rust"))
        .skip(1)
        .take_while(|l| !l.starts_with("```"))
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty() && !l.starts_with("use "))
        .collect();

    let compiled = marked_body(example, "README:begin", "README:end");
    assert_eq!(
        block, compiled,
        "README.md's Rust example and examples/quickstart.rs have diverged; \
         update whichever is wrong"
    );
}

/// Prose drifts more easily than code blocks: the root README claimed `Store`
/// persisted with `save`/`load`, which it never has. Every sentence pairing a
/// type with a method name is checked against that type's real methods.
#[test]
fn documented_methods_exist() {
    let store: &[&str] = &[
        "new",
        "add",
        "add_batch",
        "get",
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
        "contains",
        "size",
        "dimension",
        "metric",
    ];
    let disk_builder: &[&str] = &["new", "add", "save", "size"];
    let all_types = ["Store", "Index", "DiskStore", "DiskStoreBuilder"];

    for readme in [
        include_str!("../../README.md"),
        include_str!("../README.md"),
    ] {
        // Markdown wraps sentences across lines, so judge sentences, not lines:
        // a type and a method sharing a line may belong to different claims.
        let flat = readme.replace('\n', " ");
        for sentence in flat.split(". ") {
            for (ty, methods) in [
                ("DiskStoreBuilder", disk_builder),
                ("DiskStore", disk),
                ("Store", store),
                ("Index", index),
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
