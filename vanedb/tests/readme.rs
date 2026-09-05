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
