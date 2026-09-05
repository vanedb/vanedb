//! The example printed in README.md, kept compiling.
//!
//! `tests/readme.rs` asserts the README's Rust block matches the body below,
//! so the crates.io front page cannot drift from the API.

use vanedb::{Index, Metric};

fn main() -> Result<(), vanedb::VaneError> {
    let embedding = vec![0.1_f32; 768];
    let query = vec![0.1_f32; 768];

    // README:begin
    let index = Index::builder(768, Metric::Cosine)
        .capacity(100_000)
        .build()?;
    index.add(1, &embedding)?;

    // Results come back nearest first.
    let hits = index.search(&query, 10)?;
    // README:end

    assert_eq!(hits[0].id, 1);
    Ok(())
}
