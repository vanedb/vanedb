use vanedb::{ApproxIndex, Metric};

fn main() -> Result<(), vanedb::VaneError> {
    let index = ApproxIndex::builder(3, Metric::Cosine)
        .capacity(100)
        .build()?;
    index.add_batch(&[101, 202], &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0])?;
    let hits = index.search(&[1.0, 0.0, 0.0], 1)?;
    assert_eq!(hits[0].id, 101);
    println!("Nearest id: {}, distance: {}", hits[0].id, hits[0].distance);
    Ok(())
}
