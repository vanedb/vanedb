use pyo3::prelude::*;

#[pymodule]
fn vanedb(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", "0.1.0")?;
    Ok(())
}
