fn main() {
    let crate_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let out = format!("{crate_dir}/include/vanedb_rs_capi.h");
    if let Ok(bindings) = cbindgen::generate(&crate_dir) {
        bindings.write_to_file(&out);
    }
    println!("cargo:rerun-if-changed=src/lib.rs");
}
