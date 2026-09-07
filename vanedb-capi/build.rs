fn main() {
    let crate_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let out = format!("{crate_dir}/include/vanedb_rs_capi.h");

    // Fail loudly. Swallowing this left the previously committed header in
    // place and the build green, so an ABI change plus a forgotten header
    // commit would ship a header that lies about the ABI. The C ABI has no
    // registry release path, so consumers vendor this file by hand and have
    // no way to notice.
    let bindings = cbindgen::generate(&crate_dir)
        .expect("cbindgen failed to generate the C header from src/lib.rs");
    bindings.write_to_file(&out);

    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=cbindgen.toml");
}
