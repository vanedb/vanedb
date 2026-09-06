fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("android") {
        // NDK r26 (used in CI) needs explicit alignment for Android 15's
        // 16 KiB page configurations. Static-library consumers set their own
        // final linker flags; these apply to the C ABI shared library.
        println!("cargo:rustc-link-arg-cdylib=-Wl,-z,max-page-size=16384");
        println!("cargo:rustc-link-arg-cdylib=-Wl,-z,common-page-size=16384");
    }
    let crate_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let out = format!("{crate_dir}/include/vanedb_rs_capi.h");
    if let Ok(bindings) = cbindgen::generate(&crate_dir) {
        bindings.write_to_file(&out);
    }
    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=cbindgen.toml");
}
