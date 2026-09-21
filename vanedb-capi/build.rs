fn main() {
    let crate_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_env = std::env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();

    if target_os == "android" {
        // NDK r26 (used in CI) needs explicit alignment for Android 15's
        // 16 KiB page configurations. Static-library consumers set their own
        // final linker flags; these apply to the C ABI shared library.
        println!("cargo:rustc-link-arg-cdylib=-Wl,-z,max-page-size=16384");
        println!("cargo:rustc-link-arg-cdylib=-Wl,-z,common-page-size=16384");
    }

    // The exported-symbol allowlist (RFC 0002 stage 1). `exports/` holds
    // three spellings of the same `vanedb_rs_*` set, generated from the
    // header by `scripts/capi_exports.py`; a test checks they match the
    // header and CI checks they match the built library. rustc already
    // restricts a cdylib's exports to its `#[no_mangle]` items; passing the
    // list explicitly pins the contract to a file a consumer can read and
    // makes the shared library's export set independent of how rustc
    // happens to do that.
    //
    // MSVC is the exception: link.exe takes one `/DEF`, rustc already passes
    // its own, and the generated `.def` there is consumed by the `dumpbin`
    // check in CI rather than by the link. Windows GNU is left to rustc for
    // the same reason. The Apple and ELF linkers accept an additional list.
    let exports = format!("{crate_dir}/exports");
    if target_os == "macos" || target_os == "ios" {
        println!(
            "cargo:rustc-link-arg-cdylib=-Wl,-exported_symbols_list,{exports}/vanedb_capi.exp"
        );
        println!("cargo:rerun-if-changed=exports/vanedb_capi.exp");
    } else if target_os != "windows" && target_env != "msvc" {
        println!("cargo:rustc-link-arg-cdylib=-Wl,--version-script={exports}/vanedb_capi.map");
        println!("cargo:rerun-if-changed=exports/vanedb_capi.map");
    }

    let out = format!("{crate_dir}/include/vanedb_rs_capi.h");
    cbindgen::generate(&crate_dir)
        .expect("failed to generate the C ABI header")
        .write_to_file(&out);

    // The header's VANEDB_RS_VERSION exists so a consumer can catch a shared
    // object that does not match the header it compiled against. cbindgen
    // copies `after_includes` verbatim, so a literal there is a second place
    // to remember at every version bump, and nothing reads the macro back --
    // the drift would be silent, in the one macro whose whole job is to detect
    // drift. Substitute the crate version instead.
    let generated = std::fs::read_to_string(&out).expect("failed to read the generated header");
    let version = std::env::var("CARGO_PKG_VERSION").unwrap();
    let stamped = generated.replace("@CARGO_PKG_VERSION@", &version);
    assert!(
        stamped != generated,
        "cbindgen.toml no longer contains the @CARGO_PKG_VERSION@ placeholder, so the \
         header's VANEDB_RS_VERSION is not being stamped; a replace that matches nothing \
         would leave it silently wrong"
    );
    std::fs::write(&out, stamped).expect("failed to write the stamped header");
    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=cbindgen.toml");
    // Belt and braces. Cargo's fingerprint already includes the package id, so
    // a version bump re-runs this script on its own -- an earlier comment here
    // claimed otherwise and was wrong. This makes the dependency explicit.
    println!("cargo:rerun-if-env-changed=CARGO_PKG_VERSION");
}
