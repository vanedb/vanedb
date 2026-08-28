use std::path::{Path, PathBuf};
use std::process::Command;

fn git_short_rev(dir: &Path) -> String {
    Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".into())
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("bench must live directly below the repository root")
        .to_path_buf()
}

fn main() {
    let root = repo_root();
    let cpp = root.join("cpp");
    let mut cfg = cmake::Config::new(&cpp);
    cfg.define("VANEDB_BUILD_CAPI", "ON")
        .define("VANEDB_BUILD_TESTS", "OFF")
        .define("VANEDB_BUILD_BENCHMARKS", "OFF")
        .define("VANEDB_BUILD_PYTHON", "OFF")
        .define("VANEDB_BUILD_EXAMPLES", "OFF")
        // Pin the C++ profile so results are comparable no matter which Rust
        // profile triggered this build.
        .profile("Release")
        .build_target("vanedb_cpp_capi");

    // vanedb-cpp gates SIMD at compile time (#ifdef __AVX2__), and its
    // CMakeLists gives -mavx2 -mfma to its own perf targets but NOT to
    // vanedb_cpp_capi. The Rust side detects AVX2 at runtime, so without
    // these flags an x86_64 harness silently compares Rust-AVX2 against
    // C++-scalar. aarch64 is unaffected (NEON is unconditional).
    if std::env::var("CARGO_CFG_TARGET_ARCH").as_deref() == Ok("x86_64") {
        if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
            cfg.cxxflag("/arch:AVX2");
        } else {
            cfg.cxxflag("-mavx2").cxxflag("-mfma");
        }
    }
    let dst = cfg.build();

    // CMake (non-install) places the archive under <dst>/build.
    println!("cargo:rustc-link-search=native={}/build", dst.display());
    println!("cargo:rustc-link-lib=static=vanedb_cpp_capi");

    // C++ standard library.
    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=dylib=c++");
    } else {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }

    // Both engines now share one immutable source revision.
    println!(
        "cargo:rustc-env=VANEDB_MONOREPO_REV={}",
        git_short_rev(&root)
    );
    println!("cargo:rerun-if-changed=Cargo.toml");

    // Watch everything the static lib is built from — a stale lib here would
    // silently benchmark old C++ code (all of src/, not just src/core:
    // src/utils headers are included too).
    println!("cargo:rerun-if-changed={}", cpp.join("capi").display());
    println!("cargo:rerun-if-changed={}", cpp.join("src").display());
    println!(
        "cargo:rerun-if-changed={}",
        cpp.join("CMakeLists.txt").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        root.join(".git/HEAD").display()
    );
}
