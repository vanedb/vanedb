use std::path::{Path, PathBuf};
use std::process::Command;

fn git_output(dir: &Path, args: &[&str]) -> Option<String> {
    Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
}

fn git_short_rev(dir: &Path) -> String {
    git_output(dir, &["rev-parse", "--short", "HEAD"]).unwrap_or_else(|| "unknown".into())
}

fn absolute_git_path(root: &Path, path: String) -> PathBuf {
    let path = PathBuf::from(path);
    if path.is_absolute() {
        path
    } else {
        root.join(path)
    }
}

fn watch_git_revision(root: &Path) {
    let Some(git_dir) = git_output(root, &["rev-parse", "--absolute-git-dir"]).map(PathBuf::from)
    else {
        return;
    };

    // Detached checkouts update HEAD directly. Branch checkouts update the
    // referenced file instead, and packed refs are the fallback when no loose
    // ref exists. Resolve these through Git so linked worktrees work too.
    println!("cargo:rerun-if-changed={}", git_dir.join("HEAD").display());
    if let Some(head_ref) = git_output(root, &["symbolic-ref", "-q", "HEAD"]) {
        if let Some(ref_path) = git_output(root, &["rev-parse", "--git-path", &head_ref]) {
            println!(
                "cargo:rerun-if-changed={}",
                absolute_git_path(root, ref_path).display()
            );
        }
    }
    if let Some(packed_refs) = git_output(root, &["rev-parse", "--git-path", "packed-refs"]) {
        println!(
            "cargo:rerun-if-changed={}",
            absolute_git_path(root, packed_refs).display()
        );
    }
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
    watch_git_revision(&root);
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
}
