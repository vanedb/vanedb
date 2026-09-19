//! Build the hnswlib C++ bridge and the sqlite-vec amalgamation.

use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let vendor = manifest_dir.join("vendor");

    // Prefer a host g++ when available: some images ship a clang that is
    // selected as `c++` but cannot find libstdc++ headers/libs.
    if env::var_os("CXX").is_none() && PathBuf::from("/usr/bin/g++").exists() {
        // SAFETY: build scripts run single-threaded before compilation.
        unsafe { env::set_var("CXX", "g++") };
    }
    if env::var_os("CC").is_none() && PathBuf::from("/usr/bin/gcc").exists() {
        unsafe { env::set_var("CC", "gcc") };
    }

    // --- hnswlib bridge -------------------------------------------------
    let hnsw_dir = vendor.join("hnswlib");
    println!(
        "cargo:rerun-if-changed={}",
        hnsw_dir.join("bridge.cpp").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        hnsw_dir.join("bridge.h").display()
    );
    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .file(hnsw_dir.join("bridge.cpp"))
        .include(&hnsw_dir)
        .flag_if_supported("-Wno-unused-parameter")
        .flag_if_supported("-Wno-sign-compare")
        .compile("hnsw_bridge");

    // --- sqlite-vec amalgamation (static, against rusqlite's bundled SQLite)
    let sqlite_vec_c = vendor.join("sqlite-vec.c");
    let sqlite_vec_h = vendor.join("sqlite-vec.h");
    println!("cargo:rerun-if-changed={}", sqlite_vec_c.display());
    println!("cargo:rerun-if-changed={}", sqlite_vec_h.display());

    let mut build = cc::Build::new();
    build
        .file(&sqlite_vec_c)
        .include(&vendor)
        .define("SQLITE_CORE", None)
        .define("SQLITE_VEC_STATIC", None)
        .flag_if_supported("-Wno-unused-parameter")
        .flag_if_supported("-Wno-sign-compare")
        .flag_if_supported("-std=c11");

    if let Ok(include) = env::var("DEP_SQLITE3_INCLUDE") {
        build.include(include);
    } else if let Ok(lib) = pkg_config::probe_library("sqlite3") {
        for path in lib.include_paths {
            build.include(path);
        }
    }

    build.compile("sqlite_vec");

    // Surface the path-dep vanedb version for result rows.
    let vanedb_toml = manifest_dir.join("../../vanedb/Cargo.toml");
    if let Ok(text) = std::fs::read_to_string(vanedb_toml) {
        for line in text.lines() {
            if let Some(rest) = line.strip_prefix("version = \"") {
                if let Some(v) = rest.strip_suffix('"') {
                    println!("cargo:rustc-env=VANEDB_VERSION={v}");
                    break;
                }
            }
        }
    }
}
