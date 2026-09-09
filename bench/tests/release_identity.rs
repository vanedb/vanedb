//! Every place the release version is written must agree.
//!
//! The version lives in eight hand-maintained places across four languages,
//! and nothing compared them. Stamping 0.1.0 updated seven and missed
//! `cpp/src/core/version.h`, whose `VERSION_STRING` is what the C++ Python
//! module reports as `__version__`. CI caught it, but only on one of the
//! fifty-odd jobs, and only because that binding happens to compare its
//! `__version__` against its wheel metadata.
//!
//! The cost of missing one is not symmetric with the cost of checking. Both
//! publish workflows gate a tag on `vanedb/Cargo.toml`, so a stale value
//! elsewhere does not stop the release — it ships inside it, and a published
//! version cannot be recalled.
//!
//! `cpp/CMakeLists.txt` is the one site that cannot carry a prerelease
//! suffix: CMake's `project(VERSION)` accepts only numeric components. It is
//! checked against the release core alone.

use std::fs;
use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("bench/ has a parent")
        .to_path_buf()
}

fn read(rel: &str) -> String {
    let path = repo_root().join(rel);
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("{rel} must be readable: {e}"))
}

/// The value of the first `version = "..."` key in a TOML file. Naive on
/// purpose: it must find the `[package]` version, which is always first in
/// this repo's manifests, and never a dependency's.
fn toml_version(rel: &str) -> String {
    read(rel)
        .lines()
        .find_map(|line| {
            let rest = line
                .strip_prefix("version")?
                .trim_start()
                .strip_prefix('=')?;
            Some(rest.trim().trim_matches('"').to_string())
        })
        .unwrap_or_else(|| panic!("{rel} declares no version"))
}

/// The numeric part, dropping any prerelease suffix. Used ONLY for the two
/// sites that cannot express one — never to compare two sites that can, or a
/// leftover `-rc.1` becomes invisible, which is the exact bug this file
/// exists to catch.
fn release_core(version: &str) -> &str {
    version.split_once('-').map_or(version, |(core, _)| core)
}

/// Every site that states the release version, as (what it is, the value).
fn declared_versions() -> Vec<(&'static str, String)> {
    let mut sites: Vec<(&'static str, String)> = [
        "vanedb/Cargo.toml",
        "vanedb-py/Cargo.toml",
        "vanedb-capi/Cargo.toml",
        "vanedb-wasm/Cargo.toml",
        "vanedb-py/pyproject.toml",
        "cpp/pyproject.toml",
    ]
    .iter()
    .map(|rel| (*rel, toml_version(rel)))
    .collect();

    // The C++ engine reports this as `vanedb_cpp.__version__`.
    let header = read("cpp/src/core/version.h");
    let string = header
        .lines()
        .find_map(|l| l.split_once("VERSION_STRING = ")?.1.split('"').nth(1))
        .expect("version.h declares VERSION_STRING")
        .to_string();
    sites.push(("cpp/src/core/version.h VERSION_STRING", string));

    // `VANEDB_RS_VERSION` in the generated C header, so a consumer can compare
    // it against `vanedb_rs_version()` at runtime. cbindgen copies its preamble
    // verbatim, so this used to be a literal in `cbindgen.toml` that drifted
    // silently — the same failure mode as the two sites above. `build.rs` now
    // stamps the crate version over a placeholder, which makes drift
    // impossible at the source; read the committed header, because that is the
    // artifact a consumer actually compiles against.
    let header = read("vanedb-capi/include/vanedb_rs_capi.h");
    let macro_version = header
        .lines()
        .find_map(|l| {
            l.split_once("#define VANEDB_RS_VERSION ")?
                .1
                .split('"')
                .nth(1)
        })
        .expect("the generated header defines VANEDB_RS_VERSION")
        .to_string();
    sites.push((
        "vanedb-capi/include/vanedb_rs_capi.h VANEDB_RS_VERSION",
        macro_version,
    ));

    // The npm package's version is not listed here. It is copied from
    // vanedb-wasm/Cargo.toml by `scripts/build_npm_package.py`, and that
    // manifest is already checked above — so the copy is pinned at its source.
    // An earlier version read `target/npm/vanedb-wasm/package.json` behind an
    // `if exists()`, which nothing in a bench test run produces: the branch was
    // dead in every context it executed in, which is the shape of test this
    // release has spent its time removing.

    // Doxygen renders this on the generated C++ docs. A free-form string, so
    // it can spell a prerelease and must match in full; it read "0.1.0"
    // throughout the 0.1.0-rc.1 period without anything noticing.
    let doxygen = read("cpp/Doxyfile")
        .lines()
        .find_map(|l| l.split_once("PROJECT_NUMBER")?.1.split('"').nth(1))
        .expect("cpp/Doxyfile declares PROJECT_NUMBER")
        .to_string();
    sites.push(("cpp/Doxyfile PROJECT_NUMBER", doxygen));

    sites
}

/// Sites that are integers or CMake components, so they can never carry a
/// prerelease suffix. Compared against the release core.
fn core_only_sites() -> Vec<(&'static str, String)> {
    let header = read("cpp/src/core/version.h");
    // The header's numeric components, which drift independently of its string.
    let mut sites = Vec::new();
    let component = |name: &str| -> String {
        header
            .lines()
            .find_map(|l| {
                let rest = l.split_once(&format!("{name} = "))?.1;
                Some(rest.trim_end_matches(';').trim().to_string())
            })
            .unwrap_or_else(|| panic!("version.h declares no {name}"))
    };
    sites.push((
        "cpp/src/core/version.h components",
        format!(
            "{}.{}.{}",
            component("VERSION_MAJOR"),
            component("VERSION_MINOR"),
            component("VERSION_PATCH")
        ),
    ));

    sites
}

#[test]
fn every_declared_version_agrees() {
    let sites = declared_versions();
    let core_only = core_only_sites();
    assert!(
        sites.len() + core_only.len() >= 8,
        "only {} version sites found; the extractor is probably broken",
        sites.len() + core_only.len()
    );

    // `vanedb/Cargo.toml` is what both publish workflows gate the tag on, so
    // it is the reference the others must match.
    let (_, expected) = &sites[0];
    let mut disagreeing: Vec<String> = sites
        .iter()
        // Compared in full: every one of these can spell a prerelease, so a
        // stale suffix must show up as a difference.
        .filter(|(_, v)| v != expected)
        .map(|(what, v)| format!("{what} says {v}, expected {expected}"))
        .collect();
    disagreeing.extend(
        core_only
            .iter()
            .filter(|(_, v)| v != release_core(expected))
            .map(|(what, v)| format!("{what} says {v}, expected {}", release_core(expected))),
    );
    assert!(
        disagreeing.is_empty(),
        "the release version disagrees across sites:\n  {}",
        disagreeing.join("\n  ")
    );
}

#[test]
fn the_cmake_project_version_matches_the_release_core() {
    // `project(vanedb VERSION 0.1.0 ...)`. CMake takes numeric components
    // only, so it can never carry `-rc.1`; the core still has to agree.
    let cmake = read("cpp/CMakeLists.txt");
    let declared = cmake
        .lines()
        // Anchored to `project(`: line 1 is `cmake_minimum_required(VERSION
        // 3.20)`, which a bare search for "VERSION " finds first.
        .find(|l| l.trim_start().starts_with("project("))
        .and_then(|l| l.split_once("VERSION ")?.1.split_whitespace().next())
        .expect("cpp/CMakeLists.txt declares a project VERSION")
        .to_string();
    let crate_version = toml_version("vanedb/Cargo.toml");
    assert_eq!(
        declared,
        release_core(&crate_version),
        "cpp/CMakeLists.txt project VERSION must match the release core"
    );
}
