//! The generated header must compile warning-free as C99, C11 and C++17 under
//! `-Wall -Wextra -pedantic -Werror` (RFC 0002 stage 1).
//!
//! It is this crate's whole public interface, and CI's C test exercises the
//! *C++* engine's hand-written header, so nothing else feeds this one to a C
//! compiler. Every compiler found on the host is tried: `cc`/`c++` (or `$CC`
//! and `$CXX`), plus `gcc`/`g++` and `clang`/`clang++` when present. MSVC's
//! `/W4 /WX` leg runs in CI on Windows through `scripts/check_capi_header.py`,
//! which carries the same translation unit; this test has no MSVC to drive.
//! A host with no compiler at all skips with a message; CI always has one.

use std::process::Command;

/// A translation unit that uses the handle typedefs, one function per handle
/// type, the filter callback and the version macros, so an incomplete or
/// conflicting declaration fails here rather than in a user's build.
const SOURCE: &str = r#"#include "vanedb_rs_capi.h"
#include <stdio.h>
static bool accepts(uint64_t id, void *data) {
    return id == *(const uint64_t *)data;
}
int main(void) {
    vanedb_rs_store s = VANEDB_RS_NULL_HANDLE;
    vanedb_rs_index h = VANEDB_RS_NULL_HANDLE;
    vanedb_rs_disk d = VANEDB_RS_NULL_HANDLE;
    vanedb_rs_handle any = s;
    float query = 0.0f, distance;
    uint64_t selected = 42, id;
    vanedb_rs_filter_fn filter = accepts;
    if (vanedb_rs_abi_version() != VANEDB_RS_ABI_VERSION) {
        return 1;
    }
    (void)any;
    (void)vanedb_rs_store_len(s);
    (void)vanedb_rs_index_len(h);
    (void)vanedb_rs_disk_len(d);
    (void)vanedb_rs_store_search_filtered(s, &query, 1, filter, &selected,
        0, 0, 0, 0, &id, &distance);
    (void)vanedb_rs_index_search_filtered(h, &query, 1, 0, filter, &selected,
        0, 0, 0, 0, &id, &distance);
    (void)vanedb_rs_disk_search_filtered(d, &query, 1, filter, &selected,
        0, 0, 0, 0, &id, &distance);
    (void)vanedb_rs_handle_count();
    printf("%s %u\n", VANEDB_RS_VERSION, (unsigned)VANEDB_RS_INVALID_HANDLE);
    return 0;
}
"#;

fn available(compiler: &str) -> bool {
    Command::new(compiler)
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn compilers(kind: &str) -> Vec<String> {
    let (env, default, extra) = match kind {
        "c" => ("CC", "cc", ["gcc", "clang"]),
        _ => ("CXX", "c++", ["g++", "clang++"]),
    };
    let mut found: Vec<String> = Vec::new();
    let mut consider = |name: String| {
        if !found.contains(&name) && available(&name) {
            found.push(name);
        }
    };
    consider(std::env::var(env).unwrap_or_else(|_| default.to_string()));
    for name in extra {
        consider(name.to_string());
    }
    found
}

fn compile(compiler: &str, standard: &str, suffix: &str) {
    let include = concat!(env!("CARGO_MANIFEST_DIR"), "/include");
    let src = std::env::temp_dir().join(format!(
        "vanedb_header_{}_{}.{suffix}",
        std::process::id(),
        standard.trim_start_matches("-std=")
    ));
    std::fs::write(&src, SOURCE).unwrap();
    let out = Command::new(compiler)
        .args([
            standard,
            "-Wall",
            "-Wextra",
            "-pedantic",
            "-Werror",
            "-fsyntax-only",
        ])
        .arg("-I")
        .arg(include)
        .arg(&src)
        .output()
        .expect("failed to run the compiler");
    let _ = std::fs::remove_file(&src);
    assert!(
        out.status.success(),
        "the generated header does not compile with {compiler} {standard}:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn the_generated_header_compiles_as_c99_and_c11() {
    let compilers = compilers("c");
    if compilers.is_empty() {
        eprintln!("skipping: no C compiler available");
        return;
    }
    for compiler in &compilers {
        for standard in ["-std=c99", "-std=c11"] {
            compile(compiler, standard, "c");
        }
    }
}

#[test]
fn the_generated_header_compiles_as_cxx17() {
    let compilers = compilers("c++");
    if compilers.is_empty() {
        eprintln!("skipping: no C++ compiler available");
        return;
    }
    for compiler in &compilers {
        compile(compiler, "-std=c++17", "cpp");
    }
}
