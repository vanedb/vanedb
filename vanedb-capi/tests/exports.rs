//! The exported-symbol allowlist under `exports/` is generated from the header
//! by `scripts/capi_exports.py`; `build.rs` hands the Apple list to the linker,
//! the plain list localizes the static library and the `.def` feeds the
//! Windows export check. If it
//! drifts from the header, either a new function is silently absent from the
//! shared library or a removed one is still promised. This holds the three
//! files to the header without needing Python;
//! CI additionally runs the script's `--verify` and checks the built library.

mod common;

use std::collections::BTreeSet;
use std::path::PathBuf;

fn exports(name: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("exports")
        .join(name);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| {
            panic!(
                "{} must exist; run scripts/capi_exports.py generate ({e})",
                path.display()
            )
        })
        // The files are pinned to LF in .gitattributes; a checkout that
        // ignores it must still compare the same symbol set.
        .replace("\r\n", "\n")
}

fn declared() -> BTreeSet<String> {
    common::declarations().into_iter().map(|d| d.name).collect()
}

#[test]
fn every_declared_function_is_namespaced() {
    for name in declared() {
        assert!(
            name.starts_with("vanedb_rs_"),
            "{name} escapes the namespace"
        );
    }
}

#[test]
fn the_apple_list_is_the_declared_functions_with_a_leading_underscore() {
    let listed: BTreeSet<String> = exports("vanedb_capi.exp")
        .lines()
        .map(|l| {
            l.strip_prefix('_')
                .expect("Mach-O C symbols carry an underscore")
        })
        .map(String::from)
        .collect();
    assert_eq!(listed, declared());
}

#[test]
fn the_module_definition_lists_exactly_the_declared_functions() {
    let text = exports("vanedb_capi.def");
    let mut lines = text.lines();
    assert_eq!(lines.next(), Some("EXPORTS"));
    let listed: BTreeSet<String> = lines.map(|l| l.trim().to_string()).collect();
    assert_eq!(listed, declared());
}

#[test]
fn the_plain_list_matches_the_header() {
    let listed: BTreeSet<String> = exports("vanedb_capi.syms")
        .lines()
        .map(String::from)
        .collect();
    assert_eq!(listed, declared());
}

/// The signature list the ABI gate diffs against a baseline release's
/// header: one prototype per declared function, no more, no fewer. The
/// exact prototype text is checked by `scripts/test_capi_abidiff.py`
/// against the generator; this holds the set of names without Python.
#[test]
fn the_signature_list_names_exactly_the_declared_functions() {
    let listed: BTreeSet<String> = exports("vanedb_capi.sigs")
        .lines()
        .map(|line| {
            let open = line.find('(').expect("a prototype has a parameter list");
            line[..open]
                .rsplit([' ', '*'])
                .next()
                .expect("a prototype names its function")
                .to_string()
        })
        .collect();
    assert_eq!(listed, declared());
}
