//! Reads the generated header so a test can enumerate the ABI rather than
//! restate it. Shared by the ABI-version, export-list and handle tests.
//!
//! The same parse lives in `scripts/capi_exports.py`; both are deliberately
//! naive and rely on cbindgen's output shape: declarations end in `;`, block
//! comments are `/* */`, and no macro takes arguments.

#![allow(dead_code)]

use std::path::PathBuf;

pub fn header_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("include/vanedb_rs_capi.h")
}

pub fn header() -> String {
    std::fs::read_to_string(header_path()).expect("the generated header must exist")
}

/// The header with every `/* ... */` comment removed, so a function named in
/// prose is not mistaken for a declaration.
pub fn strip_comments(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find("/*") {
        out.push_str(&rest[..start]);
        match rest[start..].find("*/") {
            Some(end) => rest = &rest[start + end + 2..],
            None => return out,
        }
    }
    out.push_str(rest);
    out
}

/// One declared function: its name and the text of its parameter list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Declaration {
    pub name: String,
    pub parameters: String,
}

/// Every `vanedb_rs_*` function the header declares, in header order.
///
/// A name directly followed by `(` is a declaration; the function-pointer
/// typedef `bool (*vanedb_rs_filter_fn)(...)` has its name followed by `)`
/// and the handle typedefs have no parenthesis, so neither is picked up.
pub fn declarations() -> Vec<Declaration> {
    let text = strip_comments(&header());
    let bytes = text.as_bytes();
    let mut found = Vec::new();
    let mut from = 0;
    while let Some(offset) = text[from..].find("vanedb_rs_") {
        let start = from + offset;
        let mut end = start;
        while end < bytes.len() && (bytes[end].is_ascii_alphanumeric() || bytes[end] == b'_') {
            end += 1;
        }
        from = end;
        let name = &text[start..end];
        let after = text[end..].trim_start();
        if !after.starts_with('(') {
            continue;
        }
        // A prefix character that continues an identifier would make this a
        // longer name; cbindgen never produces one, but be exact.
        if start > 0 && (bytes[start - 1].is_ascii_alphanumeric() || bytes[start - 1] == b'_') {
            continue;
        }
        let close = after.find(')').expect("a parameter list closes");
        found.push(Declaration {
            name: name.to_string(),
            parameters: after[1..close].to_string(),
        });
    }
    assert!(
        found.len() > 40,
        "found only {} declarations; the parse is broken, not the header",
        found.len()
    );
    found
}

/// Which handle type a declaration takes, if any: `store`, `index` or
/// `disk`. A handle parameter is written with its typedef, never as a bare
/// `uint64_t`, so a typedef name inside the parameter list is the signal.
pub fn handle_kind(declaration: &Declaration) -> Option<&'static str> {
    for kind in ["store", "index", "disk"] {
        let typedef = format!("vanedb_rs_{kind}");
        if declaration
            .parameters
            .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .any(|token| token == typedef)
        {
            return Some(kind);
        }
    }
    None
}

/// The integer a `#define NAME value` line carries.
pub fn defined_integer(name: &str) -> u64 {
    let needle = format!("#define {name} ");
    header()
        .lines()
        .find_map(|line| line.strip_prefix(&needle))
        .unwrap_or_else(|| panic!("the header defines {name}"))
        .trim()
        .trim_end_matches(['u', 'U', 'l', 'L'])
        .parse()
        .unwrap_or_else(|_| panic!("{name} is an integer literal"))
}
