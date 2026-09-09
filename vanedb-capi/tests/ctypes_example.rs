//! The shipped ctypes example must declare a signature for every call it makes.
//!
//! An agent reviewing this crate crashed a Python interpreter with SIGSEGV
//! calling `vanedb_rs_last_error_message` through ctypes without declaring
//! `restype`. ctypes then treats a returned `const char*` as a C `int`,
//! truncating the pointer, and the dereference faults inside libffi with no
//! Python traceback. The library was sound; the caller had no guidance.
//!
//! `examples/ctypes_quickstart.py` is that guidance, and this checks the one
//! property that makes it safe: every function it calls is bound first. That
//! is a source-level check on purpose — an earlier version of this test ran
//! the example against the cdylib and *skipped* when it was absent, which it
//! is under a plain `cargo test`, so it passed while testing nothing. CI runs
//! the example for real; this runs everywhere.

use std::collections::HashSet;
use std::path::PathBuf;

/// Drop `#` comments. Python has no block comments, and this example contains
/// no `#` inside a string literal, so dropping from the first `#` on each line
/// is sufficient and keeps the scan honest.
fn strip_comments(source: &str) -> String {
    source
        .lines()
        .map(|line| line.split('#').next().unwrap_or(""))
        .collect::<Vec<_>>()
        .join("\n")
}

fn example() -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/ctypes_quickstart.py");
    std::fs::read_to_string(&path).expect("the ctypes example must exist")
}

/// Collect `lib.<name>` occurrences, split by whether they are a binding
/// (`lib.x.restype = ...`) or a call (`lib.x(...)`).
///
/// Comments are stripped first. Without that, commenting out a binding while
/// leaving its call passes every check here *and* the CI run — the truncated
/// return happened to be a small `size_t` — and a comment merely mentioning
/// the message accessor could shadow a real `c_char_p` binding from the
/// `c_void_p` check below. Both were found by mutating this example.
fn bound_and_called(source: &str) -> (HashSet<String>, HashSet<String>) {
    let source = strip_comments(source);
    let source = source.as_str();
    let (mut bound, mut called) = (HashSet::new(), HashSet::new());
    for (index, _) in source.match_indices("lib.") {
        let rest = &source[index + 4..];
        let name: String = rest
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        if name.is_empty() {
            continue;
        }
        let after = &rest[name.len()..];
        if after.starts_with(".restype") {
            bound.insert(name);
        } else if after.starts_with('(') {
            called.insert(name);
        }
    }
    (bound, called)
}

#[test]
fn the_ctypes_example_binds_every_function_it_calls() {
    let source = example();
    let (bound, called) = bound_and_called(&source);

    assert!(
        !called.is_empty(),
        "found no calls at all; the scan is broken, not the example"
    );
    let unbound: Vec<_> = called.difference(&bound).cloned().collect();
    assert!(
        unbound.is_empty(),
        "these are called without a declared restype, which truncates a \
         returned pointer to a C int and segfaults on dereference: {unbound:?}"
    );
}

/// Every bound function must declare `argtypes` too. An undeclared pointer
/// argument is passed as an int, so a 64-bit handle arrives truncated — the
/// same defect on the way in.
#[test]
fn the_ctypes_example_declares_argtypes_for_everything_it_binds() {
    let source = example();
    let (bound, _) = bound_and_called(&source);
    let missing: Vec<_> = bound
        .iter()
        .filter(|name| !source.contains(&format!("lib.{name}.argtypes")))
        .cloned()
        .collect();
    assert!(missing.is_empty(), "bound without argtypes: {missing:?}");
}

/// The example must not use `c_char_p` for the error message. ctypes copies
/// the bytes into a Python object at the boundary, which hides that the
/// pointer is only valid until the next call on this thread — the lifetime the
/// example exists to teach.
#[test]
fn the_error_message_is_read_as_a_raw_pointer() {
    let source = example();
    let stripped = strip_comments(&source);
    let line = stripped
        .lines()
        .find(|l| l.contains("vanedb_rs_last_error_message.restype ="))
        .expect("the example must bind the message accessor");
    assert!(
        line.contains("c_void_p"),
        "expected c_void_p so the lifetime stays visible, got: {line}"
    );
}
