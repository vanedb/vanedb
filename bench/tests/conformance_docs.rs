//! The conformance specs must not describe things that do not exist.
//!
//! This has now gone wrong three times, each the same shape — documentation
//! asserting verification that was never there:
//!
//! 1. `7c938bb`: thirteen graph fixtures that no test loaded.
//! 2. `conformance/legacy_graph/generate.py` emitting five `.qvrd` files into
//!    a directory that does not exist, no test reads, and `SHA256SUMS` does
//!    not list — behind a README describing a "C++ roundtrip" that was never
//!    written.
//! 3. A bullet in that same README detailing those files, and a
//!    `ctest -R 'legacy'` in its verification block matching zero tests —
//!    both found *inside* the fix for occurrence 2.
//!
//! Prose is not compiled, so nothing caught any of them. This checks the two
//! claims a spec makes that can be checked mechanically: the paths it names
//! exist, and the commands it tells you to run refer to something real.
//!
//! It lives in `bench` because it reads the whole repository, and `bench` is
//! a separate workspace that is never published.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("bench/ has a parent")
        .to_path_buf()
}

/// Every `README.md` under `conformance/`, with its repo-relative name.
fn conformance_docs() -> Vec<(String, String)> {
    let root = repo_root();
    let mut docs = Vec::new();
    let mut dirs = vec![root.join("conformance")];
    while let Some(dir) = dirs.pop() {
        for entry in fs::read_dir(&dir).expect("conformance/ is readable") {
            let path = entry.expect("readable entry").path();
            if path.is_dir() {
                dirs.push(path);
            } else if path.file_name().is_some_and(|n| n == "README.md") {
                let rel = path
                    .strip_prefix(&root)
                    .expect("under the repo root")
                    .to_string_lossy()
                    .into_owned();
                docs.push((rel, fs::read_to_string(&path).expect("readable README")));
            }
        }
    }
    docs.sort();
    assert!(!docs.is_empty(), "no conformance READMEs found");
    docs
}

/// The repository's top-level directories. A backticked token starting with
/// one of these and containing `/` is a path claim; anything else is prose.
const TOP_LEVEL: [&str; 10] = [
    "vanedb/",
    "vanedb-py/",
    "vanedb-wasm/",
    "vanedb-capi/",
    "cpp/",
    "bench/",
    "conformance/",
    "docs/",
    "scripts/",
    ".github/",
];

/// Repo-relative paths named inside backticks.
fn path_claims(body: &str) -> BTreeSet<String> {
    body.split('`')
        .skip(1)
        .step_by(2)
        .map(|token| token.trim().trim_end_matches(&[',', '.', ';'][..]))
        // A glob stands for a set, not a file; `cpp/build` is a build output.
        .filter(|token| !token.contains('*') && !token.starts_with("cpp/build"))
        .filter(|token| TOP_LEVEL.iter().any(|top| token.starts_with(top)))
        .map(str::to_string)
        .collect()
}

#[test]
fn every_path_the_conformance_specs_name_exists() {
    let root = repo_root();
    let mut checked = 0usize;
    let mut missing = Vec::new();
    for (name, body) in conformance_docs() {
        for claim in path_claims(&body) {
            checked += 1;
            if !root.join(&claim).exists() {
                missing.push(format!("{name} names {claim}, which does not exist"));
            }
        }
    }
    assert!(
        missing.is_empty(),
        "conformance specs name paths that are not there:\n  {}",
        missing.join("\n  ")
    );
    // Without this the test passes vacuously if the extractor ever stops
    // matching — which is exactly how prose drifts back in.
    assert!(
        checked >= 10,
        "only {checked} path claims found; the extractor is probably broken"
    );
}

/// Shell lines inside fenced blocks, comments and blanks dropped.
fn commands(body: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut inside = false;
    for line in body.lines() {
        if line.starts_with("```") {
            inside = line.len() > 3 || !inside;
            if line.trim_end() == "```" {
                inside = !inside;
            }
            continue;
        }
        let line = line.trim();
        if inside && !line.is_empty() && !line.starts_with('#') {
            out.push(line.to_string());
        }
    }
    out
}

/// Every `fn` name declared anywhere under a crate directory.
fn test_names_in(dir: &Path) -> String {
    let mut all = String::new();
    let mut dirs = vec![dir.to_path_buf()];
    while let Some(d) = dirs.pop() {
        let Ok(entries) = fs::read_dir(&d) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                if path.file_name().is_some_and(|n| n == "target") {
                    continue;
                }
                dirs.push(path);
            } else if path.extension().is_some_and(|e| e == "rs") {
                all.push_str(&fs::read_to_string(&path).unwrap_or_default());
            }
        }
    }
    all
}

/// Test names `ctest` can select: Catch2 `TEST_CASE` titles, plus any
/// `add_test(NAME ...)` registered directly in CMake.
fn cpp_test_names(root: &Path) -> Vec<String> {
    let mut names = Vec::new();
    let mut dirs = vec![root.join("cpp")];
    let mut sources = String::new();
    while let Some(d) = dirs.pop() {
        let Ok(entries) = fs::read_dir(&d) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                if path.file_name().is_some_and(|n| n == "build") {
                    continue;
                }
                dirs.push(path);
            } else {
                let keep = path.extension().is_some_and(|e| e == "cpp" || e == "mm")
                    || path.file_name().is_some_and(|n| n == "CMakeLists.txt");
                if keep {
                    sources.push_str(&fs::read_to_string(&path).unwrap_or_default());
                }
            }
        }
    }
    for (marker, open) in [("TEST_CASE(", '"'), ("add_test(NAME", ' ')] {
        let mut rest = sources.as_str();
        while let Some(at) = rest.find(marker) {
            rest = &rest[at + marker.len()..];
            let title = if open == '"' {
                rest.split('"').nth(1).unwrap_or_default()
            } else {
                rest.split_whitespace().next().unwrap_or_default()
            };
            if !title.is_empty() {
                names.push(title.to_string());
            }
        }
    }
    names
}

#[test]
fn every_command_the_conformance_specs_give_refers_to_something_real() {
    let root = repo_root();
    let mut checked = 0usize;
    let mut broken = Vec::new();

    for (name, body) in conformance_docs() {
        for cmd in commands(&body) {
            let words: Vec<&str> = cmd.split_whitespace().collect();

            // `shasum -a 256 -c <path>` / `sha256sum -c <path>`
            if words[0] == "shasum" || words[0] == "sha256sum" {
                if let Some(path) = words.iter().skip_while(|w| **w != "-c").nth(1) {
                    checked += 1;
                    if !root.join(path).exists() {
                        broken.push(format!("{name}: `{cmd}` checksums a missing {path}"));
                    }
                }
            // `cargo test -p <crate> --lib <filter>`
            } else if words[0] == "cargo" && words.get(1) == Some(&"test") {
                let krate = words
                    .iter()
                    .skip_while(|w| **w != "-p")
                    .nth(1)
                    .unwrap_or(&"vanedb");
                // Only these cargo flags consume the word after them; any
                // other bare word is the test-name filter.
                const TAKES_VALUE: [&str; 8] = [
                    "-p",
                    "--package",
                    "--test",
                    "--bench",
                    "--example",
                    "--features",
                    "--manifest-path",
                    "--target",
                ];
                let mut skip_next = false;
                let filter = words[2..].iter().find(|w| {
                    let take = !skip_next && !w.starts_with('-');
                    skip_next = TAKES_VALUE.contains(w);
                    take
                });
                if let Some(filter) = filter {
                    checked += 1;
                    let sources = test_names_in(&root.join(krate));
                    if !sources.contains(&format!("fn {filter}")) {
                        broken.push(format!(
                            "{name}: `{cmd}` filters on {filter}, which names no test in {krate}"
                        ));
                    }
                }
            // `ctest --test-dir <dir> -R '<pattern>'`
            } else if words[0] == "ctest" {
                if let Some(pat) = words.iter().skip_while(|w| **w != "-R").nth(1) {
                    checked += 1;
                    let pat = pat.trim_matches(['\'', '"']).to_lowercase();
                    // `-R` filters registered test names, so match those and
                    // not the whole source: "legacy" appears in comments in
                    // two C++ files, which would make any pattern pass.
                    let names = cpp_test_names(&root);
                    if !names.iter().any(|n| n.to_lowercase().contains(&pat)) {
                        broken.push(format!(
                            "{name}: `{cmd}` selects on {pat:?}, which matches no C++ test name"
                        ));
                    }
                }
            // `python conformance/.../generate.py`
            } else if words[0].starts_with("python") {
                if let Some(script) = words.iter().find(|w| w.ends_with(".py")) {
                    checked += 1;
                    if !root.join(script).exists() {
                        broken.push(format!("{name}: `{cmd}` runs a missing {script}"));
                    }
                }
            }
        }
    }

    assert!(
        broken.is_empty(),
        "conformance specs give commands that refer to nothing:\n  {}",
        broken.join("\n  ")
    );
    assert!(
        checked >= 2,
        "only {checked} commands recognised; the extractor is probably broken"
    );
}
