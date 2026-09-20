//! RFC 0012: `docs/PLATFORMS.md` is the only place a support tier is asserted,
//! so the README may name a platform only if that page covers it. The README
//! summary is what a reader sees first, and a name that drifts out of the page
//! is a support claim with no evidence behind it.
//!
//! The check is lexical. A platform name is an operating system word followed
//! by an architecture or a version (`Linux x86-64`, `Android 15`), a bare
//! operating system word otherwise, or a runtime, browser or libc name. Every
//! such name in the README must appear verbatim in the platform page, with
//! line wraps collapsed. Anything else in the README is prose and unchecked.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// The repository root, or `None` inside a crate archive, which carries
/// neither the root README nor `docs/`. Same guard as `readme.rs`.
fn repository() -> Option<PathBuf> {
    let repository = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()?
        .to_path_buf();
    repository
        .join("vanedb-py/Cargo.toml")
        .is_file()
        .then_some(repository)
}

fn read(repository: &Path, relative: &str) -> String {
    std::fs::read_to_string(repository.join(relative))
        .unwrap_or_else(|e| panic!("{relative}: {e}"))
        .replace('\r', "")
}

/// Operating systems. A following architecture or version narrows the name.
const SYSTEMS: &[&str] = &[
    "Linux", "macOS", "Windows", "iOS", "Android", "FreeBSD", "OpenBSD", "NetBSD", "Solaris",
    "illumos", "Alpine", "Ubuntu", "Debian",
];

/// Words that narrow an operating system to a platform.
const ARCHITECTURES: &[&str] = &[
    "x86-64", "x86_64", "x64", "ARM64", "arm64", "aarch64", "Intel", "i686", "armv7", "RISC-V",
    "s390x", "ppc64le",
];

/// Runtimes, browsers and libcs whose bare name is a platform claim.
const RUNTIMES: &[&str] = &[
    "Node.js", "Deno", "Bun", "Chrome", "Chromium", "Firefox", "WebKit", "Safari", "Edge", "glibc",
    "musl", "MSVC", "MinGW",
];

/// Keeps the inner dots of `Node.js` and `3.11`; a trailing dot ends a sentence.
fn trim(token: &str) -> &str {
    token
        .trim_matches(|c: char| !c.is_alphanumeric() && c != '-' && c != '_' && c != '.')
        .trim_end_matches('.')
}

/// A `Linux x86-64/ARM64` token pair names two platforms; `macOS/Linux` names
/// two systems. Both slashes are split before matching.
fn parts(token: &str) -> impl Iterator<Item = &str> {
    token.split(['/', '–']).map(trim).filter(|p| !p.is_empty())
}

/// Every platform name the text makes, in the spelling the platform page must
/// repeat.
fn platform_names(text: &str) -> BTreeSet<String> {
    let tokens: Vec<&str> = text.split_whitespace().collect();
    let mut names = BTreeSet::new();
    for (i, token) in tokens.iter().enumerate() {
        for part in parts(token) {
            if RUNTIMES.contains(&part) {
                names.insert(part.to_string());
            }
            if !SYSTEMS.contains(&part) {
                continue;
            }
            let qualifiers: Vec<&str> = tokens
                .get(i + 1)
                .into_iter()
                .flat_map(|next| parts(next))
                .filter(|q| {
                    ARCHITECTURES.contains(q) || q.starts_with(|c: char| c.is_ascii_digit())
                })
                .collect();
            if qualifiers.is_empty() {
                names.insert(part.to_string());
            }
            for qualifier in qualifiers {
                names.insert(format!("{part} {qualifier}"));
            }
        }
    }
    names
}

#[test]
fn platform_names_are_extracted_as_documented() {
    let names = platform_names(
        "tests on Linux x86-64/ARM64, macOS Intel/ARM64 and\nWindows x64; \
         an Android 15 emulator; Linux (glibc and musl); Node.js, Chrome.",
    );
    let expected: BTreeSet<String> = [
        "Linux x86-64",
        "Linux ARM64",
        "macOS Intel",
        "macOS ARM64",
        "Windows x64",
        "Android 15",
        "Linux",
        "glibc",
        "musl",
        "Node.js",
        "Chrome",
    ]
    .into_iter()
    .map(String::from)
    .collect();
    assert_eq!(names, expected);
}

#[test]
fn every_platform_the_readme_names_is_on_the_platform_page() {
    let Some(repository) = repository() else {
        return;
    };
    let readme = read(&repository, "README.md");
    let page = read(&repository, "docs/PLATFORMS.md");
    // Markdown wraps names across lines; judge the page with wraps collapsed.
    let page_flat = page.split_whitespace().collect::<Vec<_>>().join(" ");

    let names = platform_names(&readme);
    // The summary the RFC asks for must keep naming the tested platforms, or
    // this test would pass on a README that says nothing.
    for anchor in [
        "Linux x86-64",
        "Linux ARM64",
        "macOS ARM64",
        "Windows x64",
        "iOS ARM64",
    ] {
        assert!(
            names.contains(anchor),
            "README no longer names {anchor:?}; the platform summary must stay (found {names:?})"
        );
    }
    let missing: Vec<&String> = names
        .iter()
        .filter(|n| !page_flat.contains(n.as_str()))
        .collect();
    assert!(
        missing.is_empty(),
        "README names platforms that docs/PLATFORMS.md does not: {missing:?}\n\
         Add each to the page (in the same spelling) or drop it from the README; \
         a tier is asserted only there (RFC 0012)."
    );
}

#[test]
fn readme_and_security_policy_link_the_platform_page() {
    let Some(repository) = repository() else {
        return;
    };
    for relative in ["README.md", "SECURITY.md"] {
        let text = read(&repository, relative);
        assert!(
            text.contains("](docs/PLATFORMS.md)"),
            "{relative} must link docs/PLATFORMS.md"
        );
    }
}

#[test]
fn platform_page_keeps_its_structure() {
    let Some(repository) = repository() else {
        return;
    };
    let page = read(&repository, "docs/PLATFORMS.md");
    for required in [
        "### Tested",
        "### Built and verified on emulation",
        "### Source",
        "## Floors",
        "MSRV 1.85",
        "## Intel macOS exit",
        "August 2027",
        "## Change log",
    ] {
        assert!(
            page.contains(required),
            "docs/PLATFORMS.md lost {required:?}"
        );
    }
    // Every change-log entry is dated; the newest is the page's notice date.
    let dated = page
        .split("## Change log")
        .nth(1)
        .expect("change log section")
        .lines()
        .filter(|l| l.starts_with("- **20"))
        .count();
    assert!(dated >= 1, "the change log needs at least one dated entry");
}
