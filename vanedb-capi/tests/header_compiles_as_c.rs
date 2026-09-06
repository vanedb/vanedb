//! The generated header must compile with a C compiler.
//!
//! It is this crate's whole public interface, and CI's C test exercises the
//! *C++* engine's hand-written header, so nothing else feeds this one to a C
//! compiler.

use std::process::Command;

#[test]
fn the_generated_header_compiles_as_c() {
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    if Command::new(&cc).arg("--version").output().is_err() {
        eprintln!("skipping: no C compiler ({cc}) available");
        return;
    }

    let include = concat!(env!("CARGO_MANIFEST_DIR"), "/include");
    let src = std::env::temp_dir().join(format!("vanedb_header_{}.c", std::process::id()));
    // Exercises the handle types and one function per type, so an incomplete
    // or conflicting declaration fails here rather than in a user's build.
    std::fs::write(
        &src,
        r#"#include "vanedb_rs_capi.h"
int main(void) {
    vanedb_rs_store *s = 0;
    vanedb_rs_index *h = 0;
    vanedb_rs_disk  *d = 0;
    (void)vanedb_rs_store_len(s);
    (void)vanedb_rs_index_len(h);
    (void)vanedb_rs_disk_len(d);
    return 0;
}
"#,
    )
    .unwrap();

    let out = Command::new(&cc)
        .args(["-std=c11", "-Wall", "-Wextra", "-Werror", "-fsyntax-only"])
        .arg("-I")
        .arg(include)
        .arg(&src)
        .output()
        .expect("failed to run the C compiler");
    let _ = std::fs::remove_file(&src);

    assert!(
        out.status.success(),
        "the generated header does not compile as C:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}
