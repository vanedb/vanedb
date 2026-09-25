//! `VANEDB_RS_ABI_VERSION` in the header and `vanedb_rs_abi_version()` in the
//! library must agree: a consumer compares the two at load time to reject a
//! shared object built for another ABI (RFC 0002 stage 1).

mod common;

#[test]
fn the_header_macro_and_the_function_agree() {
    let in_header = common::defined_integer("VANEDB_RS_ABI_VERSION");
    assert_eq!(in_header, u64::from(vanedb_capi::vanedb_rs_abi_version()));
    assert_eq!(in_header, u64::from(vanedb_capi::VANEDB_RS_ABI_VERSION));
}

/// The stage-1 handle change is the incompatible change that starts the
/// count at 1. A 0 here would mean the macro was never stamped.
#[test]
fn the_abi_version_is_one() {
    assert_eq!(vanedb_capi::vanedb_rs_abi_version(), 1);
}

/// Reading the version is diagnostic, like `vanedb_rs_last_error`: it must not
/// disturb the error a caller is about to report.
#[test]
fn reading_the_abi_version_leaves_the_error_state_alone() {
    assert_eq!(vanedb_capi::vanedb_rs_store_new(0, 0), 0);
    assert_eq!(
        vanedb_capi::vanedb_rs_last_error(),
        vanedb_capi::VANEDB_RS_ZERO_DIMENSION
    );
    let _ = vanedb_capi::vanedb_rs_abi_version();
    let _ = vanedb_capi::vanedb_rs_handle_count();
    assert_eq!(
        vanedb_capi::vanedb_rs_last_error(),
        vanedb_capi::VANEDB_RS_ZERO_DIMENSION
    );
}

/// `VANEDB_RS_VERSION` stays as the semver string alongside the integer.
#[test]
fn the_semver_macro_still_matches_the_library() {
    let header = common::header();
    let macro_version = header
        .lines()
        .find_map(|l| {
            l.strip_prefix("#define VANEDB_RS_VERSION ")?
                .split('"')
                .nth(1)
        })
        .expect("the header defines VANEDB_RS_VERSION");
    let reported = unsafe { std::ffi::CStr::from_ptr(vanedb_capi::vanedb_rs_version()) }
        .to_str()
        .unwrap();
    assert_eq!(macro_version, reported);
}
