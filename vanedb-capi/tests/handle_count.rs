//! `vanedb_rs_handle_count()` is the leak check a consumer's test suite runs
//! at exit. Alone in its own binary: the handle table is process-wide, and
//! tests in one binary run on parallel threads, so a count asserted next to
//! other tests would race their allocations.

use vanedb_capi::*;

#[test]
fn the_handle_count_follows_every_kind_of_handle() {
    assert_eq!(vanedb_rs_handle_count(), 0);
    let store = vanedb_rs_store_new(2, 0);
    assert_eq!(vanedb_rs_handle_count(), 1);
    let index = vanedb_rs_index_new(2, 0, 8, 4, 16, 1);
    assert_eq!(vanedb_rs_handle_count(), 2);

    let path =
        std::env::temp_dir().join(format!("vanedb-handle-count-{}.disk", std::process::id()));
    let c_path = std::ffi::CString::new(path.to_str().unwrap()).unwrap();
    let ids = [1u64];
    let v = [1.0f32, 0.0];
    unsafe {
        assert_eq!(
            vanedb_rs_disk_build(c_path.as_ptr(), 2, 0, ids.as_ptr(), v.as_ptr(), 1),
            0
        );
    }
    let disk = unsafe { vanedb_rs_disk_open(c_path.as_ptr()) };
    assert_ne!(disk, VANEDB_RS_NULL_HANDLE);
    assert_eq!(vanedb_rs_handle_count(), 3);

    // A failed constructor registers nothing.
    assert_eq!(vanedb_rs_store_new(0, 0), VANEDB_RS_NULL_HANDLE);
    assert_eq!(vanedb_rs_handle_count(), 3);

    // Neither does a rejected free, of any shape.
    vanedb_rs_store_free(index);
    assert_eq!(vanedb_rs_last_error(), VANEDB_RS_INVALID_HANDLE);
    vanedb_rs_index_free(store & 0xFFFF_FFFF);
    assert_eq!(vanedb_rs_last_error(), VANEDB_RS_INVALID_HANDLE);
    assert_eq!(vanedb_rs_handle_count(), 3);

    vanedb_rs_store_free(store);
    assert_eq!(vanedb_rs_handle_count(), 2);
    vanedb_rs_store_free(store);
    assert_eq!(vanedb_rs_last_error(), VANEDB_RS_INVALID_HANDLE);
    assert_eq!(vanedb_rs_handle_count(), 2);
    vanedb_rs_index_free(index);
    vanedb_rs_disk_free(disk);
    assert_eq!(vanedb_rs_handle_count(), 0);
    let _ = std::fs::remove_file(&path);
}
