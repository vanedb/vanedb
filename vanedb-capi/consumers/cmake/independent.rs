//! An independently compiled Rust runtime for the installed static consumer.
use std::cell::Cell;

thread_local! {
    static CALLS: Cell<u32> = const { Cell::new(0) };
}

#[no_mangle]
pub extern "C" fn independent_rust_filter(_id: u64, data: *mut std::ffi::c_void) -> bool {
    // Catch the independent runtime's panic before returning through VaneDB.
    let outcome = std::panic::catch_unwind(|| panic!("locally contained callback panic"));
    if outcome.is_err() {
        // The C acceptance caller supplies a live counter for this synchronous call.
        unsafe { *data.cast::<u32>() += 1 };
    }
    false
}

#[no_mangle]
pub extern "C" fn independent_rust_exercise() -> u32 {
    // Keep allocation, TLS, threading and unwinding live in this library.
    std::panic::catch_unwind(|| {
        CALLS.with(|calls| calls.set(calls.get() + 1));
        let worker = std::thread::spawn(|| {
            assert_eq!(CALLS.with(Cell::get), 0);
            CALLS.with(|calls| calls.set(42));
            let values = std::hint::black_box(vec![7u64; 1024]);
            assert_eq!(values.iter().sum::<u64>(), 7168);
            assert!(std::panic::catch_unwind(|| panic!("independent unwind probe")).is_err());
            CALLS.with(Cell::get)
        });
        assert_eq!(worker.join().unwrap(), 42);
        42
    })
    .unwrap_or(0)
}
