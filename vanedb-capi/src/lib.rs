//! C ABI (`vanedb_rs_*`) over the VaneDB core. Mirrors vanedb-cpp's C API.
//!
//! Handles are 64-bit ids into a process-wide table, not pointers (RFC 0002
//! stage 1): an unknown, stale, truncated or random id fails the call with
//! `VANEDB_RS_INVALID_HANDLE` instead of being dereferenced. HNSW search takes
//! a per-call ef_search — this matches the parallel C++ ABI so a benchmark
//! harness can call both through one uniform FFI. Stored vectors and queries
//! must contain only finite values; raw-pointer arguments are null-guarded.
//! An unrecognized metric value is rejected: constructors return
//! `VANEDB_RS_NULL_HANDLE`.
use std::cell::RefCell;
use std::ffi::CStr;
use std::os::raw::c_char;
use std::ptr;
use std::slice;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use vanedb::distance::distance_fn;
use vanedb::{ApproxIndex, DiskIndex, DiskIndexBuilder, FlatIndex, Metric};

/// A handle: a 64-bit id into this library's handle table, never a pointer.
///
/// The high 32 bits are a generation counter and the low 32 bits locate the
/// slot, so a truncated id (high bits cleared, as a `ctypes` caller without
/// `restype` produces) never matches a live handle, and a freed id is never
/// reissued. `VANEDB_RS_NULL_HANDLE` (0) is never a live handle.
#[allow(non_camel_case_types)]
pub type vanedb_rs_handle = u64;
/// A `FlatIndex` handle from `vanedb_rs_store_new`.
#[allow(non_camel_case_types)]
pub type vanedb_rs_store = vanedb_rs_handle;
/// An `ApproxIndex` handle from `vanedb_rs_index_new`, `_load` or
/// `_load_from_buffer`.
#[allow(non_camel_case_types)]
pub type vanedb_rs_index = vanedb_rs_handle;
/// A `DiskIndex` handle from `vanedb_rs_disk_open`.
#[allow(non_camel_case_types)]
pub type vanedb_rs_disk = vanedb_rs_handle;

/// The ABI version this header describes. Bumped only on an incompatible
/// change; compare with `vanedb_rs_abi_version()` at runtime.
pub const VANEDB_RS_ABI_VERSION: u32 = 1;

/// The value a failed constructor returns. Never a live handle: passing it
/// fails with `VANEDB_RS_NULL_ARGUMENT`, and freeing it is a no-op.
pub const VANEDB_RS_NULL_HANDLE: vanedb_rs_handle = 0;

/// A synchronous ID predicate, called on the thread performing the search.
///
/// The callback and any memory it accesses through `user_data` must remain valid
/// for the whole call. It must not free the searched handle, or modify/free any
/// search buffers; calls using other handles are allowed.
/// It must not throw a foreign exception or use `longjmp` across Rust frames.
/// A Rust callback declared `extern "C-unwind"` may panic; with unwinding enabled
/// the search reports `VANEDB_RS_PANIC` and leaves the result buffers untouched.
#[allow(non_camel_case_types)]
pub type vanedb_rs_filter_fn =
    Option<unsafe extern "C-unwind" fn(id: u64, user_data: *mut std::ffi::c_void) -> bool>;

struct CFilterClosure {
    func: Box<dyn Fn(u64) -> bool + Sync>,
}

impl CFilterClosure {
    fn new(
        cb: unsafe extern "C-unwind" fn(id: u64, user_data: *mut std::ffi::c_void) -> bool,
        user_data: *mut std::ffi::c_void,
    ) -> Self {
        let user_ptr = user_data as usize;
        Self {
            func: Box::new(move |id: u64| {
                let udata = user_ptr as *mut std::ffi::c_void;
                let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
                    cb(id, udata)
                }));
                match res {
                    Ok(b) => b,
                    Err(p) => std::panic::resume_unwind(p),
                }
            }),
        }
    }

    fn as_predicate(&self) -> &(dyn Fn(u64) -> bool + Sync) {
        self.func.as_ref()
    }
}

/// Validate optional filter arguments before constructing borrowed slices.
fn valid_filter_args(
    filter: vanedb_rs_filter_fn,
    allow: *const u64,
    allow_len: usize,
    deny: *const u64,
    deny_len: usize,
) -> bool {
    if (allow.is_null() && allow_len != 0) || (deny.is_null() && deny_len != 0) {
        return null_arg(false);
    }
    if usize::from(filter.is_some()) + usize::from(!allow.is_null()) + usize::from(!deny.is_null())
        > 1
    {
        set_code(
            VANEDB_RS_INVALID_PARAMETER,
            "specify at most one of filter, allow, or deny",
        );
        return false;
    }
    let max_len = isize::MAX as usize / std::mem::size_of::<u64>();
    if allow_len > max_len || deny_len > max_len {
        set_code(
            VANEDB_RS_INVALID_PARAMETER,
            "filter length exceeds this platform's address space",
        );
        return false;
    }
    true
}

fn from_metric(m: Metric) -> u32 {
    match m {
        Metric::Cosine => 1,
        Metric::Dot => 2,
        // Metric is #[non_exhaustive]; a variant this ABI does not define
        // cannot reach a handle built through it.
        _ => 0,
    }
}

/// `None` for a value this ABI does not define.
///
/// Mapping an unknown value to L2 would defeat `Metric`'s `#[non_exhaustive]`
/// across the boundary: a caller built against a newer header would get
/// silently wrong distances rather than a refusal.
fn to_metric(m: u32) -> Option<Metric> {
    match m {
        0 => Some(Metric::L2),
        1 => Some(Metric::Cosine),
        2 => Some(Metric::Dot),
        _ => None,
    }
}

/// `n * dim`, or `None` on overflow.
///
/// The product is a slice length handed to `from_raw_parts`. Unchecked it
/// panics in debug and wraps in release, producing a slice length unrelated to
/// the caller's buffer -- reachable on 32-bit targets such as wasm32, where
/// `n = 100_000, dim = 50_000` wraps.
fn elements(n: usize, dim: usize) -> Option<usize> {
    n.checked_mul(dim)
}

/// Success.
pub const VANEDB_RS_OK: u32 = 0;
/// A required buffer or path argument was null, or the handle was
/// `VANEDB_RS_NULL_HANDLE`. This ABI's own misuse code: it has no `VaneError`
/// counterpart because the call never reached the core.
pub const VANEDB_RS_NULL_ARGUMENT: u32 = 1;
/// The vector's length does not match the handle's dimension.
pub const VANEDB_RS_DIMENSION_MISMATCH: u32 = 2;
/// A batch's id count and vector count disagree.
pub const VANEDB_RS_BATCH_LENGTH_MISMATCH: u32 = 3;
/// A dimension of zero was requested.
pub const VANEDB_RS_ZERO_DIMENSION: u32 = 4;
/// No vector is stored under that id.
pub const VANEDB_RS_NOT_FOUND: u32 = 5;
/// That id is already stored. Retrying will not help; upsert instead.
pub const VANEDB_RS_DUPLICATE_ID: u32 = 6;
/// `k` was zero.
pub const VANEDB_RS_INVALID_K: u32 = 7;
/// A vector or query contained a NaN or an infinity.
pub const VANEDB_RS_NON_FINITE_VALUE: u32 = 8;
/// A construction parameter was out of range — including a metric value this
/// ABI does not define.
pub const VANEDB_RS_INVALID_PARAMETER: u32 = 9;
/// The file does not exist. Distinct from `CORRUPT`: building it is sensible.
pub const VANEDB_RS_FILE_NOT_FOUND: u32 = 10;
/// The file exists but its bytes are wrong. Retrying will not help.
pub const VANEDB_RS_CORRUPT: u32 = 11;
/// An I/O failure. Retrying may help.
pub const VANEDB_RS_IO: u32 = 12;
/// A compute backend was unavailable.
pub const VANEDB_RS_BACKEND: u32 = 13;
/// A panic was caught at the boundary. A bug in the engine; report it.
pub const VANEDB_RS_PANIC: u32 = 14;
/// A `VaneError` variant added after this header was generated. `VaneError` is
/// `#[non_exhaustive]`, so a caller built against an older header must treat
/// any unrecognized code as a failure rather than as success.
pub const VANEDB_RS_UNKNOWN: u32 = 15;
/// The handle is not a live handle of the kind the function expects: it was
/// never issued, was freed, was truncated on the way through an FFI that
/// guessed its type, or belongs to another handle type. The call did nothing.
pub const VANEDB_RS_INVALID_HANDLE: u32 = 16;

thread_local! {
    static LAST_ERROR: std::cell::Cell<u32> = const { std::cell::Cell::new(VANEDB_RS_OK) };
    // `None` until something fails, so the success path never allocates and
    // never drops. Clearing on entry used to build an empty `CString` and drop
    // the previous one: one malloc and one free on every call, including
    // `vanedb_rs_l2_sq`, which took it from 7.0 ns to 21.0 ns per call.
    static LAST_MESSAGE: std::cell::RefCell<Option<std::ffi::CString>> =
        const { RefCell::new(None) };
}

/// Returned when there is no message. Static, so it needs no allocation and
/// stays valid for the life of the process.
const EMPTY_MESSAGE: &[u8] = b"\0";

fn empty_message() -> *const c_char {
    EMPTY_MESSAGE.as_ptr() as *const c_char
}

fn code_for(error: &vanedb::VaneError) -> u32 {
    use vanedb::VaneError as E;
    match error {
        E::DimensionMismatch { .. } => VANEDB_RS_DIMENSION_MISMATCH,
        E::BatchLengthMismatch { .. } => VANEDB_RS_BATCH_LENGTH_MISMATCH,
        E::ZeroDimension => VANEDB_RS_ZERO_DIMENSION,
        E::NotFound { .. } => VANEDB_RS_NOT_FOUND,
        E::DuplicateId { .. } => VANEDB_RS_DUPLICATE_ID,
        E::InvalidK => VANEDB_RS_INVALID_K,
        E::NonFiniteValue { .. } => VANEDB_RS_NON_FINITE_VALUE,
        E::InvalidParameter(_) => VANEDB_RS_INVALID_PARAMETER,
        E::Validation(_) => VANEDB_RS_INVALID_PARAMETER,
        E::FileNotFound { .. } => VANEDB_RS_FILE_NOT_FOUND,
        E::Corrupt { .. } => VANEDB_RS_CORRUPT,
        E::Io { .. } => VANEDB_RS_IO,
        E::Backend { .. } => VANEDB_RS_BACKEND,
        // `VaneError` is #[non_exhaustive]; a variant this ABI predates must
        // still read as a failure.
        _ => VANEDB_RS_UNKNOWN,
    }
}

/// Clears the thread's error state. Sets one `Cell` and allocates nothing:
/// this runs on entry to every ABI call, so anything more expensive is paid by
/// every caller on the success path.
///
/// `try_with`, not `with`: a C++ caller whose `thread_local` handle is
/// destroyed at thread exit calls `vanedb_rs_*_free` from that destructor,
/// after this TLS is gone. `with` panics there, and a panic crossing an
/// `extern "C"` boundary aborts the process — this runs outside `catch_unwind`,
/// so it would take the host program down.
fn clear_error() {
    let _ = LAST_ERROR.try_with(|slot| slot.set(VANEDB_RS_OK));
}

/// Records a failure. Only the failing paths allocate.
fn set_code(code: u32, message: &str) {
    let _ = LAST_ERROR.try_with(|slot| slot.set(code));
    // A NUL inside a Display string would be a bug in the core, not the
    // caller's problem: truncate rather than lose the report entirely.
    let owned = std::ffi::CString::new(message).unwrap_or_else(|e| {
        let mut bytes = e.into_vec();
        bytes.truncate(bytes.iter().position(|&b| b == 0).unwrap_or(0));
        std::ffi::CString::new(bytes).unwrap_or_default()
    });
    let _ = LAST_MESSAGE.try_with(|slot| *slot.borrow_mut() = Some(owned));
}

/// Records `error` and yields `fallback`, so a failing arm stays one
/// expression: `Err(e) => fail(e, 1)`.
fn fail<T>(error: vanedb::VaneError, fallback: T) -> T {
    set_code(code_for(&error), &error.to_string());
    fallback
}

/// Records a null argument and yields `fallback`.
fn null_arg<T>(fallback: T) -> T {
    set_code(VANEDB_RS_NULL_ARGUMENT, "null argument");
    fallback
}

/// Records a handle that is not live, or not of the expected kind.
fn invalid_handle<T>(fallback: T) -> T {
    set_code(
        VANEDB_RS_INVALID_HANDLE,
        "not a live handle of this kind: never issued, already freed, truncated, \
         or created by another vanedb_rs_*_new",
    );
    fallback
}

/// Records a batch whose `n * dim` product overflows. Like a null argument it
/// never reaches the core, so there is no `VaneError` to map; it is the
/// 32-bit and wasm32 case that `elements` exists to catch.
fn batch_too_large<T>(fallback: T) -> T {
    set_code(
        VANEDB_RS_BATCH_LENGTH_MISMATCH,
        "batch element count overflows this platform's address space",
    );
    fallback
}

/// Records a path argument that is not valid UTF-8. The call never reached
/// the core, so there is no `VaneError` to map.
fn bad_path<T>(fallback: T) -> T {
    set_code(VANEDB_RS_INVALID_PARAMETER, "path is not valid UTF-8");
    fallback
}

/// Records a metric value this ABI does not define.
fn bad_metric<T>(fallback: T) -> T {
    set_code(
        VANEDB_RS_INVALID_PARAMETER,
        "unrecognized metric value: expected 0 (L2), 1 (Cosine) or 2 (Dot)",
    );
    fallback
}

// ---------------------------------------------------------------------------
// The handle table.
//
// Every handle crossing the boundary is an id into this table. Motivation
// (RFC 0002): a `ctypes` caller that omits `restype` truncates a returned
// pointer to a C `int` and the next call dereferences garbage; any FFI that
// guesses types does the same. A pointer registry cannot catch a truncated
// pointer that collides with a live one. An id with a generation in its high
// half can: truncation clears the generation, and generation 0 is never
// issued.
//
// Layout of an id: bits 63..32 generation (1 ..= u32::MAX - 1), bits 31..4
// slot within the shard plus one, bits 3..0 shard. The plus one keeps the
// low half nonzero for every live id: without it the first handle issued
// (shard 0, slot 0) truncates to exactly 0, the null handle, and would be
// reported as a null argument rather than an invalid handle. Sharded so that concurrent callers
// on different handles rarely contend; the lock is held only for the lookup,
// and the caller gets an `Arc` clone, so a search never holds it. Freeing a
// handle another thread is mid-call on is therefore safe: that call finishes
// on its own clone, and every later call reports `VANEDB_RS_INVALID_HANDLE`.
// ---------------------------------------------------------------------------

#[derive(Clone)]
enum Object {
    Store(Arc<FlatIndex>),
    Index(Arc<ApproxIndex>),
    Disk(Arc<DiskIndex>),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Store,
    Index,
    Disk,
}

impl Object {
    fn kind(&self) -> Kind {
        match self {
            Object::Store(_) => Kind::Store,
            Object::Index(_) => Kind::Index,
            Object::Disk(_) => Kind::Disk,
        }
    }
}

struct Slot {
    /// The generation a live id must carry. Starts at 1 and is bumped on
    /// every free, so no id ever matches a slot twice. A slot whose next
    /// generation would be `u32::MAX` is retired rather than reused:
    /// `u32::MAX` is what a sign-extended truncated id carries, and 0 is what
    /// a zero-extended one carries, so neither is ever issued.
    generation: u32,
    object: Option<Object>,
}

struct Shard {
    slots: Vec<Slot>,
    /// Indices of slots whose object is `None` and whose generation is still
    /// issuable.
    free: Vec<u32>,
}

const SHARD_BITS: u32 = 4;
const SHARDS: usize = 1 << SHARD_BITS;
const SLOT_BITS: u32 = 32 - SHARD_BITS;
/// One fewer than the field holds: slot numbers are stored plus one.
const MAX_SLOTS: usize = (1 << SLOT_BITS) - 1;
const FIRST_GENERATION: u32 = 1;
/// The last generation that can be issued; the slot retires after it.
const LAST_GENERATION: u32 = u32::MAX - 1;

static TABLE: [Mutex<Shard>; SHARDS] = [const {
    Mutex::new(Shard {
        slots: Vec::new(),
        free: Vec::new(),
    })
}; SHARDS];
/// Spreads allocations across shards. Wrapping is harmless: only the low
/// bits are used.
static NEXT_SHARD: AtomicU32 = AtomicU32::new(0);
/// Live handles, for `vanedb_rs_handle_count`.
static LIVE: AtomicUsize = AtomicUsize::new(0);

fn compose(shard: u32, slot: u32, generation: u32) -> u64 {
    (u64::from(generation) << 32) | u64::from(((slot + 1) << SHARD_BITS) | shard)
}

fn shard_of(id: u64) -> &'static Mutex<Shard> {
    &TABLE[(id as u32 & (SHARDS as u32 - 1)) as usize]
}

/// The slot an id names, or `None` for a low half that names no slot.
fn slot_of(id: u64) -> Option<usize> {
    ((id as u32) >> SHARD_BITS)
        .checked_sub(1)
        .map(|s| s as usize)
}

fn generation_of(id: u64) -> u32 {
    (id >> 32) as u32
}

/// A poisoned shard is still structurally valid: every mutation below is a
/// single field write or a `Vec` push/pop, none of which can panic halfway.
fn lock(shard: &Mutex<Shard>) -> MutexGuard<'_, Shard> {
    shard
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Registers `object` and returns its id, or `None` if the chosen shard has
/// issued all of its 2^28 slots.
fn insert(object: Object) -> Option<u64> {
    let shard_index = NEXT_SHARD.fetch_add(1, Ordering::Relaxed) % SHARDS as u32;
    let mut shard = lock(&TABLE[shard_index as usize]);
    let slot_index = match shard.free.pop() {
        Some(index) => index,
        None => {
            if shard.slots.len() >= MAX_SLOTS {
                return None;
            }
            shard.slots.push(Slot {
                generation: FIRST_GENERATION,
                object: None,
            });
            (shard.slots.len() - 1) as u32
        }
    };
    let slot = &mut shard.slots[slot_index as usize];
    debug_assert!(slot.object.is_none());
    debug_assert!((FIRST_GENERATION..=LAST_GENERATION).contains(&slot.generation));
    slot.object = Some(object);
    LIVE.fetch_add(1, Ordering::Relaxed);
    Some(compose(shard_index, slot_index, slot.generation))
}

/// The object behind a live id, cloned out from under the lock.
fn lookup(id: u64) -> Option<Object> {
    let index = slot_of(id)?;
    let shard = lock(shard_of(id));
    let slot = shard.slots.get(index)?;
    if slot.generation != generation_of(id) {
        return None;
    }
    slot.object.clone()
}

/// Unregisters a live id of `kind`, returning the object so the caller can
/// drop it outside the lock. `None` if the id is not live or is another kind:
/// `vanedb_rs_index_free` must not free a store.
fn release(id: u64, kind: Kind) -> Option<Object> {
    let index = slot_of(id)?;
    let mut shard = lock(shard_of(id));
    let slot = shard.slots.get_mut(index)?;
    if slot.generation != generation_of(id) || slot.object.as_ref()?.kind() != kind {
        return None;
    }
    let object = slot.object.take();
    if slot.generation < LAST_GENERATION {
        slot.generation += 1;
        shard.free.push(index as u32);
    } else {
        // Retired: no generation left that a truncated id cannot carry.
        slot.generation = u32::MAX;
    }
    LIVE.fetch_sub(1, Ordering::Relaxed);
    object
}

/// Registers a new object, reporting a full table as a failure rather than
/// a crash. Unreachable in practice: it takes 2^28 live handles in one shard.
fn register(object: Object) -> u64 {
    insert(object).unwrap_or_else(|| {
        set_code(
            VANEDB_RS_INVALID_PARAMETER,
            "handle table is full: free some handles first",
        );
        VANEDB_RS_NULL_HANDLE
    })
}

/// The store behind `id`, or `None` with the error recorded.
fn store(id: vanedb_rs_store) -> Option<Arc<FlatIndex>> {
    if id == VANEDB_RS_NULL_HANDLE {
        return null_arg(None);
    }
    match lookup(id) {
        Some(Object::Store(store)) => Some(store),
        _ => invalid_handle(None),
    }
}

/// The graph behind `id`, or `None` with the error recorded.
fn index(id: vanedb_rs_index) -> Option<Arc<ApproxIndex>> {
    if id == VANEDB_RS_NULL_HANDLE {
        return null_arg(None);
    }
    match lookup(id) {
        Some(Object::Index(index)) => Some(index),
        _ => invalid_handle(None),
    }
}

/// The mapped file behind `id`, or `None` with the error recorded.
fn disk(id: vanedb_rs_disk) -> Option<Arc<DiskIndex>> {
    if id == VANEDB_RS_NULL_HANDLE {
        return null_arg(None);
    }
    match lookup(id) {
        Some(Object::Disk(disk)) => Some(disk),
        _ => invalid_handle(None),
    }
}

/// Frees a handle of `kind`. The null handle is a no-op; anything else that is
/// not a live handle of that kind records `VANEDB_RS_INVALID_HANDLE`. A
/// successful free preserves the thread's error state (see `guard_preserving`).
fn free(id: vanedb_rs_handle, kind: Kind) {
    if id == VANEDB_RS_NULL_HANDLE {
        return;
    }
    match release(id, kind) {
        Some(object) => drop(object),
        None => invalid_handle(()),
    }
}

/// The code from the calling thread's most recent `vanedb_rs_*` call, or
/// `VANEDB_RS_OK` if it succeeded.
///
/// The status functions' `1` and the searches' `0` cannot say *why*, and a
/// search returning `0` is otherwise indistinguishable from an empty store.
/// This is what a caller branches on: retry an `IO`, abort a `CORRUPT`, skip a
/// `DUPLICATE_ID`. It is thread-local, so a value set on one thread is never
/// observed on another, and reading it does not clear it.
///
/// This function does not itself count as a call: it leaves the code in place.
#[no_mangle]
pub extern "C" fn vanedb_rs_last_error() -> u32 {
    LAST_ERROR
        .try_with(|slot| slot.get())
        .unwrap_or(VANEDB_RS_OK)
}

/// The message for the calling thread's most recent failure, NUL-terminated
/// and never null (the empty string after a success).
///
/// This carries the detail the code cannot — which field mismatched, what was
/// wrong with the file.
///
/// The pointer is owned by the library and stays valid only until **either**
/// the next `vanedb_rs_*` call on this thread, **or** this thread exits — the
/// buffer lives in thread-local storage and is dropped with it. Handing the
/// pointer to another thread that outlives this one is a use-after-free, and
/// so is stashing it across a join. Copy the bytes if they need to outlive
/// either event.
///
/// # Safety
/// The returned pointer must not be freed by the caller. It must not be used
/// after another `vanedb_rs_*` call on the same thread, and must not be used
/// after that thread has exited — including by a thread that joined it.
#[no_mangle]
pub extern "C" fn vanedb_rs_last_error_message() -> *const c_char {
    // Gated on the code rather than cleared with it: clearing would mean
    // dropping a `CString` on every successful call. After a success this
    // returns the static empty string, so the observable contract is the same.
    if vanedb_rs_last_error() == VANEDB_RS_OK {
        return empty_message();
    }
    LAST_MESSAGE
        .try_with(|slot| {
            slot.borrow()
                .as_ref()
                .map_or_else(empty_message, |message| message.as_ptr())
        })
        .unwrap_or_else(|_| empty_message())
}

/// This library's version, as a NUL-terminated static string.
///
/// A consumer cannot otherwise check that the shared object it loaded matches
/// the header it compiled against.
#[no_mangle]
pub extern "C" fn vanedb_rs_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const c_char
}

/// The ABI version of the loaded library. Equal to `VANEDB_RS_ABI_VERSION`
/// in the header it was built from; a consumer whose header says otherwise
/// must not call anything else. Does not count as a call: it leaves the
/// thread's error state alone.
#[no_mangle]
pub extern "C" fn vanedb_rs_abi_version() -> u32 {
    VANEDB_RS_ABI_VERSION
}

/// The number of live handles of every kind in this process, for leak tests.
/// Does not count as a call: it leaves the thread's error state alone.
#[no_mangle]
pub extern "C" fn vanedb_rs_handle_count() -> usize {
    LIVE.load(Ordering::Relaxed)
}

/// Runs `body`, returning `fallback` if it panics.
///
/// A panic unwinding out of an `extern "C"` function aborts the process, taking
/// the embedding application with it. Every entry point routes through here so
/// a bug surfaces as this ABI's ordinary failure value — 0, 1, 0 or NaN —
/// instead. vanedb-cpp wraps every entry point in try/catch for the same reason.
///
/// Entering also clears the thread's error code, which is what makes
/// `vanedb_rs_last_error` mean "the most recent call" rather than "the most
/// recent failure". Every failing path inside `body` sets it again.
/// `guard` without the clear, for entry points that cannot fail.
///
/// A `*_free` has no failure to report, and clearing would destroy the error a
/// caller is about to act on: the idiomatic C path is fail, clean up, report,
/// and routing frees through `guard` lost both code and message between the
/// failure and the report.
fn guard_preserving<T>(fallback: T, body: impl FnOnce() -> T) -> T {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(body)) {
        Ok(value) => value,
        Err(_) => {
            set_code(VANEDB_RS_PANIC, "a panic was caught at the C ABI boundary");
            fallback
        }
    }
}

fn guard<T>(fallback: T, body: impl FnOnce() -> T) -> T {
    clear_error();
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(body)) {
        Ok(value) => value,
        Err(_) => {
            set_code(VANEDB_RS_PANIC, "a panic was caught at the C ABI boundary");
            fallback
        }
    }
}

/// Copies at most `k` results into the caller's buffers and returns the count.
///
/// # Safety
/// `out_ids` and `out_dists` must each have room for `k` elements.
unsafe fn write_results(
    results: &[vanedb::SearchResult],
    k: usize,
    out_ids: *mut u64,
    out_dists: *mut f32,
) -> usize {
    let n = results.len().min(k);
    for (i, r) in results.iter().take(k).enumerate() {
        *out_ids.add(i) = r.id;
        *out_dists.add(i) = r.distance;
    }
    n
}

/// # Safety
/// `a` and `b` must each point to at least `dim` valid `f32` values.
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_l2_sq(a: *const f32, b: *const f32, dim: usize) -> f32 {
    guard(f32::NAN, || {
        distance_fn(Metric::L2)(slice::from_raw_parts(a, dim), slice::from_raw_parts(b, dim))
    })
}

/// # Safety
/// `a` and `b` must each point to at least `dim` valid `f32` values.
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_cosine_distance(
    a: *const f32,
    b: *const f32,
    dim: usize,
) -> f32 {
    guard(f32::NAN, || {
        distance_fn(Metric::Cosine)(slice::from_raw_parts(a, dim), slice::from_raw_parts(b, dim))
    })
}

/// # Safety
/// `a` and `b` must each point to at least `dim` valid `f32` values.
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_dot_product(a: *const f32, b: *const f32, dim: usize) -> f32 {
    guard(f32::NAN, || {
        // Negate to get the raw inner product (+a·b). The core's distance_fn(Dot) returns the
        // negated distance form (-a·b, lower=closer) for search ranking. This C ABI function must
        // return the raw product to match vanedb_cpp_dot_product, which returns +a·b.
        -distance_fn(Metric::Dot)(slice::from_raw_parts(a, dim), slice::from_raw_parts(b, dim))
    })
}

/// Creates a store. Returns its handle, or `VANEDB_RS_NULL_HANDLE` on error;
/// free it with `vanedb_rs_store_free`.
#[no_mangle]
pub extern "C" fn vanedb_rs_store_new(dim: usize, metric: u32) -> vanedb_rs_store {
    guard(VANEDB_RS_NULL_HANDLE, || {
        let Some(metric) = to_metric(metric) else {
            return bad_metric(VANEDB_RS_NULL_HANDLE);
        };
        match FlatIndex::new(dim, metric) {
            Ok(s) => register(Object::Store(Arc::new(s))),
            Err(e) => fail(e, VANEDB_RS_NULL_HANDLE),
        }
    })
}

/// # Safety
/// `v` must point to at least `dim` valid `f32` values (where `dim` matches the store).
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_store_add(s: vanedb_rs_store, id: u64, v: *const f32) -> i32 {
    guard(1, || {
        let Some(store) = store(s) else { return 1 };
        if v.is_null() {
            return null_arg(1);
        }
        let vec = slice::from_raw_parts(v, store.dimension());
        match store.add(id, vec) {
            Ok(()) => 0,
            Err(e) => fail(e, 1),
        }
    })
}

/// # Safety
/// `ids` must point to `n` valid `u64`s and `vecs` to `n * dim` valid `f32`s
/// (both may be null when `n` is 0).
/// All-or-nothing: on error (duplicate id, length mismatch) the store is unchanged.
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_store_add_batch(
    s: vanedb_rs_store,
    ids: *const u64,
    vecs: *const f32,
    n: usize,
) -> i32 {
    guard(1, || {
        let Some(store) = store(s) else { return 1 };
        if n != 0 && (ids.is_null() || vecs.is_null()) {
            return null_arg(1);
        }
        let (id_slice, vec_slice): (&[u64], &[f32]) = if n == 0 {
            (&[], &[])
        } else {
            let Some(len) = elements(n, store.dimension()) else {
                return batch_too_large(1);
            };
            (
                slice::from_raw_parts(ids, n),
                slice::from_raw_parts(vecs, len),
            )
        };
        match store.add_batch(id_slice, vec_slice) {
            Ok(()) => 0,
            Err(e) => fail(e, 1),
        }
    })
}

/// # Safety
/// `q` must point to `dim` valid `f32`s; `out_ids` and `out_dists` must each
/// have room for `k` elements.
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_store_search(
    s: vanedb_rs_store,
    q: *const f32,
    k: usize,
    out_ids: *mut u64,
    out_dists: *mut f32,
) -> usize {
    guard(0, || {
        let Some(store) = store(s) else { return 0 };
        if q.is_null() || out_ids.is_null() || out_dists.is_null() {
            return null_arg(0);
        }
        let query = slice::from_raw_parts(q, store.dimension());
        match store.search(query, k) {
            Ok(res) => write_results(&res, k, out_ids, out_dists),
            Err(e) => fail(e, 0),
        }
    })
}

/// Search with at most one of a callback, allow list, or deny list.
///
/// A non-null list pointer selects that filter even when its length is zero:
/// empty allow accepts nothing; empty deny accepts everything. Null list pointers
/// require zero lengths. With all three filters null the search is unfiltered.
/// Lists must be strictly ascending without duplicates. Conflicting filters,
/// invalid lengths, or unsorted lists fail with `VANEDB_RS_INVALID_PARAMETER`;
/// a null list with nonzero length fails with `VANEDB_RS_NULL_ARGUMENT`.
/// On failure returns zero and leaves result buffers untouched; inspect
/// `vanedb_rs_last_error` to distinguish failure from no matches.
///
/// # Safety
/// `q` must point to `dim` valid `f32`s; `out_ids` and `out_dists` must each
/// have room for `k` elements.
/// Each nonempty list must point to its stated number of valid `u64`s. Inputs
/// must remain valid and unmodified until return, and outputs must not overlap
/// inputs. A callback must obey `vanedb_rs_filter_fn`'s lifetime, reentrancy and
/// unwinding requirements.
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_store_search_filtered(
    s: vanedb_rs_store,
    q: *const f32,
    k: usize,
    filter: vanedb_rs_filter_fn,
    user_data: *mut std::ffi::c_void,
    allow: *const u64,
    allow_len: usize,
    deny: *const u64,
    deny_len: usize,
    out_ids: *mut u64,
    out_dists: *mut f32,
) -> usize {
    guard(0, || {
        let Some(store) = store(s) else { return 0 };
        if q.is_null() || out_ids.is_null() || out_dists.is_null() {
            return null_arg(0);
        }
        let query = slice::from_raw_parts(q, store.dimension());

        if !valid_filter_args(filter, allow, allow_len, deny, deny_len) {
            return 0;
        }
        let pred_closure;
        let c_filter = if !allow.is_null() {
            Some(vanedb::approx::Filter::Allow(if allow_len == 0 {
                &[]
            } else {
                slice::from_raw_parts(allow, allow_len)
            }))
        } else if !deny.is_null() {
            Some(vanedb::approx::Filter::Deny(if deny_len == 0 {
                &[]
            } else {
                slice::from_raw_parts(deny, deny_len)
            }))
        } else if let Some(cb) = filter {
            pred_closure = CFilterClosure::new(cb, user_data);
            Some(vanedb::approx::Filter::Predicate(
                pred_closure.as_predicate(),
            ))
        } else {
            None
        };

        let mut params = vanedb::SearchParams::new();
        if let Some(f) = c_filter {
            params = params.filter(f);
        }

        match store.search_with(query, k, &params) {
            Ok(res) => {
                let n = write_results(&res, k, out_ids, out_dists);
                // A callback may have handled a failing call on another
                // handle. Report the successful outer search, not that error.
                clear_error();
                n
            }
            Err(e) => fail(e, 0),
        }
    })
}

/// Frees a store handle. `VANEDB_RS_NULL_HANDLE` is a no-op; a handle that is
/// not a live store (freed already, or another kind) fails with
/// `VANEDB_RS_INVALID_HANDLE`. A successful free preserves the thread's error
/// state, so a caller can fail, clean up, then report.
#[no_mangle]
pub extern "C" fn vanedb_rs_store_free(s: vanedb_rs_store) {
    guard_preserving((), || free(s, Kind::Store))
}

/// Creates a graph. Returns its handle, or `VANEDB_RS_NULL_HANDLE` on error;
/// free it with `vanedb_rs_index_free`.
#[no_mangle]
pub extern "C" fn vanedb_rs_index_new(
    dim: usize,
    metric: u32,
    capacity: usize,
    m: usize,
    ef_construction: usize,
    seed: u64,
) -> vanedb_rs_index {
    guard(VANEDB_RS_NULL_HANDLE, || {
        let Some(metric) = to_metric(metric) else {
            return bad_metric(VANEDB_RS_NULL_HANDLE);
        };
        match ApproxIndex::builder(dim, metric)
            .capacity(capacity)
            .m(m)
            .ef_construction(ef_construction)
            .seed(seed)
            .build()
        {
            Ok(h) => register(Object::Index(Arc::new(h))),
            Err(e) => fail(e, VANEDB_RS_NULL_HANDLE),
        }
    })
}

/// # Safety
/// `v` must point to at least `dim` valid `f32` values (where `dim` matches the index).
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_index_add(h: vanedb_rs_index, id: u64, v: *const f32) -> i32 {
    guard(1, || {
        let Some(idx) = index(h) else { return 1 };
        if v.is_null() {
            return null_arg(1);
        }
        let vec = slice::from_raw_parts(v, idx.dimension());
        match idx.add(id, vec) {
            Ok(()) => 0,
            Err(e) => fail(e, 1),
        }
    })
}

/// # Safety
/// `ids` must point to `n` valid `u64`s and `vecs` to `n * dim` valid `f32`s
/// (both may be null when `n` is 0).
/// All-or-nothing: on error (duplicate id, capacity, length mismatch) the index is unchanged.
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_index_add_batch(
    h: vanedb_rs_index,
    ids: *const u64,
    vecs: *const f32,
    n: usize,
) -> i32 {
    guard(1, || {
        let Some(idx) = index(h) else { return 1 };
        if n != 0 && (ids.is_null() || vecs.is_null()) {
            return null_arg(1);
        }
        let (id_slice, vec_slice): (&[u64], &[f32]) = if n == 0 {
            (&[], &[])
        } else {
            let Some(len) = elements(n, idx.dimension()) else {
                return batch_too_large(1);
            };
            (
                slice::from_raw_parts(ids, n),
                slice::from_raw_parts(vecs, len),
            )
        };
        match idx.add_batch(id_slice, vec_slice) {
            Ok(()) => 0,
            Err(e) => fail(e, 1),
        }
    })
}

/// # Safety
/// `q` must point to `dim` valid `f32`s; `out_ids` and `out_dists` must each
/// have room for `k` elements.
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_index_search(
    h: vanedb_rs_index,
    q: *const f32,
    k: usize,
    ef_search: usize,
    out_ids: *mut u64,
    out_dists: *mut f32,
) -> usize {
    guard(0, || {
        let Some(idx) = index(h) else { return 0 };
        if q.is_null() || out_ids.is_null() || out_dists.is_null() {
            return null_arg(0);
        }
        let query = slice::from_raw_parts(q, idx.dimension());
        // Per-call, not a store: mutating the handle here made one caller's
        // beam width visible to every other user of the index, and `save`
        // then wrote it into the file.
        //
        // `ef_search` is required and has no default a C caller can name, so 0
        // is the idiom for "whatever the index is set to". Passing it through
        // would clamp the beam to `k` -- for k=10 that is 5x narrower than the
        // stored default of 50, returning plausible results at silently
        // degraded recall. Resolving it here reads the handle without mutating
        // it, so it stays a per-call parameter.
        let params = if ef_search == 0 {
            vanedb::SearchParams::new()
        } else {
            vanedb::SearchParams::new().ef_search(ef_search)
        };
        match idx.search_with(query, k, &params) {
            Ok(res) => write_results(&res, k, out_ids, out_dists),
            Err(e) => fail(e, 0),
        }
    })
}

/// Search with at most one of a callback, allow list, or deny list.
///
/// A non-null list pointer selects that filter even when its length is zero:
/// empty allow accepts nothing; empty deny accepts everything. Null list pointers
/// require zero lengths. With all three filters null the search is unfiltered.
/// Lists must be strictly ascending without duplicates. Conflicting filters,
/// invalid lengths, or unsorted lists fail with `VANEDB_RS_INVALID_PARAMETER`;
/// a null list with nonzero length fails with `VANEDB_RS_NULL_ARGUMENT`.
/// On failure returns zero and leaves result buffers untouched; inspect
/// `vanedb_rs_last_error` to distinguish failure from no matches.
///
/// # Safety
/// `q` must point to `dim` valid `f32`s; `out_ids` and `out_dists` must each
/// have room for `k` elements.
/// Each nonempty list must point to its stated number of valid `u64`s. Inputs
/// must remain valid and unmodified until return, and outputs must not overlap
/// inputs. A callback must obey `vanedb_rs_filter_fn`'s lifetime, reentrancy and
/// unwinding requirements.
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_index_search_filtered(
    h: vanedb_rs_index,
    q: *const f32,
    k: usize,
    ef_search: usize,
    filter: vanedb_rs_filter_fn,
    user_data: *mut std::ffi::c_void,
    allow: *const u64,
    allow_len: usize,
    deny: *const u64,
    deny_len: usize,
    out_ids: *mut u64,
    out_dists: *mut f32,
) -> usize {
    guard(0, || {
        let Some(idx) = index(h) else { return 0 };
        if q.is_null() || out_ids.is_null() || out_dists.is_null() {
            return null_arg(0);
        }
        let query = slice::from_raw_parts(q, idx.dimension());

        if !valid_filter_args(filter, allow, allow_len, deny, deny_len) {
            return 0;
        }
        let pred_closure;
        let c_filter = if !allow.is_null() {
            Some(vanedb::approx::Filter::Allow(if allow_len == 0 {
                &[]
            } else {
                slice::from_raw_parts(allow, allow_len)
            }))
        } else if !deny.is_null() {
            Some(vanedb::approx::Filter::Deny(if deny_len == 0 {
                &[]
            } else {
                slice::from_raw_parts(deny, deny_len)
            }))
        } else if let Some(cb) = filter {
            pred_closure = CFilterClosure::new(cb, user_data);
            Some(vanedb::approx::Filter::Predicate(
                pred_closure.as_predicate(),
            ))
        } else {
            None
        };

        let mut params = if ef_search == 0 {
            vanedb::SearchParams::new()
        } else {
            vanedb::SearchParams::new().ef_search(ef_search)
        };
        if let Some(f) = c_filter {
            params = params.filter(f);
        }

        match idx.search_with(query, k, &params) {
            Ok(res) => {
                let n = write_results(&res, k, out_ids, out_dists);
                // A callback may have handled a failing call on another
                // handle. Report the successful outer search, not that error.
                clear_error();
                n
            }
            Err(e) => fail(e, 0),
        }
    })
}

/// # Safety
/// `path` must be a valid NUL-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_index_save(h: vanedb_rs_index, path: *const c_char) -> i32 {
    guard(1, || {
        let Some(idx) = index(h) else { return 1 };
        if path.is_null() {
            return null_arg(1);
        }
        match CStr::from_ptr(path).to_str() {
            Ok(p) => match idx.save(p) {
                Ok(()) => 0,
                Err(e) => fail(e, 1),
            },
            Err(_) => bad_path(1),
        }
    })
}

/// # Safety
/// `path` must be a valid NUL-terminated C string. Returns a handle (or
/// `VANEDB_RS_NULL_HANDLE`) that must be freed with `vanedb_rs_index_free`.
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_index_load(path: *const c_char) -> vanedb_rs_index {
    guard(VANEDB_RS_NULL_HANDLE, || {
        if path.is_null() {
            return null_arg(VANEDB_RS_NULL_HANDLE);
        }
        match CStr::from_ptr(path).to_str() {
            Ok(p) => match ApproxIndex::load(p) {
                Ok(h) => register(Object::Index(Arc::new(h))),
                Err(e) => fail(e, VANEDB_RS_NULL_HANDLE),
            },
            Err(_) => bad_path(VANEDB_RS_NULL_HANDLE),
        }
    })
}

struct CountingWriter {
    n: usize,
}

impl std::io::Write for CountingWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.n = self.n.checked_add(buf.len()).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "serialized length overflow",
            )
        })?;
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Writes a VNDB graph into `buf`.
///
/// `written` must be non-null. On success it receives the number of bytes
/// written. If `buf` is null and `cap` is 0, this is a size query: it
/// serializes into a counter (no output buffer) and stores the required
/// length. If `buf` is non-null but `cap` is smaller than needed, it fails
/// with `VANEDB_RS_INVALID_PARAMETER` and still stores the required length
/// so the caller can allocate and retry.
///
/// # Safety
/// `written` must be a valid pointer. If `buf` is non-null it must have
/// room for `cap` bytes.
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_index_save_to_buffer(
    h: vanedb_rs_index,
    buf: *mut u8,
    cap: usize,
    written: *mut usize,
) -> i32 {
    guard(1, || {
        let Some(idx) = index(h) else { return 1 };
        if written.is_null() {
            return null_arg(1);
        }
        if buf.is_null() && cap != 0 {
            return null_arg(1);
        }
        if buf.is_null() {
            let mut counter = CountingWriter { n: 0 };
            return match idx.save_to(&mut counter) {
                Ok(()) => {
                    *written = counter.n;
                    0
                }
                Err(e) => fail(e, 1),
            };
        }
        match idx.to_bytes() {
            Ok(bytes) => {
                *written = bytes.len();
                if cap < bytes.len() {
                    set_code(
                        VANEDB_RS_INVALID_PARAMETER,
                        &format!("buffer too small: need {} bytes, cap is {cap}", bytes.len()),
                    );
                    return 1;
                }
                ptr::copy_nonoverlapping(bytes.as_ptr(), buf, bytes.len());
                0
            }
            Err(e) => fail(e, 1),
        }
    })
}

/// Reads a VNDB graph (or a legacy Rust file) from `buf`.
///
/// # Safety
/// `buf` must point to `len` valid bytes. Returns a handle (or
/// `VANEDB_RS_NULL_HANDLE`) that must be freed with `vanedb_rs_index_free`.
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_index_load_from_buffer(
    buf: *const u8,
    len: usize,
) -> vanedb_rs_index {
    guard(VANEDB_RS_NULL_HANDLE, || {
        if buf.is_null() {
            return null_arg(VANEDB_RS_NULL_HANDLE);
        }
        let bytes = slice::from_raw_parts(buf, len);
        match ApproxIndex::from_bytes(bytes) {
            Ok(h) => register(Object::Index(Arc::new(h))),
            Err(e) => fail(e, VANEDB_RS_NULL_HANDLE),
        }
    })
}

/// Frees a graph handle. `VANEDB_RS_NULL_HANDLE` is a no-op; a handle that is
/// not a live graph fails with `VANEDB_RS_INVALID_HANDLE`. A successful free
/// preserves the thread's error state.
#[no_mangle]
pub extern "C" fn vanedb_rs_index_free(h: vanedb_rs_index) {
    guard_preserving((), || free(h, Kind::Index))
}

/// # Safety
/// `path` must be a valid NUL-terminated C string; `ids` must point to `n` valid `u64`s
/// and `vecs` to `n * dim` valid `f32`s (both may be null when `n` is 0).
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_disk_build(
    path: *const c_char,
    dim: usize,
    metric: u32,
    ids: *const u64,
    vecs: *const f32,
    n: usize,
) -> i32 {
    guard(1, || {
        if path.is_null() {
            return null_arg(1);
        }
        let p = match CStr::from_ptr(path).to_str() {
            Ok(s) => s,
            Err(_) => return bad_path(1),
        };
        let Some(metric) = to_metric(metric) else {
            return bad_metric(1);
        };
        let mut b = match DiskIndexBuilder::new(dim, metric) {
            Ok(b) => b,
            Err(e) => return fail(e, 1),
        };
        if n != 0 && (ids.is_null() || vecs.is_null()) {
            return null_arg(1);
        }
        let Some(total) = elements(n, dim) else {
            return batch_too_large(1);
        };
        let id_slice: &[u64] = if n == 0 {
            &[]
        } else {
            slice::from_raw_parts(ids, n)
        };
        let vec_slice: &[f32] = if total == 0 {
            &[]
        } else {
            slice::from_raw_parts(vecs, total)
        };
        for (i, &id) in id_slice.iter().enumerate() {
            if let Err(e) = b.add(id, &vec_slice[i * dim..(i + 1) * dim]) {
                return fail(e, 1);
            }
        }
        match b.save(p) {
            Ok(()) => 0,
            Err(e) => fail(e, 1),
        }
    })
}

/// # Safety
/// `path` must be a valid NUL-terminated C string. Returns a handle (or
/// `VANEDB_RS_NULL_HANDLE`) that must be freed with `vanedb_rs_disk_free`.
/// The underlying file must not be modified or truncated from the start of
/// this call until the handle is freed. Replacing its path with a newly built
/// file is allowed; modifying the mapped file in place is not.
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_disk_open(path: *const c_char) -> vanedb_rs_disk {
    guard(VANEDB_RS_NULL_HANDLE, || {
        if path.is_null() {
            return null_arg(VANEDB_RS_NULL_HANDLE);
        }
        match CStr::from_ptr(path).to_str() {
            // The caller guarantees the mapped file remains immutable.
            Ok(p) => match unsafe { DiskIndex::open(p) } {
                Ok(m) => register(Object::Disk(Arc::new(m))),
                Err(e) => fail(e, VANEDB_RS_NULL_HANDLE),
            },
            Err(_) => bad_path(VANEDB_RS_NULL_HANDLE),
        }
    })
}

/// # Safety
/// `q` must point to `dim` valid `f32`s; `out_ids` and `out_dists` must each
/// have room for `k` elements.
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_disk_search(
    m: vanedb_rs_disk,
    q: *const f32,
    k: usize,
    out_ids: *mut u64,
    out_dists: *mut f32,
) -> usize {
    guard(0, || {
        let Some(store) = disk(m) else { return 0 };
        if q.is_null() || out_ids.is_null() || out_dists.is_null() {
            return null_arg(0);
        }
        let query = slice::from_raw_parts(q, store.dimension());
        match store.search(query, k) {
            Ok(res) => write_results(&res, k, out_ids, out_dists),
            Err(e) => fail(e, 0),
        }
    })
}

/// Search with at most one of a callback, allow list, or deny list.
///
/// A non-null list pointer selects that filter even when its length is zero:
/// empty allow accepts nothing; empty deny accepts everything. Null list pointers
/// require zero lengths. With all three filters null the search is unfiltered.
/// Lists must be strictly ascending without duplicates. Conflicting filters,
/// invalid lengths, or unsorted lists fail with `VANEDB_RS_INVALID_PARAMETER`;
/// a null list with nonzero length fails with `VANEDB_RS_NULL_ARGUMENT`.
/// On failure returns zero and leaves result buffers untouched; inspect
/// `vanedb_rs_last_error` to distinguish failure from no matches.
///
/// # Safety
/// `q` must point to `dim` valid `f32`s; `out_ids` and `out_dists` must each
/// have room for `k` elements.
/// Each nonempty list must point to its stated number of valid `u64`s. Inputs
/// must remain valid and unmodified until return, and outputs must not overlap
/// inputs. A callback must obey `vanedb_rs_filter_fn`'s lifetime, reentrancy and
/// unwinding requirements.
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_disk_search_filtered(
    m: vanedb_rs_disk,
    q: *const f32,
    k: usize,
    filter: vanedb_rs_filter_fn,
    user_data: *mut std::ffi::c_void,
    allow: *const u64,
    allow_len: usize,
    deny: *const u64,
    deny_len: usize,
    out_ids: *mut u64,
    out_dists: *mut f32,
) -> usize {
    guard(0, || {
        let Some(store) = disk(m) else { return 0 };
        if q.is_null() || out_ids.is_null() || out_dists.is_null() {
            return null_arg(0);
        }
        let query = slice::from_raw_parts(q, store.dimension());

        if !valid_filter_args(filter, allow, allow_len, deny, deny_len) {
            return 0;
        }
        let pred_closure;
        let c_filter = if !allow.is_null() {
            Some(vanedb::approx::Filter::Allow(if allow_len == 0 {
                &[]
            } else {
                slice::from_raw_parts(allow, allow_len)
            }))
        } else if !deny.is_null() {
            Some(vanedb::approx::Filter::Deny(if deny_len == 0 {
                &[]
            } else {
                slice::from_raw_parts(deny, deny_len)
            }))
        } else if let Some(cb) = filter {
            pred_closure = CFilterClosure::new(cb, user_data);
            Some(vanedb::approx::Filter::Predicate(
                pred_closure.as_predicate(),
            ))
        } else {
            None
        };

        let mut params = vanedb::SearchParams::new();
        if let Some(f) = c_filter {
            params = params.filter(f);
        }

        match store.search_with(query, k, &params) {
            Ok(res) => {
                let n = write_results(&res, k, out_ids, out_dists);
                // A callback may have handled a failing call on another
                // handle. Report the successful outer search, not that error.
                clear_error();
                n
            }
            Err(e) => fail(e, 0),
        }
    })
}

/// Frees a mapped-file handle. `VANEDB_RS_NULL_HANDLE` is a no-op; a handle
/// that is not a live mapped file fails with `VANEDB_RS_INVALID_HANDLE`. A
/// successful free preserves the thread's error state.
#[no_mangle]
pub extern "C" fn vanedb_rs_disk_free(m: vanedb_rs_disk) {
    guard_preserving((), || free(m, Kind::Disk))
}

// ---------------------------------------------------------------------------
// Introspection and mutation.
//
// Each function is a thin wrapper over the Rust method of the same name. An
// invalid handle or a null buffer returns this ABI's failure value rather
// than dereferencing, and `guard()` keeps a panic from unwinding across the
// boundary.
// ---------------------------------------------------------------------------

/// Number of vectors in the store, or 0 if `s` is not a live store.
#[no_mangle]
pub extern "C" fn vanedb_rs_store_len(s: vanedb_rs_store) -> usize {
    guard(0, || store(s).map_or(0, |store| store.len()))
}

/// Vector dimension of the store, or 0 if `s` is not a live store.
#[no_mangle]
pub extern "C" fn vanedb_rs_store_dimension(s: vanedb_rs_store) -> usize {
    guard(0, || store(s).map_or(0, |store| store.dimension()))
}

/// Metric the store was built with: 0 = L2, 1 = cosine, 2 = dot.
///
/// Returns 0 for an invalid handle, which is indistinguishable from L2 —
/// check `vanedb_rs_last_error` before trusting it, as with every other
/// accessor here.
#[no_mangle]
pub extern "C" fn vanedb_rs_store_metric(s: vanedb_rs_store) -> u32 {
    guard(0, || {
        store(s).map_or(0, |store| from_metric(store.metric()))
    })
}

/// Metric the index was built with: 0 = L2, 1 = cosine, 2 = dot.
///
/// A loaded index reads this from the file, so it is the only way a caller can
/// confirm their query convention matches what was stored. Returns 0 for an
/// invalid handle.
#[no_mangle]
pub extern "C" fn vanedb_rs_index_metric(h: vanedb_rs_index) -> u32 {
    guard(0, || index(h).map_or(0, |idx| from_metric(idx.metric())))
}

/// Metric the mapped file was written with: 0 = L2, 1 = cosine, 2 = dot.
///
/// Returns 0 for an invalid handle.
#[no_mangle]
pub extern "C" fn vanedb_rs_disk_metric(d: vanedb_rs_disk) -> u32 {
    guard(0, || disk(d).map_or(0, |disk| from_metric(disk.metric())))
}

/// Whether `id` is present. False if `s` is not a live store.
#[no_mangle]
pub extern "C" fn vanedb_rs_store_contains(s: vanedb_rs_store, id: u64) -> bool {
    guard(false, || store(s).is_some_and(|store| store.contains(id)))
}

/// Copies the vector stored under `id` into `out`. Returns 0 on success, 1 if
/// absent or on error.
///
/// # Safety
/// `out` must have room for `vanedb_rs_store_dimension(s)` floats.
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_store_get(s: vanedb_rs_store, id: u64, out: *mut f32) -> i32 {
    guard(1, || {
        let Some(store) = store(s) else { return 1 };
        if out.is_null() {
            return null_arg(1);
        }
        match store.get(id) {
            Ok(v) => {
                ptr::copy_nonoverlapping(v.as_ptr(), out, v.len());
                0
            }
            Err(e) => fail(e, 1),
        }
    })
}

/// Removes `id`. Returns 0 on success, 1 if absent or on error.
#[no_mangle]
pub extern "C" fn vanedb_rs_store_remove(s: vanedb_rs_store, id: u64) -> i32 {
    guard(1, || {
        let Some(store) = store(s) else { return 1 };
        match store.remove(id) {
            Ok(()) => 0,
            Err(e) => fail(e, 1),
        }
    })
}

/// Number of live vectors in the index, or 0 if `h` is not a live graph.
///
/// Excludes tombstones; see `vanedb_rs_index_tombstones`.
#[no_mangle]
pub extern "C" fn vanedb_rs_index_len(h: vanedb_rs_index) -> usize {
    guard(0, || index(h).map_or(0, |idx| idx.len()))
}

/// Vector dimension of the index, or 0 if `h` is not a live graph.
#[no_mangle]
pub extern "C" fn vanedb_rs_index_dimension(h: vanedb_rs_index) -> usize {
    guard(0, || index(h).map_or(0, |idx| idx.dimension()))
}

/// Whether `id` is present and not deleted. False if `h` is not a live graph.
#[no_mangle]
pub extern "C" fn vanedb_rs_index_contains(h: vanedb_rs_index, id: u64) -> bool {
    guard(false, || index(h).is_some_and(|idx| idx.contains(id)))
}

/// Copies the vector stored under `id` into `out`. Returns 0 on success, 1 if
/// absent or on error.
///
/// # Safety
/// `out` must have room for `vanedb_rs_index_dimension(h)` floats.
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_index_get_vector(
    h: vanedb_rs_index,
    id: u64,
    out: *mut f32,
) -> i32 {
    guard(1, || {
        let Some(idx) = index(h) else { return 1 };
        if out.is_null() {
            return null_arg(1);
        }
        match idx.get_vector(id) {
            Ok(v) => {
                ptr::copy_nonoverlapping(v.as_ptr(), out, v.len());
                0
            }
            Err(e) => fail(e, 1),
        }
    })
}

/// Replaces the vector under `id`, inserting it if absent. Returns 0 on
/// success, 1 on error.
///
/// # Safety
/// `v` must point to `vanedb_rs_index_dimension(h)` valid floats.
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_index_upsert(h: vanedb_rs_index, id: u64, v: *const f32) -> i32 {
    guard(1, || {
        let Some(idx) = index(h) else { return 1 };
        if v.is_null() {
            return null_arg(1);
        }
        let vec = slice::from_raw_parts(v, idx.dimension());
        match idx.upsert(id, vec) {
            Ok(()) => 0,
            Err(e) => fail(e, 1),
        }
    })
}

/// Tombstones `id`. Returns 0 on success, 1 if absent or on error.
///
/// The vector stops being returned by searches immediately; its graph links
/// are retained until `vanedb_rs_index_compact`.
#[no_mangle]
pub extern "C" fn vanedb_rs_index_remove(h: vanedb_rs_index, id: u64) -> i32 {
    guard(1, || {
        let Some(idx) = index(h) else { return 1 };
        match idx.remove(id) {
            Ok(()) => 0,
            Err(e) => fail(e, 1),
        }
    })
}

/// Number of tombstoned slots, or 0 if `h` is not a live graph.
#[no_mangle]
pub extern "C" fn vanedb_rs_index_tombstones(h: vanedb_rs_index) -> usize {
    guard(0, || index(h).map_or(0, |idx| idx.tombstones()))
}

/// Rebuilds the graph from live vectors, clearing all tombstones. Returns 0 on
/// success, 1 on error.
#[no_mangle]
pub extern "C" fn vanedb_rs_index_compact(h: vanedb_rs_index) -> i32 {
    guard(1, || {
        let Some(idx) = index(h) else { return 1 };
        match idx.compact() {
            Ok(()) => 0,
            Err(e) => fail(e, 1),
        }
    })
}

/// Number of vectors in the mapped file, or 0 if `d` is not a live mapped file.
#[no_mangle]
pub extern "C" fn vanedb_rs_disk_len(d: vanedb_rs_disk) -> usize {
    guard(0, || disk(d).map_or(0, |disk| disk.size()))
}

/// Vector dimension of the mapped file, or 0 if `d` is not a live mapped file.
#[no_mangle]
pub extern "C" fn vanedb_rs_disk_dimension(d: vanedb_rs_disk) -> usize {
    guard(0, || disk(d).map_or(0, |disk| disk.dimension()))
}

/// Whether `id` is present. False if `d` is not a live mapped file.
#[no_mangle]
pub extern "C" fn vanedb_rs_disk_contains(d: vanedb_rs_disk, id: u64) -> bool {
    guard(false, || disk(d).is_some_and(|disk| disk.contains(id)))
}

/// Copies the vector stored under `id` into `out`. Returns 0 on success, 1 if
/// absent or on error.
///
/// # Safety
/// `out` must have room for `vanedb_rs_disk_dimension(d)` floats.
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_disk_get(d: vanedb_rs_disk, id: u64, out: *mut f32) -> i32 {
    guard(1, || {
        let Some(disk) = disk(d) else { return 1 };
        if out.is_null() {
            return null_arg(1);
        }
        match disk.get(id) {
            Ok(v) => {
                ptr::copy_nonoverlapping(v.as_ptr(), out, v.len());
                0
            }
            Err(e) => fail(e, 1),
        }
    })
}

/// Reads the vector stored under `id` into `out`, which must have room for
/// `dim` floats. Returns 0, or 1 if absent.
///
/// The same operation as `vanedb_rs_index_get_vector`, under the spelling the
/// store and disk handles use. Both exist so swapping index type does not mean
/// renaming call sites (#85).
///
/// # Safety
/// `out` must point to at least `dim` writable `f32`s.
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_index_get(h: vanedb_rs_index, id: u64, out: *mut f32) -> i32 {
    vanedb_rs_index_get_vector(h, id, out)
}

/// The same operation as `vanedb_rs_store_get`, under the spelling the graph
/// handle uses (#85).
///
/// # Safety
/// `out` must point to at least `dim` writable `f32`s.
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_store_get_vector(
    s: vanedb_rs_store,
    id: u64,
    out: *mut f32,
) -> i32 {
    vanedb_rs_store_get(s, id, out)
}

/// The same operation as `vanedb_rs_disk_get`, under the spelling the graph
/// handle uses (#85).
///
/// # Safety
/// `out` must point to at least `dim` writable `f32`s.
#[no_mangle]
pub unsafe extern "C" fn vanedb_rs_disk_get_vector(
    d: vanedb_rs_disk,
    id: u64,
    out: *mut f32,
) -> i32 {
    vanedb_rs_disk_get(d, id, out)
}

/// The graph's `M`, or 0 for an invalid handle.
///
/// A handle from `vanedb_rs_index_load` read this from the file, and a caller
/// who did not build it has no other way to know what graph they are searching.
/// The same argument covers `ef_construction`, `seed` and `capacity`.
#[no_mangle]
pub extern "C" fn vanedb_rs_index_m(h: vanedb_rs_index) -> usize {
    guard(0, || index(h).map_or(0, |idx| idx.m()))
}

/// The graph's `ef_construction`, or 0 for an invalid handle.
#[no_mangle]
pub extern "C" fn vanedb_rs_index_ef_construction(h: vanedb_rs_index) -> usize {
    guard(0, || index(h).map_or(0, |idx| idx.ef_construction()))
}

/// The seed the graph was built with, or 0 for an invalid handle.
#[no_mangle]
pub extern "C" fn vanedb_rs_index_seed(h: vanedb_rs_index) -> u64 {
    guard(0, || index(h).map_or(0, |idx| idx.seed()))
}

/// The capacity the graph was built with, or 0 for an invalid handle.
///
/// This is the build-time hint, not a limit: the index grows past it, so this
/// may be smaller than `vanedb_rs_index_len`.
#[no_mangle]
pub extern "C" fn vanedb_rs_index_capacity(h: vanedb_rs_index) -> usize {
    guard(0, || index(h).map_or(0, |idx| idx.capacity()))
}

/// Sets the handle's stored `ef_search` — the beam a search gets when it
/// passes 0, and the value `vanedb_rs_index_save` writes into the file.
///
/// Search still takes `ef_search` per call and does not touch this, so a query
/// cannot disturb another thread's. This exists because without it the stored
/// value was permanently the default for a C-built index: `0` could only ever
/// mean 50, and a tuned index could not be saved from C at all.
#[no_mangle]
pub extern "C" fn vanedb_rs_index_set_ef_search(h: vanedb_rs_index, ef_search: usize) -> i32 {
    guard(1, || {
        let Some(idx) = index(h) else { return 1 };
        idx.set_ef_search(ef_search);
        0
    })
}

/// The handle's stored `ef_search` — the beam a search gets when it passes 0.
///
/// There was previously no way to read it back, so a caller could not tell what
/// a `0` would resolve to.
#[no_mangle]
pub extern "C" fn vanedb_rs_index_ef_search(h: vanedb_rs_index) -> usize {
    guard(0, || index(h).map_or(0, |idx| idx.get_ef_search()))
}

#[cfg(test)]
mod tests {
    use super::guard;

    /// The guard is what stands between an engine bug and a dead host process,
    /// so it is tested directly rather than through a contrived engine panic —
    /// an entry-point test that never actually panics would pass with the
    /// guard removed and prove nothing.
    #[test]
    fn a_panic_becomes_the_fallback() {
        assert_eq!(guard(1i32, || panic!("engine bug")), 1);
        assert_eq!(guard(0usize, || panic!("engine bug")), 0);
        assert!(guard(f32::NAN, || panic!("engine bug")).is_nan());
        assert_eq!(guard(0u64, || panic!("engine bug")), 0);
    }

    /// The introspection and mutation surface, end to end through the ABI:
    /// an embedder can read a vector back, delete one, and ask how many the
    /// handle holds.
    #[test]
    fn the_introspection_surface_round_trips() {
        unsafe {
            let s = super::vanedb_rs_store_new(4, 0);
            assert_eq!(super::vanedb_rs_store_dimension(s), 4);
            assert_eq!(super::vanedb_rs_store_len(s), 0);

            let v = [1.0f32, 2.0, 3.0, 4.0];
            assert_eq!(super::vanedb_rs_store_add(s, 7, v.as_ptr()), 0);
            assert_eq!(super::vanedb_rs_store_len(s), 1);
            assert!(super::vanedb_rs_store_contains(s, 7));
            assert!(!super::vanedb_rs_store_contains(s, 8));

            let mut out = [0.0f32; 4];
            assert_eq!(super::vanedb_rs_store_get(s, 7, out.as_mut_ptr()), 0);
            assert_eq!(out, v);
            assert_eq!(super::vanedb_rs_store_get(s, 8, out.as_mut_ptr()), 1);

            assert_eq!(super::vanedb_rs_store_remove(s, 7), 0);
            assert_eq!(super::vanedb_rs_store_remove(s, 7), 1);
            assert_eq!(super::vanedb_rs_store_len(s), 0);
            super::vanedb_rs_store_free(s);

            let h = super::vanedb_rs_index_new(4, 0, 64, 16, 200, 42);
            assert_eq!(super::vanedb_rs_index_dimension(h), 4);
            assert_eq!(super::vanedb_rs_index_add(h, 1, v.as_ptr()), 0);
            assert_eq!(super::vanedb_rs_index_len(h), 1);
            assert!(super::vanedb_rs_index_contains(h, 1));
            assert_eq!(super::vanedb_rs_index_get_vector(h, 1, out.as_mut_ptr()), 0);
            assert_eq!(out, v);

            // upsert replaces in place; len does not grow.
            let w = [9.0f32, 9.0, 9.0, 9.0];
            assert_eq!(super::vanedb_rs_index_upsert(h, 1, w.as_ptr()), 0);
            assert_eq!(super::vanedb_rs_index_len(h), 1);
            assert_eq!(super::vanedb_rs_index_get_vector(h, 1, out.as_mut_ptr()), 0);
            assert_eq!(out, w);

            // upsert tombstones the slot it replaces, so it costs space
            // until a compact: a loop of upserts grows the index.
            assert_eq!(super::vanedb_rs_index_tombstones(h), 1);

            assert_eq!(super::vanedb_rs_index_remove(h, 1), 0);
            assert_eq!(super::vanedb_rs_index_len(h), 0);
            assert_eq!(
                super::vanedb_rs_index_remove(h, 1),
                1,
                "removing twice must fail"
            );
            assert_eq!(super::vanedb_rs_index_tombstones(h), 2);
            assert_eq!(super::vanedb_rs_index_compact(h), 0);
            assert_eq!(super::vanedb_rs_index_tombstones(h), 0);
            assert_eq!(super::vanedb_rs_index_len(h), 0);
            super::vanedb_rs_index_free(h);
        }
    }

    /// The null handle or a null output pointer returns the failure value
    /// rather than dereferencing, as everywhere else in this ABI.
    #[test]
    fn the_introspection_surface_rejects_nulls() {
        use super::VANEDB_RS_NULL_HANDLE as NULL;
        unsafe {
            let mut out = [0.0f32; 4];
            assert_eq!(super::vanedb_rs_store_len(NULL), 0);
            assert_eq!(super::vanedb_rs_store_dimension(NULL), 0);
            assert!(!super::vanedb_rs_store_contains(NULL, 1));
            assert_eq!(super::vanedb_rs_store_get(NULL, 1, out.as_mut_ptr()), 1);
            assert_eq!(super::vanedb_rs_store_remove(NULL, 1), 1);

            assert_eq!(super::vanedb_rs_index_len(NULL), 0);
            assert_eq!(super::vanedb_rs_index_dimension(NULL), 0);
            assert!(!super::vanedb_rs_index_contains(NULL, 1));
            assert_eq!(
                super::vanedb_rs_index_get_vector(NULL, 1, out.as_mut_ptr()),
                1
            );
            assert_eq!(super::vanedb_rs_index_remove(NULL, 1), 1);
            assert_eq!(super::vanedb_rs_index_tombstones(NULL), 0);
            assert_eq!(super::vanedb_rs_index_compact(NULL), 1);
            assert_eq!(super::vanedb_rs_index_upsert(NULL, 1, out.as_ptr()), 1);

            assert_eq!(super::vanedb_rs_disk_len(NULL), 0);
            assert_eq!(super::vanedb_rs_disk_dimension(NULL), 0);
            assert!(!super::vanedb_rs_disk_contains(NULL, 1));
            assert_eq!(super::vanedb_rs_disk_get(NULL, 1, out.as_mut_ptr()), 1);

            // A live handle with a null output buffer must also be rejected.
            let s = super::vanedb_rs_store_new(4, 0);
            assert_eq!(super::vanedb_rs_store_get(s, 1, std::ptr::null_mut()), 1);
            super::vanedb_rs_store_free(s);
        }
    }

    /// Buffer pointers are checked on every entry point, not just the handle.
    /// Reading through a null query is undefined behaviour, which `guard()`
    /// cannot catch — it intercepts panics, not UB.
    #[test]
    fn null_data_pointers_are_rejected_by_index_and_disk() {
        unsafe {
            let h = super::vanedb_rs_index_new(4, 0, 16, 16, 200, 42);
            assert_ne!(h, super::VANEDB_RS_NULL_HANDLE);
            let (mut ids, mut ds) = ([0u64; 2], [0f32; 2]);
            assert_eq!(
                super::vanedb_rs_index_search(
                    h,
                    std::ptr::null(),
                    2,
                    50,
                    ids.as_mut_ptr(),
                    ds.as_mut_ptr()
                ),
                0
            );
            let q = [0.0f32; 4];
            assert_eq!(
                super::vanedb_rs_index_search(
                    h,
                    q.as_ptr(),
                    2,
                    50,
                    std::ptr::null_mut(),
                    ds.as_mut_ptr()
                ),
                0
            );
            assert_eq!(
                super::vanedb_rs_index_search(
                    h,
                    q.as_ptr(),
                    2,
                    50,
                    ids.as_mut_ptr(),
                    std::ptr::null_mut()
                ),
                0
            );
            super::vanedb_rs_index_free(h);

            let path = std::ffi::CString::new(
                std::env::temp_dir()
                    .join(format!("vanedb_capi_null-{}.disk", std::process::id()))
                    .to_str()
                    .unwrap(),
            )
            .unwrap();
            let ids2 = [1u64];
            assert_eq!(
                super::vanedb_rs_disk_build(
                    path.as_ptr(),
                    4,
                    0,
                    ids2.as_ptr(),
                    std::ptr::null(),
                    1
                ),
                1
            );
            assert_eq!(
                super::vanedb_rs_disk_build(path.as_ptr(), 4, 0, std::ptr::null(), q.as_ptr(), 1),
                1
            );
            // n * dim overflowing must fail rather than wrap into a bogus slice.
            assert_eq!(
                super::vanedb_rs_disk_build(
                    path.as_ptr(),
                    usize::MAX,
                    0,
                    ids2.as_ptr(),
                    q.as_ptr(),
                    3
                ),
                1
            );
        }
    }

    #[test]
    fn null_data_pointers_are_rejected() {
        unsafe {
            let s = super::vanedb_rs_store_new(4, 0);
            assert_ne!(s, super::VANEDB_RS_NULL_HANDLE);
            assert_eq!(super::vanedb_rs_store_add(s, 1, std::ptr::null()), 1);
            assert_eq!(
                super::vanedb_rs_store_add_batch(s, std::ptr::null(), std::ptr::null(), 3),
                1
            );
            let (mut ids, mut ds) = ([0u64; 2], [0f32; 2]);
            assert_eq!(
                super::vanedb_rs_store_search(
                    s,
                    std::ptr::null(),
                    2,
                    ids.as_mut_ptr(),
                    ds.as_mut_ptr()
                ),
                0
            );
            let v = [1.0f32, 0.0, 0.0, 0.0];
            assert_eq!(
                super::vanedb_rs_store_search(
                    s,
                    v.as_ptr(),
                    2,
                    std::ptr::null_mut(),
                    ds.as_mut_ptr()
                ),
                0
            );
            // A well-formed call still works.
            assert_eq!(super::vanedb_rs_store_add(s, 1, v.as_ptr()), 0);
            super::vanedb_rs_store_free(s);
        }
    }

    #[test]
    fn element_count_refuses_to_wrap() {
        // The realistic 32-bit case: wasm32 with a large n and dim.
        assert_eq!(super::elements(100_000, 8), Some(800_000));
        assert_eq!(super::elements(usize::MAX, 2), None);
        assert_eq!(super::elements(usize::MAX / 2 + 1, 2), None);
        assert_eq!(super::elements(0, usize::MAX), Some(0));
    }

    #[test]
    fn a_normal_return_passes_through_untouched() {
        assert_eq!(guard(1i32, || 0i32), 0);
        assert_eq!(guard(0usize, || 7usize), 7);
    }

    /// The id layout is what makes a truncated id unmatchable: the generation
    /// lives in the high half, starts at 1, and never reaches `u32::MAX`.
    #[test]
    fn id_layout_keeps_generation_in_the_high_half() {
        let id = super::compose(3, 5, 9);
        assert_eq!(super::generation_of(id), 9);
        assert_eq!(super::slot_of(id), Some(5));
        assert!(std::ptr::eq(super::shard_of(id), &super::TABLE[3]));
        assert_eq!(super::generation_of(id & 0xFFFF_FFFF), 0);
        // The very first id (shard 0, slot 0) must not truncate to the null
        // handle: that would read as a null argument, not an invalid handle.
        let first = super::compose(0, 0, super::FIRST_GENERATION);
        assert_ne!(first & 0xFFFF_FFFF, 0);
        assert_eq!(super::slot_of(first), Some(0));
        assert_eq!(super::slot_of(0), None, "a zero low half names no slot");
        // A zero-extended truncation carries generation 0 and a sign-extended
        // one carries u32::MAX; neither is inside the issuable range.
        let issuable = super::FIRST_GENERATION..=super::LAST_GENERATION;
        assert!(!issuable.contains(&0));
        assert!(!issuable.contains(&u32::MAX));
    }

    /// A slot on its last issuable generation retires instead of wrapping to
    /// 0 or `u32::MAX`, the two generations a truncated id can carry.
    #[test]
    fn a_slot_at_the_last_generation_retires_rather_than_wrapping() {
        let store = super::Arc::new(super::FlatIndex::new(1, super::Metric::L2).unwrap());
        let id = super::insert(super::Object::Store(store.clone())).unwrap();
        // Age the slot to its final generation by hand.
        let index = super::slot_of(id).unwrap();
        let aged = {
            let mut shard = super::lock(super::shard_of(id));
            let slot = &mut shard.slots[index];
            slot.generation = super::LAST_GENERATION;
            super::compose(
                (id as u32) & (super::SHARDS as u32 - 1),
                index as u32,
                super::LAST_GENERATION,
            )
        };
        assert!(super::lookup(id).is_none(), "the old generation is dead");
        assert!(super::lookup(aged).is_some());
        assert!(super::release(aged, super::Kind::Store).is_some());
        assert!(super::lookup(aged).is_none());
        let shard = super::lock(super::shard_of(id));
        let slot = &shard.slots[index];
        assert_eq!(slot.generation, u32::MAX, "retired");
        assert!(
            !shard.free.contains(&(index as u32)),
            "a retired slot is never handed out again"
        );
    }
}
