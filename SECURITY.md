# Security Policy

## Supported versions

Nothing is published to crates.io or PyPI yet. Until the first release, the
supported version is `main`; report against it.

| Version | Supported |
| --- | --- |
| `main` | yes |
| published releases | none yet |

## Reporting a vulnerability

Do **not** open a public issue. Either:

- open a [private security advisory](https://github.com/vanedb/vanedb/security/advisories/new), or
- email **security@tsvetkov.org**

Include what you found, how to reproduce it, and the impact you think it has.
A crafted input file is the most useful reproduction this project can receive —
attach it, or the few lines that generate it.

**Acknowledgment** within 48 hours, **initial assessment** within 7 days. Fix
timelines follow severity: critical in days, high in a week or two, lower
severity in the next release cycle.

## Scope

The loaders parse untrusted input, so that is where the interesting surface is.

**In scope**

- Memory safety, including anything reachable from safe Rust
- Malicious or corrupt index files — `VNDB` (`DiskIndex`) and `HNSW`
  (`ApproxIndex`) — that read out of bounds, over-allocate, or load as valid
- Integer overflow in size or offset arithmetic on file-controlled values
- Input validation bypasses in any binding: Rust, Python, C ABI or wasm
- Thread-safety bugs causing corruption or wrong results
- Anything that panics across the C ABI boundary rather than returning an error

**Known and documented, so not a report**

- Modifying or truncating a `DiskIndex` file while it is open. The file is
  mapped, so this raises SIGBUS and kills the process, from any binding. It is
  documented on `DiskIndex::open` and on the C and Python equivalents.
  Rebuilding through `DiskIndexBuilder::save` is safe — it renames a new file
  into place and leaves open readers on the old one.

**Out of scope**

- Denial of service through a legitimately large allocation request
- Performance
- Vulnerabilities in dependencies — report upstream, then tell us so the
  advisory ignore list in `deny.toml` can be revisited

## What the project does about this

- `cargo-deny` gates advisories, licences, bans and sources on every change
  to Rust sources or manifests; the two
  ignored advisories are both `unmaintained`, not vulnerabilities, and each
  records a reason and an exit in `deny.toml`.
- Both loaders validate magic, version, and every derived size with checked
  arithmetic before allocating or indexing. Neither accepts a file whose ids
  are not unique: `DiskIndex` checks directly, and `ApproxIndex` enforces a
  bijection between its id map and its stored ids, which forces the same thing.
  `vanedb/tests/corruption_tests.rs` pins these with crafted files.
- The C ABI routes every entry point through `catch_unwind`, so a panic returns
  an error rather than unwinding into a foreign frame.
- SIMD kernels are checked against the scalar reference at every length from 0
  to 80, which is what the loop tiers require to be covered.
- Saves write to a temporary sibling and rename it into place, so a reader
  never observes a partially written index.
- CI runs AddressSanitizer and UBSan against the C++ engine.
