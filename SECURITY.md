# Security Policy

## Supported versions

0.1.1 is the first published release of the Rust crate, the Python package and
the WebAssembly package, and is the supported version. Report issues against it
or against `main`, and include the package version or commit. Earlier `0.1.0-rc`
prereleases and the npm `0.1.0` bootstrap are historical and not supported.

During 0.x, fixes land in a new minor or patch release rather than being
backported.

## Reporting a vulnerability

Use [GitHub private vulnerability reporting](https://github.com/vanedb/vanedb/security/advisories/new).
Do not include vulnerability details in a public issue.

Include the affected version, platform, reproduction and expected impact.
For loader issues, attach a crafted input file or a short script that produces
it. Maintainers assess reports through the private advisory; response and fix
timelines depend on severity and availability.

## Scope

Please report memory-safety defects, malicious-file validation bypasses,
unchecked size arithmetic, thread-safety defects, and failures across binding
boundaries. This includes VNDB v1 disk files, VNDB v2 graph files, legacy HNSW
files, and the Rust, Python, C and WebAssembly interfaces.

Ordinary performance questions belong in public issues. Report dependency
vulnerabilities upstream and privately notify this project when VaneDB is
affected. A deliberately large valid allocation can still exhaust available
memory; checked input sizes do not impose an application memory quota.

## Validation practices

- Rust CI uses `cargo-deny` for advisories, licences, bans and sources. Exceptions
  and their rationale are recorded in `deny.toml`.
- Loader tests exercise corrupt headers, derived sizes and ID rules. Disk IDs
  must be unique; graph validation requires unique live IDs and permits reuse
  of a tombstoned ID.
- C ABI entry points contain Rust unwinding panics and return an error value.
  Callers must still satisfy documented pointer and lifetime requirements;
  invalid pointers and process aborts cannot be caught this way.
- SIMD kernels are compared with scalar implementations across boundary lengths.
- Index saves use a temporary sibling and rename it into place. This protects
  replacement atomicity; it is not a guarantee against every power-loss scenario.
- C++ reference CI includes AddressSanitizer and UndefinedBehaviorSanitizer.
