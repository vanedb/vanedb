# Vendored third-party code for `bench/compare`

| Path | Upstream | Version | License |
|---|---|---|---|
| `hnswlib/` | [nmslib/hnswlib](https://github.com/nmslib/hnswlib) | v0.9.0 | Apache-2.0 |
| `sqlite-vec.c` / `sqlite-vec.h` | [asg017/sqlite-vec](https://github.com/asg017/sqlite-vec) amalgamation | v0.1.9 | MIT OR Apache-2.0 |

Do not edit these files. Bump the pinned version and re-copy from the upstream
release when updating.

## Local patches

`sqlite-vec.c` (v0.1.9 amalgamation): removed the BSD-style
`typedef u_int8_t uint8_t;` block (and the matching `uint16_t` / `uint64_t`
lines). On glibc those typedefs conflict with `<stdint.h>` and break
`-DSQLITE_CORE` builds. Re-apply when bumping the amalgamation.

## Stable pins

hnswlib `v0.10.0-rc.2` and sqlite-vec `v0.1.10-alpha.4` are pre-releases.
The harness pins the latest stable tag of each. `instant-distance` 0.6.1 is
the latest crates.io release.
