# Vendored third-party code for `bench/compare`

| Path | Upstream | Version | License |
|---|---|---|---|
| `hnswlib/` | [nmslib/hnswlib](https://github.com/nmslib/hnswlib) | v0.8.0 | Apache-2.0 |
| `sqlite-vec.c` / `sqlite-vec.h` | [asg017/sqlite-vec](https://github.com/asg017/sqlite-vec) amalgamation | v0.1.6 | MIT OR Apache-2.0 |

Do not edit these files. Bump the pinned version and re-copy from the upstream
release when updating.

## Local patches

`sqlite-vec.c` (v0.1.6 amalgamation): removed the BSD-style
`typedef u_int8_t uint8_t;` block. On glibc those typedefs conflict with
`<stdint.h>` and break `-DSQLITE_CORE` builds. Re-apply when bumping the
amalgamation.
