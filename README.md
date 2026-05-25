# VaneDB

Embeddable vector database for edge AI — Rust crate with Python (PyO3) and WASM bindings.

## Implementations

- **Rust** (this repo) — `cargo add vanedb`, Python via `vanedb-py` (PyO3), WASM via `vanedb-wasm` (wasm-bindgen).
- **C++** — [vanedb/vanedb-cpp](https://github.com/vanedb/vanedb-cpp), header-only, no Rust toolchain required.

Both maintained side-by-side under the [@vanedb](https://github.com/vanedb) org. Features generally land in Rust first; core algorithms and persistence formats are synced to C++.

## License

MIT
