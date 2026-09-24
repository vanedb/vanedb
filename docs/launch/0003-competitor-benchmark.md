# Launch post draft — competitor benchmark + demo (RFC 0003)

**Status:** draft for maintainer review. Do **not** publish until
`bench/COMPARISON.md` has dedicated-hardware results for Apple Silicon, Linux
AVX2, and Android ARM64 (labelled), and the demo README shows one real-vault
search.

**Channels:** Show HN, r/rust, r/LocalLLaMA (adapt tone per channel; keep the
numbers and caveats identical).

---

## Title options

1. Show HN: VaneDB – embeddable vector search vs USearch, hnswlib, and sqlite-vec
2. *(Use only after COMPARISON tables are filled.)* We benchmarked our edge
   vector DB against USearch, hnswlib, instant-distance, hnsw_rs, and sqlite-vec
   on real 768-d embeddings

## Body (draft)

We built VaneDB as an embeddable nearest-neighbour library for on-device / edge
AI (Rust, Python, C, WASM). Until now the only published comparison was against
our own frozen C++ reference engine — useless to anyone choosing a stack.

We are adding a second benchmark arm that will run VaneDB against the libraries
people actually shortlist, on a fixed real-embedding fixture (nomic-embed-text,
768-d, 100k documents, 1k queries), with interleaved rounds on dedicated
hardware:

- USearch
- hnswlib
- instant-distance
- hnsw_rs
- sqlite-vec (brute-force baseline)

Measured: build time, peak RSS, index file size, latency@k=10 across an ef
sweep, recall@10 vs f64 exact search, and delete-then-search where the engine
supports delete.

**Methodology and caveats first:** https://github.com/vanedb/vanedb/blob/main/bench/COMPARISON.md

Please read the caveats before the tables. **Numbers will be pasted here only
after** dedicated-machine runs fill `bench/COMPARISON.md` (currently pending).
CI / cloud timings are discarded by policy and must never appear in the post.

**Demo:** Obsidian vault search powered by the WASM build —
https://github.com/vanedb/obsidian-vane-search — *(official 0.2.0 walkthrough
still tracked under [#242](https://github.com/vanedb/vanedb/issues/242); use
the reviewed demo PR and checklist below).* Index a vault and run one
semantic search locally (Ollama + nomic-embed-text, or any OpenAI-compatible
embeddings endpoint) once that release is live.

What we are looking for once the tables are filled: whether those dedicated-HW
results match what you see on your hardware, and which gap (filtering,
quantization, mobile SDKs, wasm persistence) would unblock a real project.

---

## Demo update checklist (`obsidian-vane-search`, separate repo)

Use the reviewed-PR sequence in
[`0003-demo-update-checklist.md`](0003-demo-update-checklist.md). The current
candidate is [demo PR #20](https://github.com/vanedb/obsidian-vane-search/pull/20)
(vault acceptance recorded on tip). Merge it, then publish the annotated
`0.2.0` tag (`maintainer_closeout_226.sh --tag --confirm-vault-walkthrough`).
Record its official release URL and walkthrough evidence on
[#242](https://github.com/vanedb/vanedb/issues/242). The historical staging
release does not meet this requirement.
