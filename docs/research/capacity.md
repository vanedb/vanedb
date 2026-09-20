# Capacity study: client corpora, memory budgets, mapped versus quantized storage

Dated 2026-09-20. Answers the questions in
[#210](https://github.com/vanedb/vanedb/issues/210), which gates
[RFC 0008](../rfcs/0008-streaming-disk-build-and-mapped-graph.md) (streaming
disk build and mapped graph) and orders it against
[RFC 0005](../rfcs/0005-quantized-storage.md) (quantized storage).
[`LIMITS.md`](../LIMITS.md) records what 0.1.1 holds today; this page records
what it needs to hold, what the alternatives hold, and which design the
numbers favour first.

Everything here is either **computed** (a formula derived from the source or a
format specification, checked against the allocator with the script in
section 4.4), **quoted** (a published claim with its URL and the date it was
read), or **assumed** (a judgement, labelled as such). Nothing is a benchmark:
no timing from any machine appears on this page, and the on-device latency
measurements the issue asks for in its question 3 are recorded as open
(section 8) rather than invented.

## 1. Scope and method

- **Segments.** The five client segments this study uses: browser extension,
  Obsidian or note vault, mobile app, desktop RAG, embedded device. They map
  onto the issue's four (browser tab; desktop RAG / note-taking plugin; mobile
  app; edge gateway / industrial) with the note vault split out because the
  `obsidian-vane-search` demo is the project's first real workload.
- **Corpus size** is counted in vectors, not documents: one vector per chunk
  for text (typically 300 to 1,500 characters), one per item for bookmarks,
  images or sensor windows. Dimensions follow the models each segment
  actually uses: 384 (MiniLM, bge-micro, gte-small), 768 (nomic-embed-text,
  EmbeddingGemma), 1536 (OpenAI `text-embedding-3-small`).
- **Memory budgets** are the resident bytes an index may take before the
  platform or the user objects, not device RAM. They come from platform
  documentation where it could be read and from recorded kills where it could
  not.
- **VaneDB's cost per vector** is derived from the structs in `vanedb/src`
  (section 4) and checked with a counting global allocator against real
  `FlatIndex`, `ApproxIndex` and `DiskIndex` instances.
- **Competitors** are described from their own repositories and
  documentation (section 5). Every claim carries its URL and the date read.
- **Units.** Index and file sizes in section 4 are decimal (MB = 10^6 bytes,
  GB = 10^9). Platform budgets in section 3 and the capacity tables in
  sections 6 and 7 are binary (MiB, GiB), because platforms state their
  limits that way; 1 GiB / 3,421 B = 314k vectors where 1 GB would give 292k.
- **Network access.** This study was written from a sandbox whose egress
  proxy blocks most vendor documentation hosts (Apple, Android, MDN,
  Chromium, Obsidian's forum, LanceDB, Chroma, ObjectBox, Turso, arXiv,
  Wikipedia, Hugging Face). GitHub repositories and issues were reachable.
  Where a figure comes only from a search-result snippet of a page that could
  not be opened it is marked **(snippet only; not verified)**; where no source
  could be read at all it is marked **(not verified)**. Section 9 lists every
  source with its status.

## 2. Client corpus sizes by segment

| Segment | Typical vectors | Upper bound seen | Dims in use | Updates | Evidence |
|---|---:|---:|---|---|---|
| Browser extension (bookmarks, history) | 1k to 20k | 100k to 1M (full history with several passages per page) | 384 in extensions; Chrome's own history embeddings use a larger model | incremental, one item at a time | `findmark` embeds bookmarks with `Xenova/all-MiniLM-L6-v2` at 384 dims and stores "~2 MB for thousands of bookmarks" after int8 quantization ([README](https://github.com/daveshenal/findmark), read 2026-09-20). Chrome ships local history embeddings since 2024 **(snippet only; not verified)**; its dimension and passages-per-page appear in a third-party analysis that could not be opened **(snippet only; not verified)**. Upper bound assumed from ~100 pages a day × 3 to 10 passages × 1 to 3 years. |
| Obsidian / note vault | 5k to 50k chunks (1k to 10k notes × 3 to 10 chunks) | 500k to 1M chunks (vaults past 100k notes exist) | 384 default in Smart Connections; 768 or 1536 in `obsidian-vane-search` | incremental, per saved note | Smart Connections' bundled local model is `TaylorAI/bge-micro-v2`, 384 dims ([adapter source](https://github.com/brianpetro/jsbrains/blob/main/smart-embed-model/adapters/transformers.js), read 2026-09-20). `obsidian-vane-search` defaults to OpenAI `text-embedding-3-small` at 1536 dims with `nomic-embed-text` at 768 as the local option ([README](https://github.com/vanedb/obsidian-vane-search), read 2026-09-20). Obsidian forum threads report vaults of ~1k notes as "fairly large" and power users past 10k notes, with stress tests near 280k files **(snippet only; not verified)**. |
| Mobile app (offline RAG, personal knowledge, on-device search) | 1k to 50k chunks | 100k to 500k (an offline manual or message archive) | 384 (gte-small, bge-small) and 768 (EmbeddingGemma, Gecko); EmbeddingGemma can truncate to 512/256/128 **(not verified; model card not fetched)** | incremental; often rebuilt when the model changes | An on-device first-aid RAG paper reports ~8,000 chunks embedded with a small encoder, int8-quantized, in a flat index under a 2 GB app budget **(snippet only; not verified)**. Google's AI Edge RAG SDK stores vectors in SQLite (`SqliteVectorStore`) **(page not fetched; not verified)**. Turso positions its React Native binding for "vector search and personal knowledge graphs on mobile" ([Turso blog, cited in `MARKET_ANALYSIS.md`](../MARKET_ANALYSIS.md); host blocked on 2026-09-20). |
| Desktop RAG (Ollama-style, single laptop) | 10k to 300k chunks | 1M to 10M (a mail archive or a code base; Chroma's guidance stops at ~7M) | 768 (nomic-embed-text) dominant; 384 for speed; 1024 to 1536 with hosted models | batch ingest, then incremental | Local RAG tutorials pair `nomic-embed-text` with Chroma or LanceDB on "a 16 GB machine" **(snippet only; not verified)**. Chroma's single-node guidance gives a capacity formula and says it tested to ~7M embeddings **(snippet only; not verified)**, which bounds what the Python audience has tried. |
| Embedded / edge gateway (Jetson, Raspberry Pi, industrial) | 10k to 300k (images, sensor windows, parts catalogues) | 1M to 10M (continuous capture over months) | 512 to 768 (CLIP, image-text models) | append-heavy, periodic compaction | `nanodb` indexes "275K images from the MS COCO image captioning dataset" on a Jetson AGX Orin ([README](https://github.com/dusty-nv/jetson-containers/blob/master/packages/vectordb/nanodb/README.md), read 2026-09-20). Actian markets VectorAI DB to Jetson and Raspberry Pi ([`MARKET_ANALYSIS.md`](../MARKET_ANALYSIS.md)). |

Acceptable latency, **assumed** because no segment publishes a number: an
interactive search must return in well under the time the embedding model
takes to encode the query, which on-device is tens to hundreds of
milliseconds, so a warm index search of a few milliseconds is invisible and
20 ms is the ceiling; a cold search (first query after launch, or after the
platform evicted the page cache) may take one to two orders of magnitude
longer once, but not on every query. RFC 0008's 20 ms p99 gate is consistent
with this and is kept.

Reading of the table: **every segment's typical corpus is below 100k
vectors, and every upper bound is between 500k and 10M.** No segment reaches
past 10M on one device, which is the revisit condition RFC 0008 set for a
DiskANN-style design.

## 3. Memory budgets per platform

| Platform | Hard limit | Practical budget for an index | Source and status |
|---|---|---|---|
| Browser tab, wasm32 | Linear memory is 32-bit addressed: 65,536 pages × 64 KiB = 4 GiB **(computed from the wasm32 address space; the Chrome and MDN pages were blocked)**. Memory64 lifts it (Chrome 133, Firefox 134) **(snippet only; not verified)**, but `@vanedb/wasm` targets wasm32. | 256 MB on a phone browser, 1 GB on desktop | EdgeVec, the closest browser competitor, states a "~1GB practical limit" for client-side search ([README](https://github.com/matte1782/edgevec), read 2026-09-20). Chrome's V8 heap is capped near 4 GB by pointer compression **(snippet only; not verified)**; iOS Safari web content is killed at device-dependent limits of roughly 0.6 to 2 GB **(snippet only; not verified)**. |
| iOS app | Jetsam `per-process-limit`, device-class dependent. A recorded kill on an iPhone 14 (6 GB) lists `lifetimeMax` 134,272 pages × 16,384 bytes ≈ 2.05 GB ([home-assistant/iOS #4475](https://github.com/home-assistant/iOS/issues/4475), read 2026-09-20). `com.apple.developer.kernel.increased-memory-limit` raises the cap on capable devices; an iPhone 17 Pro killed a process "with 11 GB of the device's 12 GB still free" without it ([boardsesh #5524](https://github.com/boardsesh/boardsesh/pull/5524), read 2026-09-20). | 500 MB to 1 GB for the index inside a 2 GB foreground cap; far less in extensions ("Extensions have a much lower limit", WWDC18 notes) | The jetsam footprint counts dirty and compressed pages; clean memory is not counted ([WWDC 2018 session 416 notes](https://gist.github.com/SheldonWangRJT/5d2ea69f78a905c76e0c36dfc994e85c), read 2026-09-20 — **a third-party gist of the session notes; Apple's own transcript could not be fetched. This rule is load-bearing for section 7 and should be confirmed against Apple's page**). A read-only file mapping is clean until written, so **mapped f32 vectors (RFC 0008) should not count toward the iOS footprint** (inferred from that rule); resident quantized vectors (RFC 0005) do. |
| Android app | No fixed native cap. `lmkd` kills by `oom_adj_score` under memory pressure; `isLowRamDevice()` marks devices with about 1 GB or less; `getMemoryClass()` bounds only the Java heap (baseline 16 MB) **(snippet only; not verified — developer.android.com and source.android.com blocked)**. | 256 MB on low-RAM devices, 1 GB mid-range, 2 GB flagship background-safe | A mobile RAG systems paper cites "approximately 2 GB per application" as the Android constraint **(snippet only; not verified)**. **(assumed)** Mapped file pages are reclaimable page cache and count toward RSS; whether `lmkd`'s pressure signal weights them like anonymous memory was not verified. |
| Desktop (laptop with a local LLM) | Process address space is not the limit; physical RAM is 8 to 32 GB. | 1 to 4 GB for the index once a 4 to 8 GB quantized LLM is loaded; up to 16 GB on a workstation | Chroma's guidance: reserve "at least a gigabyte for the system's other needs" beyond the index **(snippet only; not verified)**. Local RAG tutorials target 16 GB machines **(snippet only; not verified)**. |
| Embedded / gateway | Unified memory shared with the GPU on Jetson; 2 to 16 GB on Raspberry Pi 5 and 4 to 64 GB across the Jetson Orin range **(not verified; Wikipedia blocked)**. | 1 to 4 GB | `nanodb` runs its 275k-image index on an AGX Orin ([README](https://github.com/dusty-nv/jetson-containers/blob/master/packages/vectordb/nanodb/README.md), read 2026-09-20). |

Two facts from this table drive the decision in section 7. First, the
budgets cluster at **256 MB, 1 GB and 2 GB**, never at "as much as the machine
has". Second, iOS charges resident dirty pages but not clean file mappings,
which favours a mapped design specifically on the platform with the hardest
cap.

## 4. What VaneDB holds today

### 4.1 Per-vector cost, derived from the structs

All three indexes keep an id map, `HashMap<u64, usize>`, which is Rust's
hashbrown table: buckets are a power of two filled to at most 7/8, each
bucket holds one 16-byte `(u64, usize)` slot plus one control byte, and one
trailing 16-byte control group. Per entry that is

    H(n) = (17 × B(n) + 16) / n,   B(n) = next_power_of_two(⌈8n / 7⌉)

which oscillates between 19.4 and 38.9 bytes as `n` crosses each power of two:
22.3 B at n = 100k, 35.7 B at 1M, 28.5 B at 10M. `LIMITS.md` previously
said "`n × ~16` bytes" for the `DiskIndex` id map, up to 2.4× too low; it is
corrected alongside this study, and open question 8.3 covers reducing it.

`FlatIndex` (`flat/mod.rs`): `ids: Vec<u64>`, `data: Vec<f32>`,
`id_to_index: HashMap<u64, usize>`.

    flat(n, d) = n × (4d + 8) + 17 × B(n) + 16

Exact for one `add_batch` into an empty store, which reserves exactly.
`add` grows `ids` and `data` by `push` / `extend_from_slice`, which double
capacity, so an incrementally built `FlatIndex` can hold up to 2× the
exact-fit bytes for those two fields; `ApproxIndex`'s `ext_ids`, `levels` and
`deleted` grow the same way once past the builder's `capacity` hint. This is
the same `Vec` doubling noted for `DiskIndexBuilder` below.

`ApproxIndex` (`approx/mod.rs`, `approx/storage.rs`): vectors in
`ChunkedVectors` (chunks of `reserve_exact` size, so no doubling waste beyond
one partly filled chunk of at most 1 MiB), `ext_ids: Vec<u64>`,
`levels: Vec<i32>`, `deleted: Vec<bool>`, the id map, and
`neighbors: Vec<Vec<Vec<usize>>>`. Each node owns a `Vec` of per-layer `Vec`s
of `usize` slot numbers: 24 bytes of `Vec` header for the node, 24 bytes per
layer, 8 bytes per link slot of *capacity*. Level `l ≥ 1` is reached with
probability `M^-l` (`derive_mult` gives `mult = 1 / ln M`), so a node has
`16/15 ≈ 1.067` layers on average at M = 16; layer 0 holds at most `2M = 32`
links and upper layers at most `M = 16`.

    approx(n, d) = n × (4d + 8 + 4 + 1) + 17 × B(n) + 16 + L(n)
    L(n)         = n × (24 + 24 × 1.067) + 8 × Σ capacity(layer)
                 ≈ n × (50 + 8 × 32 + 8 × 16 × 0.067) = n × 314   (every list full, exact fit)

Lists that were extended by reverse links grow by `Vec::push`, which doubles
capacity, so the resident figure can exceed the exact-fit figure by up to 2×
on the link part; the measurement in 4.4 reports the actual number (~300 B
per node at the degree random vectors reach). The builder also reserves for
its `capacity` hint up front (default 100,000 slots, about 3.7 MB) and
allocates vectors in 1 MiB chunks; both are fixed costs, not per-vector ones.

`DiskIndex` (`disk.rs`): an `Mmap` and `id_map: HashMap<u64, usize>` built
with `with_capacity(num_vectors)`. Heap is the id map only; the file
`32 + n × (8 + 4d)` bytes is paged by the kernel as the scan touches it.

    disk_resident(n) = 17 × B(n) + 16          (plus page cache, reclaimable)

`DiskIndexBuilder` buffers `ids: Vec<u64>` and `vectors: Vec<f32>` until
`save`: `n × (4d + 8)` plus `Vec` doubling, the peak RFC 0008's streaming
builder removes.

### 4.2 Bytes per vector at the three dimensions (computed)

Id-map cost taken at its n = 1M value (35.7 B); links at the measured
~300 B per node from 4.4.

| Index | d = 384 | d = 768 | d = 1536 |
|---|---:|---:|---:|
| `FlatIndex` | 1,580 | 3,116 | 6,188 |
| `ApproxIndex` (M = 16) | ~1,885 | ~3,421 | ~6,493 |
| `DiskIndex` resident heap | 36 | 36 | 36 |
| `DiskIndex` file (paged) | 1,544 | 3,080 | 6,152 |

### 4.3 Worked capacity at d = 768 (computed)

| n | `FlatIndex` | `ApproxIndex` | `DiskIndex` resident | `DiskIndex` file |
|---:|---:|---:|---:|---:|
| 100k | 310 MB | 341 MB | 2.2 MB | 308 MB |
| 1M | 3.12 GB | 3.42 GB | 36 MB | 3.08 GB |
| 10M | 31.1 GB | 34.1 GB | 285 MB | 30.8 GB |

Against the budgets of section 3, today's resident indexes at 768-d hold about
78k vectors in 256 MiB, 314k in 1 GiB and 628k in 2 GiB. That covers every
segment's *typical* corpus and none of the upper bounds.

### 4.4 Sanity check against the allocator (measured byte counts, not timings)

A temporary crate in the session scratchpad (not committed) wrapped the
system allocator in a counting `GlobalAlloc`, built each index from
deterministic pseudo-random vectors through the public API, read the live
heap delta while the index was alive, and parsed the saved VNDB v2 file
(per the layout in `conformance/graph/README.md`) to count the graph's
layers and links exactly. `n = 8192` is a power of two so `Vec` doubling
lands exactly on the length; `n = 10,000` shows the doubling overhead.

```rust
// Counting allocator (excerpt); byte counts only, no timings.
struct Counting;
static LIVE: AtomicUsize = AtomicUsize::new(0);
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = System.alloc(l);
        if !p.is_null() { LIVE.fetch_add(l.size(), Relaxed); }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        System.dealloc(p, l);
        LIVE.fetch_sub(l.size(), Relaxed);
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        let q = System.realloc(p, l, new);
        if !q.is_null() { LIVE.fetch_add(new, Relaxed); LIVE.fetch_sub(l.size(), Relaxed); }
        q
    }
}
#[global_allocator]
static GA: Counting = Counting;

fn hashmap_bytes(n: usize) -> usize {
    let buckets = (n.div_ceil(7) * 8).next_power_of_two().max(4);
    buckets * (size_of::<(u64, usize)>() + 1) + 16
}
// flat:   FlatIndex::new(d, L2) + add_batch(ids, data)      -> live() delta
// disk:   DiskIndexBuilder -> save -> unsafe DiskIndex::open  -> live() delta
// approx: ApproxIndex::builder(d, L2).seed(7).build() + add_batch -> live() delta,
//         then save() and parse the v2 file for layers and degrees.
```

Output (Linux x86-64, Rust 1.94, `size_of::<usize>() == 8`):

```text
size_of: f32=4 u64=8 usize=8 i32=4 bool=1 Vec<usize> header=24 (u64,usize)=16
hashmap n=8192:  measured=278544 formula=278544 (34.0 B/entry)
hashmap n=10000: measured=278544 formula=278544 (27.9 B/entry)
flat  n=8192  dim=384:  measured=12926992 formula=12926992 per_vec=1578.0
flat  n=8192  dim=768:  measured=25509904 formula=25509904 per_vec=3114.0
flat  n=8192  dim=1536: measured=50675728 formula=50675728 per_vec=6186.0
flat  n=10000 dim=384:  measured=15718544 formula=15718544 per_vec=1571.9
disk  n=8192  dim=384:  heap=278544 (=hashmap) per_vec=34.0; file=12648480 (=32+n(8+4d))
disk  n=8192  dim=768:  heap=278544 (=hashmap) per_vec=34.0; file=25231392 (=32+n(8+4d))
disk  n=8192  dim=1536: heap=278544 (=hashmap) per_vec=34.0; file=50397216 (=32+n(8+4d))
disk  n=10000 dim=384:  heap=278544 (=hashmap) per_vec=27.9; file=15440032 (=32+n(8+4d))
empty approx dim=768: builder default capacity(100k) reserves 3709384 B; capacity(8192) reserves 303872 B
approx n=8192 dim=384  (capacity(n)): measured=15409168 per_vec=1881.0; non_link_formula=12967952;
        links_measured=2441216 (298.0 B/node); links_exact_fit=2047864 (250.0 B/node); links_u32_flat=953972 (116.5 B/node)
        graph: mean_deg0=23.97 mean_links_per_node=25.05 layers_per_node=1.068 nodes_by_level=[7672, 489, 29, 2]
approx n=8192 dim=768  (capacity(n)): measured=27973648 per_vec=3414.8; non_link_formula=25550864;
        links_measured=2422784 (295.8 B/node); links_exact_fit=2029576 (247.8 B/node); links_u32_flat=944828 (115.3 B/node)
        graph: mean_deg0=23.69 mean_links_per_node=24.77 layers_per_node=1.068 nodes_by_level=[7672, 489, 29, 2]
approx n=8192 dim=1536 (default capacity): measured=56529152 per_vec=6900.5; links_measured=5812464 (709.5 B/node)
        graph: mean_deg0=23.45 mean_links_per_node=24.53 layers_per_node=1.068 nodes_by_level=[7672, 489, 29, 2]
approx n=10000 dim=384 (default capacity): measured=22437152 per_vec=2243.7; links_measured=6668608 (666.9 B/node)
        graph: mean_deg0=23.77 mean_links_per_node=24.83 layers_per_node=1.066 nodes_by_level=[9377, 585, 35, 3]
```

Reading the output:

- `FlatIndex`, `DiskIndex` and the id map match the formula **to the byte**
  at every dimension and at both `n`. The hashbrown per-entry cost is 34.0 B
  at 8192 and 27.9 B at 10,000 entries, as the bucket rule predicts.
- The level distribution matches `M^-l`: 7672 / 489 / 29 / 2 nodes at levels
  0 to 3 out of 8192 is 1.068 layers per node against the expected 1.067.
- On random vectors layer 0 settles at a mean degree of **23.7** against the
  cap of 32; the link term today is **~297 B per node** in the
  `Vec<Vec<Vec<usize>>>` layout, 20% above the exact-fit 248 B because of
  `Vec` capacity rounding, and would be **~116 B per node** as flat `u32`
  arrays (RFC 0008's layout). At a full 32-degree layer 0 the exact-fit
  figures rise by about 66 B and 33 B respectively, so embedding corpora
  should be planned at **~300 to 360 B per node today and ~120 to 150 B
  mapped**, which brackets both `LIMITS.md`'s "≈ 280" and RFC 0008's
  "about 140".
- The two runs marked "default capacity" show a fixed cost, not a per-node
  one: `ApproxIndex::builder` reserves for its default `capacity` hint of
  100,000 slots (`ext_ids`, `levels`, `neighbors` headers, `deleted`), about
  **3.7 MB before the first vector**, and `ChunkedVectors` allocates in
  1 MiB chunks. Spread over 8192 nodes that looked like 700 B per node; with
  `capacity(n)` it disappears. A phone app holding a few thousand vectors
  should pass `capacity`.

The link term is the only one the formula cannot pin, because it depends on
the achieved degree and on `Vec` capacity; the rest of this page uses the
measured ~300 B per node for today's layout and ~120 B for the mapped
layout, and treats both as floors for real embeddings at the same M.

## 5. Competitor capacity

Date read for every row: 2026-09-20. "Per-vector resident" is what the
engine's own layout implies at M = 16 (or its fixed setting) and d = 768;
overheads of the host language are excluded.

| Engine | Max practical corpus on one device | Per-vector resident (d = 768) | Vectors on disk? | Quantization | Published limits | Sources and status |
|---|---|---|---|---|---|---|
| **hnswlib** 0.8 | RAM-bound; `max_elements` is a fixed, resizable capacity | `4d + 8` label + level-0 links `(2M × 4 + 4)` = 3,212 B, plus `(M × 4 + 4)` per upper layer; docs summarise links as "M × 8-10 bytes per stored element" | No: `load_index` restores into memory | None (f32 only) | `max_elements` must be set; `mark_deleted` omits from results without reclaiming | [ALGO_PARAMS.md](https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md), [hnswalg.h](https://github.com/nmslib/hnswlib/blob/master/hnswlib/hnswalg.h) (`typedef unsigned int tableint; maxM0_ = M_ * 2; size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint); size_data_per_element_ = size_links_level0_ + data_size_ + sizeof(labeltype)`), [README](https://github.com/nmslib/hnswlib/blob/master/README.md). Fetched. |
| **USearch** 2.x | "billions or even trillions" via multi-index; single index RAM- or mmap-bound | vectors at the scalar width (`bytes_per_vector` 768 for bf16 at 384-d, 384 for i8 in a ClickHouse report) plus one allocation per node: key + level + per layer `u32` count and `u32` slots; `default_connectivity() = 16`, base layer `connectivity * 2` | Yes: `Index.restore(path, view=True)` serves from a mapping; `view()` pre-computes one 8-byte pointer per vector | f64, f32, bf16, f16, e5m2/e4m3/e3m2/e2m3, u8, i8, b1x8 | `uint40_t` keys for "over 4 Billion entries" | [README](https://github.com/unum-cloud/usearch/blob/main/README.md), [index.hpp](https://github.com/unum-cloud/usearch/blob/main/include/usearch/index.hpp), [ClickHouse #78056](https://github.com/ClickHouse/ClickHouse/issues/78056). Fetched. |
| **sqlite-vec** 0.1.6 | Brute force; the author's ANN tracking issue says it "slows down on large datasets (>1M w/ large dimensions)" | Vectors live in SQLite shadow tables, paged by SQLite's cache: `4d` for float, `d` for int8, `d/8` for bit | Yes (SQLite pages) | int8 and bit vector columns | `chunk_size` and max dimension **(not verified: docs host blocked; not found in the visible part of `sqlite-vec.c`)**. The source now carries `VEC0_DISKANN_DEFAULT_N_NEIGHBORS 72` constants, so ANN is in progress **(status not verified)** | [README](https://github.com/asg017/sqlite-vec/blob/main/README.md), [issue #25](https://github.com/asg017/sqlite-vec/issues/25), [vec0.md](https://github.com/asg017/sqlite-vec/blob/main/site/features/vec0.md) (metadata limits only). Fetched. |
| **instant-distance** 0.6 | RAM-bound | Fixed arrays: `ZeroNode([PointId; M * 2])` and `UpperNode([PointId; M])` with `M = 32` and `PointId(u32)`: 256 B at layer 0, 128 B per upper layer, always fully allocated; plus the caller's `Vec` of points (`4d`) | No | None | `M` is a compile-time constant | [types.rs](https://github.com/instant-labs/instant-distance/blob/main/instant-distance/src/types.rs), [lib.rs](https://github.com/instant-labs/instant-distance/blob/main/instant-distance/src/lib.rs) (`const M: usize = 32`). Fetched. |
| **hnsw_rs** 0.3 | Benchmarked on SIFT1M and "the first 10 Million points" of BIGANN | **Not published (not verified)** | Data (not graph) can be mmapped after dump: "use mmap on dumped data (not on graph part)"; the graph can be reloaded without the data | Generic over numeric types (f32, f64, u8, …); no quantizer | none stated | [README](https://github.com/jean-pierreBoth/hnswlib-rs/blob/master/README.md). Fetched. |
| **LanceDB** | Disk-based IVF-PQ; billion-scale marketing | PQ codes only in memory; "128x reduction" claimed for PQ; `num_sub_vectors` guidance `dim / 8` | Yes by design | PQ (and SQ) | none stated | docs.lancedb.com and lancedb.github.io blocked; GitHub docs path 404 **(snippet only; not verified)**. |
| **Chroma** | Single node tested "up through about 7 million embeddings" | HNSW index must reside in RAM; capacity rule of thumb `N = R × 0.245` (millions per GB) at 1024-d with small metadata | No (HNSW resident) | None on the embedded path | reserve ≥ 1 GB for the system | docs.trychroma.com and cookbook.chromadb.dev blocked **(snippet only; not verified)**. |
| **ObjectBox** 4.x | "millions of entries" on device | Claims vectors need not all be in memory ("if a vector is not in-memory, ObjectBox fetches it from disk"); binary ~3 MB | Yes (its own store) | none stated (f32 property index) | none stated | docs.objectbox.io and objectbox.io blocked; the Java README says only "handling millions of objects" ([README](https://github.com/objectbox/objectbox-java/blob/main/README.md), fetched). Vector claims **(snippet only; not verified)**. |
| **libSQL / Turso** | DiskANN on SQLite pages | one node per row; "80,704 bytes per node" at f32 for a 20k-d example falls to "5,392" with `compress_neighbors=float1bit` | Yes (SQLite pages) | F1BIT, F8, FB16, F16, F32, F64 storage types; 1-bit neighbour compression | ≤ 65,536 dimensions; Euclidean not supported for 1-bit | docs.turso.tech and turso.tech blocked **(snippet only; not verified)**. |
| **EdgeVec** (browser) | "~1GB practical limit" in the browser | "3.03 GB -> 832 MB at 1M vectors" with SQ8; "~300 MB (F32) to approximately 10 MB with BQ" at 100k × 768 | IndexedDB or filesystem save/load | SQ8, binary with rescoring ("~95% recall") | 217 KB gzipped bundle | [README](https://github.com/matte1782/edgevec). Fetched. |

What the table says: the two engines a buyer will hold VaneDB against on
memory are USearch (every scalar width, plus a mapped view) and EdgeVec
(binary with rescoring, in the browser). Both reach small footprints by
**quantizing first**; mapping is USearch's second lever and is absent from
EdgeVec entirely because a browser has no page cache to map. Nobody in the
in-process group ships a mapped *graph* with the vectors on disk except
USearch's view; the disk-graph engines (LanceDB, libSQL) are different index
structures on a database's page cache.

## 6. Mapped graph versus quantized storage

Per-vector resident bytes at d = 768, M = 16, id map at 36 B, links at the
section 4.4 figures: ~300 B in today's `Vec` layout and ~120 B in RFC 0008's
flat `u32` layout (the RFC's "about 140" is the full-degree case).

| Design | RFC | Resident B/vector | 100k | 1M | 10M | Build time | Recall | Complexity |
|---|---|---:|---:|---:|---:|---|---|---|
| Resident f32 (today) | — | ~3,420 | 342 MB | 3.4 GB | 34 GB | baseline | baseline | shipped |
| int8 resident, f32 dropped | 0005 | ~1,120 (`768 + 4` norm for cosine + 13 + 36 + 300 links) | 112 MB | 1.12 GB | 11.2 GB | unchanged plus quantization pass | "small, documented per metric" | new kernels (i8 dot on NEON `sdot`, AVX2 `vpmaddubsw`), `quant_params` section |
| Binary resident, no rescoring | 0005 | ~450 | 45 MB | 450 MB | 4.5 GB | unchanged | large loss; RFC says "use with rescoring" | popcount kernels |
| Binary resident + **resident** f32 rescoring | 0005 alone | ~3,520 | 352 MB | 3.5 GB | 35 GB | unchanged | ≈ f32 at rescoring factor 4 to 16 | no memory saving: the f32 copy is resident |
| Mapped f32 + resident `u32` links | 0008 | ~169 (+ page cache) | 17 MB | 169 MB | 1.7 GB | unchanged; needs a v3 file (0013) | identical to resident (same graph) | `open_mapped`, `ReadOnly` error in every binding, cold-cache spike, mmap safety |
| Binary resident navigation + **mapped** f32 rescoring | 0005 + 0008 | ~265 (+ page cache) | 27 MB | 265 MB | 2.7 GB | unchanged | ≈ f32 with rescoring | both of the above |
| int8 resident navigation + mapped f32 rescoring | 0005 + 0008 | ~940 (+ page cache) | 94 MB | 940 MB | 9.4 GB | unchanged | ≈ f32 | both |
| Streaming `DiskIndexBuilder` (exact scan) | 0008, first half | build peak `n × 8` ids instead of `n × (4d + 8)` | 0.8 MB | 8 MB | 80 MB | unchanged | exact | small: two-file write and rename |
| IVF over mapped `DiskIndex` (fallback) | 0008 fallback | centroids only | few MB | few MB | tens of MB | k-means pass | lower at equal latency | a `clusters` section, sequential reads |

Corpus a budget holds. Per-vector resident bytes scale differently with the
dimension: resident f32 is `4d + 349`, int8 is `d + 353`, mapped f32 is
**169 at every d** (links, level, tombstone and id map only; the vectors are
paged), and binary navigation with mapped rescoring is `d/8 + 169`. Budgets
are binary (256 MiB, 1 GiB, 2 GiB, 4 GiB).

| Budget | d | Resident f32 | int8 (0005) | Mapped f32 (0008) | Binary + mapped rescoring (0005 + 0008) |
|---|---:|---:|---:|---:|---:|
| 256 MiB | 384 | 142k | 364k | 1.6M | 1.2M |
| 256 MiB | 768 | 78k | 239k | 1.6M | 1.0M |
| 256 MiB | 1536 | 41k | 142k | 1.6M | 744k |
| 1 GiB | 384 | 570k | 1.46M | 6.4M | 4.9M |
| 1 GiB | 768 | 314k | 958k | 6.4M | 4.1M |
| 1 GiB | 1536 | 165k | 568k | 6.4M | 2.97M |
| 2 GiB | 384 | 1.14M | 2.91M | 12.7M | 9.9M |
| 2 GiB | 768 | 628k | 1.92M | 12.7M | 8.1M |
| 2 GiB | 1536 | 331k | 1.14M | 12.7M | 5.95M |
| 4 GiB | 384 | 2.28M | 5.83M | 25.4M | 19.8M |
| 4 GiB | 768 | 1.26M | 3.83M | 25.4M | 16.2M |
| 4 GiB | 1536 | 661k | 2.27M | 25.4M | 11.9M |

Three observations.

1. **Quantization is the cheaper 4×.** int8 alone moves every segment's
   *typical* corpus (≤ 100k) under 256 MB at 768-d and every *upper bound*
   below ~1M under 1 GB, with no new failure mode: no page faults, no
   `unsafe` open, no read-only mode, no v3 dependency beyond the encoding
   sections. It also serves the browser, where RFC 0008 cannot apply at all
   (no file mapping in wasm) and where the tightest budget lives.
2. **Mapping is the bigger 15×, and the ordering is about page faults, not
   bytes.** On paper mapped f32 alone (RFC 0008) holds the most at 768-d:
   1.6M in 256 MiB and 12.7M in 2 GiB, against 1.0M and 8.1M for binary
   navigation with mapped rescoring. But a mapped-f32 walk reads one f32 row
   per visited node, roughly `ef_search × (levels + 1)` random page touches
   per query, so a cold cache costs a page fault per visited node; that is
   the risk RFC 0008's spike measures against the 20 ms p99 gate. Binary
   navigation keeps the walk in resident memory and touches the mapping only
   for the `rescore × k` final candidates, so its cold-cache cost is bounded
   by `k` rather than by `ef`. Binary without rescoring loses too much recall
   to ship as a default, and binary with a *resident* f32 copy saves nothing;
   the design worth shipping is therefore RFC 0008's "navigate on quantized,
   rescore from the mapping", which needs RFC 0005 first.
3. **The `DiskIndexBuilder` half of RFC 0008 is independent of all of this.**
   It fixes a contradiction in the shipped product (the "larger than RAM"
   index needs RAM to build), touches no graph, and has no spike gate.

## 7. Decision

**RFC 0005 first, then RFC 0008; and RFC 0008 should be split.**

- Every segment's typical corpus (1k to 100k vectors) fits the resident
  index today except under the 256 MB browser and low-RAM mobile budgets at
  768-d, and those are exactly the budgets int8 fixes. Quantized storage
  therefore unblocks users at 0.3.0 without waiting for a device spike, and
  it is what USearch, EdgeVec and Turso are compared on. The existing
  roadmap order (0013 and 0005 share order 5 at 0.3.0, 0013 landing first
  per its own milestone note; 0008 is order 8 at 0.4.0) is confirmed by the
  numbers rather than merely by preference.
- The mapped graph earns its place only at the segments' *upper* bounds:
  desktop RAG and embedded capture between 1M and 10M vectors, and mobile
  archives between 500k and 1M on a 1 GB budget. Those exist but are not the
  first user. Of the mapped designs in section 6, mapped f32 alone holds
  12.7M at 768-d in 2 GiB and binary navigation with mapped rescoring holds
  8.1M; it is worth doing after 0005 because only the latter bounds the
  cold-cache page faults by the rescoring set rather than by the beam
  (section 6, observation 2), and because iOS does not appear to charge clean
  file mappings to the jetsam footprint (section 3, secondary source), which
  would make the mapped copy nearly free on the platform with the hardest
  cap.
- **Thresholds, at d = 768, binary budgets.** Resident f32 wins below ~314k
  vectors on 1 GiB (nothing to change). int8 wins from there to ~958k on
  1 GiB, or ~1.9M on 2 GiB. Above ~1M on 1 GiB, or ~2M on 2 GiB, only the
  mapped designs hold the corpus, and the quantized-navigation variant holds
  it with exact rescoring. The resident f32 thresholds scale with `1/d`
  (halve at 1536-d, double at 384-d) and int8 nearly so; the mapped f32
  design is dimension-independent at 169 B per vector, and binary with
  mapped rescoring moves only with `d/8`; the per-dimension table in
  section 6 gives each row.
- **Split RFC 0008.** Land the streaming `DiskIndexBuilder` on its own (it
  has no gate, and RFC 0010's write-path work is adjacent), and keep the
  mapped graph gated on (a) RFC 0005 shipped, so the spike can measure
  binary navigation rather than only f32, and (b) the cold-cache spike on one
  Android device and one NVMe laptop at 1M × 768-d against the 20 ms p99
  gate, which this study confirms as the right bar (section 2). The IVF
  fallback remains the fallback; nothing in the corpus data asks for it
  ahead of the spike.
- **No segment justifies DiskANN.** The largest upper bound is 10M on a
  workstation or gateway; the RFC's revisit condition ("past 10M vectors on
  one device") is not met.

## 8. Open questions

1. **The measurements in issue question 3 are not done.** Resident memory,
   p50/p99 warm and cold, and recall@10 for resident f32, mapped f32 and
   binary-plus-rescoring at 100k, 1M and 10M on one Android device and one
   NVMe laptop need the RFC 0003 fixture (`embeddings.vnef`) and dedicated
   hardware; neither exists in a cloud sandbox. They belong to RFC 0008's
   spike and to RFC 0005's recall table, and this page should be amended with
   the table when they land.
2. **Real-embedding link density.** The graph's link memory (section 4.4) was
   measured on pseudo-random vectors; embedding corpora at the same M usually
   fill layer 0 closer to the 2M cap. Re-run the allocator check on the
   fixture once it is hosted.
3. **The id map costs 19 to 39 bytes per vector in every index, including
   `DiskIndex`** (now stated in `LIMITS.md`). At 10M vectors that is
   190 to 390 MB of resident memory on an index whose vectors are otherwise
   paged. A sorted `u64` array with binary search (8 B per vector,
   `O(log n)` lookups) or a `u32` slot table would cut it 2 to 5×; this is
   adjacent to RFC 0010's hasher change and should be decided there.
4. **Browser budgets.** EdgeVec's "~1 GB practical" figure and the Safari
   per-device kills come from sources that could not be opened; the wasm
   persistence work (RFC 0006) should record what `@vanedb/wasm` can actually
   allocate on a mid-range Android Chrome and on iOS Safari before a browser
   capacity claim is made.
5. **Android per-app figures** rest on a snippet. Record `ActivityManager`
   memory classes and an `lmkd` kill threshold on the emulator that CI already
   runs, so the mobile SDK RFC (0007) can quote a measured budget.
6. **RFC 0008 amendment.** RFC 0008's capacity-study section already names
   this page. Amending the RFC with these findings and its status (streaming
   builder to `accepted`; mapped graph gated on RFC 0005 and the spike) is a
   separate PR, per the one-document rule.

## 9. Sources

Status on 2026-09-20: **fetched** means the page was read in full;
**snippet** means only a search-result excerpt was available because the host
is blocked from this sandbox; **blocked** means nothing was read.

VaneDB: [`vanedb/src/flat/mod.rs`](../../vanedb/src/flat/mod.rs),
[`vanedb/src/approx/mod.rs`](../../vanedb/src/approx/mod.rs),
[`vanedb/src/approx/storage.rs`](../../vanedb/src/approx/storage.rs),
[`vanedb/src/disk.rs`](../../vanedb/src/disk.rs),
[`conformance/graph/README.md`](../../conformance/graph/README.md),
[`docs/LIMITS.md`](../LIMITS.md), [`bench/COMPARISON.md`](../../bench/COMPARISON.md),
[`bench/README.md`](../../bench/README.md).

Competitors, fetched:
[hnswlib ALGO_PARAMS.md](https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md) ·
[hnswlib hnswalg.h](https://github.com/nmslib/hnswlib/blob/master/hnswlib/hnswalg.h) ·
[hnswlib README](https://github.com/nmslib/hnswlib/blob/master/README.md) ·
[USearch README](https://github.com/unum-cloud/usearch/blob/main/README.md) ·
[USearch index.hpp](https://github.com/unum-cloud/usearch/blob/main/include/usearch/index.hpp) ·
[ClickHouse #78056, usearch memory accounting](https://github.com/ClickHouse/ClickHouse/issues/78056) ·
[sqlite-vec README](https://github.com/asg017/sqlite-vec/blob/main/README.md) ·
[sqlite-vec #25, ANN tracking](https://github.com/asg017/sqlite-vec/issues/25) ·
[sqlite-vec vec0.md](https://github.com/asg017/sqlite-vec/blob/main/site/features/vec0.md) ·
[instant-distance types.rs](https://github.com/instant-labs/instant-distance/blob/main/instant-distance/src/types.rs) ·
[instant-distance lib.rs](https://github.com/instant-labs/instant-distance/blob/main/instant-distance/src/lib.rs) ·
[hnsw_rs README](https://github.com/jean-pierreBoth/hnswlib-rs/blob/master/README.md) ·
[EdgeVec README](https://github.com/matte1782/edgevec) ·
[objectbox-java README](https://github.com/objectbox/objectbox-java/blob/main/README.md)

Competitors, snippet only (host blocked):
[Chroma single-node performance](https://docs.trychroma.com/deployment/performance) ·
[Chroma cookbook resources](https://cookbook.chromadb.dev/core/resources/) ·
[LanceDB IVF-PQ concepts](https://docs.lancedb.com/indexing/vector-index) ·
[Turso AI & embeddings](https://docs.turso.tech/features/ai-and-embeddings) ·
[Turso, space complexity of vector indexes](https://turso.tech/blog/the-space-complexity-of-vector-indexes-in-libsql) ·
[ObjectBox on-device vector search](https://docs.objectbox.io/on-device-vector-search) ·
[sqlite-vec vec0 reference](https://alexgarcia.xyz/sqlite-vec/features/vec0.html)

Platforms, fetched:
[home-assistant/iOS #4475, jetsam per-process-limit on iPhone 14](https://github.com/home-assistant/iOS/issues/4475) ·
[boardsesh #5524, increased-memory-limit entitlement](https://github.com/boardsesh/boardsesh/pull/5524) ·
[WWDC 2018 session 416 notes, footprint definition](https://gist.github.com/SheldonWangRJT/5d2ea69f78a905c76e0c36dfc994e85c)

Platforms, snippet only or blocked:
[os_proc_available_memory](https://developer.apple.com/documentation/os/os_proc_available_memory) (page renders no content) ·
[Apple forum: jetsam per-process-limit on 4 GB iPhones](https://developer.apple.com/forums/thread/688973) ·
[Android ActivityManager](https://developer.android.com/reference/android/app/ActivityManager) ·
[Android low memory killer daemon](https://source.android.com/docs/core/perf/lmkd) ·
[Android low-RAM configuration](https://source.android.com/docs/core/perf/low-ram) ·
[V8: up to 4 GB of memory in WebAssembly](https://v8.dev/blog/4gb-wasm-memory) ·
[MDN WebAssembly.Memory](https://developer.mozilla.org/en-US/docs/WebAssembly/JavaScript_interface/Memory) ·
[Chrome Status: WebAssembly Memory64](https://chromestatus.com/feature/5070065734516736) ·
[Chromium issue 40691287, 4 GB per tab](https://issues.chromium.org/issues/40691287) ·
[Browser memory limits (textslashplain)](https://textslashplain.com/2020/09/15/browser-memory-limits/) ·
[WebKit RAM internals (catchmetrics)](https://www.catchmetrics.io/blog/deep-dive-ram-internals-webkit)

Segments, fetched:
[findmark README](https://github.com/daveshenal/findmark) ·
[Smart Connections embedding adapter](https://github.com/brianpetro/jsbrains/blob/main/smart-embed-model/adapters/transformers.js) ·
[obsidian-vane-search README](https://github.com/vanedb/obsidian-vane-search) ·
[jetson-containers nanodb README](https://github.com/dusty-nv/jetson-containers/blob/master/packages/vectordb/nanodb/README.md)

Segments, snippet only or blocked:
[Obsidian forum: how many notes do you have?](https://forum.obsidian.md/t/how-many-notes-do-you-have/36987) ·
[Obsidian forum: maximum number of notes in vault](https://forum.obsidian.md/t/maximum-number-of-notes-in-vault/1509) ·
[Obsidian forum: performance on large vaults](https://forum.obsidian.md/t/performance-on-large-vaults/114864) ·
[Chrome history embeddings analysis](https://dejan.ai/blog/inside-chromes-semantic-engine-a-technical-analysis-of-history-embeddings/) ·
[Pocket RAG, on-device RAG for first aid (arXiv 2602.13229)](https://arxiv.org/abs/2602.13229) ·
[On-device RAG on a mobile NPU (arXiv 2606.11257)](https://arxiv.org/abs/2606.11257) ·
[Google AI Edge RAG guide for Android](https://ai.google.dev/edge/mediapipe/solutions/genai/rag/android) ·
[EmbeddingGemma model card](https://ai.google.dev/gemma/docs/embeddinggemma) ·
[On-device RAG for Android (dev.to)](https://dev.to/software_mvp-factory/on-device-rag-for-android-4a7g) ·
[Local RAG with Ollama and Chroma (sitepoint)](https://www.sitepoint.com/local-rag-private-documents/) ·
[Raspberry Pi 5](https://en.wikipedia.org/wiki/Raspberry_Pi_5) ·
[Nvidia Jetson](https://en.wikipedia.org/wiki/Nvidia_Jetson)
