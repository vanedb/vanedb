# VaneDB market analysis

Dated 2026-09-13, three days after the 0.1.1 release. Covers audience, market
size, competitors, feedback and roadmap guidance. Inputs: the repository
and its 44 issues and 100+ pull requests, crates.io/PyPI/npm registry metadata,
and public material on competing products and on the vector-database and edge-AI
markets. Every number that came from this repository or a registry is marked as
measured; analyst figures are quoted with their source and should be read as
ranges, not facts.

## 1. Position in one paragraph

VaneDB is a correct, hardened, bring-your-own-embeddings nearest-neighbour
library with an unusually strong verification story (cross-engine conformance,
mutation testing, fuzzed loaders, a signed release record) and no users yet.
It ships `(u64 id, f32 vector)` storage with exact, HNSW and memory-mapped
indexes for Rust, Python, C and WebAssembly, and filtered search (0.2.0,
unreleased) over application-owned ID lists and predicates, with browser
persistence of the wasm graph index over IndexedDB (0.2.0, unreleased). It
does not ship metadata storage, quantization, mobile SDKs, or a comparison
against any competitor. Every one of those is something the
on-device market already expects from the incumbents. The engine is not the
gap; the product surface around it is.

## 2. What the registries say (measured)

| Signal | Value | Source |
|---|---|---|
| GitHub stars / forks / watchers | 0 / 0 / 0 | GitHub API, 2026-09-13 |
| Repository created | 2026-03-29 | GitHub API |
| Issues | 44, all opened by the maintainer | GitHub |
| External issues, PRs or discussions | none | GitHub |
| crates.io downloads (all versions) | 40 | crates.io API |
| PyPI releases | 0.1.0rc1, 0.1.0rc2, 0.1.1 (29 files) | PyPI JSON |
| npm `@vanedb/wasm` | 0.1.0, rc.1, rc.2, 0.1.1; 364 KB unpacked | npm registry |
| Python wheel size | 370–620 KB | PyPI JSON |
| Rust crate source | 133 KB packed | crates.io |
| Public mentions (HN, Reddit, X, blogs) | none found | web search |

Comparable Rust crates, 90-day downloads (crates.io API, measured):

| Crate | Recent downloads | Notes |
|---|---:|---|
| `instant-distance` | 746k | pure-Rust HNSW, minimal |
| `lancedb` | 664k | embedded columnar vector DB |
| `usearch` | 404k | C++ core, ten language bindings |
| `hnsw_rs` | 380k | pure-Rust HNSW |
| `arroy` | 83k | Meilisearch's on-disk ANN |
| `hora` | 5k | dormant |
| `vanedb` | 40 | 3 days old |

Interpretation: a pure in-process Rust ANN crate with no company behind it can
reach ~750k downloads per quarter. That is the ceiling for the crate channel.
There is no external feedback to review yet; section 6 substitutes the
maintainer's own product-review issues and the complaint history of the
competitors' users.

## 3. Target audience

Ranked by fit with what 0.1.1 actually does, not by market size.

1. **Rust application developers who need ANN in-process** (desktop tools,
   CLIs, agents, game and creative tools). Fit today: high. VaneDB's
   distinguishing assets for them are the hardened loaders, the specified
   VNDB formats, delete/upsert/compact on the graph, and a `disk` feature for
   corpora larger than RAM. Competitors: `instant-distance`, `hnsw_rs`,
   `usearch`, `lancedb`.
2. **Python developers building local RAG on a laptop or a single box**
   (Ollama-style setups, note-taking plugins, personal knowledge tools). Fit
   today: medium. The Python binding is complete and typed, but the audience
   defaults to `chromadb`, `lancedb`, `faiss`, `hnswlib` or `sqlite-vec`, all
   of which store payloads and most of which filter. The getting-started guide
   targets exactly this persona.
3. **Browser and Electron developers** doing client-side semantic search.
   Fit today: medium-low. The wasm package works, is small, and has full
   graph CRUD, but exposes no persistence. EdgeVec, voy, `client-vector-search`
   and `usearch-wasm` all persist to IndexedDB; EdgeVec also quantizes and
   filters.
4. **Mobile developers (Swift, Kotlin, Flutter, React Native)** building
   offline RAG. Fit today: low. Only the C ABI exists, verified on simulators
   and emulators. The market's reference checklist for an on-device vector DB
   (persist locally, vector + metadata, mobile SDKs, incremental CRUD, bounded
   RAM, offline, sync) is met on two of seven points. ObjectBox, Couchbase
   Lite, libSQL/Turso, `sqlite-vec` via op-sqlite, and Google's AI Edge RAG
   SDK (`SqliteVectorStore`) are the defaults here.
5. **Embedded and industrial edge (Jetson, Raspberry Pi, gateways, robots).**
   Fit today: low-medium through the C ABI; no ARM Linux C archive story beyond
   glibc, no CUDA. Actian VectorAI DB and ObjectBox market directly to this
   segment.

The stated positioning, "edge AI", points at audiences 4 and 5. The shipped
surface serves audiences 1 and 2. Either the positioning or the roadmap has
to move.

## 4. Market size

Analyst figures for the vector-database category (servers and managed
services, which is where the revenue is):

| Estimate | 2026 | Later | CAGR | Source |
|---|---|---|---|---|
| Vector DB market | $3.1–3.7B | $8.7B (2030), $18.4B (2034) | 23–24% | Research and Markets, GM Insights, Fortune Business Insights |
| Edge AI market (hardware + software) | $26–48B | $103B (2030), $245B (2040) | 17–29% | Roots Analysis, GM Insights, Grand View, Fortune |

Neither is VaneDB's addressable market. Nearly all vector-DB revenue is
managed cloud (Pinecone, Zilliz, Weaviate, Qdrant, Chroma Cloud, LanceDB Cloud)
where the customer pays for hosting, not for the engine. Edge AI revenue is
silicon, inference runtimes and integrators.

Bottom-up view of the segment VaneDB can actually sell into, embedded and
on-device data infrastructure:

- SQLite, the most deployed database in existence, earns under $5M a year
  (ZoomInfo estimate) from consortium memberships (~$75k/year), the Encryption
  Extension ($2,000 perpetual) and support.
- Realm, the embedded mobile database with 100k developers and 2B downloads,
  sold to MongoDB for $39M in 2019; MongoDB shut its successor sync product
  down on 2025-09-30 after failing to make it pay.
- ObjectBox, the closest "edge vector database" comparable, funds itself from
  paid Sync with no public price list.

Working estimate: worldwide spend on embedded and on-device vector search
software specifically is in the low tens of millions of dollars a year through
2028 and is fragmented across sync, encryption, support and enterprise
licences. A realistic serviceable market for a single-engine project is a few
million dollars a year at maturity, and only via services attached to the
engine rather than the engine itself. The category's strategic value
(ecosystem position, acquisition interest, consulting flow) exceeds its direct
licensing value.

## 5. Competitors

Grouped by what the buyer is actually comparing. Stars are public GitHub
counts quoted from search results and are approximate.

### In-process ANN libraries (VaneDB's direct peers)

| Product | Language surface | Delete | Filter | Quantization | Persistence | Notes |
|---|---|---|---|---|---|---|
| **USearch** (~3.7k stars) | C++, C, Python, JS, Rust, Java, ObjC, Swift, C#, Go, Wolfram | yes | predicate callback | f16, i8, binary | file, mmap view | Broadest binding matrix; Swift/Java packages make it the de facto mobile ANN library |
| **hnswlib** (~5.1k) | C++, Python | mark-deleted | callback | no | file | Reference HNSW; no mobile packaging |
| **FAISS** | C++, Python | limited | id selectors | PQ, SQ, binary, GPU | file | Too large for mobile; the server standard |
| `instant-distance`, `hnsw_rs`, `hnswlib-rs` | Rust | partial | no | no | file | Rust-only, minimal |
| **VaneDB** | Rust, Python, C, wasm | yes (tombstone + compact) | yes (0.2.0, unreleased): predicate, allow/deny ID lists | no | VNDB v1/v2 file; wasm graph save/load over IndexedDB (0.2.0, unreleased) | Strongest loader hardening and format spec in the group |

### Vectors inside SQLite (the "you already ship SQLite" argument)

| Product | Index | Filter | Mobile | Notes |
|---|---|---|---|---|
| **sqlite-vec** (~8.1k) | brute force only; ANN is tracking issue #25 | SQL `WHERE`, partition keys | Android, iOS, op-sqlite, Flutter | Largest mindshare; users complain about scale past ~1M and lack of pagination/thresholds (#165) |
| **vectorlite** | HNSW (hnswlib) | SQL | build yourself | 3–30x faster than sqlite-vec on ANN |
| **sqlite-vector** (sqlite.ai) | quantized brute force | SQL | yes | Commercial vendor |
| **libSQL / Turso** | DiskANN, native column type, f32/f16/bf16/1-bit | SQL | iOS, Android, React Native | "You don't need a separate vector database" is their pitch |
| DuckDB VSS | HNSW | SQL | desktop | analytics-oriented |

### Embedded vector databases (payload-aware, Python-first)

| Product | Model | Filter | Notes |
|---|---|---|---|
| **LanceDB** | Lance columnar files, IVF-PQ, FTS, SQL | yes | Open source + Cloud/Enterprise; 664k crate downloads/quarter |
| **Chroma** | embedded or server, same API | yes | Chroma Cloud metered ($0.33/GiB-month); the default for Python RAG tutorials |
| **chromem-go**, **vectorgo** | Go, zero-dependency | yes | Go ecosystem equivalents |

### Mobile-first and edge products

| Product | Surface | Sync | Business model |
|---|---|---|---|
| **ObjectBox** (~4.6k Java) | Java/Kotlin, Swift, Dart/Flutter, C/C++, Python; HNSW; metadata | paid | Core free, Sync paid, no public price |
| **Couchbase Lite** | Swift, Kotlin, C, .NET; vector search in-device | paid, incl. peer-to-peer | Enterprise licence |
| **Google AI Edge RAG SDK** | Android (Kotlin), Flutter plugin; `SqliteVectorStore`; EmbeddingGemma | no | free; platform play, iOS "planned" |
| **Actian VectorAI DB** (Apr 2026) | Jetson, Raspberry Pi, regulated/disconnected | — | enterprise |
| **VecturaKit**, SVDB, SimilaritySearchKit | Swift, MLX/MLTensor | no | hobby-scale |

### Browser and WebAssembly

| Product | Index | Persistence | Bundle | Notes |
|---|---|---|---|---|
| **EdgeVec** (Show HN Dec 2025) | HNSW, SQ8, binary | IndexedDB, filesystem | 148 KB gz | metadata filtering, soft delete, P99 tracking; the newest and most complete |
| **voy** | k-d tree | serialize | small | rebuild on update |
| **client-vector-search** | brute force | IndexedDB | small | embeds via transformers.js |
| **usearch-wasm** | HNSW | manual | larger | |
| **@vanedb/wasm** | HNSW + flat | none | 364 KB unpacked | full graph CRUD, no save/load |

### What VaneDB has that none of the above lead with

- A written, fixture-backed file format read by two independent engines, with
  corruption tests, overflow checks and a documented undetectable-corruption
  boundary.
- A sixteen-code C error channel, thread-local, panic-boundaried.
- Release evidence that names the runs, hashes and simulator results.

That is a trust asset for regulated, safety-conscious and long-lived
deployments (medical devices, automotive, industrial). It is invisible to the
tutorial-following Python developer and is not on any comparison chart.

## 6. Feedback review

### 6.1 External feedback

None exists. Zero stars, zero third-party issues, zero mentions. The release is
three days old and has had no launch (no Show HN, no r/rust post, no blog).
Any roadmap decision made now is made without a user signal; the first
roadmap item is therefore to generate one (section 7, item 0).

### 6.2 The maintainer's own product review (issues #89–#111, #153, #182)

The issue history records a product review and a senior-engineer review that
were run before release. What they found, and what remains:

| Finding | Status | Residual |
|---|---|---|
| No delete or update on the graph "rules out every stated use case" (#91) | fixed: tombstones, upsert, compact | space is not reclaimed without a full-rebuild compact under the write lock |
| Metadata filtering is "the other big gap" (#91) | fixed: RFC 0004 (#199) | filtered search via predicates, allow/deny ID lists, automatic widening |
| Empty index reserved ~295 MB and could not grow (#90) | fixed | |
| No type stubs (#105), no ARM/musl wheels "the edge is ARM" (#104) | fixed | |
| Stringly-typed errors, no `Debug` (#93, #94) | fixed | |
| Loader aborts on hostile file (#89), unchecked products (#97, #130) | fixed | |
| Rust trails C++ on `store_add` 1.58x and `disk_build` 2.18x (#77, #109) | open | SipHash on the id map; writer buffering |
| Binding parity (#85, #86) | mostly closed | `get` raises vs returns `None` deferred to 0.2 |
| macOS Intel runner exit (#47) | open | |

Personas named in #91: an Obsidian/Logseq plugin (the downstream
`obsidian-vane-search` app), a RAG corpus, and a mobile app. All three change
their corpus constantly and all three need to attach text to an id. The second
need is still unmet.

### 6.3 What the README already concedes

The README's own disclaimers are the shortest list of what a buyer will hold
against the product:

- no metadata or payload storage (filtered search supported via ID sets/predicates in RFC 0004);
- WebAssembly has no persistence;
- `DiskIndex` build buffers every vector in memory, and `open` is `unsafe`
  because the mapping cannot defend itself against a concurrent writer;
- deleted graph entries keep their storage until a full rebuild;
- mobile is simulator-verified only;
- no CUDA; Metal exposes distance scans but does not accelerate the indexes.

### 6.4 What competitors' users ask for (proxy feedback)

The complaints raised against the incumbents are the wants VaneDB would inherit
the moment it has users:

- **Filtering combined with vector search.** usearch #348, ObjectBox
  objectbox-dart #658, sqlite-vec #165 (distance thresholds, pagination). This
  is the single most common request across every embedded vector project.
- **ANN when brute force stops scaling.** sqlite-vec #25 is a multi-year
  tracking issue; vectorlite exists because of it. VaneDB has this.
- **Memory.** Browser and mobile users ask for int8 and binary quantization
  first; EdgeVec's headline is "32x memory reduction", Turso's is 1-bit
  vectors, USearch ships f16/i8/b1. VaneDB stores f32 only. Per-segment
  budgets and corpus sizes: [capacity study](research/capacity.md).
- **Persistence that survives a page reload or an app restart** (browser:
  IndexedDB; mobile: a file the platform can back up and encrypt).
- **A real SDK, not a C header.** The 2026 on-device checklist names
  Java/Kotlin, Swift and Flutter explicitly. USearch's breadth of bindings is
  why it wins mobile evaluations over hnswlib.
- **Lifecycle, not algorithms.** Mobile practitioners report that "the ANN
  math is the easy part"; the hard parts are thermal throttling, encrypted
  storage, background limits, and keeping derived vectors in sync with source
  documents.
- **Hybrid search** (BM25 + vectors) in the LanceDB and Chroma communities.
  Lower priority for an in-process library, but it is where payload storage
  leads.

## 7. Roadmap guidance

Ordered. Each item names the audience it unlocks and the evidence for it.
Each now has an RFC in [`docs/rfcs/`](rfcs/README.md) and a tracking issue;
[`ROADMAP.md`](ROADMAP.md) is the index.

0. **Generate a signal before committing engineering.** Publish a competitor
   benchmark (usearch, hnswlib, sqlite-vec, instant-distance, EdgeVec) on the
   existing `bench/` methodology: interleaved runs, recall at fixed ef, real
   embedding vectors (768-d nomic or EmbeddingGemma, not uniform random), on an
   M-series laptop and on one Android device. Post it with the
   `obsidian-vane-search` demo. The current benchmark compares VaneDB only to
   its own frozen C++ engine, which no buyer cares about. This is the cheapest
   item on the list and the only one that produces feedback.

1. **Filtered search.** `SearchParams<'a>` already carries the lifetime
   reserved for a borrowed filter; ship it. Start with a predicate
   `Fn(u64) -> bool` and an id allow/deny list, applied inside the graph walk
   (not post-hoc), with the beam widened automatically so k results survive.
   Every persona in #91 needs it, every competitor community asks for it, and
   the cross-engine invariants are untouched because the graph does not
   change. Expose it in all four bindings in one release.

2. **Quantized storage: int8 scalar, then binary with f32 rescoring.** This
   turns "edge AI" from a tagline into a spec: 100k × 768-d vectors is 307 MB
   in f32, 77 MB in int8, 9.6 MB binary. New VNDB kind identifiers, old
   readers retained, per the persistence invariant. SIMD kernels gain i8 and
   popcount paths on NEON and AVX2 with the scalar reference first.

3. **WebAssembly persistence.** `save` to bytes and `load` from bytes are all
   the wasm binding needs; IndexedDB wrapping is a fifty-line JavaScript
   helper. Without it the wasm package loses every browser evaluation to
   EdgeVec on day one. Also publish gzipped bundle size on the README.

4. **Mobile SDKs, not just a C ABI.** A Swift package (xcframework via
   `cargo-xcframework` or `uniffi`) and a Kotlin AAR, each with a
   ten-line README, each tested on the simulators the CI already runs. Add
   a Flutter plugin only when the Dart audience asks. Then, and only then, the
   physical-device follow-up in the roadmap has a reason to exist.

5. **Streaming `DiskIndex` build and a mapped graph.** Building a disk index
   must not require the whole corpus in RAM; that contradicts the index's
   stated purpose. A memory-mapped HNSW (vectors on disk, links in RAM, or
   both mapped) is the feature that makes "corpus larger than RAM" true for
   approximate search too. Turso's DiskANN and arroy show the demand. The
   [capacity study](research/capacity.md) (2026-09-20) sizes
   the corpora and memory budgets this item must serve and orders it after
   item 2.

6. **Optional payload column.** A bytes blob per id, stored beside the vector
   in VNDB, returned with results. Filtering on it is a later step; storing
   it removes the "keep your own id-to-document mapping" sentence from the
   README, which is the sentence that loses the Python audience to Chroma.

7. **Close the write-path gaps (#77, #109).** Identity hashing on the internal
   maps and buffered disk writes. Cheap, already diagnosed, and the C++
   comparison is the only benchmark currently published.

Deprioritise or reframe:

- **CUDA.** The roadmap calls it "required, high priority". It contradicts
  the positioning: edge AI runs on NPUs, mobile GPUs and CPU SIMD, not on
  discrete NVIDIA cards, and no edge competitor in section 5 offers CUDA. The
  server audience that wants CUDA uses FAISS or cuVS. Keep the roadmap's
  evidence bar, but move CUDA behind items 1–6 or scope it to Jetson, where
  it is an edge story. Do not let it consume the next release.
- **The frozen C++ engine.** It costs CI minutes and parity issues (#85, #86,
  #100) and produces a benchmark row nobody outside the project reads. Keep
  it as the conformance oracle; stop reporting it as the headline comparison.
- **Metal.** A distance-scan kernel that does not accelerate an index is not
  a feature buyers can use. Either finish it (GPU brute-force `FlatIndex` on
  Apple Silicon, which is a real on-device story) or remove it from the README.

## 8. Summary of decisions this analysis asks for

1. Choose the audience the next two releases serve: in-process Rust/Python
   (what ships) or mobile/edge (what the tagline says). Section 7 assumes the
   second, because that is where the differentiation and the paying buyers are.
2. Ship filtered search, quantization and wasm persistence before CUDA.
3. Publish one honest benchmark against the incumbents and one demo, then
   read the feedback that produces before committing further roadmap.

## Sources

Market size: [Research and Markets, vector DB](https://www.researchandmarkets.com/reports/5948613/vector-database-market-report) ·
[GM Insights, vector DB](https://www.gminsights.com/industry-analysis/vector-database-market) ·
[Fortune Business Insights, vector DB](https://www.fortunebusinessinsights.com/vector-database-market-112428) ·
[Roots Analysis, edge AI](https://www.rootsanalysis.com/edge-ai-market) ·
[GM Insights, edge AI](https://www.gminsights.com/industry-analysis/edge-ai-market) ·
[Grand View Research, edge AI](https://www.grandviewresearch.com/industry-analysis/edge-ai-market-report) ·
[Fortune Business Insights, edge AI](https://www.fortunebusinessinsights.com/edge-ai-market-107023)

Competitors and their users: [ObjectBox, on-device vector databases in 2026](https://objectbox.io/262454-2/) ·
[ObjectBox Sync pricing](https://objectbox.io/sync-pricing/) ·
[objectbox-dart #658](https://github.com/objectbox/objectbox-dart/issues/658) ·
[sqlite-vec #25, ANN tracking](https://github.com/asg017/sqlite-vec/issues/25) ·
[sqlite-vec #165, distance constraints](https://github.com/asg017/sqlite-vec/issues/165) ·
[sqlite-vec on Android and iOS](https://alexgarcia.xyz/sqlite-vec/android-ios.html) ·
[vectorlite](https://github.com/1yefuwang1/vectorlite) ·
[usearch #348, filtering](https://github.com/unum-cloud/usearch/issues/348) ·
[USearch](https://github.com/unum-cloud/usearch) ·
[Turso native vector search](https://turso.tech/vector) ·
[Turso, vector search on mobile with React Native](https://turso.tech/blog/building-vector-search-and-personal-knowledge-graphs-on-mobile-with-libsql-and-react-native) ·
[EdgeVec](https://github.com/matte1782/edgevec) ·
[EdgeVec Show HN](https://news.ycombinator.com/item?id=46249896) ·
[voy](https://github.com/tantaraio/voy) ·
[client-vector-search](https://github.com/yusufhilmi/client-vector-search) ·
[Google AI Edge RAG guide for Android](https://ai.google.dev/edge/mediapipe/solutions/genai/rag/android) ·
[ai_edge_rag Flutter plugin](https://pub.dev/packages/ai_edge_rag) ·
[Couchbase Mobile vs MongoDB Atlas Device Sync](https://www.couchbase.com/comparing-couchbase-vs-mongodb-mobile/) ·
[Actian VectorAI DB launch](https://www.actian.com/company/press-releases/actian-launches-vectorai-db-with-22x-faster-vector-search-for-production-ai-anywhere-including-the-edge/) ·
[LanceDB pricing](https://costbench.com/software/vector-databases/lancedb/) ·
[Chroma vs LanceDB cost](https://aibizhub.io/articles/chroma-vs-lancedb-cost-2026/) ·
[RAG on mobile, 2026](https://dev.to/devin-rosario/rag-on-mobile-local-vector-dbs-and-smart-search-2026-1ad7) ·
[react-native-rag op-sqlite store](https://github.com/software-mansion-labs/react-native-rag/blob/main/packages/op-sqlite/README.md)

Registry data: [crates.io vanedb](https://crates.io/crates/vanedb) ·
[PyPI vanedb](https://pypi.org/project/vanedb/) ·
[npm @vanedb/wasm](https://www.npmjs.com/package/@vanedb/wasm)
