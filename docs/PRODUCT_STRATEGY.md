# VaneDB product strategy review

Dated 2026-09-30, seventeen days after [`MARKET_ANALYSIS.md`](MARKET_ANALYSIS.md)
and three days after the 0.2.0-rc.1 packages reached crates.io, PyPI and npm.
Inputs: this repository (README, roadmap, RFCs 0001–0013, LIMITS, capacity
study, 0.2.0 readiness record, the 29 open issues), live registry and GitHub
API reads on 2026-09-30, and web research on competitors, platform vendors and
comparable businesses. Figures read from a registry or the GitHub API are
marked **measured**; everything else carries a URL in the sources section and
is marked **unverified** where only a search snippet could be read.

This page is analysis and a recommendation. Nothing here changes a roadmap
line until the corresponding RFC or issue is amended.

## 1. Verdict

The 2026-09-13 analysis said "the engine is not the gap; the product surface
around it is". Seventeen days later the product surface has moved (filtered
search, wasm persistence, byte-level save/load, the C ABI stage 1 and signed
archives are all in rc.1) and the gap has moved with it. The gap is now:

1. **A release that does not ship.** 0.2.0 has been "not ready" since
   2026-09-21 behind gates that produce no user: three dedicated-hardware
   competitor tables, an official demo tag, and a full-matrix QA campaign.
   Meanwhile rc.1 QA found a correctness defect (#299, live vectors
   unreachable after reverse-link truncation, present since 0.1.1 and in the
   frozen C++ engine too) that contradicts the project's one differentiator,
   verification.
2. **A positioning that points at the segment with the most incumbents.**
   "Edge AI" reads as mobile. In the last twelve months mobile gained Chroma's
   Swift and Android betas (UniFFI over a Rust core, exactly RFC 0007's
   design), Couchbase Lite 4.1, Google's AI Edge RAG SDK, Apple's WWDC26
   `SpotlightSearchTool` for "fully local RAG", and ObjectBox's HNSW. VaneDB's
   shipped surface (Rust, Python, C, wasm on desktop and Node) serves a
   different buyer, and that buyer's demand grew faster.
3. **No signal.** 0 stars, 0 forks, 60 all-time crate downloads
   (**measured**, 2026-09-30), no external issue in six months and 225 pull
   requests. The project has more release-readiness prose than users.

Recommendation in one line: ship 0.2.0 within two weeks on a correctness
gate, not a benchmark gate; re-aim the next two releases at **local memory
for agents and local-first desktop tools**, where the demand signal is
loudest and the incumbent (sqlite-vec) is stalled; move the payload column
forward and the mobile SDKs back; park CUDA formally and delete Metal; keep
the verification story and make it a spec buyers can cite, not a workflow
only the maintainer reads.

## 2. What changed since 2026-09-13

### 2.1 Registry and repository (measured, 2026-09-30)

| Signal | 2026-09-13 | 2026-09-30 |
|---|---|---|
| GitHub stars / forks / watchers | 0 / 0 / 0 | 0 / 0 / 0 |
| crates.io downloads, all versions | 40 | 60 (`0.2.0-rc.1` is `max_version`) |
| PyPI | 0.1.1 | 0.1.1 latest; `0.2.0rc1` published |
| npm `@vanedb/wasm` | 0.1.1 | `latest` 0.1.1; `next` 0.2.0-rc.1 |
| Open issues | 44 (all maintainer-opened) | 29, all maintainer- or agent-opened |
| Merged history | 100+ PRs | 225 PRs, 362 commits, one human author plus agents |
| Peer crates, 90-day downloads | instant-distance 746k, lancedb 664k, usearch 404k, hnsw_rs 380k | instant-distance 679k, lancedb 756k, usearch 633k, hnsw_rs 435k |

The peer numbers show the channel is alive and growing (USearch +57% in
seventeen days is release-driven noise, but LanceDB and hnsw_rs both moved up).
VaneDB's 20 new downloads are the rc.1 publication run.

### 2.2 Competitors

| Product | Change since the September 13 analysis | Consequence for VaneDB |
|---|---|---|
| **Qdrant Edge** | $50M Series B closed 2026-03-12 ($87.8M total); Edge is an in-process Rust crate with Python bindings, ~11 MB, still private beta by most sources; Qdrant states it is "not pursuing device-only", Edge is a device-as-cache for a Qdrant server | The best-funded direct competitor has chosen not to be a standalone on-device engine. Standalone is open. |
| **sqlite-vec** | Original maintainer stalled (#226); community forks (`vlasky`, PhotoStructure); ANN only in `0.1.10-alpha.4` (May 2026); precompiled iOS/Android still an open 2024 issue (#71); September 2026 bugs in bit-quantized thresholds, cosine on tiny vectors, musl and NEON builds; OpenClaw's built-in memory depends on it and files recurring "sqlite-vec unavailable" bugs on Windows and macOS | The mindshare leader for local vector search is unmaintained and breaks on install. Cross-compiled, dependency-free wheels and npm packages are exactly VaneDB's strength. |
| **Chroma** | `chroma-swift` 1.0.2 beta (iOS 17+, UniFFI over the Rust core) and `chroma-android` 0.0.1 beta; company focus is Chroma Cloud; last stable 1.5.9 (May 2026) | RFC 0007's design shipped first under a brand every Python developer knows. A VaneDB Swift package would be second and unknown. |
| **LanceDB** | v0.39.0 (2026-09-17); RaBitQ 1–8 bit (`IVF_RQ`); repositioned as "multimodal AI lakehouse"; $30M Series A; no wasm or mobile | Has left the on-device segment. Its Rust crate remains the download ceiling for the channel. |
| **USearch** | v2.26.2 (2026-08-31); Rust compact binding; open asks for int4, DiskANN, `compact()` memory reclaim, "do not early-exit filtered search" | Still the reference in-process ANN with the widest binding matrix. Its open issues are VaneDB's shipped features (compact, filter widening). |
| **EdgeVec** | No release since v0.9.0 (2026-02-27), 98 stars | Browser leader stalled; the browser space is crowded (ferrovec, SCMP, altor-vec, voy, Orama) and low-value. |
| **hnswlib** | v0.10.0-rc.2 (2026-09-14); PyPI still 0.8.0 (2023) | Unchanged threat. |
| **FAISS** | v1.15.x: Metal IVF-PQ backend, RaBitQ FastScan | Server standard adds an Apple GPU path; not an edge product. |
| **Apple** | WWDC26: Foundation Models opens to third-party LLMs; `SpotlightSearchTool` gives apps "fully local RAG" over Core Spotlight; `NLContextualEmbedding` (512-d) is the embedder; no vector-store API | The platform absorbs the simple iOS case. A third-party store must offer what Spotlight does not: custom embeddings, custom corpus, cross-platform files. |
| **Google** | AI Edge RAG SDK (`SqliteVectorStore`) last updated 2026-03-06, Android-only; weight moved to LiteRT-LM (no vector store); EmbeddingGemma 308M (768-d, MRL to 128) is the reference on-device embedder | The Android simple case is served by a free Google SDK. |
| **Actian VectorAI DB** | GA 2026-04-28; server in a container for Jetson/Raspberry Pi; free to 5k vectors then from $417/month; sells HIPAA/GDPR/SOC 2 alignment | Validates a regulated-edge buyer who pays. Sells IT-compliance certificates, not functional-safety qualification. |
| **New Rust entrants** | EmbedVec ("SQLite of vector search", RaBitQ, metadata filtering, PyO3), Vq, vecq, ruvector, hnswx, khive-hnsw | Low barrier to entry. A pure algorithm library does not differentiate; verification, packaging and format do. |

### 2.3 Demand

- **Agent memory is the loudest 2026 signal.** 244 "memory" MCP servers
  are listed on mcpservers.org; OpenClaw 2026.8.1 ships built-in memory with
  local embeddings and imports from Codex and Claude Code; ClawMem,
  `agent-memory-mcp` (LanceDB + MiniLM ONNX, no network) and
  `mcp-memory-service` are independent stores. Corpora are 10³ to 10⁵ chunks
  at 384 to 1,024 dimensions; every one of them needs persistence, delete and
  upsert, a payload next to the vector, and an install that works on Windows.
- **On-device RAG papers and vendor benchmarks** cluster at 8k to 40k chunks
  (MobileRAG, a first-aid RAG at ~120 MB, a Snapdragon X Elite NPU benchmark
  at ~40k chunks). Brute force still wins at that scale; Actian's Q2 2026
  report frames edge as "10k to 1M vectors, under 4 GB, sub-5 ms, offline"
  (**unverified**).
- **Embedders**: `all-MiniLM-L6-v2` (384-d) is the most-downloaded model on
  Hugging Face; `nomic-embed-text-v1.5` (768-d, MRL) and EmbeddingGemma
  (768-d, MRL) dominate local RAG; Apple's `NLContextualEmbedding` is 512-d;
  Qwen3-Embedding-0.6B is 1,024-d and slow on CPU. The capacity study's
  dimension assumptions hold.
- **Developer asks** across competitor trackers in 2026: quantization (int4,
  binary, RaBitQ), precompiled mobile packages, disk-resident ANN under a RAM
  cap, filtered-search correctness ("do not early-exit before k"), delete with
  memory reclaim, crash-safe browser persistence, hybrid BM25 + vector, and
  build hygiene (AVX baked from the build host, NEON and musl failures).
  VaneDB already ships four of these (filter widening, compact, crash-safe
  save, cross-compiled artifacts) and none of them is on its README's first
  screen.
- **Local LLM tooling**: Ollama ~170k stars; LM Studio has built-in RAG
  (Element Labs, $19.3M Series B); Windows AI Foundry / Foundry Local ships
  Rust, C++, Python and JS packages in 2026, while Copilot+ PC share stays
  under 10% of shipments.

### 2.4 Monetization precedents (all 2024–2026)

| Company | Model | Outcome |
|---|---|---|
| SQLite | Consortium patronage, $120k per member per year | sustains a three-person team; not a template for a new project |
| Realm / MongoDB Atlas Device Sync | vendor-owned mobile sync | shut down 2025-09-30 by a company with $2B revenue; SDKs "keep the lights on" |
| ObjectBox | free DB, paid Sync | ~$0.7M revenue, 6–8 staff (**unverified**, GetLatka) |
| Turso | $7M seed (2022), no later round; Rust SQLite rewrite pre-1.0 | July 2026 pivot to a Postgres frontend on the same core |
| ElectricSQL | embeddable wasm Postgres + sync | acquired by Databricks 2026-08-11; PGlite weekly downloads 1M → 13M in a year |
| LanceDB | OSS + Cloud/Enterprise | $41M raised, ~$2.3M revenue (**unverified**), left the edge segment |
| Chroma | OSS + usage-priced Cloud | no round since the 2023 seed; Cloud launched 2025-08 |
| Qdrant | OSS + Cloud, Edge as device cache | $87.8M raised |
| Couchbase | Enterprise licence incl. Lite | taken private for $1.5B, 2025-09 |
| TigerBeetle | support and licensing, "paying customers since day one" | $30M raised |
| Actian VectorAI DB | free to 5k vectors, then from $417/month | too new to judge |

Reading: nobody has built a business on an on-device-only vector engine, and
the two companies that tried the mobile-database-plus-sync model earned
under $1M or shut the product down. Money followed **embeddable + wasm +
agent sandbox** (Electric) and **cloud data platform** (LanceDB, Qdrant).
Support and qualification licensing (TigerBeetle, SQLite Encryption
Extension) is the only precedent that fits a single-engine project.

### 2.5 Regulated edge

EU AI Act applies in full from 2026-08-02; the Cyber Resilience Act's
vulnerability-reporting duty started 2026-09-11 and its SBOM requirement
applies from 2027-12-11; CISA published 2026 SBOM minimum elements in July;
IEC 62304 edition 2 adds an AI lifecycle for health software. Ferrocene is a
TÜV-qualified Rust toolchain for ISO 26262 ASIL D, IEC 61508 SIL 3 and
IEC 62304 Class C. No vector-database vendor claims functional-safety
qualification; Actian and ObjectBox sell IT-compliance alignment. VaneDB's
signed archives, per-target CycloneDX SBOMs, specified file formats, fuzzed
loaders and documented corruption boundary are already most of a CRA-ready
component. Demand here is inferred from regulation, not observed from buyers.

## 3. Diagnosis

1. **The verification asset is real and is currently invisible.** The
   README's first screen is a quick start and a disclaimer list. Signed
   releases, SBOMs, the fixture-backed format spec, mutation testing and the
   fuzzed loaders appear nowhere a buyer looks first. Every competitor's 2026
   bug list (sqlite-vec's NEON/musl/AVX build failures, USearch's unreachable
   nodes under concurrent add, hnswlib's brute-force filter bug) is the kind
   of defect this project's process exists to prevent.
2. **#299 undercuts that asset until it is fixed.** A hardened loader on top
   of a graph that silently loses live vectors is the wrong order of
   priorities. It predates rc.1, so it is not a regression, but it must ship
   fixed in 0.2.0 or the "correct, hardened" sentence in the analysis is
   false. The fix (a reachability guard when truncating a reverse list, or a
   repair pass in `compact()`) is a graph-construction change under the
   cross-engine invariant, so it needs the conformance suite and interleaved
   benchmarks, not a dedicated-hardware competitor table.
3. **Release gates are inverted.** 0.2.0 waits on three dedicated-hardware
   competitor tables, an official demo release and a full QA campaign.
   Those are publication conditions for a launch post, not conditions for
   publishing a 0.x crate to zero users. The benchmark methodology is a
   strength; using it as a gate on every minor release is what turned a
   two-week release into a month.
4. **The process is sized for a team the project does not have.** 225 PRs,
   "ten-role reviews", stacked candidate branches and a readiness record
   that tracks integration evidence commit by commit are appropriate for a
   1.x with paying users. At 0.x with one maintainer and agents they consume
   the only scarce resource. Keep the invariants and the CI; cut the
   ceremony.
5. **The tagline and the shipped surface serve different buyers, and the
   tagline's buyer is the better-served one.** Mobile has five incumbents and
   two platform owners. Desktop, Node and Python local memory has one
   stalled incumbent, no platform owner, and a demand curve driven by
   agents. The 2026-09-13 analysis chose mobile "because that is where the
   differentiation and the paying buyers are"; the seventeen days since
   produced evidence against both halves: Chroma took the differentiation
   and the paying-buyer precedents are ObjectBox and Realm.
6. **The roadmap order is right on engineering grounds and wrong on
   audience grounds.** RFC 0013 → 0005 → 0007 (0.3.0) → 0008/0009 (0.4.0)
   is the order that minimises format churn. It also means the payload
   column, the feature that removes the sentence which loses every Python
   and agent-memory user before the quick start, ships last, after a mobile
   SDK whose audience is already served.

## 4. Strategy

### 4.1 Position

**"The vector index you can ship inside anything: one specified file,
verified loaders, no runtime dependencies, on every desktop OS, in Python,
Rust, Node and C."** Edge AI stays as a use case, not as the headline. The
mobile claim returns when a Swift or Kotlin developer asks for it, and the
evidence bar in RFC 0007 stays.

### 4.2 Audiences, re-ranked

| Rank | Audience | Why now | What they need that is not shipped |
|---|---|---|---|
| 1 | **Agent memory and local-first desktop tools** (MCP memory servers, OpenClaw, Claude Code and Codex memory backends, Obsidian/Raycast-style plugins, Electron apps) | loudest 2026 demand; incumbent stalled and breaks on Windows; corpora fit resident f32 today | payload beside the vector (RFC 0009); a reference MCP memory server; a Node-native package or a documented wasm-in-Node path with persistence; typed errors (#262) |
| 2 | **Python developers doing local RAG on one machine** | same engine, same wheels, same payload gap | RFC 0009; `chromadb`-shaped convenience layer in the Python package (collection of id + vector + JSON payload) |
| 3 | **Rust application developers** | crate channel ceiling ~750k/quarter; VaneDB's invariants and `disk` feature are the pitch | int8 (RFC 0005) for the 256 MB and 1 GB budgets; streaming disk build (RFC 0008 part 1) |
| 4 | **Regulated and safety-conscious edge** (medical, industrial, automotive suppliers) | regulation timetable; nobody sells qualification; the evidence already exists | a page that states it; a Ferrocene build check; a CRA conformance statement; later, paid qualification kit and support |
| 5 | **Mobile (Swift, Kotlin, Flutter, React Native)** | served by Chroma, ObjectBox, Couchbase, Google, Apple; physical-device evidence is expensive | RFC 0007, unchanged, on pull |
| 6 | **Browser** | crowded, stalled leader, low willingness to pay; wasm persistence already shipped | nothing further until a user asks |

### 4.3 Release plan

**0.2.0 — ship within two weeks of this page.**

- Fix #299 under the cross-engine conformance suite; add a stranded-vector
  invariant test to `conformance/graph/`. Fix #300 (DOT NaN, kernel
  disagreement) and #301 (wasm id range) because they are rc.1 QA findings.
- Publish with the Apple Silicon competitor table only, the other two
  labelled "not yet measured". Move the Linux AVX2 and Android ARM64 tables
  and the official demo tag out of the release gate and into the launch
  post's checklist (RFC 0003 amendment).
- Release gate from here on: CI green on the tag, rc QA of installed
  artifacts, no open `release-blocker`. Dedicated-hardware numbers are
  publications, never gates, except for changes to the two performance
  invariants in `CLAUDE.md`.

**0.2.x — cheap follow-ups that ride the launch feedback.**

- #304 (signal that the beam gave up) and #305 (C `max_ef_search`): both are
  API additions, both small, both requested by rc.1 QA.
- #262 typed errors: every audience-1 integration wraps them.
- README first screen: install, one search, then the three sentences that
  differentiate (one specified file read by two engines; signed archives and
  SBOMs; fuzzed loaders and documented corruption boundary). The
  disclaimers move below the fold.

**0.3.0 — payload and container, then quantization.**

- RFC 0013 container (unchanged, first).
- **RFC 0009 payload column moves from 0.4.0 to 0.3.0**, immediately after
  0013, with the Python and JavaScript JSON conveniences. This is the single
  change that most widens audiences 1 and 2.
- RFC 0005 int8 in 0.3.0; binary-plus-rescoring may slip to 0.4.0 without
  breaking any claim, since it targets the mobile budgets, not the desktop
  ones. The capacity study's thresholds (resident f32 wins below ~314k
  vectors at 768-d on 1 GiB) say audience 1 does not need it yet.
- RFC 0002 stages 2–4 (xcframework, AAR, vcpkg, Conan): keep the vcpkg and
  Conan stage (audience 3 and 4); the Apple and Android packaging follows
  RFC 0007's pull condition.
- **RFC 0007 mobile SDKs move to 0.4.0, gated on an external request** (an
  issue, a discussion, or a downstream project asking). The RFC stays
  accepted; the evidence bar stays.

**0.4.0 — disk and mobile on evidence.**

- RFC 0008 part 1 (streaming builder) unchanged; part 2 (mapped graph) stays
  gated on the cold-cache spike. #285's physical-device matrix runs only
  once RFC 0007 has a requester.
- RFC 0007 if pulled.

**Park and delete.**

- RFC 0001 CUDA: status `parked` with the reason the 2026-09-13 analysis
  already gives. Nothing in seventeen days of research produced an edge
  buyer who wants discrete NVIDIA GPUs; Jetson buyers want a container
  (Actian) or a CPU library.
- `gpu-metal`: delete after 0.3.0 (#257 decided now, executed then). A
  distance scan that accelerates no index is a maintenance cost and a README
  disclaimer. FAISS 1.15 has a Metal IVF-PQ path; that is the reference for
  anyone who needs it.
- The frozen C++ engine: keep as the conformance oracle for v1 and v2 files;
  stop reporting the Rust-vs-C++ row as a headline. #299 shows the value of
  the oracle (it shares the bug, which proves the bug is in the design) and
  the cost (two engines to fix).

### 4.4 Distribution

Zero stars after six months is a distribution problem, not an engineering
one. In order:

1. **Launch 0.2.0** with the existing draft post
   ([`launch/0003-competitor-benchmark.md`](launch/0003-competitor-benchmark.md)),
   one table, the Obsidian demo, r/rust and r/LocalLLaMA first, Show HN
   second. Ask one question: "which missing feature stops you using it".
2. **Build one integration in the audience-1 ecosystem** within thirty days
   of launch: a reference MCP memory server on `vanedb` (Python, under 300
   lines, persistence, payload via a sidecar JSON until RFC 0009), or an
   OpenClaw memory backend. Integrations produce issues; issues produce a
   roadmap.
3. **Answer the incumbents' open issues** where VaneDB already has the
   answer: sqlite-vec #71 (mobile packaging), #25 (ANN), USearch #726
   (compact), with a factual comment and a link, not a pitch.
4. **Publish the trust page** (section 4.5) and submit the project to the
   lists that regulated buyers read (OpenSSF Scorecard badge, CRA readiness
   statement, a `SECURITY.md` that names the SBOM and signature verification
   steps, which already exist).

### 4.5 Monetization, later

Do not plan revenue from the engine. The precedents say the engine is free
and the money, if any, is in one of:

- **Qualification and support** for regulated edge: a Ferrocene-built core,
  a requirements-to-tests trace for the persistence contract, a signed
  conformance report per release, a support contract. This is the
  TigerBeetle and SQLite Encryption Extension model; it fits a one-person
  project and needs no cloud. First step now: a CI job that builds `vanedb`
  with Ferrocene and a page that says so.
- **Sync** is what ObjectBox, Couchbase, Turso and Qdrant chose and what
  Realm died on. Not for this project.
- **Cloud** is LanceDB's and Chroma's path and requires a company. Not for
  this project.

### 4.6 Ninety-day success criteria

Measured from the 0.2.0 tag. Missing all four is the signal to reconsider
the audience choice, not to add engineering.

| Metric | Target | Source |
|---|---|---|
| First external issue, PR or discussion | ≥ 10 from ≥ 5 people | GitHub |
| Stars | ≥ 200 | GitHub |
| Downstream integrations | ≥ 1 not authored by the maintainer (MCP server, plugin, app) | GitHub search, issues |
| Registry | crates.io recent downloads ≥ 2,000; PyPI ≥ 1,000/month | crates.io API, pypistats |

### 4.7 Process

- One release, one page: `docs/release/<version>-readiness.md` lists blockers
  and their state, nothing else. Integration evidence lives in CI runs, not
  in prose.
- Reviews: one review per PR from a person or an agent, plus CI. "Ten-role
  reviews" are retired until there is a second maintainer.
- RFC lifecycle rule (already written): an RFC neither implemented nor
  parked within two milestones gets `parked`. Apply it to RFC 0001 now.
- Keep: the invariants in `CLAUDE.md`, the benchmark discipline for
  performance claims, the fixture-per-compatibility-fix rule, the signed
  release pipeline.

## 5. Decisions requested

1. Adopt a correctness-and-CI release gate for 0.x; amend RFC 0003 so the
   competitor tables are launch material, not a release condition.
2. Fix #299, #300 and #301 in 0.2.0; ship 0.2.0 by 2026-10-14.
3. Re-rank audiences per section 4.2; change the README's first screen and
   the tagline accordingly.
4. Move RFC 0009 to 0.3.0 and RFC 0007 to 0.4.0 on pull; keep RFC 0013 and
   RFC 0005 int8 in 0.3.0.
5. Park RFC 0001; decide #257 as delete-after-0.3.0.
6. Commit to one audience-1 integration within thirty days of the 0.2.0
   launch and to the section 4.6 criteria as the test of the strategy.

## Sources

Registry and repository (measured 2026-09-30): crates.io API for `vanedb`,
`instant-distance`, `lancedb`, `usearch`, `hnsw_rs`, `arroy`; PyPI JSON for
`vanedb`; npm registry for `@vanedb/wasm`; GitHub API for `vanedb/vanedb`
(stars, forks, issues, contributors, pull-request count).

Competitors: [USearch releases](https://github.com/unum-cloud/usearch/releases) ·
[USearch issues](https://github.com/unum-cloud/usearch/issues) ·
[sqlite-vec releases](https://github.com/asg017/sqlite-vec/releases) ·
[sqlite-vec #71 mobile packaging](https://github.com/asg017/sqlite-vec/issues/71) ·
[sqlite-vec #226 maintenance](https://github.com/asg017/sqlite-vec/issues/226) ·
[vlasky/sqlite-vec fork](https://github.com/vlasky/sqlite-vec) ·
[OpenClaw sqlite-vec issues](https://github.com/openclaw/openclaw/issues/68892) ·
[libSQL](https://github.com/tursodatabase/libsql) · [Turso](https://github.com/tursodatabase/turso) ·
[Turso DiskANN #832](https://github.com/tursodatabase/turso/issues/832) ·
[LanceDB releases](https://github.com/lancedb/lancedb/releases) ·
[LanceDB July 2026 newsletter](https://www.lancedb.com/blog/newsletter-july-2026) ·
[LanceDB Series A](https://www.lancedb.com/blog/series-a-funding) ·
[Chroma releases](https://github.com/chroma-core/chroma/releases) ·
[chroma-swift](https://github.com/chroma-core/chroma-swift) · [chroma-android](https://github.com/chroma-core/chroma-android) ·
[Chroma Cloud pricing](https://docs.trychroma.com/cloud/pricing) ·
[ObjectBox Dart releases](https://github.com/objectbox/objectbox-dart/releases) ·
[ObjectBox Sync pricing](https://objectbox.io/sync-pricing/) (unverified) ·
[Couchbase Lite 4.1](https://docs.couchbase.com/couchbase-lite/current/cbl-whatsnew.html) ·
[Haveli completes Couchbase acquisition](https://www.couchbase.com/press-releases/haveli-investments-completes-acquisition-of-couchbase/) ·
[Google AI Edge APIs](https://github.com/google-ai-edge/ai-edge-apis) · [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) ·
[EmbeddingGemma](https://developers.googleblog.com/en/introducing-embeddinggemma/) ·
[EdgeVec changelog](https://github.com/matte1782/edgevec/blob/main/CHANGELOG.md) ·
[ferrovec](https://github.com/singhpratech/ferrovec) · [vectorlite](https://github.com/1yefuwang1/vectorlite) ·
[hnswlib v0.10.0-rc.2](https://github.com/nmslib/hnswlib/releases/tag/v0.10.0-rc.2) ·
[FAISS releases](https://github.com/facebookresearch/faiss/releases) ·
[Qdrant Edge announcement](https://www.businesswire.com/news/home/20250729908555/en/Qdrant-Announces-Qdrant-Edge-The-First-Vector-Search-Engine-for-Embedded-AI) ·
[Qdrant Edge discussion 7534](https://github.com/orgs/qdrant/discussions/7534) ·
[Qdrant Series B](https://tech.eu/2026/03/12/qdrant-closes-50m-series-b-to-expand-vector-search-infrastructure/) ·
[Actian VectorAI DB launch](https://www.actian.com/company/press-releases/actian-launches-vectorai-db-with-22x-faster-vector-search-for-production-ai-anywhere-including-the-edge/) ·
[EmbedVec](https://lib.rs/crates/embedvec) · [VecturaKit](https://github.com/rryam/VecturaKit) ·
[react-native-rag](https://github.com/software-mansion-labs/react-native-rag)

Platforms and demand: [Apple Foundation Models, WWDC26 session 241](https://developer.apple.com/videos/play/wwdc2026/241/) ·
[WWDC26 session 246](https://developer.apple.com/videos/play/wwdc2026/246/) ·
[NLContextualEmbedding](https://developer.apple.com/documentation/naturallanguage/nlcontextualembedding) ·
[Android AICore developer preview, April 2026](https://android-developers.googleblog.com/2026/04/AI-Core-Developer-Preview.html) ·
[Foundry Local](https://learn.microsoft.com/en-us/windows/ai/foundry-local/get-started) ·
[Copilot+ PC share](https://www.tomshardware.com/laptops/copilot-pcs-represent-only-a-tiny-fraction-of-laptop-sales-compatible-laptops-accounted-for-less-than-10-percent-of-total-shipments-in-3q24) ·
[MCP memory servers](https://mcpservers.org/category/memory) ·
[OpenClaw 2026.8.1 memory](https://docs.openclaw.ai/releases/2026.8.1/memory) ·
[agent-memory-mcp](https://github.com/adamrdrew/agent-memory-mcp) · [ClawMem](https://github.com/yoloshii/ClawMem) ·
[MobileRAG](https://arxiv.org/html/2507.01079) · [on-device first-aid RAG](https://arxiv.org/pdf/2602.13229) ·
[Hugging Face model download statistics](https://huggingface.co/blog/lbourdois/huggingface-models-stats) (snapshot date unverified) ·
[Qwen3-Embedding-0.6B GGUF](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF)

Business precedents: [SQLite Consortium](https://sqlite.org/consortium.html) ·
[Atlas Device Sync end of life](https://www.mongodb.com/community/forums/t/atlas-device-sync-end-of-life-and-deprecation/296687) ·
[PowerSync as Atlas Device Sync alternative](https://powersync.com/blog/powersync-as-alternative-to-mongodb-atlas-device-sync) ·
[Databricks acquires Electric](https://www.databricks.com/blog/electric-joins-databricks-bring-wasm-postgres-ai-agent-sandboxes) ·
[Turso Postgres pivot](https://turso.tech/blog/a-new-modern-version-of-postgres-in-rust) ·
[Turso pricing](https://turso.tech/pricing) ·
[ObjectBox on Crunchbase](https://www.crunchbase.com/organization/objectbox) · [ObjectBox on GetLatka](https://getlatka.com/companies/objectbox) (unverified) ·
[LanceDB on GetLatka](https://getlatka.com/companies/lancedb.com) (unverified) ·
[TigerBeetle Series A](https://www.fintechfutures.com/data-privacy-security/financial-transactions-database-tigerbeetle-raises-24m-series-a-funding) ·
[MotherDuck funding](https://tracxn.com/d/companies/motherduck/__ImNOuR4_9UpxigSXghehK9xIMsE-BU-RyEsr6aHQ6_M)

Regulated edge: [EU CRA compliance guide](https://www.mend.io/blog/eu-cyber-resilience-act-compliance-guide/) ·
[CISA 2026 SBOM minimum elements](https://www.cisa.gov/sites/default/files/2026-07/2026_cisa_sbom_minimum_elements_508c.pdf) ·
[Actian VectorAI DB for regulated edge](https://www.hpcwire.com/bigdatawire/this-just-in/actian-introduces-vectorai-db-for-edge-and-regulated-ai-deployments/) ·
[Ferrocene](https://ferrocene.dev/) · [cargo-auditable](https://github.com/rust-secure-code/cargo-auditable) ·
[cargo-sbom](https://crates.io/crates/cargo-sbom)

Rust on mobile: [UniFFI](https://github.com/mozilla/uniffi-rs) ·
[Bitwarden sdk-internal](https://github.com/bitwarden/sdk-internal) ·
[xcframework crate](https://crates.io/crates/xcframework) · [cargo-swift](https://crates.io/crates/cargo-swift) ·
[Ferrostar](https://stadiamaps.com/blog/ferrostar-building-a-cross-platform-navigation-sdk-in-rust-part-2/)
