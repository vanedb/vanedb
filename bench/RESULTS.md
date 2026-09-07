# VaneDB Benchmark Results

> **Superseded. Do not quote these numbers.**
>
> This snapshot was taken at monorepo `80066a2` (2026-09-03) by a harness that
> has since been fixed for the specific defects that produced it:
>
> - `ad62ddf` — *sweep a query set instead of timing one query.* The search
>   rows below are each a single query against two differently-seeded graphs,
>   so they measure graph luck as much as engine speed.
> - `f33a9a0` / `96ad2b8` — rank ground truth by the metric under test.
> - `a7555e1` — report per-call timings and cover every recall metric.
>
> The file was last touched by `2cf275a`, a rename that changed one word in
> the recall line and did not regenerate the numbers. Its own header still
> describes "medians of 501 interleaved paired samples (one query)" while the
> current `report.rs` sweeps 32 queries and reports time per query.
>
> **Use [the criterion table in `README.md`](README.md#headline-snapshot-apple-m4-pro-2026-09-monorepo-af05db0) instead.** It
> is what this file already called canonical, it carries per-row caveats, and
> it disagrees with the table below on two of three rows — reporting
> `store_search` at 0.98 (Rust ahead) and flagging `index_search`'s ratio as
> measuring nothing until re-run.
>
> Regenerate with `cargo run --release --bin report` on dedicated hardware,
> then delete this banner. Until then the numbers below are kept only so the
> commits that cite them remain readable.

Engines: vanedb-cpp (CMake Release) and vanedb (Rust), monorepo 80066a2.
Workload: dim=128, n=10000, k=10, L2. Latencies are medians of 501 interleaved paired samples (one query) after a joint warmup; recall is averaged over 100 queries. Both engines' data stays resident in one process (interleaved construction).

Covers l2_sq, store_search, and index_search + recall@10 only; every other operation is criterion-only (see README).

Criterion is canonical; see the README table. This bin times l2_sq in batches of 1000 calls, which inlines differently from criterion's per-call harness.

| Op | C++ (ns) | Rust (ns) | ratio (rs/cpp) |
|---|---:|---:|---:|
| l2_sq | 14 | 17 | 1.15 |
| store_search | 79500 | 88250 | 1.11 |
| index_search | 20042 | 21834 | 1.09 |

ApproxIndex recall@10: C++ 0.689, Rust 0.700

**The recall line is a single sample and the difference is noise.** Both
engines build one graph at seed 7 and score it; there is no seed sweep. Swept
over 100 construction seeds per engine on this workload the means are 0.6927
(Rust, sd 0.0055) and 0.6924 (C++, sd 0.0052) — a true difference of +0.0003,
95% CI [-0.0012, +0.0018], p = 0.69. Seed 7 is a draw where Rust landed +1.3 sd
and C++ -0.7 sd; the 0.011 gap shown above sits 14 standard errors outside that
interval. The same discipline the latency rows already got — sweep the
randomness, publish an interval — has to reach the quality metric before this
line means anything.
