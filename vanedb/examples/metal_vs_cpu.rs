//! Time `gpu-metal` against the CPU search vanedb actually runs.
//!
//! `FlatIndex`, `ApproxIndex` and `DiskIndex` never call the Metal code.
//! Enabling the feature does not move an index onto the GPU. What `gpu-metal`
//! exposes is `MetalCompute`: upload a row-major `f32` matrix, then
//! `distances` / `search` for one query at a time. This example calls that
//! API directly and compares it with `FlatIndex::search`, which is the exact
//! scan a caller runs today. Approximate search walks a graph and is a
//! different operation; this program does not time it.
//!
//! Metal steady-state time includes every per-query cost inside `search`:
//! copying the query, the dispatch, waiting for the device, copying all `n`
//! distances back, and the CPU top-k. `MetalCompute::new` (device and shader
//! setup) and `upload` (the corpus copy) are timed separately and also folded
//! into `with_setup`, as if this row were the whole session. There is no
//! batch kernel: a batch is that many sequential scans.
//!
//! The distance check uses the same CPU kernels `FlatIndex` uses. Values are
//! drawn from `[0.1, 1.1)` so the run does not trip Metal's scalar fallback
//! for tiny components.
//!
//! ```text
//! cargo run -p vanedb --release --locked --features gpu-metal --example metal_vs_cpu
//! ```
//!
//! Without `--features gpu-metal` the same binary times only the CPU path and
//! says the Metal code was not compiled. Shapes whose two resident copies
//! would not fit in available RAM are skipped. `--force` runs them anyway.

use std::collections::BTreeSet;
use std::env;
use std::hint::black_box;
use std::io::{self, Write};
#[cfg(target_os = "macos")]
use std::process::Command;
use std::time::Instant;

use vanedb::distance::Metric;
use vanedb::{FlatIndex, SearchResult};

const DEFAULT_SIZES: &[usize] = &[10_000, 100_000, 1_000_000];
const DEFAULT_DIMS: &[usize] = &[384, 768, 1536];
const DEFAULT_BATCHES: &[usize] = &[1, 32];
const DEFAULT_K: usize = 10;
const QUICK_N: usize = 4_096;
const QUICK_DIM: usize = 384;
/// Bytes left free for the hash map, result buffers and the rest of the process.
const HEADROOM_BYTES: u64 = 512 * 1024 * 1024;
/// Pass the distance check when absolute error is tiny or relative error is
/// within 0.1%. A wrong metric misses both.
const REL_TOLERANCE: f32 = 1e-3;
const ABS_TOLERANCE: f32 = 1e-4;

macro_rules! emit {
    ($($arg:tt)*) => {{
        println!($($arg)*);
        let _ = io::stdout().flush();
    }};
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    L2,
    Cosine,
    Dot,
}

impl Kind {
    fn metric(self) -> Metric {
        match self {
            Kind::L2 => Metric::L2,
            Kind::Cosine => Metric::Cosine,
            Kind::Dot => Metric::Dot,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Kind::L2 => "l2",
            Kind::Cosine => "cosine",
            Kind::Dot => "dot",
        }
    }

    fn parse(text: &str) -> Result<Self, String> {
        match text {
            "l2" | "L2" => Ok(Kind::L2),
            "cosine" | "Cosine" => Ok(Kind::Cosine),
            "dot" | "Dot" => Ok(Kind::Dot),
            other => Err(format!(
                "unknown metric '{other}' (expected l2, cosine, dot)"
            )),
        }
    }
}

struct Config {
    sizes: Vec<usize>,
    dims: Vec<usize>,
    metrics: Vec<Kind>,
    batches: Vec<usize>,
    k: usize,
    repeats: Option<usize>,
    seed: u64,
    force: bool,
    sizes_set: bool,
    dims_set: bool,
    metrics_set: bool,
    batches_set: bool,
    quick: bool,
}

impl Config {
    fn parse(args: &[String]) -> Result<Self, String> {
        let mut cfg = Config {
            sizes: DEFAULT_SIZES.to_vec(),
            dims: DEFAULT_DIMS.to_vec(),
            metrics: vec![Kind::L2, Kind::Cosine, Kind::Dot],
            batches: DEFAULT_BATCHES.to_vec(),
            k: DEFAULT_K,
            repeats: None,
            seed: 0xC0FFEE,
            force: false,
            sizes_set: false,
            dims_set: false,
            metrics_set: false,
            batches_set: false,
            quick: false,
        };
        let mut index = 0;
        while index < args.len() {
            let arg = &args[index];
            if arg == "--help" || arg == "-h" {
                print_help();
                std::process::exit(0);
            } else if arg == "--quick" {
                cfg.quick = true;
            } else if arg == "--force" {
                cfg.force = true;
            } else if let Some(value) = take_value(args, &mut index, "--sizes")? {
                cfg.sizes = parse_usize_list(value)?;
                cfg.sizes_set = true;
            } else if let Some(value) = take_value(args, &mut index, "--dims")? {
                cfg.dims = parse_usize_list(value)?;
                cfg.dims_set = true;
            } else if let Some(value) = take_value(args, &mut index, "--metrics")? {
                cfg.metrics = parse_metrics(value)?;
                cfg.metrics_set = true;
            } else if let Some(value) = take_value(args, &mut index, "--batches")? {
                cfg.batches = parse_usize_list(value)?;
                cfg.batches_set = true;
            } else if let Some(value) = take_value(args, &mut index, "--k")? {
                cfg.k = parse_usize(value, "--k")?;
            } else if let Some(value) = take_value(args, &mut index, "--repeats")? {
                cfg.repeats = Some(parse_usize(value, "--repeats")?);
            } else if let Some(value) = take_value(args, &mut index, "--seed")? {
                cfg.seed = parse_u64(value, "--seed")?;
            } else {
                return Err(format!("unknown argument '{arg}' (try --help)"));
            }
            index += 1;
        }
        if cfg.quick {
            if !cfg.sizes_set {
                cfg.sizes = vec![QUICK_N];
            }
            if !cfg.dims_set {
                cfg.dims = vec![QUICK_DIM];
            }
            if !cfg.metrics_set {
                cfg.metrics = vec![Kind::L2, Kind::Cosine, Kind::Dot];
            }
            if !cfg.batches_set {
                cfg.batches = vec![1, 8];
            }
            if cfg.repeats.is_none() {
                cfg.repeats = Some(3);
            }
        }
        cfg.validate()?;
        Ok(cfg)
    }

    fn validate(&self) -> Result<(), String> {
        if self.sizes.is_empty()
            || self.dims.is_empty()
            || self.metrics.is_empty()
            || self.batches.is_empty()
        {
            return Err("sizes, dims, metrics and batches must be non-empty".into());
        }
        if self.sizes.contains(&0) || self.batches.contains(&0) {
            return Err("sizes and batches must be non-zero".into());
        }
        if self.dims.iter().any(|&dim| dim == 0 || dim % 4 != 0) {
            return Err(
                "every dim must be non-zero and divisible by 4 (Metal's float4 kernels)".into(),
            );
        }
        if self.k == 0 {
            return Err("--k must be at least 1".into());
        }
        if self.repeats == Some(0) {
            return Err("--repeats must be at least 1".into());
        }
        Ok(())
    }

    fn max_batch(&self) -> usize {
        self.batches.iter().copied().max().unwrap_or(1)
    }

    fn repeats_for(&self, nbytes: usize) -> usize {
        if let Some(repeats) = self.repeats {
            return repeats;
        }
        if nbytes <= 64 * 1024 * 1024 {
            7
        } else if nbytes <= 512 * 1024 * 1024 {
            5
        } else {
            3
        }
    }
}

fn print_help() {
    println!(
        "\
Compare MetalCompute::search with FlatIndex::search.

  cargo run -p vanedb --release --locked --features gpu-metal --example metal_vs_cpu

Options:
  --quick                 4096 x 384, batches 1 and 8, 3 repeats
  --sizes N,N,N           vector counts (default 10000,100000,1000000)
  --dims D,D,D            dimensions, each divisible by 4 (default 384,768,1536)
  --metrics l2,cosine,dot
  --batches N,N           query counts; Metal scans once per query
  --k N                   neighbours (default 10)
  --repeats N             measured passes after the first query (default depends on size)
  --seed N
  --force                 run a shape even when it looks larger than RAM
"
    );
}

fn take_value<'a>(
    args: &'a [String],
    index: &mut usize,
    flag: &str,
) -> Result<Option<&'a str>, String> {
    let arg = &args[*index];
    if let Some(rest) = arg.strip_prefix(flag) {
        if rest.is_empty() {
            *index += 1;
            let value = args
                .get(*index)
                .ok_or_else(|| format!("{flag} needs a value"))?;
            return Ok(Some(value));
        }
        if let Some(value) = rest.strip_prefix('=') {
            return Ok(Some(value));
        }
    }
    Ok(None)
}

fn parse_usize_list(text: &str) -> Result<Vec<usize>, String> {
    if text.is_empty() {
        return Err("empty list".into());
    }
    text.split(',')
        .map(|part| parse_usize(part, "list"))
        .collect()
}

fn parse_metrics(text: &str) -> Result<Vec<Kind>, String> {
    if text.is_empty() {
        return Err("--metrics is empty".into());
    }
    text.split(',').map(Kind::parse).collect()
}

fn parse_usize(text: &str, flag: &str) -> Result<usize, String> {
    text.parse::<usize>()
        .map_err(|_| format!("{flag}: '{text}' is not an integer"))
}

fn parse_u64(text: &str, flag: &str) -> Result<u64, String> {
    text.parse::<u64>()
        .map_err(|_| format!("{flag}: '{text}' is not an integer"))
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Values in `[0.1, 1.1)`. Ordinary normals: Metal will not take its scalar path.
fn fill_ordinary(dst: &mut [f32], mut seed: u64) -> Result<(), String> {
    let scale = 1.0 / ((1u64 << 24) as f32);
    for value in dst.iter_mut() {
        let bits = splitmix64(&mut seed) >> 40;
        *value = 0.1 + (bits as f32) * scale;
    }
    if dst
        .iter()
        .any(|value| !value.is_finite() || !(0.05..1.2).contains(value))
    {
        return Err("generator produced a value outside the ordinary range".into());
    }
    Ok(())
}

fn corpus_bytes(n: usize, dim: usize) -> Result<usize, String> {
    n.checked_mul(dim)
        .and_then(|len| len.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| format!("n={n} dim={dim} overflows"))
}

/// Two resident copies (source plus index, or source plus the Metal buffer)
/// plus a rough id-map allowance.
fn peak_bytes(n: usize, dim: usize) -> Result<u64, String> {
    let matrix = corpus_bytes(n, dim)? as u64;
    let ids = (n as u64).saturating_mul(48);
    Ok(matrix.saturating_mul(2).saturating_add(ids))
}

fn available_ram() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let text = std::fs::read_to_string("/proc/meminfo").ok()?;
        for line in text.lines() {
            if let Some(rest) = line.strip_prefix("MemAvailable:") {
                let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
                return Some(kb.saturating_mul(1024));
            }
        }
        None
    }
    #[cfg(target_os = "macos")]
    {
        if let Some(bytes) = macos_available() {
            return Some(bytes);
        }
        // Half of physical RAM when vm_stat cannot be parsed.
        sysctl_u64("hw.memsize").map(|total| total / 2)
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        None
    }
}

#[cfg(target_os = "macos")]
fn macos_available() -> Option<u64> {
    let out = Command::new("vm_stat").output().ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8(out.stdout).ok()?;
    let page = text
        .split("page size of ")
        .nth(1)?
        .split_whitespace()
        .next()?
        .parse::<u64>()
        .ok()?;
    let mut pages = 0u64;
    for key in ["Pages free", "Pages inactive", "Pages speculative"] {
        let line = text
            .lines()
            .find(|line| line.trim_start().starts_with(key))?;
        pages = pages.saturating_add(vm_stat_pages(line)?);
    }
    Some(pages.saturating_mul(page))
}

#[cfg(target_os = "macos")]
fn vm_stat_pages(line: &str) -> Option<u64> {
    let digits: String = line
        .split(':')
        .nth(1)?
        .chars()
        .filter(|c| c.is_ascii_digit())
        .collect();
    digits.parse().ok()
}

#[cfg(target_os = "macos")]
fn sysctl_u64(name: &str) -> Option<u64> {
    let out = Command::new("sysctl").args(["-n", name]).output().ok()?;
    if !out.status.success() {
        return None;
    }
    String::from_utf8(out.stdout).ok()?.trim().parse().ok()
}

fn cpu_brand() -> String {
    #[cfg(target_os = "linux")]
    {
        if let Ok(text) = std::fs::read_to_string("/proc/cpuinfo") {
            for line in text.lines() {
                if let Some(rest) = line.strip_prefix("model name") {
                    if let Some(value) = rest.split(':').nth(1) {
                        return value.trim().to_string();
                    }
                }
            }
        }
    }
    #[cfg(target_os = "macos")]
    {
        if let Ok(out) = Command::new("sysctl")
            .args(["-n", "machdep.cpu.brand_string"])
            .output()
        {
            if out.status.success() {
                let brand = String::from_utf8_lossy(&out.stdout).trim().to_string();
                if !brand.is_empty() {
                    return brand;
                }
            }
        }
    }
    env::consts::ARCH.to_string()
}

fn repeats_median(
    repeats: usize,
    mut body: impl FnMut() -> Result<(), String>,
) -> Result<u128, String> {
    let mut samples = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let start = Instant::now();
        body()?;
        samples.push(start.elapsed().as_nanos());
    }
    samples.sort_unstable();
    Ok(samples[repeats / 2])
}

fn time_once(mut body: impl FnMut() -> Result<(), String>) -> Result<u128, String> {
    let start = Instant::now();
    body()?;
    Ok(start.elapsed().as_nanos())
}

struct CpuBatch {
    batch: usize,
    first_ns: u128,
    batch_ns: u128,
    /// Kept for the Metal top-k comparison. Unread when that feature is off.
    #[cfg_attr(not(feature = "gpu-metal"), allow(dead_code))]
    hits: Vec<Vec<SearchResult>>,
}

struct CpuMetric {
    kind: Kind,
    build_ns: u128,
    batches: Vec<CpuBatch>,
}

fn search_cpu(index: &FlatIndex, query: &[f32], k: usize) -> Result<Vec<SearchResult>, String> {
    let hits = index
        .search(black_box(query), k)
        .map_err(|err| err.to_string())?;
    Ok(black_box(hits))
}

fn measure_cpu(
    index: &FlatIndex,
    queries: &[Vec<f32>],
    checked: &[usize],
    k: usize,
    repeats: usize,
) -> Result<CpuBatch, String> {
    let batch = queries.len();
    let first_ns = time_once(|| {
        let _ = search_cpu(index, &queries[0], k)?;
        Ok(())
    })?;
    let batch_ns = repeats_median(repeats, || {
        for query in queries {
            let _ = search_cpu(index, query, k)?;
        }
        Ok(())
    })?;
    let mut hits = Vec::with_capacity(checked.len());
    for &slot in checked {
        let got = search_cpu(index, &queries[slot], k)?;
        check_cpu_hits(&got, k)?;
        // Identical results must pass the same comparison the Metal rows use,
        // and this keeps that comparison compiled on hosts without the feature.
        if !topk_agrees(&got, &got, 0.0) || !distances_agree(0.0, 0.0) {
            return Err("agreement check rejected identical CPU results".into());
        }
        hits.push(got);
    }
    Ok(CpuBatch {
        batch,
        first_ns,
        batch_ns,
        hits,
    })
}

fn check_cpu_hits(hits: &[SearchResult], k: usize) -> Result<(), String> {
    if hits.is_empty() || hits.len() > k {
        return Err(format!("CPU search returned {} hits", hits.len()));
    }
    if hits.iter().any(|hit| !hit.distance.is_finite()) {
        return Err("CPU search returned a non-finite distance".into());
    }
    for pair in hits.windows(2) {
        if pair[1].distance < pair[0].distance {
            return Err("CPU search results are not nearest-first".into());
        }
    }
    Ok(())
}

fn checked_slots(batch: usize, n: usize) -> Vec<usize> {
    if n <= 100_000 || batch <= 3 {
        return (0..batch).collect();
    }
    vec![0, batch / 2, batch - 1]
}

fn rel_err(cpu: f32, other: f32) -> f32 {
    let scale = cpu.abs().max(other.abs()).max(1e-6);
    (cpu - other).abs() / scale
}

fn within_tolerance(cpu: f32, other: f32) -> bool {
    let abs = (cpu - other).abs();
    abs <= ABS_TOLERANCE || rel_err(cpu, other) <= REL_TOLERANCE
}

fn distances_agree(max_abs: f32, max_rel: f32) -> bool {
    max_abs <= ABS_TOLERANCE || max_rel <= REL_TOLERANCE
}

/// Same neighbour set, allowing a swap only inside the distance-error window.
///
/// Each distance can move by `slack`, so two neighbours closer than that can
/// trade places without either kernel being wrong. A clear gap must keep both
/// the set and the order.
fn topk_agrees(cpu: &[SearchResult], gpu: &[SearchResult], slack: f32) -> bool {
    if cpu.len() != gpu.len() || cpu.is_empty() {
        return false;
    }
    let cutoff = cpu[cpu.len() - 1].distance;
    let cpu_ids: BTreeSet<u64> = cpu.iter().map(|hit| hit.id).collect();
    let gpu_ids: BTreeSet<u64> = gpu.iter().map(|hit| hit.id).collect();
    for hit in gpu {
        if let Some(reference) = cpu.iter().find(|item| item.id == hit.id) {
            if !within_tolerance(reference.distance, hit.distance) {
                return false;
            }
        } else if (hit.distance - cutoff).abs() > slack {
            return false;
        }
    }
    for hit in cpu {
        if !gpu_ids.contains(&hit.id) && (hit.distance - cutoff).abs() > slack {
            return false;
        }
    }
    let clear_gap = cpu
        .windows(2)
        .all(|pair| (pair[0].distance - pair[1].distance).abs() > slack);
    if cpu_ids == gpu_ids && clear_gap {
        return cpu.iter().zip(gpu).all(|(left, right)| left.id == right.id);
    }
    true
}

struct Row {
    n: usize,
    dim: usize,
    metric: &'static str,
    batch: usize,
    repeats: usize,
    cpu_first_ns: u128,
    cpu_batch_ns: u128,
    build_ns: u128,
    metal_first_ns: Option<u128>,
    metal_batch_ns: Option<u128>,
    upload_ns: Option<u128>,
    setup_ns: Option<u128>,
    max_abs: Option<f32>,
    max_rel: Option<f32>,
    topk_ok: Option<bool>,
    note: &'static str,
}

impl Row {
    fn cpu_per_query(&self) -> u128 {
        self.cpu_batch_ns / self.batch as u128
    }

    fn metal_per_query(&self) -> Option<u128> {
        self.metal_batch_ns.map(|total| total / self.batch as u128)
    }

    /// Setup and corpus upload charged across this batch, plus one steady query.
    fn with_setup_per_query(&self) -> Option<u128> {
        let steady = self.metal_per_query()?;
        let fixed = self.setup_ns?.saturating_add(self.upload_ns?);
        Some(fixed / self.batch as u128 + steady)
    }

    /// One query from a cold process: shader setup, corpus copy, first dispatch.
    fn cold_ns(&self) -> Option<u128> {
        Some(
            self.setup_ns?
                .saturating_add(self.upload_ns?)
                .saturating_add(self.metal_first_ns?),
        )
    }
}

fn ns_ms(ns: u128) -> f64 {
    ns as f64 / 1_000_000.0
}

fn fmt_opt_ms(value: Option<u128>) -> String {
    match value {
        Some(ns) => format!("{:.3}", ns_ms(ns)),
        None => "-".to_string(),
    }
}

fn fmt_opt_f(value: Option<f32>) -> String {
    match value {
        Some(number) => format!("{number:.3e}"),
        None => "-".to_string(),
    }
}

fn fmt_ratio(numer: Option<u128>, denom: u128) -> String {
    match numer {
        Some(value) if denom > 0 => format!("{:.3}", value as f64 / denom as f64),
        _ => "-".to_string(),
    }
}

fn print_row(row: &Row) {
    let steady = row.metal_per_query();
    let with_setup = row.with_setup_per_query();
    emit!(
        "csv,{n},{dim},{metric},{batch},{repeats},{cpu_ns},{cpu_first_ns},{build_ns},{metal_ns},{metal_first_ns},{upload_ns},{setup_ns},{with_setup_ns},{cold_ns},{max_abs},{max_rel},{topk_ok},{note}",
        n = row.n,
        dim = row.dim,
        metric = row.metric,
        batch = row.batch,
        repeats = row.repeats,
        cpu_ns = row.cpu_per_query(),
        cpu_first_ns = row.cpu_first_ns,
        build_ns = row.build_ns,
        metal_ns = option_ns(steady),
        metal_first_ns = option_ns(row.metal_first_ns),
        upload_ns = option_ns(row.upload_ns),
        setup_ns = option_ns(row.setup_ns),
        with_setup_ns = option_ns(with_setup),
        cold_ns = option_ns(row.cold_ns()),
        max_abs = fmt_opt_f(row.max_abs),
        max_rel = fmt_opt_f(row.max_rel),
        topk_ok = match row.topk_ok {
            Some(true) => "yes",
            Some(false) => "no",
            None => "na",
        },
        note = row.note,
    );
}

fn option_ns(value: Option<u128>) -> String {
    match value {
        Some(ns) => ns.to_string(),
        None => "-".to_string(),
    }
}

fn print_table(rows: &[Row]) {
    emit!("table_header n dim metric batch cpu_ms metal_ms with_setup_ms cold_ms upload_ms steady/cpu with_setup/cpu max_rel topk note");
    for row in rows {
        let steady = row.metal_per_query();
        emit!(
            "table {n:>8} {dim:>4} {metric:<6} {batch:>5} {cpu:>10} {metal:>10} {with_setup:>14} {cold:>10} {upload:>10} {ratio:>10} {setup_ratio:>14} {max_rel:>10} {topk:>4} {note}",
            n = row.n,
            dim = row.dim,
            metric = row.metric,
            batch = row.batch,
            cpu = format!("{:.3}", ns_ms(row.cpu_per_query())),
            metal = fmt_opt_ms(steady),
            with_setup = fmt_opt_ms(row.with_setup_per_query()),
            cold = fmt_opt_ms(row.cold_ns()),
            upload = fmt_opt_ms(row.upload_ns),
            ratio = fmt_ratio(steady, row.cpu_per_query()),
            setup_ratio = fmt_ratio(row.with_setup_per_query(), row.cpu_per_query()),
            max_rel = fmt_opt_f(row.max_rel),
            topk = match row.topk_ok {
                Some(true) => "yes",
                Some(false) => "no",
                None => "na",
            },
            note = row.note,
        );
    }
}

struct Session {
    /// Device and shader setup. Read when the Metal path runs.
    #[cfg_attr(not(feature = "gpu-metal"), allow(dead_code))]
    setup_ns: Option<u128>,
    #[cfg(feature = "gpu-metal")]
    compute: Option<vanedb::gpu::MetalCompute>,
}

fn open_session() -> Result<Session, String> {
    #[cfg(feature = "gpu-metal")]
    {
        emit!("metal_feature: on");
        let Some(device) = metal::Device::system_default() else {
            emit!("metal_device: unavailable");
            emit!("setup_ms: na");
            return Ok(Session {
                setup_ns: None,
                compute: None,
            });
        };
        let name = device.name().to_string();
        let headless = device.is_headless();
        let unified = device.has_unified_memory();
        emit!(
            "metal_device: available name={name} headless={headless} low_power={} unified={unified} working_set_bytes={} registry_id={}",
            device.is_low_power(),
            device.recommended_max_working_set_size(),
            device.registry_id(),
        );
        if headless || !unified {
            emit!(
                "metal_device_note: headless or not unified memory; these timings are not a laptop GPU"
            );
        }
        drop(device);
        let start = Instant::now();
        let compute = vanedb::gpu::MetalCompute::new().map_err(|err| err.to_string())?;
        let setup_ns = start.elapsed().as_nanos();
        emit!("setup_ms: {:.3}", ns_ms(setup_ns));
        Ok(Session {
            setup_ns: Some(setup_ns),
            compute: Some(compute),
        })
    }
    #[cfg(not(feature = "gpu-metal"))]
    {
        emit!("metal_feature: off");
        emit!("metal_device: not_compiled");
        emit!("setup_ms: na");
        Ok(Session { setup_ns: None })
    }
}

fn run(cfg: &Config) -> Result<(), String> {
    emit!("metal_vs_cpu");
    emit!(
        "level: MetalCompute::search versus FlatIndex::search; no index calls Metal, and a batch is one scan per query"
    );
    let session = open_session()?;
    let ram = available_ram();
    emit!("arch: {}", env::consts::ARCH);
    emit!("cpu: {}", cpu_brand());
    emit!(
        "threads: {}",
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    );
    emit!(
        "ram_available_bytes: {}",
        ram.map(|bytes| bytes.to_string())
            .unwrap_or_else(|| "unknown".into())
    );
    emit!("cpu_search: single-threaded FlatIndex::search (SIMD kernels, k-element heap)");
    emit!(
        "legend: cpu_ms and metal_ms are steady per-query times; metal_ms includes the query copy, dispatch, wait, full distance readback and top-k; with_setup_ms adds MetalCompute::new and the corpus upload spread across this batch; cold_ms is setup + upload + the first dispatch; a steady/cpu ratio below 1 means Metal was faster; index build is not in cpu_ms; ratios inside a few percent are noise, and a hosted runner is not evidence"
    );
    emit!(
        "csv,n,dim,metric,batch,repeats,cpu_ns,cpu_first_ns,build_ns,metal_ns,metal_first_ns,upload_ns,setup_ns,with_setup_ns,cold_ns,max_abs,max_rel,topk_ok,note"
    );

    let mut rows = Vec::new();
    let mut skipped = 0usize;
    let mut mismatched = 0usize;
    for &n in &cfg.sizes {
        for &dim in &cfg.dims {
            match bench_shape(cfg, &session, n, dim)? {
                ShapeOutcome::Skipped { peak, have } => {
                    skipped += 1;
                    emit!(
                        "skip n={n} dim={dim} peak_bytes={peak} ram_available_bytes={} reason=two resident copies plus headroom exceed available RAM; pass --force to run anyway",
                        have.map(|bytes| bytes.to_string()).unwrap_or_else(|| "unknown".into())
                    );
                }
                ShapeOutcome::Rows(shape_rows) => {
                    for row in &shape_rows {
                        if row.note == "mismatch" {
                            mismatched += 1;
                        }
                        print_row(row);
                    }
                    rows.extend(shape_rows);
                }
            }
        }
    }

    emit!("rows: {}", rows.len());
    emit!("skipped_shapes: {skipped}");
    emit!("mismatches: {mismatched}");
    print_table(&rows);
    if rows.is_empty() {
        return Err("no shape fit in available RAM".into());
    }
    if rows.iter().any(|row| row.cpu_per_query() == 0) {
        return Err("CPU search measured 0 ns; the timed call was optimized away".into());
    }
    if mismatched > 0 {
        return Err(format!(
            "{mismatched} Metal result sets did not match the CPU path"
        ));
    }
    Ok(())
}

enum ShapeOutcome {
    Skipped { peak: u64, have: Option<u64> },
    Rows(Vec<Row>),
}

fn bench_shape(
    cfg: &Config,
    session: &Session,
    n: usize,
    dim: usize,
) -> Result<ShapeOutcome, String> {
    let nbytes = corpus_bytes(n, dim)?;
    let peak = peak_bytes(n, dim)?;
    let have = available_ram();
    let exceeds_ram = have.is_some_and(|free| peak.saturating_add(HEADROOM_BYTES) > free);
    if !cfg.force && exceeds_ram {
        return Ok(ShapeOutcome::Skipped { peak, have });
    }
    emit!(
        "shape n={n} dim={dim} bytes={nbytes} peak_bytes={peak} repeats={}",
        cfg.repeats_for(nbytes)
    );
    let mut corpus = vec![0.0f32; n * dim];
    fill_ordinary(
        &mut corpus,
        cfg.seed ^ (n as u64).wrapping_mul(0x1000) ^ dim as u64,
    )?;
    let mut queries = Vec::with_capacity(cfg.max_batch());
    for slot in 0..cfg.max_batch() {
        let mut query = vec![0.0f32; dim];
        fill_ordinary(
            &mut query,
            cfg.seed
                .wrapping_add(0x9E37)
                .wrapping_add((slot as u64).wrapping_mul(0x51)),
        )?;
        queries.push(query);
    }
    let ids: Vec<u64> = (0..n as u64).collect();
    let repeats = cfg.repeats_for(nbytes);

    let mut cpu_metrics = Vec::with_capacity(cfg.metrics.len());
    for &kind in &cfg.metrics {
        let start = Instant::now();
        let index = FlatIndex::new(dim, kind.metric()).map_err(|err| err.to_string())?;
        index
            .add_batch(&ids, &corpus)
            .map_err(|err| err.to_string())?;
        let build_ns = start.elapsed().as_nanos();
        let mut batches = Vec::with_capacity(cfg.batches.len());
        for &batch in &cfg.batches {
            let slots = checked_slots(batch, n);
            let measured = measure_cpu(&index, &queries[..batch], &slots, cfg.k, repeats)?;
            batches.push(measured);
        }
        cpu_metrics.push(CpuMetric {
            kind,
            build_ns,
            batches,
        });
    }

    let shape = ShapeInput {
        n,
        dim,
        corpus: &corpus,
        queries: &queries,
        ids: &ids,
        cpu: &cpu_metrics,
    };
    let metal_rows = measure_metal(session, cfg, &shape)?;
    if metal_rows.is_empty() {
        let mut rows = Vec::new();
        for cpu in &cpu_metrics {
            for batch in &cpu.batches {
                rows.push(Row {
                    n,
                    dim,
                    metric: cpu.kind.name(),
                    batch: batch.batch,
                    repeats,
                    cpu_first_ns: batch.first_ns,
                    cpu_batch_ns: batch.batch_ns,
                    build_ns: cpu.build_ns,
                    metal_first_ns: None,
                    metal_batch_ns: None,
                    upload_ns: None,
                    setup_ns: None,
                    max_abs: None,
                    max_rel: None,
                    topk_ok: None,
                    note: "no_metal",
                });
            }
        }
        return Ok(ShapeOutcome::Rows(rows));
    }
    Ok(ShapeOutcome::Rows(metal_rows))
}

#[cfg_attr(not(feature = "gpu-metal"), allow(dead_code))]
struct ShapeInput<'a> {
    n: usize,
    dim: usize,
    corpus: &'a [f32],
    queries: &'a [Vec<f32>],
    ids: &'a [u64],
    cpu: &'a [CpuMetric],
}

#[cfg(not(feature = "gpu-metal"))]
fn measure_metal(
    _session: &Session,
    _cfg: &Config,
    _shape: &ShapeInput<'_>,
) -> Result<Vec<Row>, String> {
    Ok(Vec::new())
}

#[cfg(feature = "gpu-metal")]
fn measure_metal(
    session: &Session,
    cfg: &Config,
    shape: &ShapeInput<'_>,
) -> Result<Vec<Row>, String> {
    let Some(compute) = session.compute.as_ref() else {
        return Ok(Vec::new());
    };
    let start = Instant::now();
    let buffer = compute
        .upload(shape.corpus, shape.n, shape.dim)
        .map_err(|err| err.to_string())?;
    let upload_ns = start.elapsed().as_nanos();
    emit!("upload_ms: {:.3}", ns_ms(upload_ns));
    let mut rows = Vec::new();
    let cost = MetalCost {
        upload_ns,
        setup_ns: session.setup_ns,
    };
    for cpu_metric in shape.cpu {
        rows.extend(measure_metal_metric(
            compute, &buffer, cfg, shape, cpu_metric, cost,
        )?);
    }
    Ok(rows)
}

#[cfg(feature = "gpu-metal")]
#[derive(Clone, Copy)]
struct MetalCost {
    upload_ns: u128,
    setup_ns: Option<u128>,
}

#[cfg(feature = "gpu-metal")]
fn measure_metal_metric(
    compute: &vanedb::gpu::MetalCompute,
    buffer: &vanedb::gpu::metal::GpuBuffer,
    cfg: &Config,
    shape: &ShapeInput<'_>,
    cpu_metric: &CpuMetric,
    cost: MetalCost,
) -> Result<Vec<Row>, String> {
    let gpu_metric = vanedb::gpu::GpuMetric::from(cpu_metric.kind.metric());
    // First dispatch, before the correctness scan, so cold_ms includes the
    // real launch rather than a kernel that has already run.
    let metal_first_ns = time_once(|| {
        let hits = compute
            .search(&shape.queries[0], shape.ids, buffer, cfg.k, gpu_metric)
            .map_err(|err| err.to_string())?;
        drop(black_box(hits));
        Ok(())
    })?;
    let repeats = cfg.repeats_for(corpus_bytes(shape.n, shape.dim)?);
    let mut timed = Vec::with_capacity(cpu_metric.batches.len());
    for batch in &cpu_metric.batches {
        let query_slice = &shape.queries[..batch.batch];
        let metal_batch_ns = repeats_median(repeats, || {
            for query in query_slice {
                let hits = compute
                    .search(query, shape.ids, buffer, cfg.k, gpu_metric)
                    .map_err(|err| err.to_string())?;
                drop(black_box(hits));
            }
            Ok(())
        })?;
        timed.push(metal_batch_ns);
    }

    let (max_abs, max_rel) =
        distance_error(compute, buffer, shape, cpu_metric.kind.metric(), gpu_metric)?;
    let distances_ok = distances_agree(max_abs, max_rel);
    // Each side's error can move a neighbour, so a swap is possible across
    // twice the largest absolute gap.
    let slack = (max_abs * 2.0).max(ABS_TOLERANCE);
    if !distances_ok {
        emit!(
            "distance_mismatch n={} dim={} metric={} max_abs={max_abs:.3e} max_rel={max_rel:.3e}",
            shape.n,
            shape.dim,
            cpu_metric.kind.name(),
        );
    }

    let mut rows = Vec::with_capacity(cpu_metric.batches.len());
    for (batch, metal_batch_ns) in cpu_metric.batches.iter().zip(timed) {
        let mut topk_ok = distances_ok;
        let slots = checked_slots(batch.batch, shape.n);
        for (ordinal, slot) in slots.iter().copied().enumerate() {
            let gpu_hits = compute
                .search(&shape.queries[slot], shape.ids, buffer, cfg.k, gpu_metric)
                .map_err(|err| err.to_string())?;
            if !topk_agrees(&batch.hits[ordinal], &gpu_hits, slack) {
                topk_ok = false;
                emit!(
                    "mismatch n={} dim={} metric={} batch={} query={slot} cpu={} gpu={}",
                    shape.n,
                    shape.dim,
                    cpu_metric.kind.name(),
                    batch.batch,
                    format_hits(&batch.hits[ordinal]),
                    format_hits(&gpu_hits),
                );
            }
        }
        rows.push(Row {
            n: shape.n,
            dim: shape.dim,
            metric: cpu_metric.kind.name(),
            batch: batch.batch,
            repeats,
            cpu_first_ns: batch.first_ns,
            cpu_batch_ns: batch.batch_ns,
            build_ns: cpu_metric.build_ns,
            metal_first_ns: Some(metal_first_ns),
            metal_batch_ns: Some(metal_batch_ns),
            upload_ns: Some(cost.upload_ns),
            setup_ns: cost.setup_ns,
            max_abs: Some(max_abs),
            max_rel: Some(max_rel),
            topk_ok: Some(topk_ok),
            note: if distances_ok && topk_ok {
                "ok"
            } else {
                "mismatch"
            },
        });
    }
    Ok(rows)
}

#[cfg(feature = "gpu-metal")]
fn distance_error(
    compute: &vanedb::gpu::MetalCompute,
    buffer: &vanedb::gpu::metal::GpuBuffer,
    shape: &ShapeInput<'_>,
    cpu_metric: Metric,
    gpu_metric: vanedb::gpu::GpuMetric,
) -> Result<(f32, f32), String> {
    let gpu_distances = compute
        .distances(&shape.queries[0], buffer, gpu_metric)
        .map_err(|err| err.to_string())?;
    if gpu_distances.len() != shape.n {
        return Err(format!(
            "Metal returned {} distances for {} vectors",
            gpu_distances.len(),
            shape.n
        ));
    }
    let distance = vanedb::distance::distance_fn(cpu_metric);
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for (slot, gpu_distance) in gpu_distances.iter().copied().enumerate() {
        let start = slot * shape.dim;
        let cpu_distance = distance(&shape.queries[0], &shape.corpus[start..start + shape.dim]);
        max_abs = max_abs.max((gpu_distance - cpu_distance).abs());
        max_rel = max_rel.max(rel_err(cpu_distance, gpu_distance));
    }
    Ok((max_abs, max_rel))
}

#[cfg(feature = "gpu-metal")]
fn format_hits(hits: &[SearchResult]) -> String {
    hits.iter()
        .map(|hit| format!("{}:{:.6}", hit.id, hit.distance))
        .collect::<Vec<_>>()
        .join(",")
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    let cfg = match Config::parse(&args) {
        Ok(cfg) => cfg,
        Err(err) => {
            eprintln!("error: {err}");
            std::process::exit(2);
        }
    };
    if let Err(err) = run(&cfg) {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}
