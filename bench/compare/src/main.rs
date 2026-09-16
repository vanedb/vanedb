//! One-command competitor benchmark (RFC 0003).

use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};
use vanedb_compare::engines::{BuildParams, EngineKind, MetricKind};
use vanedb_compare::fixture::{
    classify_fixture, default_fixture_dir, load_fixture, verify_sha256sums, write_smoke_fixture,
    FixtureMeta, FixtureRole, PUBLISH_MIN_DOCS, PUBLISH_MIN_QUERIES,
};
use vanedb_compare::report::render_machine_section;
use vanedb_compare::run::{run_comparison, write_json_report, RunConfig};

#[derive(Parser, Debug)]
#[command(
    name = "compare",
    about = "VaneDB competitor benchmark harness (RFC 0003)"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Run every selected engine on a checksummed fixture (interleaved rounds).
    Run {
        /// Path to a `.vnef` fixture. Defaults to fixtures/embeddings.vnef.
        #[arg(long)]
        fixture: Option<PathBuf>,
        /// Allow the deterministic smoke fixture (NOT for COMPARISON.md).
        #[arg(long, default_value_t = false)]
        allow_smoke: bool,
        /// Verify against fixtures/SHA256SUMS when present.
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        verify_checksum: bool,
        /// Skip SHA256SUMS verification (not allowed with --markdown).
        #[arg(long, default_value_t = false)]
        no_verify_checksum: bool,
        #[arg(long, value_enum, default_value_t = MetricArg::Cosine)]
        metric: MetricArg,
        #[arg(long, default_value_t = 10)]
        k: usize,
        /// Comma-separated ef_search sweep, e.g. 16,32,50,100
        #[arg(long, default_value = "16,32,50,100")]
        ef: String,
        #[arg(long, default_value_t = 16)]
        m: usize,
        #[arg(long, default_value_t = 200)]
        ef_construction: usize,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// Interleaved rounds (A-B-C-… repeated). Use ≥2 on dedicated hardware.
        #[arg(long, default_value_t = 2)]
        rounds: usize,
        /// Limit queries (smoke / debugging).
        #[arg(long)]
        max_queries: Option<usize>,
        /// Restrict to a subset of engines (default: all).
        #[arg(long, value_enum)]
        engine: Vec<EngineKind>,
        #[arg(long)]
        out_dir: Option<PathBuf>,
        #[arg(long)]
        json_out: Option<PathBuf>,
        #[arg(long, default_value_t = false)]
        skip_save: bool,
        #[arg(long, default_value_t = false)]
        skip_delete: bool,
        /// Also print a markdown section for pasting into COMPARISON.md.
        /// Refuses smoke/dev fixtures and requires VANEDB_COMPARE_HW + checksum.
        #[arg(long, default_value_t = false)]
        markdown: bool,
        /// Include sqlite-vec on cosine runs (harness-side f32 scan, not vec0).
        /// Default: skip sqlite-vec when --metric cosine so published rows are honest.
        #[arg(long, default_value_t = false)]
        force_sqlite_vec_cosine: bool,
    },
    /// Write a tiny deterministic smoke fixture (not for published numbers).
    WriteSmoke {
        #[arg(long, default_value = "fixtures/smoke.vnef")]
        out: PathBuf,
        #[arg(long, default_value_t = 256)]
        n_docs: usize,
        #[arg(long, default_value_t = 16)]
        n_queries: usize,
        #[arg(long, default_value_t = 768)]
        dim: usize,
    },
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum MetricArg {
    L2,
    Cosine,
}

impl From<MetricArg> for MetricKind {
    fn from(value: MetricArg) -> Self {
        match value {
            MetricArg::L2 => MetricKind::L2,
            MetricArg::Cosine => MetricKind::Cosine,
        }
    }
}

fn main() {
    if let Err(e) = real_main() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn real_main() -> Result<(), String> {
    let cli = Cli::parse();
    match cli.command {
        Command::WriteSmoke {
            out,
            n_docs,
            n_queries,
            dim,
        } => {
            if let Some(parent) = out.parent() {
                std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
            }
            let sha = write_smoke_fixture(&out, n_docs, n_queries, dim)?;
            let meta = FixtureMeta {
                model: "none (deterministic smoke)".into(),
                corpus: "synthetic".into(),
                dim: dim as u32,
                n_docs: n_docs as u32,
                n_queries: n_queries as u32,
                metric_native: "cosine".into(),
                generator: "compare write-smoke".into(),
                notes: "NOT for published COMPARISON.md numbers".into(),
            };
            let meta_path = out.with_file_name("metadata.smoke.json");
            std::fs::write(
                &meta_path,
                serde_json::to_string_pretty(&meta).map_err(|e| e.to_string())?,
            )
            .map_err(|e| e.to_string())?;
            println!("wrote {} sha256={sha}", out.display());
            Ok(())
        }
        Command::Run {
            fixture,
            allow_smoke,
            verify_checksum,
            no_verify_checksum,
            metric,
            k,
            ef,
            m,
            ef_construction,
            seed,
            rounds,
            max_queries,
            engine,
            out_dir,
            json_out,
            skip_save,
            skip_delete,
            markdown,
            force_sqlite_vec_cosine,
        } => {
            let fixture_path = resolve_fixture(fixture, allow_smoke)?;
            let verify_checksum = verify_checksum && !no_verify_checksum;
            let mut checksum_verified = false;
            if verify_checksum {
                let sums = fixture_path
                    .parent()
                    .unwrap_or_else(|| std::path::Path::new("."))
                    .join("SHA256SUMS");
                if sums.exists() {
                    verify_sha256sums(&fixture_path, &sums)?;
                    checksum_verified = true;
                    eprintln!("checksum ok for {}", fixture_path.display());
                } else if markdown {
                    return Err(format!(
                        "refusing --markdown without SHA256SUMS beside {}",
                        fixture_path.display()
                    ));
                } else {
                    eprintln!(
                        "warning: no SHA256SUMS beside {}; skipping verify",
                        fixture_path.display()
                    );
                }
            } else if markdown {
                return Err("refusing --markdown with --verify-checksum=false".into());
            }

            let fixture = load_fixture(&fixture_path)?;
            let role = classify_fixture(&fixture);
            if role.requires_allow_smoke() && !allow_smoke {
                return Err(format!(
                    "refusing {} fixture (n_docs={}, need ≥{PUBLISH_MIN_DOCS} for publish) \
                     without --allow-smoke — renaming the file does not bypass this",
                    role.as_str(),
                    fixture.n_docs()
                ));
            }
            eprintln!(
                "fixture role={} dim={} docs={} queries={} sha256={}",
                role.as_str(),
                fixture.dim,
                fixture.n_docs(),
                fixture.n_queries(),
                fixture.sha256
            );

            if markdown {
                if role != FixtureRole::Publish {
                    return Err(format!(
                        "refusing --markdown for fixture_role={} (n_docs={}, n_queries={}). \
                         Publish only checksummed embeddings.vnef with ≥{PUBLISH_MIN_DOCS} docs \
                         and ≥{PUBLISH_MIN_QUERIES} queries",
                        role.as_str(),
                        fixture.n_docs(),
                        fixture.n_queries()
                    ));
                }
                if !checksum_verified {
                    return Err(
                        "refusing --markdown unless the fixture hash is listed in SHA256SUMS"
                            .into(),
                    );
                }
                let hw = std::env::var("VANEDB_COMPARE_HW").unwrap_or_default();
                if hw.is_empty() || hw == "unlabelled" {
                    return Err(
                        "refusing --markdown without VANEDB_COMPARE_HW set to a real label \
                         (e.g. linux-avx2, apple-silicon, android-arm64-emulator)"
                            .into(),
                    );
                }
                if rounds < 2 {
                    return Err(
                        "refusing --markdown with --rounds < 2 (dedicated interleaved runs only)"
                            .into(),
                    );
                }
                if let Some(mq) = max_queries {
                    if mq < fixture.n_queries() {
                        return Err(format!(
                            "refusing --markdown with --max-queries {mq} < fixture n_queries={} \
                             (publish must use the full query set)",
                            fixture.n_queries()
                        ));
                    }
                }
                if fixture.n_queries() < PUBLISH_MIN_QUERIES {
                    return Err(format!(
                        "refusing --markdown: fixture n_queries={} < {PUBLISH_MIN_QUERIES}",
                        fixture.n_queries()
                    ));
                }
                if force_sqlite_vec_cosine {
                    return Err("refusing --markdown with --force-sqlite-vec-cosine \
                         (harness-side cosine scan is not a COMPARISON.md row; use --metric l2)"
                        .into());
                }
                if skip_delete {
                    return Err(
                        "refusing --markdown with --skip-delete (publish rows must exercise delete)"
                            .into(),
                    );
                }
                if skip_save {
                    return Err(
                        "refusing --markdown with --skip-save (publish rows must record file size)"
                            .into(),
                    );
                }
                if fixture.meta.is_none() {
                    return Err(
                        "refusing --markdown without metadata.json beside the fixture".into(),
                    );
                }
            }

            let ef_sweep = parse_ef_list(&ef)?;
            let metric_kind: MetricKind = metric.into();
            let mut engines = if engine.is_empty() {
                EngineKind::all().to_vec()
            } else {
                engine
            };
            if metric_kind == MetricKind::Cosine
                && !force_sqlite_vec_cosine
                && engines.contains(&EngineKind::SqliteVec)
            {
                engines.retain(|e| *e != EngineKind::SqliteVec);
                eprintln!(
                    "skipping sqlite-vec on cosine (not a native vec0 metric); \
                     pass --force-sqlite-vec-cosine to include the harness-side scan, \
                     or run --metric l2 for native sqlite-vec"
                );
            }
            let out_dir = out_dir.unwrap_or_else(|| {
                PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/compare-out")
            });
            std::fs::create_dir_all(&out_dir).map_err(|e| e.to_string())?;

            let cfg = RunConfig {
                engines,
                metric: metric_kind,
                k,
                ef_sweep,
                params: BuildParams {
                    m,
                    ef_construction,
                    ef_search: 50,
                    seed,
                },
                rounds,
                out_dir: out_dir.clone(),
                max_queries,
                skip_save,
                skip_delete,
                fixture_role: role,
            };

            let report = run_comparison(&fixture, &cfg)?;
            let json_path = json_out.unwrap_or_else(|| out_dir.join("report.json"));
            write_json_report(&report, &json_path)?;
            eprintln!("wrote {}", json_path.display());
            if markdown {
                print!("{}", render_machine_section(&report));
            }
            Ok(())
        }
    }
}

fn resolve_fixture(explicit: Option<PathBuf>, allow_smoke: bool) -> Result<PathBuf, String> {
    if let Some(p) = explicit {
        return Ok(p);
    }
    let dir = default_fixture_dir();
    let full = dir.join("embeddings.vnef");
    if full.exists() {
        return Ok(full);
    }
    if !allow_smoke {
        return Err(format!(
            "fixtures/embeddings.vnef not found under {}. Generate it with \
             scripts/generate_fixture.py, or pass --fixture / --allow-smoke for harness checks.",
            dir.display()
        ));
    }
    let smoke = dir.join("smoke.vnef");
    if smoke.exists() {
        eprintln!(
            "using smoke fixture {} (--allow-smoke); not for publication",
            smoke.display()
        );
        return Ok(smoke);
    }
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    write_smoke_fixture(&smoke, 256, 16, 768)?;
    eprintln!("generated smoke fixture at {}", smoke.display());
    Ok(smoke)
}

fn parse_ef_list(s: &str) -> Result<Vec<usize>, String> {
    s.split(',')
        .map(|p| {
            p.trim()
                .parse::<usize>()
                .map_err(|e| format!("bad ef value '{p}': {e}"))
        })
        .collect()
}
