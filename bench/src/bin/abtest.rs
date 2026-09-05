//! Answers "is this delta real?" for two revisions, using the interleaved
//! procedure that has repeatedly been done by hand (#60).
//!
//! Builds each revision in its own git worktree, runs the selected benches
//! A-B-A-B, and reports each operation's median per arm, the spread each arm
//! measured for itself, and the cross-revision delta — calling a delta real
//! only when it clears both spreads and the project's noise floor.

use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

use vanedb_bench::abtest::{self, Comparison, Measurement, Verdict, NOISE_FLOOR};

const DEFAULT_BENCHES: &[&str] = &["distance", "store", "index", "disk"];

const USAGE: &str = "\
usage: abtest --a <rev> --b <rev> [options] [-- <criterion args>]

  --a <rev>        baseline revision (default: origin/main)
  --b <rev>        revision under test (default: HEAD)
  --rounds <n>     interleaved rounds; each arm runs n times (default: 2)
  --bench <name>   bench target, repeatable (default: distance store index disk)
  --workdir <dir>  where worktrees are built (default: the system temp dir)
  --keep           do not remove the worktrees on success
  -h, --help       this message

Anything after -- is passed to criterion, e.g. -- --quick

Run on idle, dedicated hardware. Timings from CI or a busy machine are not
evidence of anything.
";

struct Args {
    a: String,
    b: String,
    rounds: usize,
    benches: Vec<String>,
    workdir: PathBuf,
    keep: bool,
    criterion: Vec<String>,
}

fn parse_args() -> Result<Option<Args>, String> {
    let mut args = Args {
        a: "origin/main".into(),
        b: "HEAD".into(),
        rounds: 2,
        benches: Vec::new(),
        workdir: std::env::temp_dir(),
        keep: false,
        criterion: Vec::new(),
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut value = |name: &str| it.next().ok_or_else(|| format!("{name} needs a value"));
        match arg.as_str() {
            "-h" | "--help" => return Ok(None),
            "--a" => args.a = value("--a")?,
            "--b" => args.b = value("--b")?,
            "--rounds" => {
                args.rounds = value("--rounds")?
                    .parse()
                    .map_err(|_| "--rounds needs a number".to_string())?
            }
            "--bench" => args.benches.push(value("--bench")?),
            "--workdir" => args.workdir = PathBuf::from(value("--workdir")?),
            "--keep" => args.keep = true,
            "--" => {
                args.criterion.extend(it.by_ref());
                break;
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    if args.rounds == 0 {
        return Err("--rounds must be at least 1".into());
    }
    if args.benches.is_empty() {
        args.benches = DEFAULT_BENCHES.iter().map(|s| (*s).to_string()).collect();
    }
    Ok(Some(args))
}

fn git(repo: &Path, args: &[&str]) -> Result<String, String> {
    let out = Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(args)
        .output()
        .map_err(|e| format!("git {}: {e}", args.join(" ")))?;
    if !out.status.success() {
        return Err(format!(
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("bench lives directly below the repository root")
        .to_path_buf()
}

/// A detached worktree at `rev`, reused when one is already there so repeated
/// comparisons against the same baseline do not rebuild the C++ engine.
fn worktree(repo: &Path, workdir: &Path, rev: &str) -> Result<(PathBuf, String), String> {
    let sha = git(repo, &["rev-parse", rev])?;
    let short = git(repo, &["rev-parse", "--short", rev])?;
    let dir = workdir.join(format!("vanedb-abtest-{short}"));
    if dir.join(".git").exists() {
        let existing = git(&dir, &["rev-parse", "HEAD"])?;
        if existing == sha {
            return Ok((dir, short));
        }
        return Err(format!(
            "{} holds {existing}, not {sha}; remove it or pass --workdir",
            dir.display()
        ));
    }
    git(
        repo,
        &["worktree", "add", "--detach", &dir.to_string_lossy(), &sha],
    )?;
    Ok((dir, short))
}

fn bench(worktree: &Path, args: &Args) -> Result<Vec<Measurement>, String> {
    let mut cmd = Command::new("cargo");
    cmd.arg("bench")
        .arg("--manifest-path")
        .arg(worktree.join("bench").join("Cargo.toml"))
        .arg("--locked");
    for b in &args.benches {
        cmd.arg("--bench").arg(b);
    }
    cmd.arg("--").arg("--noplot").args(&args.criterion);
    let out = cmd.output().map_err(|e| format!("cargo bench: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "cargo bench in {} failed:\n{}",
            worktree.display(),
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    Ok(abtest::parse_run(&String::from_utf8_lossy(&out.stdout)))
}

fn report(rows: &[Comparison], a_label: &str, b_label: &str) {
    let width = rows.iter().map(|r| r.id.len()).max().unwrap_or(2).max(9);
    println!(
        "\n{:<width$}  {:>12}  {:>12}  {:>9}  {:>8}  {:>8}  verdict",
        "operation",
        format!("A {a_label}"),
        format!("B {b_label}"),
        "delta",
        "A spread",
        "B spread"
    );
    for row in rows {
        let cell = |arm: &Option<abtest::Arm>| {
            arm.as_ref()
                .map(|a| abtest::format_ns(a.median_ns))
                .unwrap_or_else(|| "-".into())
        };
        let pct = |v: Option<f64>| {
            v.map(|v| format!("{:+.1}%", v * 100.0))
                .unwrap_or_else(|| "-".into())
        };
        let spread = |arm: &Option<abtest::Arm>| {
            arm.as_ref()
                .map(|a| format!("{:.1}%", a.spread * 100.0))
                .unwrap_or_else(|| "-".into())
        };
        println!(
            "{:<width$}  {:>12}  {:>12}  {:>9}  {:>8}  {:>8}  {}",
            row.id,
            cell(&row.a),
            cell(&row.b),
            pct(row.delta),
            spread(&row.a),
            spread(&row.b),
            match row.verdict {
                Verdict::Significant => "SIGNIFICANT",
                Verdict::Noise => "noise",
                Verdict::AOnly => "A only",
                Verdict::BOnly => "B only",
            }
        );
    }
    let real = rows
        .iter()
        .filter(|r| r.verdict == Verdict::Significant)
        .count();
    println!(
        "\n{real} of {} operations moved beyond the noise each arm measured \
         (floor {:.0}%).",
        rows.len(),
        NOISE_FLOOR * 100.0
    );
}

fn run() -> Result<(), String> {
    let Some(args) = parse_args()? else {
        print!("{USAGE}");
        return Ok(());
    };
    let repo = repo_root();
    let (a_dir, a_label) = worktree(&repo, &args.workdir, &args.a)?;
    let (b_dir, b_label) = worktree(&repo, &args.workdir, &args.b)?;
    if a_label == b_label {
        return Err(format!("--a and --b are both {a_label}"));
    }

    // Build both before timing anything: a compile inside the interleaving
    // would land entirely in whichever arm went first.
    for (dir, label) in [(&a_dir, &a_label), (&b_dir, &b_label)] {
        eprintln!("building {label}...");
        let mut cmd = Command::new("cargo");
        cmd.arg("bench")
            .arg("--manifest-path")
            .arg(dir.join("bench").join("Cargo.toml"))
            .arg("--locked")
            .arg("--no-run");
        for b in &args.benches {
            cmd.arg("--bench").arg(b);
        }
        let status = cmd.status().map_err(|e| format!("cargo bench: {e}"))?;
        if !status.success() {
            return Err(format!("building {label} failed"));
        }
    }

    let mut a_rounds = Vec::new();
    let mut b_rounds = Vec::new();
    for round in 1..=args.rounds {
        eprintln!("round {round}/{}: A {a_label}", args.rounds);
        a_rounds.push(bench(&a_dir, &args)?);
        eprintln!("round {round}/{}: B {b_label}", args.rounds);
        b_rounds.push(bench(&b_dir, &args)?);
    }

    report(&abtest::compare(&a_rounds, &b_rounds), &a_label, &b_label);

    if !args.keep {
        for dir in [&a_dir, &b_dir] {
            let _ = git(
                &repo,
                &["worktree", "remove", "--force", &dir.to_string_lossy()],
            );
        }
    } else {
        eprintln!("\nworktrees kept: {}, {}", a_dir.display(), b_dir.display());
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("abtest: {err}");
            ExitCode::FAILURE
        }
    }
}
