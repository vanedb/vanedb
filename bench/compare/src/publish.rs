//! Publish-path policy for COMPARISON.md (independent of fixture role).

/// Canonical methodology knobs (must match `bench/COMPARISON.md`).
pub const PUBLISH_M: usize = 16;
pub const PUBLISH_EF_CONSTRUCTION: usize = 200;
pub const PUBLISH_K: usize = 10;
pub const PUBLISH_SEED: u64 = 42;
pub const PUBLISH_EF_SWEEP: &[usize] = &[16, 32, 50, 100];

/// Cosine publish rows omit sqlite-vec (no native vec0 cosine).
pub const PUBLISH_ENGINES_COSINE: &[&str] = &[
    "vanedb",
    "usearch",
    "hnswlib",
    "instant-distance",
    "hnsw_rs",
];

/// L2 publish rows include native sqlite-vec.
pub const PUBLISH_ENGINES_L2: &[&str] = &[
    "vanedb",
    "usearch",
    "hnswlib",
    "instant-distance",
    "hnsw_rs",
    "sqlite-vec",
];

/// Flag checks that must fail before any fixture-role messaging, so CI can
/// assert them on the smoke fixture without the role gate masking them.
#[derive(Clone, Copy, Debug)]
pub struct MarkdownFlagGate {
    pub rounds: usize,
    pub force_sqlite_vec_cosine: bool,
    pub skip_delete: bool,
    pub skip_save: bool,
}

pub fn refuse_markdown_flags(g: MarkdownFlagGate) -> Result<(), String> {
    // Order matters for tests/CI: force-sqlite and skip-* before rounds so
    // each refusal reason is independently observable.
    if g.force_sqlite_vec_cosine {
        return Err("refusing --markdown with --force-sqlite-vec-cosine \
             (harness-side cosine scan is not a COMPARISON.md row; use --metric l2)"
            .into());
    }
    if g.skip_delete {
        return Err(
            "refusing --markdown with --skip-delete (publish rows must exercise delete)".into(),
        );
    }
    if g.skip_save {
        return Err(
            "refusing --markdown with --skip-save (publish rows must record file size)".into(),
        );
    }
    if g.rounds < 2 {
        return Err(
            "refusing --markdown with --rounds < 2 (dedicated interleaved runs only)".into(),
        );
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub struct CanonicalParamsGate {
    pub m: usize,
    pub ef_construction: usize,
    pub k: usize,
    pub seed: u64,
    pub ef_sweep: Vec<usize>,
}

pub fn refuse_noncanonical_params(g: &CanonicalParamsGate) -> Result<(), String> {
    if g.m != PUBLISH_M {
        return Err(format!(
            "refusing --markdown with --m {} (COMPARISON methodology requires M={PUBLISH_M})",
            g.m
        ));
    }
    if g.ef_construction != PUBLISH_EF_CONSTRUCTION {
        return Err(format!(
            "refusing --markdown with --ef-construction {} \
             (COMPARISON methodology requires ef_construction={PUBLISH_EF_CONSTRUCTION})",
            g.ef_construction
        ));
    }
    if g.k != PUBLISH_K {
        return Err(format!(
            "refusing --markdown with --k {} (COMPARISON methodology requires k={PUBLISH_K})",
            g.k
        ));
    }
    if g.seed != PUBLISH_SEED {
        return Err(format!(
            "refusing --markdown with --seed {} (COMPARISON methodology requires seed={PUBLISH_SEED})",
            g.seed
        ));
    }
    if g.ef_sweep.as_slice() != PUBLISH_EF_SWEEP {
        return Err(format!(
            "refusing --markdown with --ef {:?} \
             (COMPARISON methodology requires {:?})",
            g.ef_sweep, PUBLISH_EF_SWEEP
        ));
    }
    Ok(())
}

/// Refuse pasteable markdown from CI / GitHub Actions hosts (defense in depth).
pub fn refuse_ci_env_for_markdown() -> Result<(), String> {
    let ci = std::env::var("CI").unwrap_or_default();
    let gha = std::env::var("GITHUB_ACTIONS").unwrap_or_default();
    if ci == "true" || gha == "true" {
        return Err(
            "refusing --markdown under CI/GITHUB_ACTIONS (dedicated hardware only; \
             AGENTS.md forbids publishing shared-runner timings)"
                .into(),
        );
    }
    Ok(())
}

pub fn required_publish_engines(metric: &str) -> Result<&'static [&'static str], String> {
    match metric {
        "cosine" => Ok(PUBLISH_ENGINES_COSINE),
        "l2" => Ok(PUBLISH_ENGINES_L2),
        other => Err(format!("unknown publish metric {other}")),
    }
}

pub fn refuse_incomplete_engine_set(
    metric: &str,
    engines: &[impl AsRef<str>],
) -> Result<(), String> {
    let required = required_publish_engines(metric)?;
    let got: Vec<&str> = engines.iter().map(|e| e.as_ref()).collect();
    let mut missing = Vec::new();
    for name in required {
        if !got.iter().any(|g| *g == *name) {
            missing.push(*name);
        }
    }
    if !missing.is_empty() {
        return Err(format!(
            "refusing --markdown: incomplete engine set for metric={metric}; missing {:?}. \
             Publish runs must include the full fair set (no --engine cherry-pick)",
            missing
        ));
    }
    // Extra engines (e.g. sqlite-vec on cosine) are also refused.
    let mut extras = Vec::new();
    for g in &got {
        if !required.iter().any(|r| r == g) {
            extras.push((*g).to_string());
        }
    }
    if !extras.is_empty() {
        return Err(format!(
            "refusing --markdown: unexpected engines {:?} for metric={metric}",
            extras
        ));
    }
    if got.len() != required.len() {
        return Err(format!(
            "refusing --markdown: engine count {} != {} for metric={metric}",
            got.len(),
            required.len()
        ));
    }
    Ok(())
}

/// Engines that must report a file size on the publish path.
pub fn save_required_engines() -> &'static [&'static str] {
    &["vanedb", "usearch", "hnswlib", "sqlite-vec"]
}

pub fn refuse_incomplete_save_rows<'a, I>(engines: I) -> Result<(), String>
where
    I: IntoIterator<Item = (&'a str, Option<u64>)>,
{
    for (name, size) in engines {
        if save_required_engines().contains(&name) && size.is_none() {
            return Err(format!(
                "refusing --markdown: engine {name} supports save but file_size_bytes is missing"
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn force_sqlite_refused_before_rounds() {
        let err = refuse_markdown_flags(MarkdownFlagGate {
            rounds: 4,
            force_sqlite_vec_cosine: true,
            skip_delete: false,
            skip_save: false,
        })
        .unwrap_err();
        assert!(err.contains("force-sqlite-vec-cosine"), "{err}");
    }

    #[test]
    fn skip_delete_refused() {
        let err = refuse_markdown_flags(MarkdownFlagGate {
            rounds: 4,
            force_sqlite_vec_cosine: false,
            skip_delete: true,
            skip_save: false,
        })
        .unwrap_err();
        assert!(err.contains("skip-delete"), "{err}");
    }

    #[test]
    fn skip_save_refused() {
        let err = refuse_markdown_flags(MarkdownFlagGate {
            rounds: 4,
            force_sqlite_vec_cosine: false,
            skip_delete: false,
            skip_save: true,
        })
        .unwrap_err();
        assert!(err.contains("skip-save"), "{err}");
    }

    #[test]
    fn rounds_lt_2_refused() {
        let err = refuse_markdown_flags(MarkdownFlagGate {
            rounds: 1,
            force_sqlite_vec_cosine: false,
            skip_delete: false,
            skip_save: false,
        })
        .unwrap_err();
        assert!(err.contains("rounds"), "{err}");
    }

    #[test]
    fn ok_when_clean() {
        refuse_markdown_flags(MarkdownFlagGate {
            rounds: 2,
            force_sqlite_vec_cosine: false,
            skip_delete: false,
            skip_save: false,
        })
        .unwrap();
    }

    #[test]
    fn incomplete_hnswlib_save_refused() {
        let err = refuse_incomplete_save_rows([
            ("vanedb", Some(1)),
            ("hnswlib", None),
            ("usearch", Some(2)),
        ])
        .unwrap_err();
        assert!(err.contains("hnswlib"), "{err}");
    }

    fn canonical() -> CanonicalParamsGate {
        CanonicalParamsGate {
            m: PUBLISH_M,
            ef_construction: PUBLISH_EF_CONSTRUCTION,
            k: PUBLISH_K,
            seed: PUBLISH_SEED,
            ef_sweep: PUBLISH_EF_SWEEP.to_vec(),
        }
    }

    #[test]
    fn noncanonical_m_refused() {
        let mut g = canonical();
        g.m = 32;
        let err = refuse_noncanonical_params(&g).unwrap_err();
        assert!(err.contains("--m"), "{err}");
    }

    #[test]
    fn noncanonical_ef_refused() {
        let mut g = canonical();
        g.ef_sweep = vec![50];
        let err = refuse_noncanonical_params(&g).unwrap_err();
        assert!(err.contains("--ef"), "{err}");
    }

    #[test]
    fn canonical_params_ok() {
        refuse_noncanonical_params(&canonical()).unwrap();
    }

    #[test]
    fn cosine_engine_cherry_pick_refused() {
        let err = refuse_incomplete_engine_set("cosine", &["vanedb", "usearch"]).unwrap_err();
        assert!(err.contains("incomplete"), "{err}");
    }

    #[test]
    fn cosine_with_sqlite_refused() {
        let mut engines = PUBLISH_ENGINES_COSINE.to_vec();
        engines.push("sqlite-vec");
        let err = refuse_incomplete_engine_set("cosine", &engines).unwrap_err();
        assert!(err.contains("unexpected"), "{err}");
    }

    #[test]
    fn l2_full_set_ok() {
        refuse_incomplete_engine_set("l2", PUBLISH_ENGINES_L2).unwrap();
    }

    #[test]
    fn cosine_full_set_ok() {
        refuse_incomplete_engine_set("cosine", PUBLISH_ENGINES_COSINE).unwrap();
    }
}
