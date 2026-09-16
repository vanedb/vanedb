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

/// True when the process looks like a shared CI/cloud runner.
pub fn shared_runner_env() -> bool {
    let truthy = |k: &str| {
        matches!(
            std::env::var(k)
                .unwrap_or_default()
                .to_ascii_lowercase()
                .as_str(),
            "1" | "true" | "yes"
        )
    };
    // Presence flags: empty string must not count (CI YAML `VAR: ""` still sets the key).
    let present = |k: &str| std::env::var(k).map(|v| !v.is_empty()).unwrap_or(false);
    truthy("CI")
        || truthy("GITHUB_ACTIONS")
        || truthy("GITLAB_CI")
        || truthy("CIRCLECI")
        || truthy("BUILDKITE")
        || truthy("TF_BUILD")
        || present("CURSOR_AGENT")
        || present("CODESPACES")
}

/// Maintainer attestation that this host is idle dedicated hardware.
pub fn dedicated_hw_attested() -> bool {
    matches!(
        std::env::var("VANEDB_COMPARE_DEDICATED")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes"
    )
}

/// Refuse pasteable markdown from CI / GitHub Actions hosts (defense in depth).
pub fn refuse_ci_env_for_markdown() -> Result<(), String> {
    if shared_runner_env() {
        return Err("refusing --markdown under a shared CI/cloud runner env \
             (dedicated hardware only; AGENTS.md forbids publishing shared-runner timings)"
            .into());
    }
    Ok(())
}

/// Refuse --markdown unless the operator attests dedicated hardware.
pub fn refuse_unattested_dedicated_hw() -> Result<(), String> {
    refuse_unless_dedicated_attested(dedicated_hw_attested())
}

pub fn refuse_unless_dedicated_attested(attested: bool) -> Result<(), String> {
    if !attested {
        return Err("refusing --markdown without VANEDB_COMPARE_DEDICATED=1 \
             (operator attestation that this is an idle dedicated machine)"
            .into());
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
        if !got.contains(name) {
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

pub fn refuse_non_publish_role(
    role: crate::fixture::FixtureRole,
    n_docs: usize,
    n_queries: usize,
) -> Result<(), String> {
    use crate::fixture::{FixtureRole, PUBLISH_MIN_DOCS, PUBLISH_MIN_QUERIES};
    if role != FixtureRole::Publish {
        return Err(format!(
            "refusing --markdown for fixture_role={} (n_docs={n_docs}, n_queries={n_queries}). \
             Publish only checksummed embeddings.vnef with ≥{PUBLISH_MIN_DOCS} docs \
             and ≥{PUBLISH_MIN_QUERIES} queries",
            role.as_str(),
        ));
    }
    Ok(())
}

/// Engines that must report a file size on the publish path.
pub fn save_required_engines() -> &'static [&'static str] {
    &["vanedb", "usearch", "hnswlib", "sqlite-vec"]
}

/// Engines that must report delete_ok on the publish path.
pub fn delete_required_engines() -> &'static [&'static str] {
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

pub fn refuse_incomplete_delete_rows<'a, I>(engines: I) -> Result<(), String>
where
    I: IntoIterator<Item = (&'a str, Option<bool>)>,
{
    for (name, delete_ok) in engines {
        if delete_required_engines().contains(&name) && delete_ok != Some(true) {
            return Err(format!(
                "refusing --markdown: engine {name} must report delete_ok=true (got {delete_ok:?})"
            ));
        }
    }
    Ok(())
}

/// Allowed `VANEDB_COMPARE_HW` prefixes for the three #198 hardware classes.
pub fn refuse_bad_hw_label(hw: &str) -> Result<(), String> {
    if hw.starts_with("android-arm64-") {
        if !android_host() {
            return Err(
                "refusing --markdown with android-* off an Android host \
                 (need /system/build.prop, or follow bench/compare/ANDROID.md on-device)"
                    .into(),
            );
        }
        return Ok(());
    }
    let ok = hw.starts_with("apple-") || hw.starts_with("linux-avx2");
    if !ok {
        return Err(format!(
            "refusing --markdown with VANEDB_COMPARE_HW={hw:?}; \
             use a label starting with apple-, linux-avx2, or android-arm64- \
             (e.g. apple-m4-pro, linux-avx2, android-arm64-device)"
        ));
    }
    Ok(())
}

fn android_host() -> bool {
    std::path::Path::new("/system/build.prop").exists()
        || std::path::Path::new("/system/bin/app_process").exists()
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

    #[test]
    fn incomplete_delete_refused() {
        let err =
            refuse_incomplete_delete_rows([("vanedb", None), ("usearch", Some(true))]).unwrap_err();
        assert!(err.contains("delete_ok"), "{err}");
        let err = refuse_incomplete_delete_rows([("vanedb", Some(false))]).unwrap_err();
        assert!(err.contains("delete_ok=true"), "{err}");
    }

    #[test]
    fn bad_hw_label_refused() {
        let err = refuse_bad_hw_label("cloud-box").unwrap_err();
        assert!(err.contains("apple-"), "{err}");
        refuse_bad_hw_label("linux-avx2").unwrap();
        refuse_bad_hw_label("apple-m4-pro").unwrap();
        // android-* requires an Android host filesystem; this CI/dev Linux box must refuse.
        let err = refuse_bad_hw_label("android-arm64-emulator").unwrap_err();
        assert!(err.contains("android"), "{err}");
    }

    #[test]
    fn refuse_non_publish_role_smoke() {
        use crate::fixture::FixtureRole;
        let err = refuse_non_publish_role(FixtureRole::Smoke, 256, 16).unwrap_err();
        assert!(err.contains("fixture_role=smoke"), "{err}");
        refuse_non_publish_role(FixtureRole::Publish, 100_000, 1_000).unwrap();
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

    #[test]
    fn unattested_dedicated_refused() {
        let err = refuse_unless_dedicated_attested(false).unwrap_err();
        assert!(err.contains("VANEDB_COMPARE_DEDICATED"), "{err}");
        refuse_unless_dedicated_attested(true).unwrap();
    }
}
