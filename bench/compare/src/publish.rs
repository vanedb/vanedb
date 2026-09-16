//! Publish-path policy for COMPARISON.md (independent of fixture role).

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
}
