#!/usr/bin/env python3
"""Replace a *Pending.* COMPARISON.md slot with gated publish markdown.

Reads one or more harness JSON reports (same gates as render_comparison_md.py),
maps hardware_label + metric to the matching Results subsection, and replaces
that subsection's *Pending.* body. Refuses to overwrite a non-Pending slot
unless --force. Never invents timings — JSON must already pass publish gates.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

# Reuse the renderer's gates + markdown body.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_comparison_md as render  # noqa: E402

COMPARISON_MD = Path(__file__).resolve().parents[2] / "COMPARISON.md"

# (section heading fragment, subsection) — first match wins under ## Results.
HW_SECTION = (
    ("apple-", "### Apple Silicon (dedicated laptop)"),
    ("linux-avx2", "### Linux x86-64 AVX2 (dedicated box)"),
    ("android-arm64-", "### Android ARM64 (device or emulator)"),
)
# Only these ### headings close a HW section — rendered machine titles also use
# ### and must NOT truncate the section when Cosine is filled before L2.
HW_HEADINGS = tuple(heading for _, heading in HW_SECTION)

METRIC_SUB = {
    "cosine": "#### Cosine",
    "l2": "#### Squared L2",
}


def section_for(hw: str, metric: str) -> tuple[str, str]:
    for prefix, heading in HW_SECTION:
        if hw.startswith(prefix):
            sub = METRIC_SUB.get(metric)
            if sub is None:
                raise SystemExit(f"unknown metric {metric!r}")
            return heading, sub
    raise SystemExit(
        f"hardware_label={hw!r} does not map to a COMPARISON.md Results section"
    )


def _end_at_known_hw_or_h2(text: str) -> int:
    """Index of next HW-class ### sibling or ## heading; else len(text).

    Must not treat arbitrary `###` (rendered machine titles inside a slot) as
    section boundaries — that broke Cosine-then-L2 fills.
    """
    end = len(text)
    idx = text.find("\n## ")
    if idx != -1:
        end = min(end, idx)
    for heading in HW_HEADINGS:
        needle = "\n" + heading
        idx = text.find(needle)
        if idx != -1:
            end = min(end, idx)
    return end


def render_body(report: dict) -> str:
    """Run render gates; return markdown body (no trailing extra blank lines)."""
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
        json.dump(report, tmp)
        tmp_path = tmp.name
    old_argv = sys.argv
    try:
        buf = io.StringIO()
        sys.argv = [render.__file__, tmp_path]
        with redirect_stdout(buf):
            rc = render.main()
        if rc != 0:
            raise SystemExit(rc)
        return buf.getvalue().rstrip() + "\n"
    finally:
        sys.argv = old_argv
        Path(tmp_path).unlink(missing_ok=True)


def replace_slot(text: str, section: str, subsection: str, body: str, force: bool) -> str:
    if section not in text:
        raise SystemExit(f"COMPARISON.md missing section {section!r}")
    if section not in HW_HEADINGS:
        raise SystemExit(f"section {section!r} is not a known HW Results heading")
    sec_start = text.index(section)
    rest = text[sec_start + len(section) :]
    next_sec = _end_at_known_hw_or_h2(rest)
    sec_block = rest[:next_sec]
    if subsection not in sec_block:
        raise SystemExit(f"section {section!r} missing subsection {subsection!r}")
    sub_start = sec_block.index(subsection)
    after_sub = sec_block[sub_start + len(subsection) :]
    # End this metric slot at the next #### metric heading, or HW/## boundary.
    next_sub = len(after_sub)
    idx = after_sub.find("\n#### ")
    if idx != -1:
        next_sub = min(next_sub, idx)
    next_sub = min(next_sub, _end_at_known_hw_or_h2(after_sub))
    old_body = after_sub[:next_sub]
    stripped = old_body.strip()
    if stripped != "*Pending.*" and not force:
        raise SystemExit(
            f"refusing to overwrite non-Pending slot under {section} / {subsection} "
            f"(got {stripped[:60]!r}…); pass --force to replace"
        )
    trailing_nl = ""
    if next_sub < len(after_sub):
        trailing_nl = "\n"
    new_sub_block = subsection + "\n\n" + body.rstrip() + "\n" + trailing_nl
    new_sec_block = sec_block[:sub_start] + new_sub_block + after_sub[next_sub:]
    return text[:sec_start] + section + new_sec_block + rest[next_sec:]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "json_paths",
        nargs="+",
        type=Path,
        help="publish JSON from record_publish_run.sh / compare --json-out",
    )
    ap.add_argument(
        "--comparison",
        type=Path,
        default=COMPARISON_MD,
        help=f"path to COMPARISON.md (default: {COMPARISON_MD})",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="overwrite a slot that is not *Pending.*",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="print which slots would fill; do not write",
    )
    args = ap.parse_args()
    text = args.comparison.read_text()
    for path in args.json_paths:
        report = json.loads(path.read_text())
        hw = report.get("hardware_label", "")
        metric = report.get("metric", "")
        section, subsection = section_for(hw, metric)
        body = render_body(report)
        print(f"fill {path} → {section} / {subsection}", file=sys.stderr)
        if args.dry_run:
            continue
        text = replace_slot(text, section, subsection, body, force=args.force)
    if args.dry_run:
        return 0
    args.comparison.write_text(text)
    print(f"wrote {args.comparison}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
