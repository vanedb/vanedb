# Recorded runs

JSON reports from dedicated-hardware runs go here. Name them
`{hardware_label}-{metric}-{YYYYMMDD}.json`.

**Evidence rule for #198:** before pasting into
[`../../COMPARISON.md`](../../COMPARISON.md), keep the matching JSON in this
directory (or attach it on the issue/PR). Re-render only via
`scripts/render_comparison_md.py`, which refuses forged params, incomplete
engine sets, and fixture hashes not listed in `fixtures/SHA256SUMS`.

Raw timings from CI or shared cloud runners must not be committed or pasted.
