//! Interleaved A-B comparison of two revisions.
//!
//! Raw bench numbers repeatedly produced false regressions that only manual
//! A-B-A reruns disproved, and one real regression that only an interleaved
//! rerun confirmed (#60). This module holds the parts of that procedure worth
//! testing: reading criterion's output, and deciding whether a delta exceeds
//! the noise the run itself measured.

use std::collections::BTreeMap;

/// The project's measured noise floor on dedicated hardware. A run whose own
/// spread comes out smaller than this is not evidence of a tighter machine,
/// so the floor is what a delta must clear.
pub const NOISE_FLOOR: f64 = 0.03;

/// One criterion measurement: the benchmark id and its reported median.
#[derive(Clone, Debug, PartialEq)]
pub struct Measurement {
    pub id: String,
    pub median_ns: f64,
}

/// One revision's results for one benchmark id, across rounds.
#[derive(Clone, Debug, PartialEq)]
pub struct Arm {
    pub median_ns: f64,
    /// (max - min) / min across rounds: the noise this run actually measured.
    pub spread: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// The delta exceeds both arms' spread and the documented floor.
    Significant,
    /// Indistinguishable from the run's own noise.
    Noise,
    /// The benchmark exists in only one revision.
    AOnly,
    BOnly,
}

/// One row of the comparison table.
#[derive(Clone, Debug, PartialEq)]
pub struct Comparison {
    pub id: String,
    pub a: Option<Arm>,
    pub b: Option<Arm>,
    pub delta: Option<f64>,
    pub verdict: Verdict,
}

/// Every `time:` measurement in one criterion run's output.
pub fn parse_run(output: &str) -> Vec<Measurement> {
    let mut measurements = Vec::new();
    let mut last_name = String::new();
    for line in output.lines() {
        // Criterion prints a long benchmark id alone, then its measurement on
        // the next line. Ids never contain whitespace, which is what separates
        // them from criterion's prose ("Found 2 outliers among ...").
        if !line.is_empty() && !line.contains(char::is_whitespace) {
            last_name = line.to_string();
            continue;
        }
        let Some((prefix, rest)) = line.split_once("time:") else {
            continue;
        };
        let id = if prefix.trim().is_empty() {
            last_name.clone()
        } else {
            prefix.trim().to_string()
        };
        if !id.is_empty() {
            if let Some(median_ns) = parse_median(rest) {
                measurements.push(Measurement { id, median_ns });
            }
        }
    }
    measurements
}

/// The middle of criterion's `[lower estimate upper]` triple, in nanoseconds.
fn parse_median(rest: &str) -> Option<f64> {
    let open = rest.find('[')?;
    let close = rest.find(']')?;
    let tokens: Vec<&str> = rest.get(open + 1..close)?.split_whitespace().collect();
    let [_, _, value, unit, _, _] = tokens[..] else {
        return None;
    };
    Some(value.parse::<f64>().ok()? * unit_ns(unit)?)
}

fn unit_ns(unit: &str) -> Option<f64> {
    Some(match unit {
        "ps" => 1e-3,
        "ns" => 1.0,
        // U+00B5 MICRO SIGN and U+03BC GREEK SMALL LETTER MU both appear in
        // the wild depending on the terminal criterion wrote through.
        "us" | "\u{b5}s" | "\u{3bc}s" => 1e3,
        "ms" => 1e6,
        "s" => 1e9,
        _ => return None,
    })
}

/// Median and spread over one arm's rounds.
pub fn summarize(samples: &[f64]) -> Arm {
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("criterion medians are finite"));
    let Some(&min) = sorted.first() else {
        return Arm {
            median_ns: 0.0,
            spread: 0.0,
        };
    };
    let max = sorted[sorted.len() - 1];
    let mid = sorted.len() / 2;
    let median = if sorted.len().is_multiple_of(2) {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    };
    let spread = if min > 0.0 { (max - min) / min } else { 0.0 };
    Arm {
        median_ns: median,
        spread,
    }
}

/// Rows for every benchmark seen in either revision, sorted by id.
pub fn compare(a_rounds: &[Vec<Measurement>], b_rounds: &[Vec<Measurement>]) -> Vec<Comparison> {
    let a = collect(a_rounds);
    let b = collect(b_rounds);
    let mut ids: Vec<&String> = a.keys().chain(b.keys()).collect();
    ids.sort();
    ids.dedup();

    ids.into_iter()
        .map(|id| {
            let arm_a = a.get(id).map(|s| summarize(s));
            let arm_b = b.get(id).map(|s| summarize(s));
            match (arm_a, arm_b) {
                (Some(x), Some(y)) => {
                    let delta = (y.median_ns - x.median_ns) / x.median_ns;
                    // A delta counts only when it clears the noise this run
                    // measured for itself, and never below the floor: a run
                    // that happened to repeat exactly has not proved the
                    // machine is quieter than it is known to be.
                    let threshold = x.spread.max(y.spread).max(NOISE_FLOOR);
                    let verdict = if delta.abs() > threshold {
                        Verdict::Significant
                    } else {
                        Verdict::Noise
                    };
                    Comparison {
                        id: id.clone(),
                        a: Some(x),
                        b: Some(y),
                        delta: Some(delta),
                        verdict,
                    }
                }
                (Some(x), None) => Comparison {
                    id: id.clone(),
                    a: Some(x),
                    b: None,
                    delta: None,
                    verdict: Verdict::AOnly,
                },
                (None, Some(y)) => Comparison {
                    id: id.clone(),
                    a: None,
                    b: Some(y),
                    delta: None,
                    verdict: Verdict::BOnly,
                },
                (None, None) => unreachable!("id came from one of the two maps"),
            }
        })
        .collect()
}

fn collect(rounds: &[Vec<Measurement>]) -> BTreeMap<String, Vec<f64>> {
    let mut by_id: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for round in rounds {
        for m in round {
            by_id.entry(m.id.clone()).or_default().push(m.median_ns);
        }
    }
    by_id
}

/// A duration in the unit a reader expects, as criterion prints it.
pub fn format_ns(ns: f64) -> String {
    let (value, unit) = if ns >= 1e9 {
        (ns / 1e9, "s")
    } else if ns >= 1e6 {
        (ns / 1e6, "ms")
    } else if ns >= 1e3 {
        (ns / 1e3, "us")
    } else {
        (ns, "ns")
    };
    format!("{value:.2} {unit}")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Real criterion output, including the shapes that break naive parsers:
    /// a wrapped benchmark name, throughput and change lines, outlier notes.
    const RUN: &str = "\
Benchmarking mmap_build/cpp: Warming up for 3.0000 s
mmap_build/cpp          time:   [4.7941 ms 4.8888 ms 4.9125 ms]
                        change: [-1.2345% +0.1234% +1.5678%] (p = 0.42 > 0.05)
                        No change in performance detected.
Found 2 outliers among 10 measurements (20.00%)
  1 (10.00%) high mild
store_search/n=10000/cpp
                        time:   [79.527 us 79.732 us 79.783 us]
store_add/n=10000/cpp   time:   [771.40 \u{b5}s 775.36 \u{b5}s 776.35 \u{b5}s]
                        thrpt:  [12.881 Melem/s 12.897 Melem/s 12.963 Melem/s]
l2_sq/dim=128/rs/128    time:   [16.472 ns 16.706 ns 16.764 ns]
";

    #[test]
    fn reads_the_median_of_each_measurement() {
        let m = parse_run(RUN);
        assert_eq!(m[0].id, "mmap_build/cpp");
        assert!(
            (m[0].median_ns - 4_888_800.0).abs() < 1e-6,
            "got {}",
            m[0].median_ns
        );
        assert_eq!(m[3].id, "l2_sq/dim=128/rs/128");
        assert!(
            (m[3].median_ns - 16.706).abs() < 1e-9,
            "got {}",
            m[3].median_ns
        );
    }

    #[test]
    fn a_name_wrapped_onto_its_own_line_still_belongs_to_its_measurement() {
        let m = parse_run(RUN);
        assert_eq!(m[1].id, "store_search/n=10000/cpp");
        assert!(
            (m[1].median_ns - 79_732.0).abs() < 1e-6,
            "got {}",
            m[1].median_ns
        );
    }

    #[test]
    fn throughput_change_and_outlier_lines_are_not_measurements() {
        assert_eq!(parse_run(RUN).len(), 4);
    }

    #[test]
    fn both_spellings_of_microseconds_are_understood() {
        let m = parse_run(RUN);
        assert!(
            (m[2].median_ns - 775_360.0).abs() < 1e-6,
            "got {}",
            m[2].median_ns
        );
    }

    #[test]
    fn an_arm_reports_its_median_and_the_noise_it_measured() {
        let arm = summarize(&[100.0, 110.0]);
        assert!((arm.median_ns - 105.0).abs() < 1e-9);
        assert!((arm.spread - 0.1).abs() < 1e-9, "spread {}", arm.spread);
    }

    fn rounds(values: &[f64]) -> Vec<Vec<Measurement>> {
        values
            .iter()
            .map(|&v| {
                vec![Measurement {
                    id: "op".into(),
                    median_ns: v,
                }]
            })
            .collect()
    }

    #[test]
    fn a_delta_inside_the_measured_spread_is_noise() {
        // A swings 20%; B sits 10% above A's median — inside that swing.
        let rows = compare(&rounds(&[100.0, 120.0]), &rounds(&[121.0, 121.0]));
        assert_eq!(rows[0].verdict, Verdict::Noise);
    }

    #[test]
    fn a_delta_beyond_both_spreads_is_significant() {
        let rows = compare(&rounds(&[100.0, 101.0]), &rounds(&[200.0, 202.0]));
        assert_eq!(rows[0].verdict, Verdict::Significant);
        assert!((rows[0].delta.unwrap() - 1.0).abs() < 0.02);
    }

    #[test]
    fn a_quiet_run_still_cannot_claim_less_noise_than_the_documented_floor() {
        // Both arms perfectly repeatable, delta 1%: below the 3% floor.
        let rows = compare(&rounds(&[100.0, 100.0]), &rounds(&[101.0, 101.0]));
        assert_eq!(rows[0].verdict, Verdict::Noise);
    }

    #[test]
    fn a_benchmark_missing_from_one_revision_is_flagged_not_compared() {
        let a = vec![vec![Measurement {
            id: "only_in_a".into(),
            median_ns: 1.0,
        }]];
        let b = vec![vec![Measurement {
            id: "only_in_b".into(),
            median_ns: 1.0,
        }]];
        let rows = compare(&a, &b);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].verdict, Verdict::AOnly);
        assert_eq!(rows[1].verdict, Verdict::BOnly);
        assert!(rows[0].delta.is_none());
    }

    #[test]
    fn durations_print_in_the_unit_a_reader_expects() {
        assert_eq!(format_ns(16.706), "16.71 ns");
        assert_eq!(format_ns(79_732.0), "79.73 us");
        assert_eq!(format_ns(4_888_800.0), "4.89 ms");
        assert_eq!(format_ns(1_196_000_000.0), "1.20 s");
    }
}
