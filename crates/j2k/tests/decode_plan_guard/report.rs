// SPDX-License-Identifier: MIT OR Apache-2.0

//! The decode guard's table, its comparison with the baseline, and the
//! summary of a run.

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;

use super::decode::{Budget, Measured, Outcome, Output, Request};
use super::fixtures::{Fixture, COMPLEMENTARY_STYLES};
use super::{Case, NOISY_REPEATS, POOL_REPEATS, POOL_THREADS, SLACK_BYTES};

pub(super) struct Row {
    pub(super) id: String,
    pub(super) measured: Measured,
    /// Highest less lowest peak over the repeats.
    pub(super) spread: usize,
    pub(super) expected: Outcome,
}

/// The table in the baseline's format: a header and one line per case.
pub(super) fn table(rows: &[Row]) -> String {
    let mut table = String::new();
    let _ = writeln!(
        table,
        "# Decode guard table: case, outcome (ok:<FNV-1a 64 of the output> or err:<error class>),"
    );
    let _ = writeln!(
        table,
        "# peak heap bytes above the bytes live before the decode (native planes the decode returns are"
    );
    let _ = writeln!(
        table,
        "# inside the peak; a caller's buffer is not), peak spread over repeats, output bytes. Pool"
    );
    let _ = writeln!(
        table,
        "# cases keep their highest peak of {POOL_REPEATS} runs, or of {NOISY_REPEATS} on large fixtures or when those"
    );
    let _ = writeln!(
        table,
        "# spread by more than the slack. A later peak passes unless it exceeds this peak by more than this spread plus"
    );
    let _ = writeln!(
        table,
        "# 64 KiB plus 1/64 of this peak. Case: fixture/request/output/parallelism/budget."
    );
    let _ = writeln!(
        table,
        "# target {}; routes and bytes covered by crates/j2k-native/src/tests/multi_segment.rs, not heap peaks: {}",
        target(),
        COMPLEMENTARY_STYLES.join(", ")
    );
    let _ = writeln!(
        table,
        "# Accepted pool-scratch change: two RGB16 row cases add at most 74 KiB and six TERMALL full-image cases"
    );
    let _ = writeln!(
        table,
        "# add at most 105 KiB because pool routes reserve one bounded Tier-1 scratch owner per worker up front."
    );
    for row in rows {
        let _ = writeln!(
            table,
            "{:<64} {:<21} {:>10} {:>8} {:>8}",
            row.id,
            row.measured.outcome.to_string(),
            row.measured.peak,
            row.spread,
            row.measured.output
        );
    }
    table
}

/// The target and profile a table is measured on, as its header names them.
fn target() -> String {
    format!(
        "{}-{}, {} profile",
        std::env::consts::ARCH,
        std::env::consts::OS,
        profile()
    )
}

const fn profile() -> &'static str {
    if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    }
}

#[derive(Debug)]
struct Baseline {
    target: String,
    /// Outcome, peak and peak spread by case.
    rows: HashMap<String, (String, usize, usize)>,
}

impl Baseline {
    /// This target's baseline: peaks differ between targets (SIMD widths
    /// pad buffers differently), bytes do not.
    fn path() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/decode_plan_guard")
            .join(format!(
                "baseline-{}-{}.txt",
                std::env::consts::ARCH,
                std::env::consts::OS
            ))
    }

    fn read() -> Result<Option<Self>, Vec<String>> {
        match std::fs::read_to_string(Self::path()) {
            Ok(text) => Self::parse(&text).map(Some),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(vec![format!(
                "could not read {}: {error}",
                Self::path().display()
            )]),
        }
    }

    fn parse(text: &str) -> Result<Self, Vec<String>> {
        let mut target = None;
        let mut rows = HashMap::new();
        let mut errors = Vec::new();
        for (index, line) in text.lines().enumerate() {
            let line_number = index + 1;
            if let Some(value) = line.strip_prefix("# target ") {
                let value = value.split(';').next().unwrap_or_default().to_string();
                if target.replace(value).is_some() {
                    errors.push(format!("line {line_number}: duplicate target header"));
                }
            } else if !line.starts_with('#') && !line.trim().is_empty() {
                match parse_baseline_row(line_number, line) {
                    Ok((id, accepted)) => {
                        if rows.insert(id.clone(), accepted).is_some() {
                            errors.push(format!("line {line_number}: duplicate case {id}"));
                        }
                    }
                    Err(error) => errors.push(error),
                }
            }
        }
        let Some(target) = target else {
            errors.push("missing target header".to_string());
            return Err(errors);
        };
        if errors.is_empty() {
            Ok(Self { target, rows })
        } else {
            Err(errors)
        }
    }
}

fn parse_baseline_row(
    line_number: usize,
    line: &str,
) -> Result<(String, (String, usize, usize)), String> {
    let mut fields = line.split_whitespace();
    let id = fields
        .next()
        .ok_or_else(|| format!("line {line_number}: missing case id"))?;
    let outcome = fields
        .next()
        .ok_or_else(|| format!("line {line_number} case {id}: missing outcome"))?;
    let peak = fields
        .next()
        .ok_or_else(|| format!("line {line_number} case {id}: missing peak"))?
        .parse()
        .map_err(|_| format!("line {line_number} case {id}: invalid peak"))?;
    let spread = fields
        .next()
        .ok_or_else(|| format!("line {line_number} case {id}: missing spread"))?
        .parse()
        .map_err(|_| format!("line {line_number} case {id}: invalid spread"))?;
    fields
        .next()
        .ok_or_else(|| format!("line {line_number} case {id}: missing output bytes"))?
        .parse::<usize>()
        .map_err(|_| format!("line {line_number} case {id}: invalid output bytes"))?;
    if fields.next().is_some() {
        return Err(format!("line {line_number} case {id}: extra fields"));
    }
    let valid_outcome = outcome
        .strip_prefix("ok:")
        .is_some_and(|hash| hash.len() == 16 && hash.bytes().all(|byte| byte.is_ascii_hexdigit()))
        || outcome.strip_prefix("err:").is_some_and(|class| {
            !class.is_empty() && class.bytes().all(|byte| byte.is_ascii_alphanumeric())
        });
    if !valid_outcome {
        return Err(format!("line {line_number} case {id}: invalid outcome"));
    }
    Ok((id.to_string(), (outcome.to_string(), peak, spread)))
}

/// What the run found besides byte mismatches, for the summary.
#[derive(Default)]
pub(super) struct Findings {
    pub(super) outcome_changes: Vec<String>,
    pub(super) baseline_changes: Vec<String>,
    pub(super) peaks_above_baseline: Vec<String>,
    pub(super) missing_from_baseline: Vec<String>,
    pub(super) baseline_errors: Vec<String>,
}

impl Findings {
    pub(super) fn is_empty(&self) -> bool {
        self.outcome_changes.is_empty()
            && self.baseline_changes.is_empty()
            && self.peaks_above_baseline.is_empty()
            && self.missing_from_baseline.is_empty()
            && self.baseline_errors.is_empty()
    }

    pub(super) fn failures(&self) -> Vec<String> {
        [
            ("outcome differs from planes", &self.outcome_changes),
            ("outcome differs from baseline", &self.baseline_changes),
            ("peak exceeds baseline", &self.peaks_above_baseline),
            ("case missing from baseline", &self.missing_from_baseline),
            ("baseline error", &self.baseline_errors),
        ]
        .into_iter()
        .flat_map(|(kind, lines)| lines.iter().map(move |line| format!("{kind}: {line}")))
        .collect()
    }
}

fn committed_baseline_target() -> Option<String> {
    matches!(
        (std::env::consts::ARCH, std::env::consts::OS),
        ("aarch64", "macos") | ("x86_64", "linux")
    )
    .then(|| {
        format!(
            "{}-{}, release profile",
            std::env::consts::ARCH,
            std::env::consts::OS
        )
    })
}

fn compare_rows(rows: &[Row], baseline: &Baseline, findings: &mut Findings, peaks_compare: bool) {
    let mut measured_ids = HashSet::new();
    for row in rows {
        if !measured_ids.insert(row.id.as_str()) {
            findings
                .baseline_errors
                .push(format!("duplicate measured case {}", row.id));
        }
        let Some(&(ref outcome, peak, spread)) = baseline.rows.get(&row.id) else {
            findings.missing_from_baseline.push(row.id.clone());
            continue;
        };
        let now = row.measured.outcome.to_string();
        if &now != outcome {
            findings
                .baseline_changes
                .push(format!("{}: {outcome} -> {now}", row.id));
        }
        let limit = peak
            .saturating_add(spread)
            .saturating_add(SLACK_BYTES)
            .saturating_add(peak / 64);
        if peaks_compare && row.measured.peak > limit {
            findings.peaks_above_baseline.push(format!(
                "{}: {} -> {} bytes (limit {limit})",
                row.id, peak, row.measured.peak,
            ));
        }
    }

    let mut stale: Vec<_> = baseline
        .rows
        .keys()
        .filter(|id| !measured_ids.contains(id.as_str()))
        .collect();
    stale.sort_unstable();
    findings.baseline_errors.extend(
        stale
            .into_iter()
            .map(|id| format!("baseline case {id} was not measured")),
    );
}

pub(super) fn compare_with_baseline(rows: &[Row], findings: &mut Findings) -> String {
    let baseline = match Baseline::read() {
        Ok(Some(baseline)) => baseline,
        Ok(None) => {
            if committed_baseline_target().is_some() {
                findings.baseline_errors.push(format!(
                    "missing committed baseline at {}",
                    Baseline::path().display()
                ));
            }
            return format!("no baseline at {}", Baseline::path().display());
        }
        Err(errors) => {
            findings.baseline_errors.extend(errors);
            return format!("invalid baseline at {}", Baseline::path().display());
        }
    };
    if let Some(expected) = committed_baseline_target() {
        if baseline.target != expected {
            findings.baseline_errors.push(format!(
                "baseline target is {}, expected {expected}",
                baseline.target
            ));
        }
    }
    let target = target();
    let peaks_compare = baseline.target == target;
    compare_rows(rows, &baseline, findings, peaks_compare);
    let measured_on = baseline.target;
    if peaks_compare {
        format!("baseline of {measured_on}: peaks compared")
    } else {
        format!("baseline of {measured_on}, this run {target}: peaks not compared")
    }
}

/// Per fixture: the planes floor the native decoder plans before a
/// full-image decode, and the measured peaks of the 8-bit full-image decode
/// on the planes and with the planes budget at zero.
fn estimates(fixtures: &[Fixture], rows: &[Row], cases: &[Case]) -> String {
    let mut out = String::from(
        "fixture                                       planes floor   planes serial/pool4   budget-0 serial/pool4\n",
    );
    for (index, fixture) in fixtures.iter().enumerate() {
        let floor = j2k_native::Image::new(&fixture.bytes, &j2k_native::DecodeSettings::default())
            .ok()
            .and_then(|image| image.full_decode_planes_floor(0).ok().flatten());
        let peak = |threads, budget| {
            cases
                .iter()
                .zip(rows)
                .find(|(case, _)| {
                    case.fixture == index
                        && case.request == Request::Full
                        && case.output == Output::Packed(fixture.formats[0])
                        && case.threads == threads
                        && case.budget == budget
                })
                .map_or(0, |(_, row)| row.measured.peak)
        };
        let _ = writeln!(
            out,
            "{:<45} {:>12}   {:>9} / {:>9}   {:>9} / {:>9}",
            fixture.name,
            floor.map_or_else(|| "-".to_string(), |floor| floor.to_string()),
            peak(1, Budget::Planes(usize::MAX)),
            peak(POOL_THREADS, Budget::Planes(usize::MAX)),
            peak(1, Budget::Planes(0)),
            peak(POOL_THREADS, Budget::Planes(0)),
        );
    }
    out
}

pub(super) fn write_report(table: &str, summary: &str) {
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("decode_plan_guard");
    std::fs::create_dir_all(&dir).expect("create the guard report directory");
    std::fs::write(dir.join("table.txt"), table).expect("write the guard table");
    std::fs::write(dir.join("summary.txt"), summary).expect("write the guard summary");
    println!("{table}\n{summary}\nwritten to {}", dir.display());
}

/// What one run found, for its summary.
pub(super) struct Report<'a> {
    pub(super) rows: &'a [Row],
    pub(super) fixtures: &'a [Fixture],
    pub(super) cases: &'a [Case],
    pub(super) mismatches: &'a [String],
    pub(super) findings: &'a Findings,
}

impl Report<'_> {
    pub(super) fn summary(&self, baseline: &str, seconds: f64) -> String {
        let (rows, findings) = (self.rows, self.findings);
        let errors = rows
            .iter()
            .filter(|row| matches!(row.measured.outcome, Outcome::Error(_)))
            .count();
        let planes_errors = rows
            .iter()
            .filter(|row| matches!(row.expected, Outcome::Error(_)))
            .count();
        let mut summary = format!(
            "{} cases over {} fixtures in {seconds:.1} s: {} decoded, {errors} failed \
             ({planes_errors} where the planes route fails); {} byte mismatches, {} outcomes other \
             than the planes route's\n{baseline}: {} outcome changes, {} peaks above it, {} cases \
             not in it, {} baseline errors\n\n",
            rows.len(),
            self.fixtures.len(),
            rows.len() - errors,
            self.mismatches.len(),
            findings.outcome_changes.len(),
            findings.baseline_changes.len(),
            findings.peaks_above_baseline.len(),
            findings.missing_from_baseline.len(),
            findings.baseline_errors.len(),
        );
        for (title, lines) in [
            ("byte mismatches", self.mismatches),
            (
                "outcomes other than the planes route's",
                &findings.outcome_changes,
            ),
            (
                "outcome changes from the baseline",
                &findings.baseline_changes,
            ),
            ("peaks above the baseline", &findings.peaks_above_baseline),
            (
                "cases missing from the baseline",
                &findings.missing_from_baseline,
            ),
            ("baseline errors", &findings.baseline_errors),
        ] {
            if !lines.is_empty() {
                let _ = writeln!(summary, "{title}:\n  {}\n", lines.join("\n  "));
            }
        }
        summary.push_str(&estimates(self.fixtures, rows, self.cases));
        summary
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(id: &str, outcome: Outcome, peak: usize) -> Row {
        Row {
            id: id.to_string(),
            measured: Measured {
                outcome: outcome.clone(),
                peak,
                output: 0,
            },
            spread: 0,
            expected: outcome,
        }
    }

    #[test]
    fn baseline_parser_rejects_malformed_and_duplicate_rows() {
        let text =
            "# target aarch64-macos, release profile; code-block styles covered by j2k-native unit tests: BYPASS, RESET\n\
                    case-a ok:0000000000000001 100 2 16\n\
                    case-a ok:0000000000000001 100 2 16\n\
                    missing-fields ok:0000000000000002 100\n\
                    bad-number ok:0000000000000003 many 2 16\n";

        let errors = Baseline::parse(text).expect_err("malformed baseline must fail");

        assert!(errors
            .iter()
            .any(|error| error.contains("duplicate case case-a")));
        assert!(errors.iter().any(|error| error.contains("missing-fields")));
        assert!(errors.iter().any(|error| error.contains("bad-number")));
    }

    #[test]
    fn committed_baselines_parse_and_cover_the_same_cases() {
        let arm = Baseline::parse(include_str!("baseline-aarch64-macos.txt"))
            .expect("ARM64 baseline parses");
        let x86 = Baseline::parse(include_str!("baseline-x86_64-linux.txt"))
            .expect("x86-64 baseline parses");

        assert_eq!(arm.rows.len(), 1_444);
        assert_eq!(
            arm.rows.keys().collect::<HashSet<_>>(),
            x86.rows.keys().collect()
        );
    }

    #[test]
    fn comparison_rejects_outcome_peak_and_case_set_drift() {
        let accepted_peak = 1_000;
        let accepted_spread = 50;
        let limit = accepted_peak + accepted_spread + SLACK_BYTES + accepted_peak / 64;
        let baseline = Baseline {
            target: target(),
            rows: HashMap::from([
                (
                    "changed".to_string(),
                    (
                        "ok:0000000000000001".to_string(),
                        accepted_peak,
                        accepted_spread,
                    ),
                ),
                (
                    "at-limit".to_string(),
                    (
                        "ok:0000000000000002".to_string(),
                        accepted_peak,
                        accepted_spread,
                    ),
                ),
                (
                    "stale".to_string(),
                    (
                        "ok:0000000000000003".to_string(),
                        accepted_peak,
                        accepted_spread,
                    ),
                ),
            ]),
        };
        let rows = [
            row(
                "changed",
                Outcome::Error("Unsupported".to_string()),
                limit + 1,
            ),
            row(
                "changed",
                Outcome::Error("Unsupported".to_string()),
                limit + 1,
            ),
            row("at-limit", Outcome::Bytes(2), limit),
            row("missing", Outcome::Bytes(4), 0),
        ];
        let mut findings = Findings::default();

        compare_rows(&rows, &baseline, &mut findings, true);

        assert_eq!(findings.baseline_changes.len(), 2);
        assert_eq!(findings.peaks_above_baseline.len(), 2);
        assert_eq!(findings.missing_from_baseline, ["missing".to_string()]);
        assert!(findings
            .baseline_errors
            .iter()
            .any(|error| error.contains("duplicate measured case changed")));
        assert!(findings
            .baseline_errors
            .iter()
            .any(|error| error.contains("baseline case stale was not measured")));
    }
}
