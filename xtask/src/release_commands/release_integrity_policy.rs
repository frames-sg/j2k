// SPDX-License-Identifier: MIT OR Apache-2.0

mod markdown;

use markdown::content_lines as markdown_content_lines;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ReleaseIntegrityMode {
    PreCandidate,
    Publish,
}

impl ReleaseIntegrityMode {
    pub(super) fn parse(mut args: impl Iterator<Item = String>) -> Result<Self, String> {
        match (args.next().as_deref(), args.next()) {
            (None, None) => Ok(Self::PreCandidate),
            (Some("--publish"), None) => Ok(Self::Publish),
            (Some(argument), None) => Err(format!(
                "unknown release-integrity argument `{argument}`; expected --publish"
            )),
            _ => Err("release-integrity accepts only the optional --publish flag".to_string()),
        }
    }
}

/// `CHANGELOG.md` holds only the changes for the next release. Published
/// release notes live on GitHub Releases, so the file has exactly one
/// `## Unreleased` heading and no version headings. A publish also needs at
/// least one entry under `Unreleased`, so a release can't go out without notes.
pub(super) fn validate_changelog_state(
    changelog: &str,
    version: &str,
    mode: ReleaseIntegrityMode,
) -> Result<(), String> {
    let markdown_lines = markdown_content_lines(changelog);

    let unreleased_count = exact_line_count(&markdown_lines, UNRELEASED_HEADING);
    if unreleased_count != 1 {
        return Err(format!(
            "expected exactly one `{UNRELEASED_HEADING}` heading; found {unreleased_count}"
        ));
    }

    let version_headings = markdown_lines
        .iter()
        .copied()
        .filter(|line| is_version_heading(line))
        .collect::<Vec<_>>();
    if !version_headings.is_empty() {
        return Err(format!(
            "release sections belong in GitHub Releases, not the changelog; remove {}",
            version_headings
                .iter()
                .map(|heading| format!("`{heading}`"))
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }

    if mode == ReleaseIntegrityMode::Publish {
        let has_entry =
            markdown_section(&markdown_lines, UNRELEASED_HEADING).is_some_and(|section| {
                section
                    .iter()
                    .any(|line| !line.trim().is_empty() && markdown_heading_level(line).is_none())
            });
        if !has_entry {
            return Err(format!(
                "`{UNRELEASED_HEADING}` has no entries to publish as the {version} release notes"
            ));
        }
    }
    Ok(())
}

const UNRELEASED_HEADING: &str = "## Unreleased";

/// A level-2 heading naming a release, such as `## [0.7.0] - 2026-07-10`,
/// `## 0.7.0`, or `## v0.7.0`.
fn is_version_heading(line: &str) -> bool {
    line.strip_prefix("## ").is_some_and(|title| {
        title
            .trim_start_matches('[')
            .trim_start_matches('v')
            .starts_with(|character: char| character.is_ascii_digit())
    })
}

pub(super) fn validate_patch_provenance(provenance: &str) -> Result<(), String> {
    let mut errors = Vec::new();
    let markdown_lines = markdown_content_lines(provenance);
    let approval_heading_count = exact_line_count(&markdown_lines, "## Release approval");
    if approval_heading_count != 1 {
        errors.push(format!(
            "expected exactly one `## Release approval` section; found {approval_heading_count}"
        ));
    }

    if approval_heading_count == 1 {
        if let Some(approval) = markdown_section(&markdown_lines, "## Release approval") {
            match unique_field(&approval, "- Reviewer identity:") {
                Ok(reviewer) if reviewer_is_credible(reviewer) => {}
                Ok(_) => errors.push(
                    "Reviewer identity must name a real reviewer, not a placeholder".to_string(),
                ),
                Err(error) => errors.push(error),
            }
            match unique_field(&approval, "- Approval date:") {
                Ok(date) if is_calendar_date(date) => {}
                Ok(date) => errors.push(format!(
                    "Approval date `{date}` is not a calendar-valid YYYY-MM-DD date"
                )),
                Err(error) => errors.push(error),
            }
        } else {
            errors.push("release approval section body is missing".to_string());
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors.join("; "))
    }
}

fn markdown_section<'a>(lines: &[&'a str], heading: &str) -> Option<Vec<&'a str>> {
    let mut lines = lines.iter().copied().skip_while(|line| *line != heading);
    lines.next()?;
    Some(
        lines
            .take_while(|line| !matches!(markdown_heading_level(line), Some(1 | 2)))
            .collect(),
    )
}

fn markdown_heading_level(line: &str) -> Option<usize> {
    let level = line.bytes().take_while(|byte| *byte == b'#').count();
    if (1..=6).contains(&level)
        && line
            .as_bytes()
            .get(level)
            .is_none_or(u8::is_ascii_whitespace)
    {
        Some(level)
    } else {
        None
    }
}

fn exact_line_count(lines: &[&str], expected: &str) -> usize {
    lines.iter().filter(|line| **line == expected).count()
}

fn unique_field<'a>(lines: &[&'a str], label: &str) -> Result<&'a str, String> {
    let values = lines
        .iter()
        .filter_map(|line| line.strip_prefix(label))
        .map(|value| trim_code_span(value.trim()))
        .collect::<Vec<_>>();
    if values.len() != 1 {
        return Err(format!(
            "expected exactly one `{label} <value>` field; found {}",
            values.len()
        ));
    }
    Ok(values[0])
}

fn trim_code_span(value: &str) -> &str {
    value
        .strip_prefix('`')
        .and_then(|value| value.strip_suffix('`'))
        .unwrap_or(value)
        .trim()
}

fn reviewer_is_credible(reviewer: &str) -> bool {
    let normalized = reviewer.trim().to_ascii_lowercase();
    let exact_placeholders = [
        "pending",
        "tbd",
        "todo",
        "none",
        "n/a",
        "unknown",
        "unassigned",
        "not recorded",
    ];
    let placeholder_tokens = ["pending", "tbd", "todo", "none", "unknown", "unassigned"];
    reviewer.chars().filter(char::is_ascii_alphanumeric).count() >= 3
        && !exact_placeholders.contains(&normalized.as_str())
        && !normalized
            .split(|character: char| !character.is_ascii_alphanumeric())
            .any(|token| placeholder_tokens.contains(&token))
        && !normalized.contains("placeholder")
        && !normalized.contains("replace me")
}

pub(super) fn is_calendar_date(date: &str) -> bool {
    if date.len() != 10
        || !date.bytes().enumerate().all(|(index, byte)| match index {
            4 | 7 => byte == b'-',
            _ => byte.is_ascii_digit(),
        })
    {
        return false;
    }
    let Ok(year) = date[0..4].parse::<u32>() else {
        return false;
    };
    let Ok(month) = date[5..7].parse::<u32>() else {
        return false;
    };
    let Ok(day) = date[8..10].parse::<u32>() else {
        return false;
    };
    let leap_year =
        year.is_multiple_of(4) && (!year.is_multiple_of(100) || year.is_multiple_of(400));
    let days_in_month = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if leap_year => 29,
        2 => 28,
        _ => return false,
    };
    year != 0 && (1..=days_in_month).contains(&day)
}

#[cfg(test)]
mod tests;
