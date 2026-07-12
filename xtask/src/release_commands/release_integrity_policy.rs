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

pub(super) fn validate_changelog_state(
    changelog: &str,
    version: &str,
    mode: ReleaseIntegrityMode,
) -> Result<(), String> {
    let markdown_lines = markdown_content_lines(changelog);
    let version_marker = format!("## [{version}");
    let version_headings = markdown_lines
        .iter()
        .copied()
        .filter(|line| line.starts_with(&version_marker))
        .collect::<Vec<_>>();

    if mode == ReleaseIntegrityMode::Publish || !version_headings.is_empty() {
        return validate_publish_changelog(&markdown_lines, version, &version_headings);
    }

    let unreleased_count = exact_line_count(&markdown_lines, "## [Unreleased]");
    let staged_version = format!("Staged workspace version: `{version}`.");
    let staged_count = exact_line_count(&markdown_lines, &staged_version);
    if unreleased_count == 1 && staged_count == 1 {
        Ok(())
    } else {
        Err(format!(
            "pre-candidate state requires exactly one `## [Unreleased]` heading and one `{staged_version}` line; found {unreleased_count} and {staged_count}"
        ))
    }
}

fn validate_publish_changelog(
    markdown_lines: &[&str],
    version: &str,
    version_headings: &[&str],
) -> Result<(), String> {
    if version_headings.len() != 1 {
        return Err(format!(
            "publish state requires exactly one `## [{version}] - YYYY-MM-DD` heading; found {}",
            version_headings.len()
        ));
    }

    let prefix = format!("## [{version}] - ");
    let Some(date) = version_headings[0].strip_prefix(&prefix) else {
        return Err(format!(
            "publish heading must be exactly `## [{version}] - YYYY-MM-DD`"
        ));
    };
    if !is_calendar_date(date) {
        return Err(format!(
            "publish heading date `{date}` is not a calendar-valid YYYY-MM-DD date"
        ));
    }

    let staged_version = format!("Staged workspace version: `{version}`.");
    let unreleased_count = exact_line_count(markdown_lines, "## [Unreleased]");
    let staged_count = exact_line_count(markdown_lines, &staged_version);
    if unreleased_count != 0 || staged_count != 0 {
        return Err(format!(
            "publish state must not retain provisional Unreleased or staged-version markers; found {unreleased_count} and {staged_count}"
        ));
    }
    Ok(())
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
