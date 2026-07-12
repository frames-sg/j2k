// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{is_calendar_date, validate_changelog_state, ReleaseIntegrityMode};
use super::VERSION;

#[test]
fn pre_candidate_changelog_accepts_exact_unreleased_state() {
    let changelog = "# Changelog\n\n## [Unreleased]\n\nStaged workspace version: `0.7.0`.\n";
    assert_eq!(
        validate_changelog_state(changelog, VERSION, ReleaseIntegrityMode::PreCandidate),
        Ok(())
    );
}

#[test]
fn pre_candidate_changelog_rejects_inline_examples_and_duplicates() {
    let inline = "At release use `## [0.7.0] - 2026-07-10`.\n";
    assert!(validate_changelog_state(inline, VERSION, ReleaseIntegrityMode::PreCandidate).is_err());

    let duplicated = "## [Unreleased]\n## [Unreleased]\nStaged workspace version: `0.7.0`.\n";
    assert!(
        validate_changelog_state(duplicated, VERSION, ReleaseIntegrityMode::PreCandidate).is_err()
    );

    let malformed = "## [Unreleased]\nStaged workspace version: `0.7.0`.\n## [0.7.0 - 2026-07-10\n";
    assert!(
        validate_changelog_state(malformed, VERSION, ReleaseIntegrityMode::PreCandidate).is_err()
    );

    let fenced_only = "```markdown\n## [Unreleased]\nStaged workspace version: `0.7.0`.\n```\n";
    assert!(
        validate_changelog_state(fenced_only, VERSION, ReleaseIntegrityMode::PreCandidate).is_err()
    );

    let html_comment_only = "<!--\n## [Unreleased]\nStaged workspace version: `0.7.0`.\n-->\n";
    assert!(validate_changelog_state(
        html_comment_only,
        VERSION,
        ReleaseIntegrityMode::PreCandidate,
    )
    .is_err());
}

#[test]
fn publish_changelog_accepts_one_calendar_valid_dated_heading() {
    let changelog = "# Changelog\n\n## [0.7.0] - 2028-02-29\n";
    assert_eq!(
        validate_changelog_state(changelog, VERSION, ReleaseIntegrityMode::Publish),
        Ok(())
    );
}

#[test]
fn publish_changelog_rejects_provisional_or_duplicate_state() {
    let provisional =
        "## [Unreleased]\nStaged workspace version: `0.7.0`.\n## [0.7.0] - 2026-07-10\n";
    assert!(validate_changelog_state(provisional, VERSION, ReleaseIntegrityMode::Publish).is_err());

    let duplicated = "## [0.7.0] - 2026-07-10\n## [0.7.0] - 2026-07-11\n";
    assert!(validate_changelog_state(duplicated, VERSION, ReleaseIntegrityMode::Publish).is_err());

    for code_only in [
        "```markdown\n## [0.7.0] - 2026-07-10\n```\n",
        "    ## [0.7.0] - 2026-07-10\n",
        "<!--\n## [0.7.0] - 2026-07-10\n-->\n",
        "<div class=\"example\">\n## [0.7.0] - 2026-07-10\n</div>\n",
    ] {
        assert!(
            validate_changelog_state(code_only, VERSION, ReleaseIntegrityMode::Publish).is_err()
        );
    }

    let real_with_fenced_example =
        "## [0.7.0] - 2026-07-10\n```markdown\n## [0.7.0] - 2026-07-11\n```\n";
    assert_eq!(
        validate_changelog_state(
            real_with_fenced_example,
            VERSION,
            ReleaseIntegrityMode::Publish,
        ),
        Ok(())
    );

    for tag in ["pre", "script", "style", "textarea"] {
        let raw_html = format!("<{tag}>\n## [0.7.0] - 2026-07-10\n</{tag}>\n");
        assert!(
            validate_changelog_state(&raw_html, VERSION, ReleaseIntegrityMode::Publish).is_err(),
            "{tag} HTML content must not count as a release heading"
        );
    }

    let real_after_div = "<div>\n## [0.7.0] - 2026-07-09\n</div>\n\n## [0.7.0] - 2026-07-10\n";
    assert_eq!(
        validate_changelog_state(real_after_div, VERSION, ReleaseIntegrityMode::Publish),
        Ok(())
    );
}

#[test]
fn calendar_dates_reject_invalid_months_days_and_non_leap_dates() {
    assert!(is_calendar_date("2028-02-29"));
    for invalid in [
        "2025-02-29",
        "2026-04-31",
        "2026-13-01",
        "2026-00-01",
        "0000-01-01",
        "2026-07-1",
    ] {
        assert!(!is_calendar_date(invalid), "{invalid} must be rejected");
    }
}

#[test]
fn release_integrity_mode_rejects_unknown_or_extra_arguments() {
    assert_eq!(
        ReleaseIntegrityMode::parse(std::iter::empty()),
        Ok(ReleaseIntegrityMode::PreCandidate)
    );
    assert_eq!(
        ReleaseIntegrityMode::parse(["--publish".to_string()].into_iter()),
        Ok(ReleaseIntegrityMode::Publish)
    );
    assert!(ReleaseIntegrityMode::parse(["--other".to_string()].into_iter()).is_err());
    assert!(ReleaseIntegrityMode::parse(
        ["--publish".to_string(), "extra".to_string()].into_iter()
    )
    .is_err());
}
