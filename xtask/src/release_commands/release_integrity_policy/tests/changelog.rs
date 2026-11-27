// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{is_calendar_date, validate_changelog_state, ReleaseIntegrityMode};
use super::VERSION;

const WITH_ENTRY: &str = "# Changelog\n\nChanges for the next release.\n\n## Unreleased\n\n### Decoding\n\n- Faster region decode.\n";
const EMPTY: &str =
    "# Changelog\n\nChanges for the next release.\n\n## Unreleased\n\n### Decoding\n";

#[test]
fn changelog_accepts_one_unreleased_section_in_both_modes() {
    for mode in [
        ReleaseIntegrityMode::PreCandidate,
        ReleaseIntegrityMode::Publish,
    ] {
        assert_eq!(validate_changelog_state(WITH_ENTRY, VERSION, mode), Ok(()));
    }
}

#[test]
fn empty_unreleased_section_is_fine_between_releases_but_cannot_publish() {
    assert_eq!(
        validate_changelog_state(EMPTY, VERSION, ReleaseIntegrityMode::PreCandidate),
        Ok(())
    );
    let error = validate_changelog_state(EMPTY, VERSION, ReleaseIntegrityMode::Publish)
        .expect_err("publishing needs release notes");
    assert!(error.contains("0.7.0"), "unexpected error: {error}");

    let headings_then_next_section = "## Unreleased\n### Decoding\n## Notes\n- not release notes\n";
    assert!(validate_changelog_state(
        headings_then_next_section,
        VERSION,
        ReleaseIntegrityMode::Publish,
    )
    .is_err());
}

#[test]
fn changelog_rejects_missing_duplicate_or_hidden_unreleased_headings() {
    for invalid in [
        "# Changelog\n",
        "## [Unreleased]\n- Old-style heading.\n",
        "## Unreleased\n- One.\n## Unreleased\n- Two.\n",
        "```markdown\n## Unreleased\n- Fenced.\n```\n",
        "<!--\n## Unreleased\n- Commented.\n-->\n",
        "    ## Unreleased\n",
    ] {
        for mode in [
            ReleaseIntegrityMode::PreCandidate,
            ReleaseIntegrityMode::Publish,
        ] {
            assert!(
                validate_changelog_state(invalid, VERSION, mode).is_err(),
                "{invalid:?} must be rejected in {mode:?}"
            );
        }
    }
}

#[test]
fn changelog_rejects_release_sections() {
    for release in [
        "## [0.7.0] - 2026-07-10",
        "## [0.6.2] - 2026-06-28",
        "## 0.7.0",
        "## v0.7.0",
    ] {
        let changelog = format!("{WITH_ENTRY}\n{release}\n\n- Shipped.\n");
        for mode in [
            ReleaseIntegrityMode::PreCandidate,
            ReleaseIntegrityMode::Publish,
        ] {
            let error = validate_changelog_state(&changelog, VERSION, mode)
                .expect_err("release sections belong in GitHub Releases");
            assert!(error.contains(release), "unexpected error: {error}");
        }
    }

    let fenced_example = format!("{WITH_ENTRY}\n```markdown\n## [0.7.0] - 2026-07-10\n```\n");
    assert_eq!(
        validate_changelog_state(&fenced_example, VERSION, ReleaseIntegrityMode::Publish),
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
