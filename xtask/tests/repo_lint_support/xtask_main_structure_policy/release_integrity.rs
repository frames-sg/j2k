// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{assert_pattern_checks, PatternCheck, XtaskSources};

pub(super) fn assert_ownership(sources: &XtaskSources) {
    assert_pattern_checks(&[
        PatternCheck::new(
            "release integrity pure policy ownership",
            &sources.release_integrity_policy,
        )
        .required(&[
            "mod markdown;",
            "enum ReleaseIntegrityMode",
            "fn validate_changelog_state(",
            "fn validate_patch_provenance(",
            "fn is_calendar_date(",
            "mod tests;",
        ]),
        PatternCheck::new(
            "release integrity Markdown evidence ownership",
            &sources.release_integrity_markdown,
        )
        .required(&[
            "mod html;",
            "fn content_lines(",
            "fn non_indented_line(",
            "fn opening_fence(",
            "fn closes_fence(",
        ]),
        PatternCheck::new(
            "release integrity HTML evidence ownership",
            &sources.release_integrity_html,
        )
        .required(&[
            "enum HtmlBlock",
            "fn opening(",
            "fn closes_on(",
            "fn starts_open_tag(",
            "fn starts_block_tag(",
        ]),
    ]);
}

pub(super) fn assert_regressions(sources: &XtaskSources) {
    let regressions = [
        sources.release_integrity_changelog_tests.as_str(),
        sources.release_integrity_metadata_tests.as_str(),
        sources.release_integrity_provenance_tests.as_str(),
    ]
    .join("\n");
    assert_pattern_checks(&[PatternCheck::new(
        "release integrity behavior regressions",
        &regressions,
    )
    .required(&[
        "pre_candidate_changelog_accepts_exact_unreleased_state",
        "publish_changelog_accepts_one_calendar_valid_dated_heading",
        "publish_changelog_rejects_provisional_or_duplicate_state",
        "fenced_only",
        "html_comment_only",
        "real_with_fenced_example",
        "real_after_div",
        "patch_provenance_accepts_structured_reviewer_and_calendar_date",
        "patch_provenance_rejects_missing_duplicate_or_placeholder_fields",
        "fenced_approval",
        "indented_fields",
        "html_comment_approval",
        "raw_html_approval",
        "div_html_approval",
    ])]);
}
