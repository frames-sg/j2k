// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::validate_patch_provenance;

#[test]
fn patch_provenance_accepts_structured_reviewer_and_calendar_date() {
    let provenance = "## Release approval\n\n- Reviewer identity: `@release-reviewer`\n- Approval date: `2026-07-10`\n";
    assert_eq!(validate_patch_provenance(provenance), Ok(()));
}

#[test]
fn patch_provenance_rejects_missing_duplicate_or_placeholder_fields() {
    let missing = "## Release approval\n- Reviewer identity: `@release-reviewer`\n";
    assert!(validate_patch_provenance(missing).is_err());

    let duplicate = "## Release approval\n- Reviewer identity: `@one`\n- Reviewer identity: `@two`\n- Approval date: `2026-07-10`\n";
    assert!(validate_patch_provenance(duplicate).is_err());

    let placeholder =
        "## Release approval\n- Reviewer identity: `PENDING`\n- Approval date: `PENDING`\n";
    assert!(validate_patch_provenance(placeholder).is_err());

    let disguised_placeholder = "## Release approval\n- Reviewer identity: `pending-reviewer`\n- Approval date: `2026-07-10`\n";
    assert!(validate_patch_provenance(disguised_placeholder).is_err());

    let fields_outside_section = "- Reviewer identity: `@release-reviewer`\n- Approval date: `2026-07-10`\n\n## Release approval\n\nNo approval fields here.\n";
    assert!(validate_patch_provenance(fields_outside_section).is_err());

    let fenced_approval = "```markdown\n## Release approval\n- Reviewer identity: `@release-reviewer`\n- Approval date: `2026-07-10`\n```\n";
    assert!(validate_patch_provenance(fenced_approval).is_err());

    let indented_fields = "## Release approval\n    - Reviewer identity: `@release-reviewer`\n    - Approval date: `2026-07-10`\n";
    assert!(validate_patch_provenance(indented_fields).is_err());

    let html_comment_approval = "<!--\n## Release approval\n- Reviewer identity: `@release-reviewer`\n- Approval date: `2026-07-10`\n-->\n";
    assert!(validate_patch_provenance(html_comment_approval).is_err());

    let raw_html_approval = "<textarea>\n## Release approval\n- Reviewer identity: `@release-reviewer`\n- Approval date: `2026-07-10`\n</textarea>\n";
    assert!(validate_patch_provenance(raw_html_approval).is_err());

    let div_html_approval = "<div>\n## Release approval\n- Reviewer identity: `@release-reviewer`\n- Approval date: `2026-07-10`\n</div>\n";
    assert!(validate_patch_provenance(div_html_approval).is_err());
}

#[test]
fn patch_provenance_rejects_calendar_invalid_approval_date() {
    let provenance = "## Release approval\n- Reviewer identity: `@release-reviewer`\n- Approval date: `2026-04-31`\n";
    assert!(validate_patch_provenance(provenance).is_err());
}
