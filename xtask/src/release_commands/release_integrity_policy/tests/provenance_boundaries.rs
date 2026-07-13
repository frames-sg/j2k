// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::validate_patch_provenance;

#[test]
fn patch_provenance_requires_one_bounded_release_approval_section() {
    let fields = "- Reviewer identity: `@release-reviewer`\n- Approval date: `2026-07-10`\n";
    assert!(validate_patch_provenance(fields).is_err());

    let duplicate_headings =
        format!("## Release approval\n{fields}\n## Release approval\n{fields}");
    let error = validate_patch_provenance(&duplicate_headings)
        .expect_err("duplicate approval sections must reject");
    assert!(error.contains("found 2"));

    let fields_after_section =
        format!("## Release approval\nNo approval yet.\n\n## Another section\n{fields}");
    let error = validate_patch_provenance(&fields_after_section)
        .expect_err("fields in a later section must not satisfy approval");
    assert!(error.contains("Reviewer identity"));
    assert!(error.contains("Approval date"));

    let lower_heading_inside_section =
        format!("## Release approval\n### Approval record\n{fields}");
    assert_eq!(
        validate_patch_provenance(&lower_heading_inside_section),
        Ok(())
    );
}

#[test]
fn patch_provenance_rejects_ambiguous_reviewer_placeholders() {
    for reviewer in [
        "x",
        "n/a",
        "unknown",
        "release-todo-owner",
        "placeholder reviewer",
        "replace me",
    ] {
        let provenance = format!(
            "## Release approval\n- Reviewer identity: `{reviewer}`\n- Approval date: `2026-07-10`\n"
        );
        assert!(
            validate_patch_provenance(&provenance).is_err(),
            "accepted placeholder reviewer {reviewer:?}"
        );
    }
}
