// SPDX-License-Identifier: MIT OR Apache-2.0

use super::test_support::{captured_profile_lines, use_test_profile_sink};
use super::{emit_formatted, emit_profile_line, emit_profile_row_now};
use crate::ProfileError;

#[test]
fn immediate_profile_rows_are_formatted_and_emitted_in_order() {
    let _sink = use_test_profile_sink();

    emit_profile_line("preformatted profile row");
    emit_profile_row_now(
        "jpeg",
        "decode",
        "tile/0",
        &[("rows", "4"), ("elapsed_us", "12")],
    );

    assert_eq!(
        captured_profile_lines(),
        [
            "preformatted profile row",
            "j2k_profile codec=jpeg op=decode path=tile/0 rows=4 elapsed_us=12",
        ]
    );
}

#[test]
fn formatted_profile_errors_emit_typed_diagnostics() {
    let _sink = use_test_profile_sink();

    emit_formatted(
        "test_row",
        Err(ProfileError::InvalidInput {
            what: "test invalid field",
        }),
    );

    assert_eq!(
        captured_profile_lines(),
        ["j2k_profile_error operation=test_row error=invalid profile input: test invalid field"]
    );
}

#[test]
fn profile_sink_is_nested_and_transactional() {
    let _outer = use_test_profile_sink();
    emit_profile_line("outer-before");
    assert!(std::thread::spawn(captured_profile_lines)
        .join()
        .expect("profile sink child thread")
        .is_empty());

    {
        let _inner = use_test_profile_sink();
        emit_profile_line("inner");
        assert_eq!(captured_profile_lines(), ["inner"]);
    }

    emit_profile_line("outer-after");
    assert_eq!(captured_profile_lines(), ["outer-before", "outer-after"]);
}
