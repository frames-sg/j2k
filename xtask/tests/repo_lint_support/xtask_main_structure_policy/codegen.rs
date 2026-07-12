// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{assert_module_stays_focused, read};
use crate::repo_lint_support::{assert_pattern_checks, PatternCheck};

pub(super) fn assert_ownership_and_focus() {
    let commands = read("xtask/src/codegen_commands.rs");
    let transaction = read("xtask/src/codegen_commands/transaction.rs");

    assert_module_stays_focused("xtask/src/codegen_commands.rs", &commands, 400);
    assert_module_stays_focused(
        "xtask/src/codegen_commands/transaction.rs",
        &transaction,
        350,
    );
    assert_pattern_checks(&[
        PatternCheck::new("xtask codegen command ownership", &commands).required(&[
            "mod transaction;",
            "pub(super) fn stable_api(",
            "pub(super) fn codec_math_codegen(",
            "fn render_codec_math_dwt97_metal_fragment()",
            "fn render_stable_api_snapshots()",
            "write_generated_pair_transactionally(&snapshots)",
            "write_generated_pair_transactionally(&fragments)",
        ]),
        PatternCheck::new("xtask generated-pair transaction ownership", &transaction).required(&[
            "pub(super) fn write_generated_pair_transactionally(",
            "fn stage_generated_entries(",
            "fn stage_generated_file(",
            "fn restore_originals(",
            "pub(super) fn rollback_generated_pair_install(",
            "fn sync_generated_directories(",
        ]),
    ]);
}
