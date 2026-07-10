// SPDX-License-Identifier: MIT OR Apache-2.0

use super::classification::{CorpusInputClass, DecodeMode};
use super::full_frame_policy::should_compare_full_frame_for_policy;

pub(crate) fn should_compare_full_frame(mode: DecodeMode, input_class: CorpusInputClass) -> bool {
    should_compare_full_frame_for_policy(mode, input_class, force_full_frame_compare())
}

pub(super) fn force_full_frame_compare() -> bool {
    std::env::var_os("J2K_FORCE_FULL_FRAME")
        .is_some_and(|value| !matches!(value.to_str(), Some("0" | "false" | "FALSE" | "False")))
}
