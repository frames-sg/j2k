// SPDX-License-Identifier: MIT OR Apache-2.0

use super::classification::{CorpusInputClass, DecodeMode};

pub(crate) fn should_compare_full_frame_for_policy(
    mode: DecodeMode,
    input_class: CorpusInputClass,
    force_full_frame: bool,
) -> bool {
    match input_class {
        CorpusInputClass::BoundedFullFrame => true,
        CorpusInputClass::VeryLarge => {
            force_full_frame && matches!(mode, DecodeMode::Gray | DecodeMode::Rgb)
        }
    }
}
