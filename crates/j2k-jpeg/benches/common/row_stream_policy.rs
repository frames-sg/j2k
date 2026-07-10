// SPDX-License-Identifier: MIT OR Apache-2.0

use super::classification::{CorpusInputClass, DecodeMode};

pub(crate) fn should_bench_decode_rows_rgb_for_policy(
    mode: DecodeMode,
    input_class: CorpusInputClass,
    force_full_frame: bool,
) -> bool {
    if force_full_frame {
        return false;
    }
    mode == DecodeMode::Rgb && input_class == CorpusInputClass::VeryLarge
}
