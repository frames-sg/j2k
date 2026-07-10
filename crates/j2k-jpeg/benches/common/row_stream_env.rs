// SPDX-License-Identifier: MIT OR Apache-2.0

use super::classification::{CorpusInputClass, DecodeMode};
use super::full_frame_env::force_full_frame_compare;
use super::row_stream_policy::should_bench_decode_rows_rgb_for_policy;

pub(crate) fn should_bench_decode_rows_rgb(
    mode: DecodeMode,
    input_class: CorpusInputClass,
) -> bool {
    should_bench_decode_rows_rgb_for_policy(mode, input_class, force_full_frame_compare())
}
