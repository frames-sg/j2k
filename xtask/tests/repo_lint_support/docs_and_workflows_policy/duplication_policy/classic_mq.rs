// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::repo_lint_support::{assert_file_pattern_checks, repo_root, FilePatternCheck};

#[test]
fn mq_qe_table_is_shared_by_encoder_decoder_and_cuda() {
    assert_file_pattern_checks(
        repo_root(),
        &[
            FilePatternCheck::new("crates/j2k-codec-math/src/classic.rs")
                .named("shared classic Tier-1 table module")
                .required(&[
                    "pub struct ClassicMqState",
                    "pub const MQ_STATES: [ClassicMqState; 47]",
                    "pub const MQ_QE_VALUES: [u32; 47]",
                    "pub const PACKED_MQ_TRANSITION_VALUES: [u32; 47]",
                ]),
            FilePatternCheck::new("crates/j2k-native/src/j2c/mq.rs")
                .named("native MQ table module")
                .required(&["j2k_codec_math::classic::MQ_STATES as QE_TABLE"])
                .forbidden(&["struct QeData", "static QE_TABLE", "const QE_TABLE"]),
            FilePatternCheck::new("crates/j2k-native/src/j2c/arithmetic_decoder.rs")
                .named("native arithmetic decoder")
                .required(&["use super::mq::QE_TABLE;"])
                .forbidden(&["struct QeData", "static QE_TABLE"]),
            FilePatternCheck::new("crates/j2k-native/src/j2c/arithmetic_encoder.rs")
                .named("native arithmetic encoder")
                .required(&["use super::mq::QE_TABLE;"])
                .forbidden(&["struct QeData", "static QE_TABLE"]),
            FilePatternCheck::new("crates/j2k-cuda-runtime/src/classic_decode/abi.rs")
                .named("CUDA classic Tier-1 table consumer")
                .required(&[
                    "MQ_QE_VALUES, PACKED_MQ_TRANSITION_VALUES, PACKED_SIGN_CONTEXT_LOOKUP",
                    "ZERO_CTX_HL_LOOKUP, ZERO_CTX_LL_LH_LOOKUP",
                ])
                .forbidden(&["const MQ_QE_VALUES", "const PACKED_MQ_TRANSITION_VALUES"]),
        ],
    );
}
