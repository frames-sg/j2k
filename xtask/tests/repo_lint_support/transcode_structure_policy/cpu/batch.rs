// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{assert_line_budget, read};
use super::{assert_pattern_checks, PatternCheck};

struct BatchSources {
    batch: String,
    result_slots: String,
    result_slot_tests: String,
    actual_live: String,
    individual: String,
    prepare: String,
    group_budget: String,
    transform: String,
    accelerated_storage: String,
    storage: String,
    encode: String,
    float97_input: String,
    precomputed: String,
    encode_live: String,
}

impl BatchSources {
    fn read() -> Self {
        Self {
            batch: read("crates/j2k-transcode/src/jpeg_to_htj2k/batch.rs"),
            result_slots: read("crates/j2k-transcode/src/jpeg_to_htj2k/batch/result_slots.rs"),
            result_slot_tests: read(
                "crates/j2k-transcode/src/jpeg_to_htj2k/batch/result_slots/tests.rs",
            ),
            actual_live: read("crates/j2k-transcode/src/jpeg_to_htj2k/batch/actual_live.rs"),
            individual: read("crates/j2k-transcode/src/jpeg_to_htj2k/batch/individual.rs"),
            prepare: read("crates/j2k-transcode/src/jpeg_to_htj2k/batch/prepare.rs"),
            group_budget: read("crates/j2k-transcode/src/jpeg_to_htj2k/batch/group_budget.rs"),
            transform: read("crates/j2k-transcode/src/jpeg_to_htj2k/batch/transform.rs"),
            accelerated_storage: read(
                "crates/j2k-transcode/src/jpeg_to_htj2k/batch/accelerated_storage.rs",
            ),
            storage: read("crates/j2k-transcode/src/jpeg_to_htj2k/batch/storage.rs"),
            encode: read("crates/j2k-transcode/src/jpeg_to_htj2k/batch/encode.rs"),
            float97_input: read(
                "crates/j2k-transcode/src/jpeg_to_htj2k/batch/encode/float97_input.rs",
            ),
            precomputed: read("crates/j2k-transcode/src/jpeg_to_htj2k/batch/encode/precomputed.rs"),
            encode_live: read("crates/j2k-transcode/src/jpeg_to_htj2k/batch/encode/live.rs"),
        }
    }

    fn assert_line_budgets(&self) {
        for (path, source, max_lines) in [
            ("jpeg_to_htj2k/batch.rs", self.batch.as_str(), 350),
            (
                "jpeg_to_htj2k/batch/result_slots.rs",
                self.result_slots.as_str(),
                125,
            ),
            (
                "jpeg_to_htj2k/batch/result_slots/tests.rs",
                self.result_slot_tests.as_str(),
                100,
            ),
            (
                "jpeg_to_htj2k/batch/actual_live.rs",
                self.actual_live.as_str(),
                100,
            ),
            (
                "jpeg_to_htj2k/batch/individual.rs",
                self.individual.as_str(),
                125,
            ),
            ("jpeg_to_htj2k/batch/prepare.rs", self.prepare.as_str(), 325),
            (
                "jpeg_to_htj2k/batch/group_budget.rs",
                self.group_budget.as_str(),
                75,
            ),
            (
                "jpeg_to_htj2k/batch/transform.rs",
                self.transform.as_str(),
                425,
            ),
            (
                "jpeg_to_htj2k/batch/accelerated_storage.rs",
                self.accelerated_storage.as_str(),
                475,
            ),
            ("jpeg_to_htj2k/batch/storage.rs", self.storage.as_str(), 475),
            ("jpeg_to_htj2k/batch/encode.rs", self.encode.as_str(), 600),
            (
                "jpeg_to_htj2k/batch/encode/float97_input.rs",
                self.float97_input.as_str(),
                150,
            ),
            (
                "jpeg_to_htj2k/batch/encode/precomputed.rs",
                self.precomputed.as_str(),
                350,
            ),
            (
                "jpeg_to_htj2k/batch/encode/live.rs",
                self.encode_live.as_str(),
                250,
            ),
        ] {
            assert_line_budget(path, source, max_lines);
        }
    }

    fn assert_contracts(&self) {
        assert_pattern_checks(&[
            PatternCheck::new("JPEG-to-HTJ2K batch facade", &self.batch)
                .required(&[
                    "mod prepare;",
                    "mod group_budget;",
                    "mod result_slots;",
                    "mod transform;",
                    "mod accelerated_storage;",
                    "mod storage;",
                    "mod encode;",
                ])
                .forbidden(&[
                    "struct IntegerBatchTile",
                    "fn transform_float97_batch_tiles(",
                    "fn store_integer_batch_wavelet(",
                    "fn encode_float97_batch_tile(",
                    "tile_results[",
                ]),
            PatternCheck::new("batch result-slot invariants", &self.result_slots).required(&[
                "struct BatchResultSlots<T>",
                "fn insert(",
                "mod tests;",
                "fn into_results(",
                "JpegToHtj2kError::InternalInvariant",
            ]),
            PatternCheck::new("batch result-slot regressions", &self.result_slot_tests).required(
                &[
                    "fn missing_worker_result_is_an_internal_invariant(",
                    "fn duplicate_worker_result_is_an_internal_invariant(",
                    "fn out_of_range_worker_result_is_an_internal_invariant(",
                    "fn complete_worker_results_preserve_input_order(",
                ],
            ),
            PatternCheck::new("batch 9/7 encode-input ownership", &self.float97_input)
                .required(&[
                    "enum Float97BatchEncodingInput",
                    "fn select_float97_batch_encoding(",
                    "PreencodedHtj2k97CompactImage",
                    "PrequantizedHtj2k97Image",
                ])
                .forbidden(&["include!(", "use super::*;"]),
            PatternCheck::new("JPEG-to-HTJ2K batch preparation ownership", &self.prepare).required(
                &[
                    "struct IntegerBatchTile",
                    "fn prepare_float97_batch_tile(",
                    "fn batch_component_groups(",
                ],
            ),
            PatternCheck::new(
                "JPEG-to-HTJ2K batch group budget ownership",
                &self.group_budget,
            )
            .required(&[
                "fn batch_component_count(",
                "fn validate_group_workspace(",
                "fn next_group_len(",
            ]),
            PatternCheck::new("JPEG-to-HTJ2K batch transform ownership", &self.transform).required(
                &[
                    "fn transform_integer_batch_tiles",
                    "fn float97_wavelets_for_batch_group",
                    "fn record_cpu_fallback",
                ],
            ),
            PatternCheck::new("accelerated batch storage", &self.accelerated_storage).required(&[
                "fn store_compact_preencoded_component(",
                "fn try_store_grouped_i16_preencoded_float97_batches",
                "fn try_store_prequantized_float97_batch_group",
            ]),
            PatternCheck::new("wavelet batch storage", &self.storage).required(&[
                "fn store_integer_batch_wavelet(",
                "fn store_float97_batch_wavelet(",
            ]),
            PatternCheck::new("JPEG-to-HTJ2K batch encode ownership", &self.encode).required(&[
                "fn record_encode_dispatch_delta(",
                "mod precomputed;",
                "fn encode_float97_batch_tile",
            ]),
            PatternCheck::new(
                "JPEG-to-HTJ2K precomputed batch encode ownership",
                &self.precomputed,
            )
            .required(&[
                "fn encode_float97_precomputed_tiles_batch",
                "fn can_encode_float97_precomputed_tiles_batch",
            ]),
        ]);
    }
}

pub(super) fn assert_batch_structure() {
    let sources = BatchSources::read();
    sources.assert_line_budgets();
    sources.assert_contracts();
}
