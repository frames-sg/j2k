// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use crate::accelerator::{
    DctGridI16ToHtj2k97CodeBlockBatch, PreencodedHtj2k97CodeBlock,
    PreencodedHtj2k97CompactCodeBlock, PreencodedHtj2k97CompactComponent,
    PreencodedHtj2k97CompactResolution, PreencodedHtj2k97CompactSubband,
    PreencodedHtj2k97Resolution, PreencodedHtj2k97Subband,
};
use j2k::{EncodedHtJ2kCodeBlock, J2kHtCodeBlockEncodeJob};
use j2k_jpeg::transcode::JpegDctCodingMode;
use j2k_jpeg::ColorSpace;
use j2k_test_support::{JPEG_BASELINE_420_16X16, JPEG_GRAYSCALE_8X8};

#[test]
fn timing_report_add_assign_saturates_and_adds_all_counter_kinds() {
    let mut report = TranscodeTimingReport {
        source_raw_probe_us: u128::MAX - 1,
        dwt97_batch_ht_codeblock_dispatches: usize::MAX - 1,
        tile_count: 2,
        accelerator_jobs: 3,
        cpu_fallback_jobs: 4,
        ..TranscodeTimingReport::default()
    };
    report.add_assign(TranscodeTimingReport {
        source_raw_probe_us: 10,
        dwt97_batch_ht_codeblock_dispatches: 10,
        tile_count: 5,
        accelerator_jobs: 7,
        cpu_fallback_jobs: 11,
        ..TranscodeTimingReport::default()
    });

    assert_eq!(report.source_raw_probe_us, u128::MAX);
    assert_eq!(report.dwt97_batch_ht_codeblock_dispatches, usize::MAX);
    assert_eq!(report.tile_count, 7);
    assert_eq!(report.accelerator_jobs, 10);
    assert_eq!(report.cpu_fallback_jobs, 15);
}

#[test]
fn timing_report_classifies_accelerator_work_from_dispatch_and_resident_counters() {
    assert!(!TranscodeTimingReport::default().accelerator_work_observed());

    assert!(TranscodeTimingReport {
        accelerator_dispatches: 1,
        ..TranscodeTimingReport::default()
    }
    .accelerator_work_observed());

    assert!(TranscodeTimingReport {
        dwt97_batch_pack_upload_bytes: 1,
        ..TranscodeTimingReport::default()
    }
    .accelerator_work_observed());

    assert!(TranscodeTimingReport {
        dwt97_batch_ht_output_readback_transfers: 1,
        ..TranscodeTimingReport::default()
    }
    .accelerator_work_observed());
}

#[test]
fn stateful_transcoder_reuses_dct_block_scratch_across_tiles() {
    let options = JpegToHtj2kOptions {
        coefficient_path: JpegToHtj2kCoefficientPath::FloatDirectLinear53,
        ..JpegToHtj2kOptions::default()
    };
    let mut transcoder = JpegToHtj2kTranscoder::default();

    let larger = transcoder
        .transcode(JPEG_BASELINE_420_16X16, &options)
        .expect("stateful transcode accepts 4:2:0 JPEG");
    let capacity_after_larger = transcoder.scratch.dct_blocks_f64.capacity();
    assert!(capacity_after_larger >= 4);

    let smaller = transcoder
        .transcode(JPEG_GRAYSCALE_8X8, &options)
        .expect("stateful transcode accepts grayscale JPEG");
    let stateless = jpeg_to_htj2k(JPEG_GRAYSCALE_8X8, &options)
        .expect("stateless transcode accepts grayscale JPEG");

    assert_eq!(larger.report.component_count, 3);
    assert_eq!(smaller.report.component_count, 1);
    assert_eq!(
        transcoder.scratch.dct_blocks_f64.capacity(),
        capacity_after_larger
    );
    assert_eq!(smaller.codestream, stateless.codestream);
}

#[test]
fn stateful_transcoder_reuses_integer_idct_block_scratch_across_tiles() {
    let options = JpegToHtj2kOptions::default();
    let mut transcoder = JpegToHtj2kTranscoder::default();

    let larger = transcoder
        .transcode(JPEG_BASELINE_420_16X16, &options)
        .expect("stateful integer-direct transcode accepts 4:2:0 JPEG");
    let capacity_after_larger = transcoder.scratch.integer_idct_blocks.capacity();
    assert!(capacity_after_larger >= 4);

    let smaller = transcoder
        .transcode(JPEG_GRAYSCALE_8X8, &options)
        .expect("stateful integer-direct transcode accepts grayscale JPEG");
    let stateless = jpeg_to_htj2k(JPEG_GRAYSCALE_8X8, &options)
        .expect("stateless integer-direct transcode accepts grayscale JPEG");

    assert_eq!(larger.report.component_count, 3);
    assert_eq!(smaller.report.component_count, 1);
    assert_eq!(
        transcoder.scratch.integer_idct_blocks.capacity(),
        capacity_after_larger
    );
    assert_eq!(smaller.codestream, stateless.codestream);
}

#[test]
fn transcode_batch_profile_row_preserves_labels_and_metric_rollups() {
    let report = BatchTranscodeReport {
        tile_count: 2,
        successful_tiles: 2,
        failed_tiles: 0,
        transformed_components: 6,
        reversible_dwt53_batches: 1,
        reversible_dwt53_batch_jobs: 6,
        extract_us: 10,
        transform_us: 20,
        encode_us: 30,
        timings: TranscodeTimingReport {
            jpeg_dct_extract_us: 11,
            dct_to_wavelet_total_us: 22,
            dwt97_batch_pack_upload_transfers: 1,
            dwt97_batch_pack_upload_bytes: 8,
            dwt97_batch_resident_dct_handoff_count: 3,
            dwt97_batch_resident_dwt_handoff_count: 4,
            dwt97_batch_ht_status_readback_transfers: 2,
            dwt97_batch_ht_status_readback_bytes: 16,
            dwt97_batch_ht_output_readback_transfers: 3,
            dwt97_batch_ht_output_readback_bytes: 24,
            dwt97_batch_readback_transfers: 5,
            dwt97_batch_readback_bytes: 40,
            htj2k_encode_us: 33,
            component_count: 6,
            batch_count: 1,
            batch_jobs: 6,
            accelerator_dispatches: 1,
            accelerator_dispatched_jobs: 6,
            cpu_fallback_jobs: 0,
            ..TranscodeTimingReport::default()
        },
        coefficient_path: JpegToHtj2kCoefficientPath::IntegerDirect53,
    };

    let row = report.profile_row("fixture batch", TranscodeBatchProfileRequest::MetalAuto);
    let fields = row.fields();
    let get = |key: &str| {
        fields
            .iter()
            .find_map(|(field_key, value)| (*field_key == key).then_some(value.as_str()))
            .unwrap_or_else(|| panic!("missing profile field {key}"))
    };

    assert_eq!(fields[0].0, "codec");
    assert_eq!(fields[1].0, "op");
    assert_eq!(fields[2].0, "request");
    assert_eq!(fields[3].0, "path");
    assert_eq!(fields[4].0, "pipeline");
    assert_eq!(fields[5].0, "context");
    assert_eq!(get("codec"), "transcode");
    assert_eq!(get("op"), "transcode_batch");
    assert_eq!(get("request"), "metal_auto");
    assert_eq!(get("path"), "auto");
    assert_eq!(get("pipeline"), "jpeg_to_htj2k");
    assert_eq!(get("context"), "fixture_batch");
    assert_eq!(get("coefficient_path"), "IntegerDirect53");
    assert_eq!(get("extract_processor"), "cpu");
    assert_eq!(get("transform_processor"), "metal");
    assert_eq!(get("encode_processor"), "cpu");
    assert_eq!(get("tile_count"), "2");
    assert_eq!(get("successful_tiles"), "2");
    assert_eq!(get("transformed_components"), "6");
    assert_eq!(get("total_us"), "60");
    assert_eq!(get("jpeg_dct_extract_us"), "11");
    assert_eq!(get("dct_to_wavelet_total_us"), "22");
    assert_eq!(get("htj2k_encode_us"), "33");
    assert_eq!(get("host_to_device_transfer_count"), "1");
    assert_eq!(get("host_to_device_transfer_bytes"), "8");
    assert_eq!(get("device_to_host_transfer_count"), "10");
    assert_eq!(get("device_to_host_transfer_bytes"), "80");
    assert_eq!(get("accelerator_dispatches"), "1");
    assert_eq!(get("cpu_fallback_jobs"), "0");
    assert_eq!(row.codec(), "transcode");
    assert_eq!(row.op(), "transcode_batch");
    assert_eq!(row.path(), "auto");

    assert_eq!(
        TranscodeBatchProfileRequest::MetalExplicit.profile_path(&TranscodeTimingReport::default()),
        "cpu"
    );
    assert_eq!(
        TranscodeBatchProfileRequest::Cpu.profile_path(&report.timings),
        "cpu"
    );
}

#[derive(Default)]
struct GroupedI16Accelerator {
    grouped_calls: usize,
    single_calls: usize,
    grouped_lengths: Vec<Vec<usize>>,
}

impl DctToWaveletStageAccelerator for GroupedI16Accelerator {
    fn supports_htj2k97_i16_preencoded_batch(&self) -> bool {
        true
    }

    fn dct_grid_i16_to_htj2k97_preencoded_batch(
        &mut self,
        jobs: &[DctGridI16ToHtj2k97CodeBlockJob<'_>],
        _options: Htj2k97CodeBlockOptions,
    ) -> Result<Option<Vec<PreencodedHtj2k97Component>>, TranscodeStageError> {
        self.single_calls = self.single_calls.saturating_add(1);
        Ok(Some(
            jobs.iter()
                .map(|job| dummy_preencoded_component(job.x_rsiz, job.y_rsiz))
                .collect(),
        ))
    }

    fn dct_grid_i16_to_htj2k97_preencoded_batch_groups(
        &mut self,
        groups: &[DctGridI16ToHtj2k97CodeBlockBatch<'_, '_>],
        _options: Htj2k97CodeBlockOptions,
    ) -> Result<Option<Vec<Vec<PreencodedHtj2k97Component>>>, TranscodeStageError> {
        self.grouped_calls = self.grouped_calls.saturating_add(1);
        self.grouped_lengths
            .push(groups.iter().map(|group| group.jobs.len()).collect());
        Ok(Some(
            groups
                .iter()
                .map(|group| {
                    group
                        .jobs
                        .iter()
                        .map(|job| dummy_preencoded_component(job.x_rsiz, job.y_rsiz))
                        .collect()
                })
                .collect(),
        ))
    }
}

#[test]
fn float97_batch_offers_i16_preencoded_geometry_groups_together() {
    let mut tiles = vec![test_float97_tile()];
    let options = JpegToHtj2kOptions::lossy_97();
    let mut scratch = JpegToHtj2kScratch::default();
    let mut accelerator = GroupedI16Accelerator::default();
    let mut timings = TranscodeTimingReport::default();

    let (batch_count, job_count) = transform_float97_batch_tiles(
        &mut tiles,
        &options,
        &mut scratch,
        &mut accelerator,
        &mut timings,
    )
    .expect("grouped i16 preencoded transform");

    assert_eq!(batch_count, 2);
    assert_eq!(job_count, 3);
    assert_eq!(accelerator.grouped_calls, 1);
    assert_eq!(accelerator.single_calls, 0);
    assert_eq!(accelerator.grouped_lengths, vec![vec![1, 2]]);
    assert!(tiles[0].preencoded_components.iter().all(Option::is_some));
}

#[derive(Default)]
struct CountingHtBatchEncodeAccelerator {
    batches: usize,
    jobs: usize,
    single_blocks: usize,
}

impl J2kEncodeStageAccelerator for CountingHtBatchEncodeAccelerator {
    fn encode_ht_code_blocks(
        &mut self,
        jobs: &[J2kHtCodeBlockEncodeJob<'_>],
    ) -> Result<Option<Vec<EncodedHtJ2kCodeBlock>>, &'static str> {
        self.batches = self.batches.saturating_add(1);
        self.jobs = self.jobs.saturating_add(jobs.len());
        Ok(None)
    }

    fn encode_ht_code_block(
        &mut self,
        _job: J2kHtCodeBlockEncodeJob<'_>,
    ) -> Result<Option<EncodedHtJ2kCodeBlock>, &'static str> {
        self.single_blocks = self.single_blocks.saturating_add(1);
        Ok(None)
    }
}

#[test]
fn float97_precomputed_prepared_tiles_offer_all_tiles_to_one_ht_batch() {
    let tiles = vec![
        test_float97_precomputed_tile(0),
        test_float97_precomputed_tile(1),
    ];
    let mut options = JpegToHtj2kOptions::lossy_97();
    options.encode_options.code_block_width_exp = 2;
    options.encode_options.code_block_height_exp = 2;
    let mut accelerator = CountingHtBatchEncodeAccelerator::default();

    let encoded_tiles = encode_float97_prepared_tiles(tiles, &options, &mut accelerator);

    assert_eq!(encoded_tiles.len(), 2);
    for (expected_tile_index, (actual_tile_index, encoded)) in encoded_tiles.into_iter().enumerate()
    {
        assert_eq!(actual_tile_index, expected_tile_index);
        let encoded = encoded.expect("precomputed batch tile encodes");
        assert!(encoded.codestream.starts_with(&[0xff, 0x4f]));
    }
    assert_eq!(accelerator.batches, 1);
    assert!(accelerator.jobs > 0);
    assert_eq!(accelerator.single_blocks, accelerator.jobs);
}

#[test]
fn compact_preencoded_component_storage_rebases_ranges_into_tile_payload() {
    let mut tile = test_float97_tile();
    let batch_payload = vec![1, 2, 3, 4, 5, 6];
    let component = PreencodedHtj2k97CompactComponent {
        x_rsiz: 1,
        y_rsiz: 1,
        resolutions: vec![PreencodedHtj2k97CompactResolution {
            subbands: vec![PreencodedHtj2k97CompactSubband {
                sub_band_type: crate::accelerator::J2kSubBandType::LowLow,
                num_cbs_x: 2,
                num_cbs_y: 1,
                total_bitplanes: 1,
                code_blocks: vec![
                    PreencodedHtj2k97CompactCodeBlock {
                        width: 1,
                        height: 1,
                        payload_range: 1..3,
                        cleanup_length: 2,
                        refinement_length: 0,
                        num_coding_passes: 1,
                        num_zero_bitplanes: 0,
                    },
                    PreencodedHtj2k97CompactCodeBlock {
                        width: 1,
                        height: 1,
                        payload_range: 3..6,
                        cleanup_length: 3,
                        refinement_length: 0,
                        num_coding_passes: 1,
                        num_zero_bitplanes: 0,
                    },
                ],
            }],
        }],
    };

    store_compact_preencoded_component(&mut tile, 1, &batch_payload, component)
        .expect("compact component storage");

    let stored = tile.preencoded_compact_components[1]
        .as_ref()
        .expect("stored compact component");
    assert_eq!(tile.preencoded_compact_payload, vec![2, 3, 4, 5, 6]);
    assert_eq!(
        stored.resolutions[0].subbands[0].code_blocks[0].payload_range,
        0..2
    );
    assert_eq!(
        stored.resolutions[0].subbands[0].code_blocks[1].payload_range,
        2..5
    );
}

fn test_float97_tile() -> Float97BatchTile {
    let components = vec![
        test_component(0, 16, 16, 2, 2),
        test_component(1, 8, 8, 1, 1),
        test_component(2, 8, 8, 1, 1),
    ];
    Float97BatchTile {
        tile_index: 0,
        jpeg: JpegDctImage {
            width: 16,
            height: 16,
            color_space: ColorSpace::YCbCr,
            coding_mode: JpegDctCodingMode::BaselineSequential,
            scan_count: 1,
            components,
            restart_index: None,
        },
        component_sampling: vec![(1, 1), (2, 2), (2, 2)],
        decomposition_levels: 1,
        all_unit_sampled: false,
        component_reports: Vec::new(),
        precomputed_components: vec![None, None, None],
        preencoded_compact_payload: Vec::new(),
        preencoded_compact_components: vec![None, None, None],
        preencoded_components: vec![None, None, None],
        prequantized_components: vec![None, None, None],
        float_validation_actual: Vec::new(),
        float_validation_expected: Vec::new(),
        timings: TranscodeTimingReport::default(),
    }
}

fn test_float97_precomputed_tile(tile_index: usize) -> Float97BatchTile {
    let width = 17;
    let height = 13;
    let component = test_component(0, width, height, 1, 1);
    Float97BatchTile {
        tile_index,
        jpeg: JpegDctImage {
            width,
            height,
            color_space: ColorSpace::Grayscale,
            coding_mode: JpegDctCodingMode::BaselineSequential,
            scan_count: 1,
            components: vec![component],
            restart_index: None,
        },
        component_sampling: vec![(1, 1)],
        decomposition_levels: 1,
        all_unit_sampled: true,
        component_reports: vec![TranscodeComponentReport {
            component_index: 0,
            width,
            height,
            block_cols: width.div_ceil(8),
            block_rows: height.div_ceil(8),
            x_rsiz: 1,
            y_rsiz: 1,
        }],
        precomputed_components: vec![Some(dummy_precomputed_component(1, 1, width, height))],
        preencoded_compact_payload: Vec::new(),
        preencoded_compact_components: vec![None],
        preencoded_components: vec![None],
        prequantized_components: vec![None],
        float_validation_actual: Vec::new(),
        float_validation_expected: Vec::new(),
        timings: TranscodeTimingReport::default(),
    }
}

fn test_component(
    component_index: usize,
    width: u32,
    height: u32,
    h_samp: u8,
    v_samp: u8,
) -> JpegDctComponent {
    let block_cols = width.div_ceil(8);
    let block_rows = height.div_ceil(8);
    let block_count = (block_cols * block_rows) as usize;
    JpegDctComponent {
        component_index,
        width,
        height,
        h_samp,
        v_samp,
        block_cols,
        block_rows,
        quant_table: [1u16; 64],
        quantized_blocks: vec![[0i16; 64]; block_count],
        dequantized_blocks: vec![[0i16; 64]; block_count],
    }
}

fn dummy_precomputed_component(
    x_rsiz: u8,
    y_rsiz: u8,
    width: u32,
    height: u32,
) -> PrecomputedHtj2k97Component {
    let low_width = width.div_ceil(2);
    let low_height = height.div_ceil(2);
    let high_width = width / 2;
    let high_height = height / 2;
    PrecomputedHtj2k97Component {
        x_rsiz,
        y_rsiz,
        dwt: J2kForwardDwt97Output {
            ll: sample_f32_coefficients(low_width * low_height, 0.25),
            ll_width: low_width,
            ll_height: low_height,
            levels: vec![J2kForwardDwt97Level {
                hl: sample_f32_coefficients(high_width * low_height, -0.75),
                lh: sample_f32_coefficients(low_width * high_height, 1.25),
                hh: sample_f32_coefficients(high_width * high_height, -1.5),
                width,
                height,
                low_width,
                low_height,
                high_width,
                high_height,
            }],
        },
    }
}

#[expect(
    clippy::cast_precision_loss,
    reason = "small deterministic fixture indices are intentionally mapped into an f32 signal"
)]
fn sample_f32_coefficients(count: u32, seed: f32) -> Vec<f32> {
    (0..count)
        .map(|idx| seed + (idx as f32).sin() * 0.125)
        .collect()
}

fn dummy_preencoded_component(x_rsiz: u8, y_rsiz: u8) -> PreencodedHtj2k97Component {
    PreencodedHtj2k97Component {
        x_rsiz,
        y_rsiz,
        resolutions: vec![PreencodedHtj2k97Resolution {
            subbands: vec![PreencodedHtj2k97Subband {
                sub_band_type: crate::accelerator::J2kSubBandType::LowLow,
                num_cbs_x: 1,
                num_cbs_y: 1,
                total_bitplanes: 1,
                code_blocks: vec![PreencodedHtj2k97CodeBlock {
                    width: 1,
                    height: 1,
                    encoded: EncodedHtJ2kCodeBlock {
                        data: Vec::new(),
                        cleanup_length: 0,
                        refinement_length: 0,
                        num_coding_passes: 0,
                        num_zero_bitplanes: 1,
                    },
                }],
            }],
        }],
    }
}
