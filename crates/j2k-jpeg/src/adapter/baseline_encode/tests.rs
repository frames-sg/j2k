// SPDX-License-Identifier: MIT OR Apache-2.0

use super::planning::{
    jpeg_baseline_gpu_encode_batch_plan, jpeg_baseline_gpu_encode_params,
    jpeg_baseline_gpu_encode_tile_plan, same_source_buffer_batch_end,
};
use super::tables::jpeg_baseline_sampling_for;
use super::types::{
    JpegBaselineEncodeTables, JpegBaselineGpuEncodeBatchPlan, JpegBaselineGpuEncodeError,
    JpegBaselineGpuEncodeHostAdapter, JpegBaselineGpuEncodeTile, JpegBaselineGpuEncodeTilePlan,
};
use super::validation::validate_jpeg_baseline_gpu_encode_tile;
use super::{encode_jpeg_baseline_gpu_batch, encode_jpeg_baseline_gpu_tile};
use crate::encoder::{JpegBackend, JpegEncodeError, JpegEncodeOptions, JpegSubsampling};
use crate::PixelFormat;

#[test]
fn baseline_encode_modules_stay_focused_and_fragment_free() {
    const ROOT: &str = include_str!("../baseline_encode.rs");
    const ALLOCATION: &str = include_str!("allocation.rs");
    const FRAME: &str = include_str!("frame.rs");
    const ORCHESTRATE: &str = include_str!("orchestrate.rs");
    const ORCHESTRATE_BATCH: &str = include_str!("orchestrate/batch.rs");
    const ORCHESTRATE_BATCH_GROUP: &str = include_str!("orchestrate/batch/group.rs");
    const PLANNING: &str = include_str!("planning.rs");
    const PLANNING_BATCH: &str = include_str!("planning/batch.rs");
    const TABLES: &str = include_str!("tables.rs");
    const TYPES: &str = include_str!("types.rs");
    const VALIDATION: &str = include_str!("validation.rs");

    let modules = [
        ("baseline_encode.rs", ROOT, 45usize),
        ("baseline_encode/allocation.rs", ALLOCATION, 300),
        ("baseline_encode/frame.rs", FRAME, 220),
        ("baseline_encode/orchestrate.rs", ORCHESTRATE, 190),
        (
            "baseline_encode/orchestrate/batch.rs",
            ORCHESTRATE_BATCH,
            190,
        ),
        (
            "baseline_encode/orchestrate/batch/group.rs",
            ORCHESTRATE_BATCH_GROUP,
            130,
        ),
        ("baseline_encode/planning.rs", PLANNING, 240),
        ("baseline_encode/planning/batch.rs", PLANNING_BATCH, 120),
        ("baseline_encode/tables.rs", TABLES, 260),
        ("baseline_encode/types.rs", TYPES, 260),
        ("baseline_encode/validation.rs", VALIDATION, 190),
    ];
    for (path, source, max_lines) in modules {
        let line_count = source.lines().count();
        assert!(
            line_count <= max_lines,
            "{path} grew to {line_count} lines; split it before exceeding {max_lines}"
        );
        assert!(
            !source.contains("include!(") && !source.contains("#[path"),
            "{path} must remain a real Rust module, not a textual source fragment"
        );
        assert!(
            !source.contains("pub use ") || !source.contains("::*"),
            "{path} must not hide its API behind a wildcard re-export"
        );
    }

    for declaration in [
        "mod allocation;",
        "mod frame;",
        "mod orchestrate;",
        "mod planning;",
        "mod tables;",
        "mod types;",
        "mod validation;",
    ] {
        assert!(
            ROOT.contains(declaration),
            "baseline_encode facade lost required module boundary {declaration}"
        );
    }
    assert!(
        ORCHESTRATE.contains("mod batch;"),
        "GPU baseline orchestration lost its batch execution boundary"
    );
}

#[test]
fn baseline_encode_error_and_marker_literals_remain_path_owned() {
    const FRAME: &str = include_str!("frame.rs");
    const ORCHESTRATE: &str = include_str!("orchestrate.rs");
    const ORCHESTRATE_BATCH: &str = include_str!("orchestrate/batch.rs");
    const ORCHESTRATE_BATCH_GROUP: &str = include_str!("orchestrate/batch/group.rs");
    const PLANNING: &str = include_str!("planning.rs");
    let orchestration = [ORCHESTRATE, ORCHESTRATE_BATCH, ORCHESTRATE_BATCH_GROUP].concat();

    for marker in ["DQT", "DRI", "SOF0", "DHT", "SOS"] {
        assert_eq!(
            FRAME.matches(&format!("\"{marker}\"")).count(),
            1,
            "{marker} segment name moved or duplicated"
        );
    }
    for message in [
        "GPU JPEG baseline batch returned the wrong number of entropy chunks",
        "GPU JPEG baseline entropy output exceeded its planned capacity",
    ] {
        assert_eq!(orchestration.matches(message).count(), 1);
    }
    for message in [
        "JPEG MCU count overflow",
        "JPEG entropy capacity overflow",
        "JPEG entropy capacity exceeds usize",
    ] {
        assert_eq!(PLANNING.matches(message).count(), 1);
    }
}

fn rgb_tile() -> JpegBaselineGpuEncodeTile {
    JpegBaselineGpuEncodeTile {
        byte_offset: 32,
        width: 17,
        height: 9,
        pitch_bytes: 64,
        output_width: 32,
        output_height: 16,
        format: PixelFormat::Rgb8,
        buffer_len: 32 + 8 * 64 + 17 * 3,
    }
}

#[test]
fn gpu_encode_params_preserve_explicit_offsets() {
    let options = JpegEncodeOptions {
        subsampling: JpegSubsampling::Ybr420,
        restart_interval: Some(4),
        backend: JpegBackend::Cuda,
        ..JpegEncodeOptions::default()
    };
    let sampling = jpeg_baseline_sampling_for(options.subsampling);
    let tile = rgb_tile();

    validate_jpeg_baseline_gpu_encode_tile(tile, options, JpegBackend::Cuda).expect("valid tile");
    let params =
        jpeg_baseline_gpu_encode_params(tile, options, sampling, 4096, tile.byte_offset, 128)
            .expect("gpu params");

    assert_eq!(params.input_offset_bytes, 32);
    assert_eq!(params.entropy_offset_bytes, 128);
    assert_eq!(params.entropy_capacity, 4096);
    assert_eq!(params.format, 1);
    assert_eq!(params.components, 3);
    assert_eq!(params.mcus_per_row, 2);
    assert_eq!(params.mcu_rows, 1);
    assert_eq!(params.restart_interval_mcus, 4);
}

#[test]
fn gpu_encode_batch_plan_accumulates_offsets_in_tile_order() {
    let options = JpegEncodeOptions {
        subsampling: JpegSubsampling::Ybr420,
        restart_interval: Some(4),
        backend: JpegBackend::Cuda,
        ..JpegEncodeOptions::default()
    };
    let sampling = jpeg_baseline_sampling_for(options.subsampling);
    let first = rgb_tile();
    let mut second = rgb_tile();
    second.byte_offset = 512;
    second.buffer_len = second.byte_offset + 8 * second.pitch_bytes + 17 * 3;

    let plan =
        jpeg_baseline_gpu_encode_batch_plan(&[first, second], options, JpegBackend::Cuda, sampling)
            .expect("valid batch plan");

    assert_eq!(plan.params.len(), 2);
    assert_eq!(
        plan.params[0].input_offset_bytes,
        u32::try_from(first.byte_offset).expect("fixture offset fits in u32")
    );
    assert_eq!(plan.params[0].entropy_offset_bytes, 0);
    assert_eq!(
        plan.params[1].input_offset_bytes,
        u32::try_from(second.byte_offset).expect("fixture offset fits in u32")
    );
    assert_eq!(
        plan.params[1].entropy_offset_bytes,
        plan.params[0].entropy_capacity
    );
    assert_eq!(
        plan.total_entropy_capacity,
        usize::try_from(plan.params[0].entropy_capacity).unwrap()
            + usize::try_from(plan.params[1].entropy_capacity).unwrap()
    );
}

#[test]
fn gpu_encode_plan_rejects_entropy_bound_above_shared_host_cap() {
    let options = JpegEncodeOptions {
        quality: 100,
        subsampling: JpegSubsampling::Gray,
        restart_interval: Some(1),
        backend: JpegBackend::Cuda,
    };
    let sampling = jpeg_baseline_sampling_for(options.subsampling);
    let tile = JpegBaselineGpuEncodeTile {
        byte_offset: 0,
        width: 1,
        height: 1,
        pitch_bytes: 1,
        output_width: 8_225,
        output_height: 65_273,
        format: PixelFormat::Gray8,
        buffer_len: 1,
    };

    let error =
        jpeg_baseline_gpu_encode_tile_plan(tile, options, JpegBackend::Cuda, sampling, 0, 0)
            .expect_err("conservative GPU entropy capacity exceeds the shared host cap");
    assert!(matches!(
        error,
        JpegBaselineGpuEncodeError::Encode(JpegEncodeError::MemoryCapExceeded {
            requested,
            cap
        }) if requested > cap
    ));
}

#[test]
fn gpu_batch_plan_caps_the_combined_entropy_allocation() {
    let options = JpegEncodeOptions {
        quality: 90,
        subsampling: JpegSubsampling::Gray,
        restart_interval: None,
        backend: JpegBackend::Cuda,
    };
    let sampling = jpeg_baseline_sampling_for(options.subsampling);
    let tile = JpegBaselineGpuEncodeTile {
        byte_offset: 0,
        width: 1,
        height: 1,
        pitch_bytes: 1,
        output_width: 4_096,
        output_height: 8_192,
        format: PixelFormat::Gray8,
        buffer_len: 1,
    };

    let error =
        jpeg_baseline_gpu_encode_batch_plan(&[tile, tile], options, JpegBackend::Cuda, sampling)
            .expect_err("combined entropy capacity exceeds the shared host cap");
    assert!(matches!(
        error,
        JpegBaselineGpuEncodeError::Encode(JpegEncodeError::MemoryCapExceeded {
            requested,
            cap
        }) if requested > cap
    ));
}

#[test]
fn gpu_encode_validation_reports_short_pitch() {
    let mut tile = rgb_tile();
    tile.pitch_bytes = 50;
    let err = validate_jpeg_baseline_gpu_encode_tile(
        tile,
        JpegEncodeOptions {
            subsampling: JpegSubsampling::Ybr444,
            backend: JpegBackend::Metal,
            ..JpegEncodeOptions::default()
        },
        JpegBackend::Metal,
    )
    .expect_err("short pitch must fail");

    match err {
        JpegBaselineGpuEncodeError::PitchTooShort {
            row_bytes,
            pitch_bytes,
        } => {
            assert_eq!(row_bytes, 51);
            assert_eq!(pitch_bytes, 50);
        }
        other => panic!("unexpected validation error: {other:?}"),
    }
}

#[test]
fn gpu_encode_batch_plan_validates_every_tile() {
    let options = JpegEncodeOptions {
        subsampling: JpegSubsampling::Ybr444,
        backend: JpegBackend::Metal,
        ..JpegEncodeOptions::default()
    };
    let sampling = jpeg_baseline_sampling_for(options.subsampling);
    let mut second = rgb_tile();
    second.pitch_bytes = 50;

    let err = jpeg_baseline_gpu_encode_batch_plan(
        &[rgb_tile(), second],
        options,
        JpegBackend::Metal,
        sampling,
    )
    .expect_err("invalid second tile must fail");

    match err {
        JpegBaselineGpuEncodeError::PitchTooShort {
            row_bytes,
            pitch_bytes,
        } => {
            assert_eq!(row_bytes, 51);
            assert_eq!(pitch_bytes, 50);
        }
        other => panic!("unexpected validation error: {other:?}"),
    }
}

#[test]
fn same_source_buffer_batch_end_groups_contiguous_keys() {
    let tiles = [10u64, 10, 10, 11, 10];

    assert_eq!(same_source_buffer_batch_end(&tiles, 0, |value| *value), 3);
    assert_eq!(same_source_buffer_batch_end(&tiles, 3, |value| *value), 4);
}

#[derive(Clone, Copy)]
struct MockTile {
    source: u8,
    entropy_byte: u8,
}

#[derive(Debug, PartialEq, Eq)]
enum MockSubmission {
    Tile(u8),
    Batch(Vec<u8>),
}

#[derive(Default)]
struct RecordingAdapter {
    submissions: Vec<MockSubmission>,
    oversize_single_output: bool,
    oversize_batch_last_output: bool,
    oversize_batch_outer: bool,
    output_dimensions: Option<(u32, u32)>,
}

impl JpegBaselineGpuEncodeHostAdapter<MockTile> for RecordingAdapter {
    type Error = JpegEncodeError;
    type SourceKey = u8;

    fn backend(&self) -> JpegBackend {
        JpegBackend::Cuda
    }

    fn source_key(&self, tile: &MockTile) -> Self::SourceKey {
        tile.source
    }

    fn gpu_tile(&self, _tile: MockTile) -> Result<JpegBaselineGpuEncodeTile, Self::Error> {
        let (output_width, output_height) = self.output_dimensions.unwrap_or((8, 8));
        Ok(JpegBaselineGpuEncodeTile {
            byte_offset: 0,
            width: 1,
            height: 1,
            pitch_bytes: 1,
            output_width,
            output_height,
            format: PixelFormat::Gray8,
            buffer_len: 1,
        })
    }

    fn map_plan_error(&self, error: JpegBaselineGpuEncodeError) -> Self::Error {
        match error {
            JpegBaselineGpuEncodeError::Encode(error) => error,
            _ => JpegEncodeError::InternalInvariant {
                reason: "mock planning error",
            },
        }
    }

    fn encode_tile_entropy(
        &mut self,
        tile: MockTile,
        _tables: &JpegBaselineEncodeTables,
        plan: JpegBaselineGpuEncodeTilePlan,
    ) -> Result<Vec<u8>, Self::Error> {
        self.submissions
            .push(MockSubmission::Tile(tile.entropy_byte));
        if self.oversize_single_output {
            let mut output = Vec::with_capacity(plan.entropy_capacity + 1);
            output.push(tile.entropy_byte);
            return Ok(output);
        }
        Ok(vec![tile.entropy_byte])
    }

    fn encode_batch_entropy(
        &mut self,
        tiles: &[MockTile],
        _tables: &JpegBaselineEncodeTables,
        plan: JpegBaselineGpuEncodeBatchPlan,
    ) -> Result<Vec<Vec<u8>>, Self::Error> {
        self.submissions.push(MockSubmission::Batch(
            tiles.iter().map(|tile| tile.entropy_byte).collect(),
        ));
        let outer_capacity = tiles.len() + usize::from(self.oversize_batch_outer);
        let mut output = Vec::with_capacity(outer_capacity);
        for (index, tile) in tiles.iter().enumerate() {
            if self.oversize_batch_last_output && index + 1 == tiles.len() {
                let planned = usize::try_from(plan.params[index].entropy_capacity)
                    .expect("mock planned capacity fits usize");
                let mut entropy = Vec::with_capacity(planned + 1);
                entropy.push(tile.entropy_byte);
                output.push(entropy);
            } else {
                output.push(vec![tile.entropy_byte]);
            }
        }
        Ok(output)
    }
}

fn mock_gpu_options() -> JpegEncodeOptions {
    JpegEncodeOptions {
        subsampling: JpegSubsampling::Gray,
        backend: JpegBackend::Cuda,
        ..JpegEncodeOptions::default()
    }
}

#[test]
fn gpu_batch_preserves_order_across_multiple_source_groups() {
    let tiles = [
        MockTile {
            source: 1,
            entropy_byte: 10,
        },
        MockTile {
            source: 1,
            entropy_byte: 11,
        },
        MockTile {
            source: 2,
            entropy_byte: 12,
        },
        MockTile {
            source: 3,
            entropy_byte: 13,
        },
        MockTile {
            source: 3,
            entropy_byte: 14,
        },
    ];
    let mut adapter = RecordingAdapter::default();

    let encoded = encode_jpeg_baseline_gpu_batch(&tiles, mock_gpu_options(), &mut adapter)
        .expect("mock batch encode");

    assert_eq!(
        adapter.submissions,
        [
            MockSubmission::Batch(vec![10, 11]),
            MockSubmission::Tile(12),
            MockSubmission::Batch(vec![13, 14]),
        ]
    );
    assert_eq!(encoded.len(), tiles.len());
    for (frame, tile) in encoded.iter().zip(tiles) {
        assert_eq!(frame.data[frame.data.len() - 3], tile.entropy_byte);
    }
}

#[test]
fn gpu_single_rejects_adapter_capacity_above_plan_before_frame_copy() {
    let tile = MockTile {
        source: 1,
        entropy_byte: 10,
    };
    let mut adapter = RecordingAdapter {
        oversize_single_output: true,
        ..RecordingAdapter::default()
    };

    let error = encode_jpeg_baseline_gpu_tile(tile, mock_gpu_options(), &mut adapter)
        .expect_err("adapter output capacity exceeds its contract");

    assert!(matches!(
        error,
        JpegEncodeError::InternalInvariant { reason }
            if reason == "GPU JPEG baseline entropy output exceeded its planned capacity"
    ));
}

#[test]
fn gpu_batch_validates_every_adapter_capacity_before_frame_copy() {
    let tiles = [
        MockTile {
            source: 1,
            entropy_byte: 10,
        },
        MockTile {
            source: 1,
            entropy_byte: 11,
        },
    ];
    let mut adapter = RecordingAdapter {
        oversize_batch_last_output: true,
        ..RecordingAdapter::default()
    };

    let error = encode_jpeg_baseline_gpu_batch(&tiles, mock_gpu_options(), &mut adapter)
        .expect_err("last adapter output capacity exceeds its contract");

    assert!(matches!(
        error,
        JpegEncodeError::InternalInvariant { reason }
            if reason == "GPU JPEG baseline entropy output exceeded its planned capacity"
    ));
}

#[test]
fn gpu_batch_rejects_adapter_outer_capacity_above_the_tile_count() {
    let tiles = [
        MockTile {
            source: 1,
            entropy_byte: 10,
        },
        MockTile {
            source: 1,
            entropy_byte: 11,
        },
    ];
    let mut adapter = RecordingAdapter {
        oversize_batch_outer: true,
        ..RecordingAdapter::default()
    };

    let error = encode_jpeg_baseline_gpu_batch(&tiles, mock_gpu_options(), &mut adapter)
        .expect_err("adapter outer capacity exceeds its contract");

    assert!(matches!(
        error,
        JpegEncodeError::InternalInvariant { reason }
            if reason == "GPU JPEG baseline entropy output exceeded its planned capacity"
    ));
}

#[test]
fn gpu_batch_rejects_retained_frames_across_source_groups_before_submission() {
    let tiles = [
        MockTile {
            source: 1,
            entropy_byte: 10,
        },
        MockTile {
            source: 2,
            entropy_byte: 11,
        },
        MockTile {
            source: 3,
            entropy_byte: 12,
        },
    ];
    let mut adapter = RecordingAdapter {
        output_dimensions: Some((4_096, 4_096)),
        ..RecordingAdapter::default()
    };

    let error = encode_jpeg_baseline_gpu_batch(&tiles, mock_gpu_options(), &mut adapter)
        .expect_err("three retained frame capacities exceed the live host cap");

    assert!(matches!(
        error,
        JpegEncodeError::MemoryCapExceeded { requested, cap }
            if requested > cap && cap == j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES
    ));
    assert!(adapter.submissions.is_empty());
}
