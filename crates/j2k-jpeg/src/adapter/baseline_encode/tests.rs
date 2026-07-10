// SPDX-License-Identifier: MIT OR Apache-2.0

use super::planning::{
    jpeg_baseline_gpu_encode_batch_plan, jpeg_baseline_gpu_encode_params,
    same_source_buffer_batch_end,
};
use super::tables::jpeg_baseline_sampling_for;
use super::types::{JpegBaselineGpuEncodeError, JpegBaselineGpuEncodeTile};
use super::validation::validate_jpeg_baseline_gpu_encode_tile;
use crate::encoder::{JpegBackend, JpegEncodeOptions, JpegSubsampling};
use crate::PixelFormat;

#[test]
fn baseline_encode_modules_stay_focused_and_fragment_free() {
    const ROOT: &str = include_str!("../baseline_encode.rs");
    const FRAME: &str = include_str!("frame.rs");
    const ORCHESTRATE: &str = include_str!("orchestrate.rs");
    const PLANNING: &str = include_str!("planning.rs");
    const TABLES: &str = include_str!("tables.rs");
    const TYPES: &str = include_str!("types.rs");
    const VALIDATION: &str = include_str!("validation.rs");

    let modules = [
        ("baseline_encode.rs", ROOT, 45usize),
        ("baseline_encode/frame.rs", FRAME, 220),
        ("baseline_encode/orchestrate.rs", ORCHESTRATE, 170),
        ("baseline_encode/planning.rs", PLANNING, 240),
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
}

#[test]
fn baseline_encode_error_and_marker_literals_remain_path_owned() {
    const FRAME: &str = include_str!("frame.rs");
    const ORCHESTRATE: &str = include_str!("orchestrate.rs");
    const PLANNING: &str = include_str!("planning.rs");

    for marker in ["DQT", "DRI", "SOF0", "DHT", "SOS"] {
        assert_eq!(
            FRAME.matches(&format!("\"{marker}\"")).count(),
            1,
            "{marker} segment name moved or duplicated"
        );
    }
    assert_eq!(
        ORCHESTRATE
            .matches("GPU JPEG baseline batch returned the wrong number of entropy chunks")
            .count(),
        1
    );
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
