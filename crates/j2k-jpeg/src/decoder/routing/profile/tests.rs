// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec;
use alloc::vec::Vec;

use super::test_support::{captured_profile_rows, use_test_profile_sink, CapturedProfileRow};
use super::{DecodeOutcome, DecodeProfileRecord, DownscaleFactor, Instant, OutputFormat, Rect};
use crate::error::Warning;

fn record(
    source_roi: Option<Rect>,
    output_rect: Rect,
    fmt: OutputFormat,
    downscale: DownscaleFactor,
    stride: usize,
    scratch_bytes: usize,
) -> DecodeProfileRecord {
    let start = Instant::now();
    DecodeProfileRecord {
        total_start: start,
        decode_start: start,
        source_dimensions: (640, 480),
        output_rect,
        stride,
        bytes_per_pixel: fmt.bytes_per_pixel(),
        scratch_bytes,
        fmt,
        downscale,
        source_roi,
    }
}

fn outcome(decoded: Rect) -> DecodeOutcome {
    DecodeOutcome {
        decoded,
        warnings: vec![Warning::MissingEoi, Warning::UnknownColorProfile],
    }
}

fn normalized_fields(row: CapturedProfileRow) -> Vec<(String, String)> {
    assert_eq!(row.op, "decode");
    assert_eq!(row.path, "cpu");
    let fields = row.fields.expect("profile fields remain bounded");
    let mut decode_us = None;
    let mut total_us = None;
    let fields = fields
        .into_iter()
        .map(|(key, value)| match key.as_str() {
            "decode_us" => {
                decode_us = Some(value.parse::<u128>().expect("decode timing is numeric"));
                (key, "<micros>".to_owned())
            }
            "total_us" => {
                total_us = Some(value.parse::<u128>().expect("total timing is numeric"));
                (key, "<micros>".to_owned())
            }
            _ => (key, value),
        })
        .collect();
    assert!(
        total_us.expect("total timing field") >= decode_us.expect("decode timing field"),
        "total elapsed time cannot precede decode elapsed time"
    );
    fields
}

fn owned_fields(fields: &[(&str, &str)]) -> Vec<(String, String)> {
    fields
        .iter()
        .map(|(key, value)| ((*key).to_owned(), (*value).to_owned()))
        .collect()
}

#[test]
fn emit_dispatches_full_profile_with_exact_bounded_fields() {
    let _sink = use_test_profile_sink();
    let output_rect = Rect {
        x: 0,
        y: 0,
        w: 320,
        h: 240,
    };
    record(
        None,
        output_rect,
        OutputFormat::Rgba8Scaled {
            alpha: 200,
            factor: DownscaleFactor::Half,
        },
        DownscaleFactor::Half,
        1280,
        4096,
    )
    .emit(&outcome(Rect::full((640, 480))));

    let [row] = captured_profile_rows()
        .try_into()
        .expect("one full profile row");
    assert_eq!(row.operation, "jpeg_decode_full_fields");
    assert_eq!(
        normalized_fields(row),
        owned_fields(&[
            ("mode", "full"),
            ("fmt", "Rgba8"),
            ("downscale", "half"),
            ("source_width", "640"),
            ("source_height", "480"),
            ("output_width", "320"),
            ("output_height", "240"),
            ("stride", "1280"),
            ("bpp", "4"),
            ("scratch_bytes", "4096"),
            ("output_bytes", "307200"),
            ("decode_us", "<micros>"),
            ("total_us", "<micros>"),
            ("warnings", "2"),
        ])
    );
}

#[test]
fn emit_dispatches_region_and_scaled_region_profiles_with_exact_fields() {
    let _sink = use_test_profile_sink();
    let roi = Rect {
        x: 11,
        y: 13,
        w: 101,
        h: 51,
    };
    record(
        Some(roi),
        roi,
        OutputFormat::Rgb8,
        DownscaleFactor::Full,
        303,
        1234,
    )
    .emit(&outcome(roi));
    record(
        Some(roi),
        Rect {
            x: 2,
            y: 3,
            w: 26,
            h: 13,
        },
        OutputFormat::Rgba8Scaled {
            alpha: 200,
            factor: DownscaleFactor::Quarter,
        },
        DownscaleFactor::Quarter,
        104,
        5678,
    )
    .emit(&outcome(roi));

    let [region, scaled] = captured_profile_rows()
        .try_into()
        .expect("region modes emit two rows");
    assert_eq!(region.operation, "jpeg_decode_region_fields");
    assert_eq!(
        normalized_fields(region),
        owned_fields(&[
            ("mode", "region"),
            ("fmt", "Rgb8"),
            ("downscale", "full"),
            ("source_width", "640"),
            ("source_height", "480"),
            ("roi_x", "11"),
            ("roi_y", "13"),
            ("roi_w", "101"),
            ("roi_h", "51"),
            ("output_width", "101"),
            ("output_height", "51"),
            ("stride", "303"),
            ("bpp", "3"),
            ("scratch_bytes", "1234"),
            ("output_bytes", "15453"),
            ("decode_us", "<micros>"),
            ("total_us", "<micros>"),
            ("warnings", "2"),
        ])
    );
    assert_eq!(scaled.operation, "jpeg_decode_region_fields");
    assert_eq!(
        normalized_fields(scaled),
        owned_fields(&[
            ("mode", "region_scaled"),
            ("fmt", "Rgba8"),
            ("downscale", "quarter"),
            ("source_width", "640"),
            ("source_height", "480"),
            ("roi_x", "11"),
            ("roi_y", "13"),
            ("roi_w", "101"),
            ("roi_h", "51"),
            ("output_width", "26"),
            ("output_height", "13"),
            ("stride", "104"),
            ("bpp", "4"),
            ("scratch_bytes", "5678"),
            ("output_bytes", "1352"),
            ("decode_us", "<micros>"),
            ("total_us", "<micros>"),
            ("warnings", "2"),
        ])
    );
}

#[test]
fn profile_capture_is_thread_local_nested_and_restored() {
    let _outer = use_test_profile_sink();
    record(
        None,
        Rect::full((1, 1)),
        OutputFormat::Gray8,
        DownscaleFactor::Full,
        1,
        0,
    )
    .emit(&outcome(Rect::full((1, 1))));
    assert!(std::thread::spawn(captured_profile_rows)
        .join()
        .expect("profile sink child thread")
        .is_empty());

    {
        let _inner = use_test_profile_sink();
        record(
            Some(Rect::full((1, 1))),
            Rect::full((1, 1)),
            OutputFormat::Gray8,
            DownscaleFactor::Full,
            1,
            0,
        )
        .emit(&outcome(Rect::full((1, 1))));
        assert_eq!(captured_profile_rows().len(), 1);
    }

    record(
        None,
        Rect::full((1, 1)),
        OutputFormat::Gray8,
        DownscaleFactor::Full,
        1,
        0,
    )
    .emit(&outcome(Rect::full((1, 1))));
    assert_eq!(captured_profile_rows().len(), 2);
}
