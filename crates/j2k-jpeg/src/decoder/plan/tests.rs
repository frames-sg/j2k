// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;

use crate::{Decoder, JpegError, MarkerKind, SofKind};

#[derive(Clone, Copy)]
struct LosslessFixture {
    frame_components: u8,
    scan_components: u8,
    precision: u8,
    predictor: u8,
    spectral_end: u8,
    scan_component_id: u8,
    dc_table: u8,
}

impl Default for LosslessFixture {
    fn default() -> Self {
        Self {
            frame_components: 1,
            scan_components: 1,
            precision: 8,
            predictor: 1,
            spectral_end: 0,
            scan_component_id: 1,
            dc_table: 0,
        }
    }
}

fn push_segment(bytes: &mut Vec<u8>, marker: u8, payload: &[u8]) {
    let length = u16::try_from(payload.len() + 2).expect("test segment length fits u16");
    bytes.extend_from_slice(&[0xff, marker]);
    bytes.extend_from_slice(&length.to_be_bytes());
    bytes.extend_from_slice(payload);
}

fn lossless_jpeg(config: LosslessFixture) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&[0xff, 0xd8]);

    let mut sof = Vec::new();
    sof.extend_from_slice(&[config.precision, 0, 1, 0, 1, config.frame_components]);
    for component in 1..=config.frame_components {
        sof.extend_from_slice(&[component, 0x11, 0]);
    }
    push_segment(&mut bytes, 0xc3, &sof);

    let mut dht = Vec::new();
    dht.extend_from_slice(&[0, 1]);
    dht.extend(core::iter::repeat_n(0, 15));
    dht.push(0);
    push_segment(&mut bytes, 0xc4, &dht);

    let mut sos = Vec::new();
    sos.push(config.scan_components);
    for index in 0..config.scan_components {
        let component = if index == 0 {
            config.scan_component_id
        } else {
            index + 1
        };
        sos.extend_from_slice(&[component, config.dc_table << 4]);
    }
    sos.extend_from_slice(&[config.predictor, config.spectral_end, 0]);
    push_segment(&mut bytes, 0xda, &sos);
    bytes.extend_from_slice(&[0, 0xff, 0xd9]);
    bytes
}

#[test]
fn valid_lossless_plan_records_predictor_and_scan_geometry() {
    let bytes = lossless_jpeg(LosslessFixture::default());
    let decoder = Decoder::new(&bytes).expect("valid lossless fixture");
    let lossless = decoder.lossless_plan.as_ref().expect("lossless plan");

    assert_eq!(decoder.info().sof_kind, SofKind::Lossless);
    assert_eq!(lossless.predictor, 1);
    assert_eq!(lossless.bit_depth, 8);
    assert_eq!(lossless.dimensions, (1, 1));
    assert_eq!(decoder.plan.components.len(), 1);
    assert_eq!(decoder.plan.components[0].output_index, 0);
}

#[test]
fn lossless_plan_rejects_predictor_outside_the_spec_range() {
    let bytes = lossless_jpeg(LosslessFixture {
        predictor: 0,
        ..LosslessFixture::default()
    });

    assert_eq!(
        Decoder::new(&bytes).expect_err("predictor zero must fail"),
        JpegError::UnsupportedPredictor { predictor: 0 }
    );
}

#[test]
fn lossless_plan_rejects_nonzero_spectral_end() {
    let bytes = lossless_jpeg(LosslessFixture {
        spectral_end: 1,
        ..LosslessFixture::default()
    });

    assert_eq!(
        Decoder::new(&bytes).expect_err("lossless spectral end must be zero"),
        JpegError::NotImplemented {
            sof: SofKind::Lossless
        }
    );
}

#[test]
fn lossless_plan_rejects_scan_component_count_mismatch() {
    let bytes = lossless_jpeg(LosslessFixture {
        frame_components: 3,
        scan_components: 1,
        ..LosslessFixture::default()
    });

    assert_eq!(
        Decoder::new(&bytes).expect_err("color scan must carry three components"),
        JpegError::UnsupportedComponentCount { count: 1 }
    );
}

#[test]
fn lossless_plan_reports_unknown_scan_component() {
    let bytes = lossless_jpeg(LosslessFixture {
        scan_component_id: 2,
        ..LosslessFixture::default()
    });
    let error = Decoder::new(&bytes).expect_err("unknown scan component must fail");

    assert!(matches!(
        error,
        JpegError::UnknownScanComponent { component: 2, .. }
    ));
    assert!(error.offset().is_some());
}

#[test]
fn lossless_plan_reports_missing_huffman_table() {
    let bytes = lossless_jpeg(LosslessFixture {
        dc_table: 1,
        ..LosslessFixture::default()
    });

    assert_eq!(
        Decoder::new(&bytes).expect_err("undefined DC table must fail"),
        JpegError::MissingHuffmanTable {
            component: 1,
            class: 0,
            id: 1
        }
    );
}

#[test]
fn component_lookup_preserves_frame_order_and_missing_state() {
    assert_eq!(super::find_component_index(&[3, 1, 2], 1), Some(1));
    assert_eq!(super::find_component_index(&[3, 1, 2], 4), None);
}

#[test]
fn lossless_plan_requires_start_of_scan_state() {
    let bytes = [0xff, 0xd8, 0xff, 0xd9];
    let error = Decoder::new(&bytes).expect_err("stream without SOF/SOS must fail");
    assert!(matches!(
        error,
        JpegError::MissingMarker {
            marker: MarkerKind::Sof | MarkerKind::Sos
        }
    ));
}
