// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k::{J2kDecoder, J2kError, J2kView};
use j2k_core::{
    Colorspace, CompressedPayloadKind, CompressedTransferSyntax, InputError, PassthroughDecision,
    PassthroughRequirements,
};
use j2k_native::{encode, encode_htj2k, EncodeOptions};
use j2k_test_support::{minimal_j2k_codestream, minimal_jp2, wrap_jp2_codestream};

fn codestream_without_siz() -> Vec<u8> {
    let mut bytes = vec![0xFF, 0x4F];
    bytes.extend_from_slice(&[
        0xFF, 0x52, 0x00, 0x0C, 0x00, 0x00, 0x00, 0x01, 0x01, 0x05, 0x04, 0x04, 0x00, 0x01,
    ]);
    bytes.extend_from_slice(&[0xFF, 0x90, 0x00, 0x0A, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]);
    bytes
}

fn codestream_without_cod() -> Vec<u8> {
    let mut bytes = vec![0xFF, 0x4F];
    let mut siz = Vec::new();
    push_u16(&mut siz, 0);
    push_u32(&mut siz, 128);
    push_u32(&mut siz, 64);
    push_u32(&mut siz, 0);
    push_u32(&mut siz, 0);
    push_u32(&mut siz, 64);
    push_u32(&mut siz, 64);
    push_u32(&mut siz, 0);
    push_u32(&mut siz, 0);
    push_u16(&mut siz, 3);
    for _ in 0..3 {
        siz.extend_from_slice(&[0x07, 0x01, 0x01]);
    }
    bytes.extend_from_slice(&[0xFF, 0x51]);
    push_u16(&mut bytes, (siz.len() + 2) as u16);
    bytes.extend_from_slice(&siz);
    bytes.extend_from_slice(&[0xFF, 0x90, 0x00, 0x0A, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]);
    bytes
}

fn codestream_truncated_after_main_header() -> Vec<u8> {
    let mut bytes = minimal_j2k_codestream();
    bytes.truncate(bytes.len() - 10);
    bytes
}

fn codestream_with_component_count(component_count: u16) -> Vec<u8> {
    let mut bytes = minimal_j2k_codestream();
    let siz = bytes
        .windows(2)
        .position(|marker| marker == [0xFF, 0x51])
        .expect("SIZ marker");
    let lsiz = u16::from_be_bytes([bytes[siz + 2], bytes[siz + 3]]) as usize;
    let component_start = siz + 40;
    let component_end = siz + 2 + lsiz;
    let first_component = bytes[component_start..component_start + 3].to_vec();
    let new_lsiz = 38 + usize::from(component_count) * 3;

    bytes[siz + 2..siz + 4].copy_from_slice(&(new_lsiz as u16).to_be_bytes());
    bytes[siz + 38..siz + 40].copy_from_slice(&component_count.to_be_bytes());
    bytes.splice(
        component_start..component_end,
        first_component.repeat(usize::from(component_count)),
    );
    bytes
}

fn jp2_with_truncated_jp2h_child_box() -> Vec<u8> {
    let codestream = minimal_j2k_codestream();
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&[0, 0, 0, 12, b'j', b'P', b' ', b' ', 0x0D, 0x0A, 0x87, 0x0A]);
    bytes.extend_from_slice(&[
        0, 0, 0, 20, b'f', b't', b'y', b'p', b'j', b'p', b'2', b' ', 0, 0, 0, 0, b'j', b'p', b'2',
        b' ',
    ]);
    bytes.extend_from_slice(&[
        0, 0, 0, 16, b'j', b'p', b'2', b'h', 0, 0, 0, 32, b'i', b'h', b'd', b'r',
    ]);
    let len = (8 + codestream.len()) as u32;
    bytes.extend_from_slice(&len.to_be_bytes());
    bytes.extend_from_slice(b"jp2c");
    bytes.extend_from_slice(&codestream);
    bytes
}

fn ht_codestream() -> Vec<u8> {
    let pixels = [10_u8, 20, 30, 40];
    encode_htj2k(&pixels, 2, 2, 1, 8, false, &EncodeOptions::default()).expect("encode ht")
}

fn classic_lossless_codestream() -> Vec<u8> {
    let pixels = [10_u8, 20, 30, 40];
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    encode(&pixels, 2, 2, 1, 8, false, &options).expect("encode classic j2k")
}

fn ht_jp2() -> Vec<u8> {
    wrap_jp2_codestream(&ht_codestream(), 2, 2, 1, 8, 17)
}

fn classic_lossless_jp2() -> Vec<u8> {
    wrap_jp2_codestream(&classic_lossless_codestream(), 2, 2, 1, 8, 17)
}

fn jp2_with_jp2c_before_jp2h() -> Vec<u8> {
    let codestream = minimal_j2k_codestream();
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&[0, 0, 0, 12, b'j', b'P', b' ', b' ', 0x0D, 0x0A, 0x87, 0x0A]);
    bytes.extend_from_slice(&[
        0, 0, 0, 20, b'f', b't', b'y', b'p', b'j', b'p', b'2', b' ', 0, 0, 0, 0, b'j', b'p', b'2',
        b' ',
    ]);
    let len = (8 + codestream.len()) as u32;
    bytes.extend_from_slice(&len.to_be_bytes());
    bytes.extend_from_slice(b"jp2c");
    bytes.extend_from_slice(&codestream);
    bytes.extend_from_slice(&[
        0, 0, 0, 45, b'j', b'p', b'2', b'h', 0, 0, 0, 22, b'i', b'h', b'd', b'r', 0, 0, 0, 64, 0,
        0, 0, 128, 0, 3, 7, 7, 0, 0, 0, 0, 0, 15, b'c', b'o', b'l', b'r', 1, 0, 0, 0, 0, 0, 16,
    ]);
    bytes
}

fn push_u16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_be_bytes());
}

fn push_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_be_bytes());
}

fn rewrite_siz_u32(bytes: &mut [u8], payload_offset: usize, value: u32) {
    let siz = bytes
        .windows(2)
        .position(|marker| marker == [0xFF, 0x51])
        .expect("SIZ marker");
    let offset = siz + 4 + payload_offset;
    bytes[offset..offset + 4].copy_from_slice(&value.to_be_bytes());
}

fn rewrite_component_sampling(bytes: &mut [u8], component: usize, x_rsiz: u8, y_rsiz: u8) {
    let siz = bytes
        .windows(2)
        .position(|marker| marker == [0xFF, 0x51])
        .expect("SIZ marker");
    let component_offset = siz + 40 + component * 3;
    bytes[component_offset + 1] = x_rsiz;
    bytes[component_offset + 2] = y_rsiz;
}

#[test]
fn inspect_raw_codestream_reports_core_info() {
    let info = J2kDecoder::inspect(&minimal_j2k_codestream()).expect("codestream inspect");
    assert_eq!(info.dimensions, (128, 64));
    assert_eq!(info.components, 3);
    assert_eq!(info.bit_depth, 8);
    assert_eq!(info.colorspace, Colorspace::Rct);
    assert_eq!(info.resolution_levels, 6);
    let tiles = info.tile_layout.expect("tile layout");
    assert_eq!(tiles.tile_width, 64);
    assert_eq!(tiles.tile_height, 64);
    assert_eq!(tiles.tiles_x, 2);
    assert_eq!(tiles.tiles_y, 1);
}

#[test]
fn inspect_raw_codestream_rejects_zero_component_sampling() {
    let mut bytes = minimal_j2k_codestream();
    rewrite_component_sampling(&mut bytes, 0, 0, 1);

    let err = J2kDecoder::inspect(&bytes).expect_err("zero sampling must reject");

    assert!(matches!(err, J2kError::InvalidSiz { .. }));
}

#[test]
fn inspect_raw_codestream_rejects_oversized_dimensions() {
    let mut bytes = minimal_j2k_codestream();
    rewrite_siz_u32(&mut bytes, 2, 60_001);

    let err = J2kDecoder::inspect(&bytes).expect_err("oversized width must reject");

    assert!(matches!(err, J2kError::InvalidSiz { .. }));
}

#[test]
fn inspect_raw_codestream_rejects_bad_tile_origin() {
    let mut bytes = minimal_j2k_codestream();
    rewrite_siz_u32(&mut bytes, 26, 1);

    let err = J2kDecoder::inspect(&bytes).expect_err("bad tile origin must reject");

    assert!(matches!(err, J2kError::InvalidSiz { .. }));
}

#[test]
fn inspect_raw_codestream_rejects_tile_extent_overflow() {
    let mut bytes = minimal_j2k_codestream();
    rewrite_siz_u32(&mut bytes, 2, u32::MAX);
    rewrite_siz_u32(&mut bytes, 10, u32::MAX - 1);
    rewrite_siz_u32(&mut bytes, 18, 10);
    rewrite_siz_u32(&mut bytes, 26, u32::MAX - 2);

    let err = J2kDecoder::inspect(&bytes).expect_err("overflow must reject");

    assert!(matches!(err, J2kError::InvalidSiz { .. }));
}

#[test]
fn inspect_raw_codestream_rejects_excessive_tile_count() {
    let mut bytes = minimal_j2k_codestream();
    rewrite_siz_u32(&mut bytes, 2, 257);
    rewrite_siz_u32(&mut bytes, 6, 257);
    rewrite_siz_u32(&mut bytes, 18, 1);
    rewrite_siz_u32(&mut bytes, 22, 1);

    let err = J2kDecoder::inspect(&bytes).expect_err("tile count must reject");

    assert!(matches!(err, J2kError::InvalidSiz { .. }));
}

#[test]
fn inspect_spec_valid_component_count_above_u8_is_unsupported() {
    let bytes = codestream_with_component_count(256);
    let err = J2kDecoder::inspect(&bytes).expect_err("component count unsupported");

    let J2kError::Unsupported(unsupported) = err else {
        panic!("expected unsupported component count, got {err:?}");
    };
    assert_eq!(unsupported.what, "component count > 255");
}

#[test]
fn inspect_jp2_uses_container_colorspace() {
    let info = J2kDecoder::inspect(&minimal_jp2()).expect("jp2 inspect");
    assert_eq!(info.dimensions, (128, 64));
    assert_eq!(info.colorspace, Colorspace::SRgb);
}

#[test]
fn view_and_decoder_share_inspect_info() {
    let bytes = ht_jp2();
    let view = J2kView::parse(&bytes).expect("view");
    let dec = J2kDecoder::from_view(view).expect("decoder");
    assert_eq!(dec.info().dimensions, (2, 2));
    assert_eq!(dec.info().components, 1);
}

#[test]
fn codestream_without_siz_is_rejected() {
    let err = J2kDecoder::inspect(&codestream_without_siz()).unwrap_err();
    assert!(matches!(
        err,
        J2kError::MissingRequiredMarker { marker: "SIZ" }
    ));
}

#[test]
fn bad_jp2_signature_is_rejected() {
    let mut bad = minimal_jp2();
    bad[11] = 0x00;
    let err = J2kDecoder::inspect(&bad).unwrap_err();
    assert!(matches!(err, J2kError::InvalidBox { .. }));
}

#[test]
fn codestream_without_cod_is_rejected() {
    let err = J2kDecoder::inspect(&codestream_without_cod()).unwrap_err();
    assert!(matches!(
        err,
        J2kError::MissingRequiredMarker { marker: "COD" }
    ));
}

#[test]
fn codestream_truncated_after_main_header_is_rejected() {
    let err = J2kDecoder::inspect(&codestream_truncated_after_main_header()).unwrap_err();
    assert!(matches!(
        err,
        J2kError::Input(j2k_core::InputError::TruncatedAt { .. })
    ));
}

#[test]
fn jp2_with_truncated_nested_header_box_returns_error() {
    let err = J2kDecoder::inspect(&jp2_with_truncated_jp2h_child_box())
        .expect_err("truncated nested jp2h child box");

    assert!(matches!(
        err,
        J2kError::Input(InputError::TruncatedAt {
            segment: "box payload",
            ..
        })
    ));
}

#[test]
fn jp2_with_codestream_before_header_is_rejected() {
    let err = J2kDecoder::inspect(&jp2_with_jp2c_before_jp2h()).unwrap_err();
    assert!(matches!(err, J2kError::InvalidBox { .. }));
}

#[test]
fn inspect_ht_codestream_reports_core_info() {
    let info = J2kDecoder::inspect(&ht_codestream()).expect("ht inspect");
    assert_eq!(info.dimensions, (2, 2));
    assert_eq!(info.components, 1);
    assert_eq!(info.bit_depth, 8);
    assert_eq!(info.colorspace, Colorspace::SGray);
}

#[test]
fn inspect_ht_jp2_reports_core_info() {
    let info = J2kDecoder::inspect(&ht_jp2()).expect("ht jp2 inspect");
    assert_eq!(info.dimensions, (2, 2));
    assert_eq!(info.components, 1);
    assert_eq!(info.colorspace, Colorspace::SGray);
}

#[test]
fn j2k_view_exposes_classic_lossless_codestream_passthrough_candidate() {
    let bytes = classic_lossless_codestream();
    let view = J2kView::parse(&bytes).expect("classic j2k view");
    let candidate = view
        .passthrough_candidate()
        .expect("classic j2k passthrough candidate");
    let requirements = PassthroughRequirements::new(
        CompressedTransferSyntax::Jpeg2000Lossless,
        CompressedPayloadKind::Jpeg2000Codestream,
    )
    .with_dimensions((2, 2))
    .with_components(1)
    .with_bit_depth(8);

    assert_eq!(
        candidate.evaluate(&requirements),
        PassthroughDecision::Copy {
            bytes: bytes.as_slice()
        }
    );
    assert_eq!(
        candidate.transfer_syntax(),
        CompressedTransferSyntax::Jpeg2000Lossless
    );
    assert_eq!(
        candidate.payload_kind(),
        CompressedPayloadKind::Jpeg2000Codestream
    );
}

#[test]
fn j2k_view_exposes_ht_lossless_codestream_passthrough_candidate() {
    let bytes = ht_codestream();
    let view = J2kView::parse(&bytes).expect("htj2k view");

    assert_eq!(
        view.passthrough_candidate()
            .expect("htj2k passthrough candidate")
            .transfer_syntax(),
        CompressedTransferSyntax::HtJpeg2000Lossless
    );
}

#[test]
fn jp2_file_is_not_eligible_for_raw_dicom_codestream_copy() {
    let bytes = classic_lossless_jp2();
    let view = J2kView::parse(&bytes).expect("jp2 view");
    let requirements = PassthroughRequirements::new(
        CompressedTransferSyntax::Jpeg2000Lossless,
        CompressedPayloadKind::Jpeg2000Codestream,
    );

    assert_eq!(
        view.passthrough_candidate()
            .expect("jp2 passthrough candidate")
            .copy_bytes_if_eligible(&requirements),
        Err(j2k_core::PassthroughRejectReason::PayloadKindMismatch {
            source: CompressedPayloadKind::Jp2File,
            destination: CompressedPayloadKind::Jpeg2000Codestream,
        })
    );
}
