// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

use j2k::{
    encode_j2k_lossless, EncodeBackendPreference, J2kBlockCodingMode, J2kEncodeValidation,
    J2kLosslessEncodeOptions, J2kLosslessSamples,
};
use j2k_native::J2kCodestreamHeaderMetadata;
use j2k_test_support::wrap_jp2_codestream;

#[cfg(unix)]
use super::validate_case_encoder;
use super::{
    cod_profile, codestream_segment_payload, decode_encoded_output, parse_cod_profile,
    validate_cod_profile, validate_encoded_profile, validate_header_profile, CodProfile,
};
#[cfg(unix)]
use crate::encode_compare::EncoderTool;
use crate::encode_compare::{EncoderKind, ImageCase};

static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(0);

fn temp_dir(label: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!(
        "j2k-encode-validation-{label}-{}-{}",
        std::process::id(),
        NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&root).expect("create validation test directory");
    root
}

fn image_case(pixels: Vec<u8>) -> ImageCase {
    ImageCase {
        name: "gray-fixture".to_string(),
        input_source: "external:fixture".to_string(),
        corpus_category: "natural-image".to_string(),
        corpus_name: "fixture".to_string(),
        license_status: "cc0".to_string(),
        source_command: "fixture".to_string(),
        manifest_status: "covered".to_string(),
        source_format: "pgm".to_string(),
        width: 128,
        height: 128,
        components: 1,
        pixels,
        pnm_path: PathBuf::from("fixture.pgm"),
    }
}

fn valid_header() -> J2kCodestreamHeaderMetadata {
    J2kCodestreamHeaderMetadata {
        dimensions: (128, 128),
        components: 1,
        bit_depth: 8,
        tile_size: (128, 128),
        tile_count: (1, 1),
        component_info: Vec::new(),
        resolution_levels: 3,
        has_mct: false,
        reversible: true,
        high_throughput: false,
    }
}

fn valid_cod() -> CodProfile {
    CodProfile {
        scod: 0,
        progression_order: 0,
        decomposition_levels: 2,
        code_block_width_exp: 4,
        code_block_height_exp: 4,
        code_block_style: 0,
        transform: 1,
    }
}

#[test]
fn header_profile_accepts_contract_and_reports_each_mismatch() {
    let gray = image_case(vec![0; 256]);
    validate_header_profile(&valid_header(), &gray, EncoderKind::J2k).expect("valid header");

    let mut cases = Vec::new();
    let mut header = valid_header();
    header.dimensions = (127, 128);
    cases.push((header, "profile dimensions"));
    let mut header = valid_header();
    header.components = 3;
    cases.push((header, "profile components"));
    let mut header = valid_header();
    header.tile_count = (2, 1);
    cases.push((header, "profile tile count"));
    let mut header = valid_header();
    header.resolution_levels = 2;
    cases.push((header, "profile resolution levels"));
    let mut header = valid_header();
    header.reversible = false;
    cases.push((header, "not reversible 5/3"));
    let mut header = valid_header();
    header.high_throughput = true;
    cases.push((header, "used HT block coding"));
    let mut header = valid_header();
    header.has_mct = true;
    cases.push((header, "grayscale profile unexpectedly enables MCT"));

    for (header, expected) in cases {
        let error = validate_header_profile(&header, &gray, EncoderKind::OpenJpeg)
            .expect_err("invalid header profile");
        assert!(error.contains(expected), "unexpected error: {error}");
    }

    let mut rgb = image_case(vec![0; 16 * 16 * 3]);
    rgb.components = 3;
    let mut header = valid_header();
    header.components = 3;
    let error =
        validate_header_profile(&header, &rgb, EncoderKind::Grok).expect_err("RGB requires MCT");
    assert!(error.contains("missing RGB reversible color transform"));
}

#[test]
fn cod_profile_contract_reports_each_non_classic_option() {
    let case = image_case(vec![0; 256]);
    validate_cod_profile(&valid_cod(), &case, EncoderKind::J2k).expect("valid COD profile");

    let mut cases = Vec::new();
    let mut cod = valid_cod();
    cod.progression_order = 1;
    cases.push((cod, "progression order"));
    let mut cod = valid_cod();
    cod.decomposition_levels = 1;
    cases.push((cod, "decomposition levels"));
    let mut cod = valid_cod();
    cod.code_block_width_exp = 5;
    cases.push((cod, "code-block exponents"));
    let mut cod = valid_cod();
    cod.code_block_style = 0x40;
    cases.push((cod, "HT code-block style"));
    let mut cod = valid_cod();
    cod.transform = 0;
    cases.push((cod, "profile transform"));
    for (scod, expected) in [
        (0x01, "overrides precincts"),
        (0x02, "enables SOP markers"),
        (0x04, "enables EPH markers"),
    ] {
        let mut cod = valid_cod();
        cod.scod = scod;
        cases.push((cod, expected));
    }

    for (cod, expected) in cases {
        let error = validate_cod_profile(&cod, &case, EncoderKind::Kakadu)
            .expect_err("invalid COD profile");
        assert!(error.contains(expected), "unexpected error: {error}");
    }
}

#[test]
fn codestream_parser_handles_known_unknown_and_malformed_segments() {
    let codestream = [
        0xFF, 0x4F, 0xFF, 0x51, 0x00, 0x04, 0xAA, 0xBB, 0xFF, 0x52, 0x00, 0x0C, 0, 0, 0, 0, 0, 2,
        4, 4, 0, 1,
    ];
    let profile = cod_profile(&codestream).expect("parse COD after generic segment");
    assert_eq!(profile.progression_order, 0);
    assert_eq!(profile.decomposition_levels, 2);
    assert_eq!(profile.transform, 1);

    let mut offset = 2;
    assert_eq!(
        codestream_segment_payload(&[0, 0, 0, 4, 7, 8], &mut offset, "fixture"),
        Ok(&[7, 8][..])
    );
    assert_eq!(offset, 6);

    for (bytes, expected) in [
        (&[0x00, 0x00][..], "missing SOC marker"),
        (&[0xFF, 0x4F, 0x00, 0x51][..], "invalid codestream marker"),
        (&[0xFF, 0x4F, 0xFF, 0x90][..], "missing COD marker"),
        (
            &[0xFF, 0x4F, 0xFF, 0x51][..],
            "truncated marker segment segment length",
        ),
        (
            &[0xFF, 0x4F, 0xFF, 0x51, 0, 1][..],
            "invalid marker segment segment length",
        ),
        (
            &[0xFF, 0x4F, 0xFF, 0x51, 0, 5, 1][..],
            "truncated marker segment",
        ),
        (
            &[0xFF, 0x4F, 0xFF, 0x52, 0, 3, 0][..],
            "COD payload is shorter",
        ),
    ] {
        let error = cod_profile(bytes).err().expect("malformed codestream");
        assert!(error.contains(expected), "unexpected error: {error}");
    }

    let mut overflow_offset = usize::MAX;
    let error = codestream_segment_payload(&[], &mut overflow_offset, "fixture")
        .expect_err("length offset overflow");
    assert!(error.contains("length offset overflow"));
    assert!(parse_cod_profile(&[0; 9])
        .err()
        .expect("short COD payload error")
        .contains("shorter"));
}

fn encoded_fixture(case: &ImageCase) -> Vec<u8> {
    let samples = J2kLosslessSamples::new(
        &case.pixels,
        case.width,
        case.height,
        u16::from(case.components),
        8,
        false,
    )
    .expect("fixture samples");
    let options = J2kLosslessEncodeOptions::default()
        .with_backend(EncodeBackendPreference::CpuOnly)
        .with_block_coding_mode(J2kBlockCodingMode::Classic)
        .with_max_decomposition_levels(Some(2))
        .with_validation(J2kEncodeValidation::External);
    let encoded = encode_j2k_lossless(samples, &options).expect("encode fixture");
    wrap_jp2_codestream(
        &encoded.codestream,
        case.width,
        case.height,
        u16::from(case.components),
        8,
        16,
    )
}

#[test]
fn encoded_profile_and_decode_accept_a_local_lossless_fixture() {
    let root = temp_dir("decode");
    let pixels = (0_usize..128 * 128)
        .map(|index| u8::try_from(index % 251).expect("fixture value fits u8"))
        .collect::<Vec<_>>();
    let case = image_case(pixels.clone());
    let path = root.join("fixture.jp2");
    fs::write(&path, encoded_fixture(&case)).expect("write encoded fixture");

    validate_encoded_profile(&path, &case, EncoderKind::J2k).expect("valid encoded profile");
    assert_eq!(decode_encoded_output(&path, &case), Ok(pixels));
    let error = validate_encoded_profile(&root.join("missing.jp2"), &case, EncoderKind::J2k)
        .expect_err("missing output");
    assert!(error.contains("read"));
    let raw_codestream = root.join("raw.j2k");
    fs::write(&raw_codestream, [0xFF, 0x4F, 0xFF, 0xD9]).expect("write raw codestream");
    let error = validate_encoded_profile(&raw_codestream, &case, EncoderKind::J2k)
        .expect_err("JP2 container required");
    assert!(error.contains("not a JP2 container") || error.contains("extract"));
}

#[cfg(unix)]
#[test]
fn case_validation_runs_a_hermetic_encoder_and_detects_pixel_mismatch() {
    use std::os::unix::fs::PermissionsExt;

    let root = temp_dir("encoder");
    let pixels = (0_usize..128 * 128)
        .map(|index| u8::try_from(index % 251).expect("fixture value fits u8"))
        .collect::<Vec<_>>();
    let case = image_case(pixels);
    let encoded = root.join("source.jp2");
    fs::write(&encoded, encoded_fixture(&case)).expect("write source fixture");
    let program = root.join("encoder.sh");
    fs::write(
        &program,
        format!(
            "#!/bin/sh\nout=''\nwhile [ $# -gt 0 ]; do if [ \"$1\" = --output ]; then shift; out=$1; fi; shift; done\ncp '{}' \"$out\"\n",
            encoded.display()
        ),
    )
    .expect("write fake encoder");
    let mut permissions = fs::metadata(&program)
        .expect("encoder metadata")
        .permissions();
    permissions.set_mode(0o700);
    fs::set_permissions(&program, permissions).expect("make encoder executable");
    let tool = EncoderTool {
        kind: EncoderKind::J2k,
        program,
        available: true,
    };

    validate_case_encoder(&case, &tool, &root).expect("matching round trip");
    let mut mismatched = case;
    mismatched.pixels[0] ^= 0xFF;
    let error =
        validate_case_encoder(&mismatched, &tool, &root).expect_err("mismatched source pixels");
    assert!(error.contains("did not round-trip losslessly"));
}
