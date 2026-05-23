// SPDX-License-Identifier: Apache-2.0

use signinum_j2k_native::{DecodeSettings, Image};
use signinum_jpeg::{
    encode_jpeg_baseline, JpegBackend, JpegEncodeOptions, JpegSamples, JpegSubsampling,
};
use signinum_transcode::{jpeg_to_htj2k, EncodedTranscode, JpegToHtj2kOptions};
use std::{
    env, fs,
    path::PathBuf,
    process::{Command, Output},
    time::{SystemTime, UNIX_EPOCH},
};

#[test]
fn grayscale_8x8_jpeg_transcodes_to_decodable_htj2k() {
    let jpeg = include_bytes!("../../signinum-jpeg/fixtures/conformance/grayscale_8x8.jpg");

    let encoded = jpeg_to_htj2k(jpeg, &JpegToHtj2kOptions::default())
        .expect("transcode grayscale JPEG to HTJ2K");
    let decoded = Image::new(&encoded.codestream, &DecodeSettings::default())
        .expect("native parser accepts generated HTJ2K")
        .decode_native()
        .expect("native decoder accepts generated HTJ2K");

    assert_eq!((encoded.report.width, encoded.report.height), (8, 8));
    assert_eq!(encoded.report.component_count, 1);
    assert_eq!((decoded.width, decoded.height), (8, 8));
    assert_eq!(decoded.num_components, 1);
}

#[test]
fn grayscale_8x8_transcode_reports_opt_in_float_reference_metrics() {
    let jpeg = include_bytes!("../../signinum-jpeg/fixtures/conformance/grayscale_8x8.jpg");
    let options = JpegToHtj2kOptions {
        validate_against_float_reference: true,
        ..JpegToHtj2kOptions::default()
    };

    let encoded =
        jpeg_to_htj2k(jpeg, &options).expect("transcode grayscale JPEG with validation enabled");
    let metrics = encoded
        .report
        .float_reference_metrics
        .as_ref()
        .expect("float reference metrics are reported");

    assert_eq!(metrics.total, 64);
    assert_eq!(metrics.exact_matches, 64);
    assert_eq!(metrics.max_abs_error, 0);
}

#[test]
fn generated_htj2k_is_accepted_by_available_external_decoder() {
    let jpeg = include_bytes!("../../signinum-jpeg/fixtures/conformance/grayscale_8x8.jpg");
    let encoded = jpeg_to_htj2k(jpeg, &JpegToHtj2kOptions::default())
        .expect("transcode grayscale JPEG to HTJ2K");
    let decoders = available_external_decoders();
    if decoders.is_empty() {
        eprintln!("skipping external HTJ2K decoder check: no supported decoder executable found");
        return;
    }

    let mut failures = Vec::new();
    for decoder in decoders {
        match run_external_decoder(decoder, &encoded.codestream) {
            Ok(()) => return,
            Err(err) => failures.push(format!("{decoder:?}: {err}")),
        }
    }

    panic!(
        "generated HTJ2K codestream was rejected by all available external decoders:\n{}",
        failures.join("\n")
    );
}

#[test]
fn grayscale_multiblock_jpeg_transcodes_to_decodable_htj2k() {
    let width = 13;
    let height = 11;
    let gray = patterned_gray(width, height);
    let jpeg = encode_jpeg_baseline(
        JpegSamples::Gray8 {
            data: &gray,
            width,
            height,
        },
        JpegEncodeOptions {
            quality: 90,
            subsampling: JpegSubsampling::Gray,
            restart_interval: None,
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode grayscale JPEG fixture");

    let encoded = jpeg_to_htj2k(&jpeg.data, &JpegToHtj2kOptions::default())
        .expect("transcode multi-block grayscale JPEG to HTJ2K");
    let decoded = Image::new(&encoded.codestream, &DecodeSettings::default())
        .expect("native parser accepts generated HTJ2K")
        .decode_native()
        .expect("native decoder accepts generated HTJ2K");

    assert_eq!(
        (encoded.report.width, encoded.report.height),
        (width, height)
    );
    assert_eq!(encoded.report.component_count, 1);
    assert_eq!((decoded.width, decoded.height), (width, height));
    assert_eq!(decoded.num_components, 1);
}

#[test]
fn ycbcr_444_jpeg_transcodes_to_decodable_htj2k_without_mct() {
    let jpeg = include_bytes!("../../signinum-jpeg/fixtures/conformance/baseline_444_8x8.jpg");

    let encoded = jpeg_to_htj2k(jpeg, &JpegToHtj2kOptions::default())
        .expect("transcode 4:4:4 YCbCr JPEG to HTJ2K");
    let decoded = Image::new(&encoded.codestream, &DecodeSettings::default())
        .expect("native parser accepts generated HTJ2K")
        .decode_native()
        .expect("native decoder accepts generated HTJ2K");

    assert_eq!((encoded.report.width, encoded.report.height), (8, 8));
    assert_eq!(encoded.report.component_count, 3);
    assert_report_sampling(&encoded, &[(8, 8, 1, 1), (8, 8, 1, 1), (8, 8, 1, 1)]);
    assert_eq!((decoded.width, decoded.height), (8, 8));
    assert_eq!(decoded.num_components, 3);
}

#[test]
fn ycbcr_422_jpeg_transcodes_with_native_component_sampling() {
    let jpeg = include_bytes!("../../signinum-jpeg/fixtures/conformance/baseline_422_16x8.jpg");

    let encoded = jpeg_to_htj2k(jpeg, &JpegToHtj2kOptions::default())
        .expect("transcode 4:2:2 YCbCr JPEG to HTJ2K");
    let decoded = Image::new(&encoded.codestream, &DecodeSettings::default())
        .expect("native parser accepts generated HTJ2K")
        .decode_native()
        .expect("native decoder accepts generated HTJ2K");

    assert_eq!((encoded.report.width, encoded.report.height), (16, 8));
    assert_eq!(encoded.report.component_count, 3);
    assert_report_sampling(&encoded, &[(16, 8, 1, 1), (8, 8, 2, 1), (8, 8, 2, 1)]);
    assert_eq!((decoded.width, decoded.height), (16, 8));
    assert_eq!(decoded.num_components, 3);
    assert_component_sampling(&encoded.codestream, &[(1, 1), (2, 1), (2, 1)]);
}

#[test]
fn ycbcr_420_jpeg_transcodes_with_native_component_sampling() {
    let jpeg = include_bytes!("../../signinum-jpeg/fixtures/conformance/baseline_420_16x16.jpg");

    let encoded = jpeg_to_htj2k(jpeg, &JpegToHtj2kOptions::default())
        .expect("transcode 4:2:0 YCbCr JPEG to HTJ2K");
    let decoded = Image::new(&encoded.codestream, &DecodeSettings::default())
        .expect("native parser accepts generated HTJ2K")
        .decode_native()
        .expect("native decoder accepts generated HTJ2K");

    assert_eq!((encoded.report.width, encoded.report.height), (16, 16));
    assert_eq!(encoded.report.component_count, 3);
    assert_report_sampling(&encoded, &[(16, 16, 1, 1), (8, 8, 2, 2), (8, 8, 2, 2)]);
    assert!(encoded.report.float_reference_metrics.is_none());
    assert_eq!((decoded.width, decoded.height), (16, 16));
    assert_eq!(decoded.num_components, 3);
    assert_component_sampling(&encoded.codestream, &[(1, 1), (2, 2), (2, 2)]);
}

#[test]
fn ycbcr_420_validation_metrics_cover_native_component_coefficients() {
    let jpeg = include_bytes!("../../signinum-jpeg/fixtures/conformance/baseline_420_16x16.jpg");
    let options = JpegToHtj2kOptions {
        validate_against_float_reference: true,
        ..JpegToHtj2kOptions::default()
    };

    let encoded =
        jpeg_to_htj2k(jpeg, &options).expect("transcode 4:2:0 JPEG with validation enabled");
    let metrics = encoded
        .report
        .float_reference_metrics
        .as_ref()
        .expect("float reference metrics are reported");

    assert_eq!(metrics.total, 384);
    assert_eq!(metrics.exact_matches, 384);
    assert_eq!(metrics.max_abs_error, 0);
}

fn patterned_gray(width: u32, height: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(width as usize * height as usize);
    for y in 0..height {
        for x in 0..width {
            out.push(((x * 7 + y * 11 + 19) & 0xff) as u8);
        }
    }
    out
}

fn assert_component_sampling(codestream: &[u8], expected: &[(u8, u8)]) {
    let siz = find_marker(codestream, 0x51).expect("SIZ marker");
    let component_info = siz + 40;
    for (component_index, &(x_rsiz, y_rsiz)) in expected.iter().enumerate() {
        let offset = component_info + component_index * 3;
        assert_eq!(codestream[offset + 1], x_rsiz);
        assert_eq!(codestream[offset + 2], y_rsiz);
    }
}

fn assert_report_sampling(encoded: &EncodedTranscode, expected: &[(u32, u32, u8, u8)]) {
    assert_eq!(encoded.report.components.len(), expected.len());
    for (component, &(width, height, x_rsiz, y_rsiz)) in
        encoded.report.components.iter().zip(expected)
    {
        assert_eq!((component.width, component.height), (width, height));
        assert_eq!((component.x_rsiz, component.y_rsiz), (x_rsiz, y_rsiz));
    }
}

#[derive(Debug, Clone, Copy)]
enum ExternalDecoder {
    Grok,
    OpenJpeg,
}

fn available_external_decoders() -> Vec<ExternalDecoder> {
    let mut decoders = Vec::new();
    if Command::new("grk_decompress").arg("-h").output().is_ok() {
        decoders.push(ExternalDecoder::Grok);
    }
    if Command::new("opj_decompress").arg("-h").output().is_ok() {
        decoders.push(ExternalDecoder::OpenJpeg);
    }
    decoders
}

fn run_external_decoder(decoder: ExternalDecoder, codestream: &[u8]) -> Result<(), String> {
    let ExternalDecodeFiles {
        input_path,
        output_path,
    } = write_external_decode_input(codestream)?;
    let output = match decoder {
        ExternalDecoder::Grok => Command::new("grk_decompress")
            .arg("-i")
            .arg(&input_path)
            .arg("-o")
            .arg(&output_path)
            .arg("-O")
            .arg("PNM")
            .output(),
        ExternalDecoder::OpenJpeg => Command::new("opj_decompress")
            .arg("-quiet")
            .arg("-i")
            .arg(&input_path)
            .arg("-o")
            .arg(&output_path)
            .output(),
    };
    let output = output.map_err(|err| err.to_string());
    let _ = fs::remove_file(&input_path);
    let _ = fs::remove_file(&output_path);

    let output = output?;
    if output.status.success() {
        Ok(())
    } else {
        Err(format_command_output(&output))
    }
}

struct ExternalDecodeFiles {
    input_path: PathBuf,
    output_path: PathBuf,
}

fn write_external_decode_input(codestream: &[u8]) -> Result<ExternalDecodeFiles, String> {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|err| err.to_string())?
        .as_nanos();
    let stem = format!("signinum-transcode-{}-{unique}", std::process::id());
    let input_path = env::temp_dir().join(format!("{stem}.j2k"));
    let output_path = env::temp_dir().join(format!("{stem}.pgm"));
    fs::write(&input_path, codestream).map_err(|err| err.to_string())?;

    Ok(ExternalDecodeFiles {
        input_path,
        output_path,
    })
}

fn format_command_output(output: &Output) -> String {
    format!(
        "status: {}; stdout: {}; stderr: {}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

fn find_marker(codestream: &[u8], marker: u8) -> Option<usize> {
    codestream
        .windows(2)
        .position(|window| window == [0xff, marker])
}
