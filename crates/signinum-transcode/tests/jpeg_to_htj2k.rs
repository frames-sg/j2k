// SPDX-License-Identifier: Apache-2.0

use signinum_j2k_native::{DecodeSettings, Image};
use signinum_jpeg::{
    encode_jpeg_baseline, JpegBackend, JpegEncodeOptions, JpegSamples, JpegSubsampling,
};
use signinum_transcode::accelerator::{
    DctGridToDwt53Job, DctGridToDwt97Job, DctGridToReversibleDwt53Job,
    DctToWaveletStageAccelerator, RayonReversibleDwt53Accelerator, ReversibleDwt53FirstLevel,
};
use signinum_transcode::dct53_2d::{
    dct8x8_blocks_to_dwt53_float_linear_with_scratch, Dct53GridScratch, Dwt53TwoDimensional,
};
use signinum_transcode::dct97_2d::{
    dct8x8_blocks_to_dwt97_float_linear_with_scratch, Dct97GridScratch, Dwt97TwoDimensional,
};
use signinum_transcode::{
    jpeg_to_htj2k, EncodedTranscode, JpegToHtj2kCoefficientPath, JpegToHtj2kOptions,
    JpegToHtj2kTranscoder, TranscodeValidationClassification,
};
use std::{
    env, fs,
    path::PathBuf,
    process::{Command, Output},
    time::{SystemTime, UNIX_EPOCH},
};

#[path = "../../signinum-jpeg/tests/fixtures/mod.rs"]
mod jpeg_fixtures;

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
        coefficient_path: JpegToHtj2kCoefficientPath::FloatDirectLinear53,
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
fn grayscale_8x8_jpeg_transcodes_to_decodable_lossy_97_htj2k() {
    let jpeg = include_bytes!("../../signinum-jpeg/fixtures/conformance/grayscale_8x8.jpg");
    let options = JpegToHtj2kOptions {
        validate_against_float_reference: true,
        ..JpegToHtj2kOptions::lossy_97()
    };

    let encoded =
        jpeg_to_htj2k(jpeg, &options).expect("transcode grayscale JPEG to lossy 9/7 HTJ2K");
    let decoded = Image::new(&encoded.codestream, &DecodeSettings::default())
        .expect("native parser accepts generated 9/7 HTJ2K")
        .decode_native()
        .expect("native decoder accepts generated 9/7 HTJ2K");
    let metrics = encoded
        .report
        .float_reference_metrics
        .as_ref()
        .expect("float reference metrics are reported");

    assert_eq!(
        encoded.report.path,
        "full_resolution_components_float_direct_97"
    );
    assert_eq!(
        encoded.report.coefficient_path,
        JpegToHtj2kCoefficientPath::FloatDirectLinear97
    );
    assert_eq!(encoded.report.decomposition_levels, 1);
    assert_eq!(metrics.total, 64);
    assert_eq!(metrics.exact_matches, 64);
    assert_eq!(metrics.max_abs_error, 0);
    assert_eq!((decoded.width, decoded.height), (8, 8));
    assert_eq!(decoded.num_components, 1);
}

#[test]
fn option_constructors_select_consistent_default_codec_modes() {
    let lossless = JpegToHtj2kOptions::lossless_53();
    assert_eq!(
        lossless.coefficient_path,
        JpegToHtj2kCoefficientPath::IntegerDirect53
    );
    assert!(lossless.encode_options.reversible);
    assert!(!lossless.encode_options.use_mct);

    let lossy = JpegToHtj2kOptions::lossy_97();
    assert_eq!(
        lossy.coefficient_path,
        JpegToHtj2kCoefficientPath::FloatDirectLinear97
    );
    assert!(!lossy.encode_options.reversible);
    assert!(!lossy.encode_options.use_mct);
}

#[test]
fn transcode_rejects_inconsistent_codec_mode_options() {
    let jpeg = include_bytes!("../../signinum-jpeg/fixtures/conformance/grayscale_8x8.jpg");
    let options = JpegToHtj2kOptions {
        coefficient_path: JpegToHtj2kCoefficientPath::FloatDirectLinear97,
        ..JpegToHtj2kOptions::default()
    };

    let err = jpeg_to_htj2k(jpeg, &options).expect_err("9/7 path requires irreversible encode");

    assert!(
        err.to_string()
            .contains("9/7 coefficient path requires irreversible HTJ2K encode"),
        "{err}"
    );
}

#[test]
fn ycbcr_420_jpeg_transcodes_to_decodable_lossy_97_htj2k_with_native_sampling() {
    let jpeg = include_bytes!("../../signinum-jpeg/fixtures/conformance/baseline_420_16x16.jpg");
    let options = JpegToHtj2kOptions {
        validate_against_float_reference: true,
        ..JpegToHtj2kOptions::lossy_97()
    };

    let encoded = jpeg_to_htj2k(jpeg, &options).expect("transcode 4:2:0 JPEG to lossy 9/7 HTJ2K");
    let decoded = Image::new(&encoded.codestream, &DecodeSettings::default())
        .expect("native parser accepts generated 9/7 HTJ2K")
        .decode_native()
        .expect("native decoder accepts generated 9/7 HTJ2K");
    let metrics = encoded
        .report
        .float_reference_metrics
        .as_ref()
        .expect("float reference metrics are reported");

    assert_eq!(
        encoded.report.path,
        "native_component_sampling_float_direct_97"
    );
    assert_eq!(metrics.total, 384);
    assert_eq!(metrics.exact_matches, 384);
    assert_eq!(metrics.max_abs_error, 0);
    assert_report_sampling(&encoded, &[(16, 16, 1, 1), (8, 8, 2, 2), (8, 8, 2, 2)]);
    assert_eq!((decoded.width, decoded.height), (16, 16));
    assert_eq!(decoded.num_components, 3);
    assert_component_sampling(&encoded.codestream, &[(1, 1), (2, 2), (2, 2)]);
}

#[test]
fn grayscale_8x8_transcode_reports_opt_in_integer_reference_metrics() {
    let jpeg = include_bytes!("../../signinum-jpeg/fixtures/conformance/grayscale_8x8.jpg");
    let options = JpegToHtj2kOptions {
        validate_against_integer_reference: true,
        ..JpegToHtj2kOptions::default()
    };

    let encoded =
        jpeg_to_htj2k(jpeg, &options).expect("transcode grayscale JPEG with integer validation");
    let metrics = encoded
        .report
        .integer_reference_metrics
        .as_ref()
        .expect("integer reference metrics are reported");

    assert_eq!(metrics.total, 64);
    assert_eq!(metrics.exact_matches, metrics.total);
    assert_eq!(metrics.max_abs_error, 0);
    assert_eq!(
        encoded.report.coefficient_path,
        JpegToHtj2kCoefficientPath::IntegerDirect53
    );
    assert_eq!(
        encoded.report.integer_reference_classification,
        Some(TranscodeValidationClassification::Exact)
    );
    assert_eq!(encoded.report.float_reference_classification, None);
}

#[test]
fn default_transcode_uses_integer_direct_coefficients() {
    let jpeg = include_bytes!("../../signinum-jpeg/fixtures/conformance/baseline_420_16x16.jpg");
    let options = JpegToHtj2kOptions {
        validate_against_integer_reference: true,
        ..JpegToHtj2kOptions::default()
    };

    let encoded = jpeg_to_htj2k(jpeg, &options)
        .expect("transcode 4:2:0 JPEG with default integer direct path");
    let metrics = encoded
        .report
        .integer_reference_metrics
        .as_ref()
        .expect("integer reference metrics are reported");

    assert_eq!(
        encoded.report.path,
        "native_component_sampling_integer_direct_53"
    );
    assert_eq!(metrics.total, 384);
    assert_eq!(metrics.exact_matches, metrics.total);
    assert_eq!(metrics.max_abs_error, 0);
}

#[test]
fn grayscale_8x8_jpeg_transcodes_with_two_decomposition_levels() {
    let jpeg = include_bytes!("../../signinum-jpeg/fixtures/conformance/grayscale_8x8.jpg");
    let mut encode_options = JpegToHtj2kOptions::default().encode_options;
    encode_options.num_decomposition_levels = 2;
    let options = JpegToHtj2kOptions {
        encode_options,
        coefficient_path: JpegToHtj2kCoefficientPath::FloatDirectLinear53,
        validate_against_float_reference: true,
        ..JpegToHtj2kOptions::default()
    };

    let encoded =
        jpeg_to_htj2k(jpeg, &options).expect("transcode grayscale JPEG with two DWT levels");
    let decoded = Image::new(&encoded.codestream, &DecodeSettings::default())
        .expect("native parser accepts generated HTJ2K")
        .decode_native()
        .expect("native decoder accepts generated HTJ2K");
    let metrics = encoded
        .report
        .float_reference_metrics
        .as_ref()
        .expect("float reference metrics are reported");

    assert_eq!(encoded.report.decomposition_levels, 2);
    assert_eq!((decoded.width, decoded.height), (8, 8));
    assert_eq!(decoded.num_components, 1);
    assert_eq!(metrics.total, 64);
    assert_eq!(metrics.exact_matches, 64);
}

#[test]
fn integer_direct_transcode_matches_integer_oracle_with_two_decomposition_levels() {
    let jpeg = include_bytes!("../../signinum-jpeg/fixtures/conformance/grayscale_8x8.jpg");
    let mut encode_options = JpegToHtj2kOptions::default().encode_options;
    encode_options.num_decomposition_levels = 2;
    let options = JpegToHtj2kOptions {
        encode_options,
        validate_against_integer_reference: true,
        ..JpegToHtj2kOptions::default()
    };

    let encoded =
        jpeg_to_htj2k(jpeg, &options).expect("integer-direct transcode supports two DWT levels");
    let metrics = encoded
        .report
        .integer_reference_metrics
        .as_ref()
        .expect("integer reference metrics are reported");

    assert_eq!(
        encoded.report.path,
        "full_resolution_components_integer_direct_53"
    );
    assert_eq!(encoded.report.decomposition_levels, 2);
    assert_eq!(metrics.total, 64);
    assert_eq!(metrics.exact_matches, metrics.total);
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
fn progressive_ycbcr_420_jpeg_transcodes_with_native_component_sampling() {
    let jpeg = jpeg_fixtures::progressive_8x8_jpeg();

    let encoded = jpeg_to_htj2k(&jpeg, &JpegToHtj2kOptions::default())
        .expect("transcode progressive 4:2:0 JPEG to HTJ2K");
    let decoded = Image::new(&encoded.codestream, &DecodeSettings::default())
        .expect("native parser accepts generated HTJ2K")
        .decode_native()
        .expect("native decoder accepts generated HTJ2K");

    assert_eq!((encoded.report.width, encoded.report.height), (8, 8));
    assert_eq!(encoded.report.component_count, 3);
    assert_report_sampling(&encoded, &[(8, 8, 1, 1), (4, 4, 2, 2), (4, 4, 2, 2)]);
    assert_eq!((decoded.width, decoded.height), (8, 8));
    assert_eq!(decoded.num_components, 3);
    assert_component_sampling(&encoded.codestream, &[(1, 1), (2, 2), (2, 2)]);
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
fn integer_direct_transcode_with_rayon_accelerator_matches_scalar_for_grayscale_dimensions() {
    for (width, height) in [(8, 8), (13, 11), (16, 16)] {
        let jpeg = encoded_gray_jpeg(width, height);
        let options = JpegToHtj2kOptions {
            validate_against_integer_reference: true,
            ..JpegToHtj2kOptions::default()
        };
        let scalar = jpeg_to_htj2k(&jpeg, &options).expect("scalar integer-direct transcode");
        let mut transcoder = JpegToHtj2kTranscoder::default();
        let mut accelerator = RayonReversibleDwt53Accelerator::default();

        let accelerated = transcoder
            .transcode_with_accelerator(&jpeg, &options, &mut accelerator)
            .expect("rayon integer-direct transcode");

        assert_eq!(
            accelerated.codestream, scalar.codestream,
            "accelerated IntegerDirect53 must match scalar oracle for {width}x{height}"
        );
        assert_eq!(
            accelerated.report.integer_reference_classification,
            Some(TranscodeValidationClassification::Exact)
        );
        assert_eq!(accelerator.reversible_dwt53_attempts(), 1);
        assert_eq!(accelerator.reversible_dwt53_dispatches(), 1);
    }
}

#[test]
fn integer_direct_transcode_with_rayon_accelerator_matches_scalar_for_ycbcr_420() {
    let jpeg = include_bytes!("../../signinum-jpeg/fixtures/conformance/baseline_420_16x16.jpg");
    let options = JpegToHtj2kOptions {
        validate_against_integer_reference: true,
        ..JpegToHtj2kOptions::default()
    };
    let scalar = jpeg_to_htj2k(jpeg, &options).expect("scalar 4:2:0 integer-direct transcode");
    let mut transcoder = JpegToHtj2kTranscoder::default();
    let mut accelerator = RayonReversibleDwt53Accelerator::default();

    let accelerated = transcoder
        .transcode_with_accelerator(jpeg, &options, &mut accelerator)
        .expect("rayon 4:2:0 integer-direct transcode");

    assert_eq!(accelerated.codestream, scalar.codestream);
    assert_eq!(
        accelerated.report.integer_reference_classification,
        Some(TranscodeValidationClassification::Exact)
    );
    assert_eq!(accelerator.reversible_dwt53_attempts(), 3);
    assert_eq!(accelerator.reversible_dwt53_dispatches(), 3);
    assert_report_sampling(&accelerated, &[(16, 16, 1, 1), (8, 8, 2, 2), (8, 8, 2, 2)]);
    assert_component_sampling(&accelerated.codestream, &[(1, 1), (2, 2), (2, 2)]);
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
        coefficient_path: JpegToHtj2kCoefficientPath::FloatDirectLinear53,
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

#[test]
fn stateful_transcoder_reuses_dct_block_scratch_across_tiles() {
    let larger_jpeg =
        include_bytes!("../../signinum-jpeg/fixtures/conformance/baseline_420_16x16.jpg");
    let smaller_jpeg = include_bytes!("../../signinum-jpeg/fixtures/conformance/grayscale_8x8.jpg");
    let options = JpegToHtj2kOptions {
        coefficient_path: JpegToHtj2kCoefficientPath::FloatDirectLinear53,
        ..JpegToHtj2kOptions::default()
    };
    let mut transcoder = JpegToHtj2kTranscoder::default();

    let larger = transcoder
        .transcode(larger_jpeg, &options)
        .expect("stateful transcode accepts 4:2:0 JPEG");
    let capacity_after_larger = transcoder.dct_block_scratch_capacity();
    assert!(capacity_after_larger >= 4);

    let smaller = transcoder
        .transcode(smaller_jpeg, &options)
        .expect("stateful transcode accepts grayscale JPEG");
    let stateless =
        jpeg_to_htj2k(smaller_jpeg, &options).expect("stateless transcode accepts grayscale JPEG");

    assert_eq!(larger.report.component_count, 3);
    assert_eq!(smaller.report.component_count, 1);
    assert_eq!(
        transcoder.dct_block_scratch_capacity(),
        capacity_after_larger
    );
    assert_eq!(smaller.codestream, stateless.codestream);
}

#[test]
fn float_direct_transcode_paths_use_acceleration_hooks_when_available() {
    let jpeg = include_bytes!("../../signinum-jpeg/fixtures/conformance/grayscale_8x8.jpg");
    let mut transcoder = JpegToHtj2kTranscoder::default();
    let mut accelerator = CountingAccelerator::default();

    let options_53 = JpegToHtj2kOptions {
        coefficient_path: JpegToHtj2kCoefficientPath::FloatDirectLinear53,
        ..JpegToHtj2kOptions::default()
    };
    transcoder
        .transcode_with_accelerator(jpeg, &options_53, &mut accelerator)
        .expect("accelerated 5/3 float transcode succeeds");

    let options_97 = JpegToHtj2kOptions {
        ..JpegToHtj2kOptions::lossy_97()
    };
    transcoder
        .transcode_with_accelerator(jpeg, &options_97, &mut accelerator)
        .expect("accelerated 9/7 float transcode succeeds");

    assert_eq!(accelerator.dwt53_calls, 1);
    assert_eq!(accelerator.dwt97_calls, 1);
}

#[test]
fn integer_direct_transcode_path_uses_reversible_acceleration_hook_when_available() {
    let jpeg = include_bytes!("../../signinum-jpeg/fixtures/conformance/grayscale_8x8.jpg");
    let mut transcoder = JpegToHtj2kTranscoder::default();
    let mut accelerator = CountingAccelerator::default();

    transcoder
        .transcode_with_accelerator(jpeg, &JpegToHtj2kOptions::default(), &mut accelerator)
        .expect("accelerated integer-direct transcode succeeds");

    assert_eq!(accelerator.reversible_dwt53_calls, 1);
}

#[test]
fn stateful_transcoder_reuses_integer_idct_block_scratch_across_tiles() {
    let larger_jpeg =
        include_bytes!("../../signinum-jpeg/fixtures/conformance/baseline_420_16x16.jpg");
    let smaller_jpeg = include_bytes!("../../signinum-jpeg/fixtures/conformance/grayscale_8x8.jpg");
    let options = JpegToHtj2kOptions::default();
    let mut transcoder = JpegToHtj2kTranscoder::default();

    let larger = transcoder
        .transcode(larger_jpeg, &options)
        .expect("stateful integer-direct transcode accepts 4:2:0 JPEG");
    let capacity_after_larger = transcoder.integer_idct_block_scratch_capacity();
    assert!(capacity_after_larger >= 4);

    let smaller = transcoder
        .transcode(smaller_jpeg, &options)
        .expect("stateful integer-direct transcode accepts grayscale JPEG");
    let stateless = jpeg_to_htj2k(smaller_jpeg, &options)
        .expect("stateless integer-direct transcode accepts grayscale JPEG");

    assert_eq!(larger.report.component_count, 3);
    assert_eq!(smaller.report.component_count, 1);
    assert_eq!(
        transcoder.integer_idct_block_scratch_capacity(),
        capacity_after_larger
    );
    assert_eq!(smaller.codestream, stateless.codestream);
}

#[derive(Default)]
struct CountingAccelerator {
    reversible_dwt53_calls: usize,
    dwt53_calls: usize,
    dwt97_calls: usize,
    dwt53_scratch: Dct53GridScratch,
    dwt97_scratch: Dct97GridScratch,
}

impl DctToWaveletStageAccelerator for CountingAccelerator {
    fn dct_grid_to_reversible_dwt53(
        &mut self,
        job: DctGridToReversibleDwt53Job<'_>,
    ) -> Result<Option<ReversibleDwt53FirstLevel>, &'static str> {
        self.reversible_dwt53_calls += 1;
        Ok(Some(
            RayonReversibleDwt53Accelerator::default()
                .dct_grid_to_reversible_dwt53(job)?
                .expect("rayon accelerator handles test job"),
        ))
    }

    fn dct_grid_to_dwt53(
        &mut self,
        job: DctGridToDwt53Job<'_>,
    ) -> Result<Option<Dwt53TwoDimensional<f64>>, &'static str> {
        self.dwt53_calls += 1;
        let dwt = dct8x8_blocks_to_dwt53_float_linear_with_scratch(
            job.blocks,
            job.block_cols,
            job.block_rows,
            job.width,
            job.height,
            &mut self.dwt53_scratch,
        )
        .map_err(|_| "test DCT 5/3 grid failed")?;
        Ok(Some(dwt))
    }

    fn dct_grid_to_dwt97(
        &mut self,
        job: DctGridToDwt97Job<'_>,
    ) -> Result<Option<Dwt97TwoDimensional<f64>>, &'static str> {
        self.dwt97_calls += 1;
        let dwt = dct8x8_blocks_to_dwt97_float_linear_with_scratch(
            job.blocks,
            job.block_cols,
            job.block_rows,
            job.width,
            job.height,
            &mut self.dwt97_scratch,
        )
        .map_err(|_| "test DCT 9/7 grid failed")?;
        Ok(Some(dwt))
    }
}

fn encoded_gray_jpeg(width: u32, height: u32) -> Vec<u8> {
    let gray = patterned_gray(width, height);
    encode_jpeg_baseline(
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
    .expect("encode grayscale JPEG fixture")
    .data
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
