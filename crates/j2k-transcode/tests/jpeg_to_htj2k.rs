// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k::adapter::encode_stage::{
    EncodedHtJ2kCodeBlock, IrreversibleQuantizationSubbandScales, J2kEncodeStageAccelerator,
    J2kHtCodeBlockEncodeJob,
};
use j2k_jpeg::{
    encode_jpeg_baseline, JpegBackend, JpegEncodeOptions, JpegSamples, JpegSubsampling,
};
use j2k_native::{DecodeSettings, Image};
use j2k_transcode::accelerator::{
    DctGridToDwt53Job, DctGridToDwt97Job, DctGridToHtj2k97CodeBlockJob,
    DctGridToReversibleDwt53Job, DctToWaveletStageAccelerator, Dwt97BatchStageTimings,
    Htj2k97CodeBlockOptions, J2kSubBandType, PreencodedHtj2k97CodeBlock,
    PreencodedHtj2k97Component, PreencodedHtj2k97Resolution, PreencodedHtj2k97Subband,
    PrequantizedHtj2k97CodeBlock, PrequantizedHtj2k97Component, PrequantizedHtj2k97Resolution,
    PrequantizedHtj2k97Subband, RayonReversibleDwt53Accelerator, ReversibleDwt53FirstLevel,
    TranscodeStageError,
};
use j2k_transcode::dct53_2d::{
    dct8x8_blocks_to_dwt53_float_linear_with_scratch, Dct53GridScratch, Dwt53TwoDimensional,
};
use j2k_transcode::dct97_2d::{
    dct8x8_blocks_then_dwt97_float_with_scratch, Dct97GridScratch, Dwt97TwoDimensional,
};
use j2k_transcode::{
    jpeg_to_htj2k, EncodedTranscode, JpegTileBatchInput, JpegToHtj2kCoefficientPath,
    JpegToHtj2kOptions, JpegToHtj2kTranscoder, TranscodePipelineStageKind, TranscodeStageProcessor,
    TranscodeTimingReport, TranscodeValidationClassification,
    JPEG_TO_HTJ2K_LOSSY_97_QUANTIZATION_SCALE,
};
use std::{
    env, fs,
    path::PathBuf,
    process::{Command, Output},
    time::{SystemTime, UNIX_EPOCH},
};

#[path = "fixtures/mod.rs"]
mod jpeg_fixtures;
use j2k_test_support::{
    JPEG_BASELINE_420_16X16, JPEG_BASELINE_422_16X8, JPEG_BASELINE_444_8X8, JPEG_GRAYSCALE_8X8,
};

#[test]
fn grayscale_8x8_jpeg_transcodes_to_decodable_htj2k() {
    let jpeg = JPEG_GRAYSCALE_8X8;

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
    let jpeg = JPEG_GRAYSCALE_8X8;
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
    let jpeg = JPEG_GRAYSCALE_8X8;
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
    assert_eq!(
        lossy
            .encode_options
            .irreversible_quantization_scale
            .to_bits(),
        JPEG_TO_HTJ2K_LOSSY_97_QUANTIZATION_SCALE.to_bits()
    );
    assert_eq!(
        lossy
            .encode_options
            .irreversible_quantization_scale
            .to_bits(),
        1.9f32.to_bits()
    );
    assert_eq!(
        lossy
            .encode_options
            .irreversible_quantization_subband_scales,
        IrreversibleQuantizationSubbandScales::default()
    );
}

#[test]
fn transcode_rejects_inconsistent_codec_mode_options() {
    let jpeg = JPEG_GRAYSCALE_8X8;
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
    let jpeg = JPEG_BASELINE_420_16X16;
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
    let jpeg = JPEG_GRAYSCALE_8X8;
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
    let jpeg = JPEG_BASELINE_420_16X16;
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
    let jpeg = JPEG_GRAYSCALE_8X8;
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
    let jpeg = JPEG_GRAYSCALE_8X8;
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
    let jpeg = JPEG_GRAYSCALE_8X8;
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
        assert_eq!(accelerator.reversible_dwt53_attempts(), 0);
        assert_eq!(accelerator.reversible_dwt53_dispatches(), 0);
        assert_eq!(accelerator.reversible_dwt53_batch_attempts(), 1);
        assert_eq!(accelerator.reversible_dwt53_batch_dispatches(), 1);
    }
}

#[test]
fn integer_direct_transcode_with_rayon_accelerator_matches_scalar_for_ycbcr_420() {
    let jpeg = JPEG_BASELINE_420_16X16;
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
    assert_eq!(accelerator.reversible_dwt53_attempts(), 0);
    assert_eq!(accelerator.reversible_dwt53_dispatches(), 0);
    assert_eq!(accelerator.reversible_dwt53_batch_attempts(), 2);
    assert_eq!(accelerator.reversible_dwt53_batch_dispatches(), 2);
    assert_report_sampling(&accelerated, &[(16, 16, 1, 1), (8, 8, 2, 2), (8, 8, 2, 2)]);
    assert_component_sampling(&accelerated.codestream, &[(1, 1), (2, 2), (2, 2)]);
}

#[test]
fn ycbcr_444_jpeg_transcodes_to_decodable_htj2k_without_mct() {
    let jpeg = JPEG_BASELINE_444_8X8;

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
    let jpeg = JPEG_BASELINE_422_16X8;

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
    let jpeg = JPEG_BASELINE_420_16X16;

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
    let jpeg = JPEG_BASELINE_420_16X16;
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
    let larger_jpeg = JPEG_BASELINE_420_16X16;
    let smaller_jpeg = JPEG_GRAYSCALE_8X8;
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
    let jpeg = JPEG_GRAYSCALE_8X8;
    let mut transcoder = JpegToHtj2kTranscoder::default();
    let mut accelerator = CountingAccelerator::default();

    let options_53 = JpegToHtj2kOptions {
        coefficient_path: JpegToHtj2kCoefficientPath::FloatDirectLinear53,
        ..JpegToHtj2kOptions::default()
    };
    let encoded_53 = transcoder
        .transcode_with_accelerator(jpeg, &options_53, &mut accelerator)
        .expect("accelerated 5/3 float transcode succeeds");
    assert_eq!(encoded_53.report.timings.component_count, 1);
    assert_eq!(encoded_53.report.timings.accelerator_attempts, 1);
    assert_eq!(encoded_53.report.timings.accelerator_dispatches, 1);
    assert_eq!(encoded_53.report.timings.cpu_fallback_jobs, 0);

    let options_97 = JpegToHtj2kOptions {
        ..JpegToHtj2kOptions::lossy_97()
    };
    let encoded_97 = transcoder
        .transcode_with_accelerator(jpeg, &options_97, &mut accelerator)
        .expect("accelerated 9/7 float transcode succeeds");
    assert_eq!(encoded_97.report.timings.component_count, 1);
    assert_eq!(encoded_97.report.timings.accelerator_attempts, 1);
    assert_eq!(encoded_97.report.timings.accelerator_dispatches, 1);
    assert_eq!(encoded_97.report.timings.cpu_fallback_jobs, 0);

    assert_eq!(accelerator.dwt53_calls, 1);
    assert_eq!(accelerator.dwt97_calls, 1);
}

#[test]
fn lossy_97_cpu_report_includes_transform_fallback_timing_breakdown() {
    let jpeg = JPEG_GRAYSCALE_8X8;
    let mut transcoder = JpegToHtj2kTranscoder::default();

    let encoded = transcoder
        .transcode(jpeg, &JpegToHtj2kOptions::lossy_97())
        .expect("CPU-only 9/7 transcode succeeds");
    let timings = encoded.report.timings;

    assert_eq!(timings.jpeg_dct_extract_us, encoded.report.extract_us);
    assert_eq!(timings.dct_to_wavelet_total_us, encoded.report.transform_us);
    assert_eq!(timings.htj2k_encode_us, encoded.report.encode_us);
    assert_eq!(timings.component_count, 1);
    assert_eq!(timings.accelerator_attempts, 1);
    assert_eq!(timings.accelerator_jobs, 1);
    assert_eq!(timings.accelerator_dispatches, 0);
    assert_eq!(timings.accelerator_dispatched_jobs, 0);
    assert_eq!(timings.cpu_fallback_jobs, 1);
}

#[test]
fn transcode_pipeline_map_covers_cpu_fallback_stages() {
    let jpeg = JPEG_GRAYSCALE_8X8;
    let mut transcoder = JpegToHtj2kTranscoder::default();

    let encoded = transcoder
        .transcode(jpeg, &JpegToHtj2kOptions::lossy_97())
        .expect("CPU-only 9/7 transcode succeeds");
    let map = encoded.report.pipeline_map();
    let stages = map
        .stages
        .iter()
        .map(|stage| stage.stage)
        .collect::<Vec<_>>();

    assert_eq!(
        stages,
        vec![
            TranscodePipelineStageKind::EntropyDecode,
            TranscodePipelineStageKind::CoefficientPrep,
            TranscodePipelineStageKind::Transform,
            TranscodePipelineStageKind::QuantizationCodeBlockPrep,
            TranscodePipelineStageKind::Packetization,
            TranscodePipelineStageKind::CodestreamAssembly,
        ]
    );
    let transform = map
        .stages
        .iter()
        .find(|stage| stage.stage == TranscodePipelineStageKind::Transform)
        .expect("pipeline map includes transform stage");
    assert_eq!(transform.processor, TranscodeStageProcessor::Cpu);
    assert_eq!(
        map.recommendation.stage,
        TranscodePipelineStageKind::Transform
    );
    assert!(map.debug_report().contains("stage=transform processor=Cpu"));
}

#[test]
fn transcode_pipeline_map_reports_metal_residency_and_next_stage() {
    let timings = TranscodeTimingReport {
        jpeg_dct_extract_us: 11,
        jpeg_dct_repack_us: 7,
        dct_to_wavelet_accelerator_us: 100,
        dwt97_batch_pack_upload_us: 9,
        dwt97_batch_pack_upload_transfers: 1,
        dwt97_batch_pack_upload_bytes: 256,
        dwt97_batch_resident_dct_handoff_count: 4,
        dwt97_batch_idct_row_lift_us: 31,
        dwt97_batch_column_lift_us: 29,
        dwt97_batch_resident_dwt_handoff_count: 16,
        dwt97_batch_quantize_codeblock_us: 13,
        dwt97_batch_readback_us: 17,
        dwt97_batch_readback_transfers: 4,
        dwt97_batch_readback_bytes: 512,
        htj2k_encode_us: 101,
        accelerator_attempts: 1,
        accelerator_jobs: 4,
        accelerator_dispatches: 1,
        accelerator_dispatched_jobs: 4,
        ..TranscodeTimingReport::default()
    };

    let map = timings.pipeline_map();
    let coefficient_prep = map
        .stages
        .iter()
        .find(|stage| stage.stage == TranscodePipelineStageKind::CoefficientPrep)
        .expect("pipeline map includes coefficient prep stage");
    let transform = map
        .stages
        .iter()
        .find(|stage| stage.stage == TranscodePipelineStageKind::Transform)
        .expect("pipeline map includes transform stage");
    let code_block_prep = map
        .stages
        .iter()
        .find(|stage| stage.stage == TranscodePipelineStageKind::QuantizationCodeBlockPrep)
        .expect("pipeline map includes code-block prep stage");
    let debug = map.debug_report();

    assert_eq!(coefficient_prep.processor, TranscodeStageProcessor::Hybrid);
    assert_eq!(coefficient_prep.transfer_count, 1);
    assert_eq!(coefficient_prep.transfer_bytes, 256);
    assert_eq!(coefficient_prep.resident_handoff_count, 4);
    assert_eq!(transform.processor, TranscodeStageProcessor::Metal);
    assert_eq!(transform.transfer_count, 4);
    assert_eq!(transform.transfer_bytes, 512);
    assert_eq!(transform.resident_handoff_count, 16);
    assert_eq!(code_block_prep.processor, TranscodeStageProcessor::Metal);
    assert_eq!(
        map.recommendation.stage,
        TranscodePipelineStageKind::QuantizationCodeBlockPrep
    );
    assert!(
        map.recommendation.evidence_us >= timings.dwt97_batch_readback_us,
        "recommendation should cite readback or encode timing evidence"
    );
    assert!(debug.contains("stage=entropy_decode processor=Cpu"));
    assert!(debug.contains("stage=transform processor=Metal"));
    assert!(debug.contains("resident_handoffs=16"));
    assert!(debug.contains("transfer_count=4 transfer_bytes=512"));
    assert!(debug.contains("recommend_next_stage=quantization_code_block_prep"));
}

#[test]
fn integer_direct_transcode_path_uses_reversible_acceleration_hook_when_available() {
    let jpeg = JPEG_GRAYSCALE_8X8;
    let mut transcoder = JpegToHtj2kTranscoder::default();
    let mut accelerator = CountingAccelerator::default();

    let encoded = transcoder
        .transcode_with_accelerator(jpeg, &JpegToHtj2kOptions::default(), &mut accelerator)
        .expect("accelerated integer-direct transcode succeeds");

    assert_eq!(accelerator.reversible_dwt53_calls, 0);
    assert_eq!(accelerator.reversible_dwt53_batch_calls, 1);
    assert_eq!(accelerator.reversible_dwt53_batch_sizes, vec![1]);
    assert_eq!(encoded.report.timings.batch_count, 1);
    assert_eq!(encoded.report.timings.batch_jobs, 1);
    assert_eq!(encoded.report.timings.accelerator_attempts, 1);
    assert_eq!(encoded.report.timings.accelerator_dispatches, 1);
    assert_eq!(encoded.report.timings.cpu_fallback_jobs, 0);
}

#[test]
fn integer_direct_transcode_batches_same_geometry_components_when_available() {
    for fixture in [
        BatchFixture {
            name: "grayscale",
            jpeg: JPEG_GRAYSCALE_8X8,
            expected_batch_sizes: &[1],
        },
        BatchFixture {
            name: "ycbcr_444",
            jpeg: JPEG_BASELINE_444_8X8,
            expected_batch_sizes: &[3],
        },
        BatchFixture {
            name: "ycbcr_422",
            jpeg: JPEG_BASELINE_422_16X8,
            expected_batch_sizes: &[1, 2],
        },
        BatchFixture {
            name: "ycbcr_420",
            jpeg: JPEG_BASELINE_420_16X16,
            expected_batch_sizes: &[1, 2],
        },
    ] {
        let options = JpegToHtj2kOptions {
            validate_against_integer_reference: true,
            ..JpegToHtj2kOptions::default()
        };
        let scalar =
            jpeg_to_htj2k(fixture.jpeg, &options).expect("scalar integer-direct transcode");
        let mut transcoder = JpegToHtj2kTranscoder::default();
        let mut accelerator = CountingAccelerator::default();

        let accelerated = transcoder
            .transcode_with_accelerator(fixture.jpeg, &options, &mut accelerator)
            .expect("batched integer-direct transcode succeeds");

        assert_eq!(
            accelerated.codestream, scalar.codestream,
            "batched IntegerDirect53 must match scalar oracle for {}",
            fixture.name
        );
        assert_eq!(
            accelerated.report.integer_reference_classification,
            Some(TranscodeValidationClassification::Exact)
        );
        assert_eq!(
            accelerator.reversible_dwt53_calls, 0,
            "batch-capable accelerator should not need single-job fallback for {}",
            fixture.name
        );
        assert_eq!(
            accelerator.reversible_dwt53_batch_sizes, fixture.expected_batch_sizes,
            "unexpected same-geometry batch grouping for {}",
            fixture.name
        );
    }
}

#[test]
fn integer_direct_batch_transcode_groups_components_across_tiles() {
    for fixture in [
        BatchFixture {
            name: "grayscale",
            jpeg: JPEG_GRAYSCALE_8X8,
            expected_batch_sizes: &[4],
        },
        BatchFixture {
            name: "ycbcr_444",
            jpeg: JPEG_BASELINE_444_8X8,
            expected_batch_sizes: &[4, 4, 4],
        },
        BatchFixture {
            name: "ycbcr_422",
            jpeg: JPEG_BASELINE_422_16X8,
            expected_batch_sizes: &[4, 4, 4],
        },
        BatchFixture {
            name: "ycbcr_420",
            jpeg: JPEG_BASELINE_420_16X16,
            expected_batch_sizes: &[4, 4, 4],
        },
    ] {
        let options = JpegToHtj2kOptions {
            validate_against_integer_reference: true,
            ..JpegToHtj2kOptions::default()
        };
        let inputs = vec![
            JpegTileBatchInput {
                bytes: fixture.jpeg
            };
            4
        ];
        let expected = jpeg_to_htj2k(fixture.jpeg, &options)
            .expect("scalar integer-direct transcode succeeds");
        let mut transcoder = JpegToHtj2kTranscoder::default();
        let mut accelerator = CountingAccelerator::default();

        let batch = transcoder
            .transcode_batch_with_accelerator(&inputs, &options, &mut accelerator)
            .expect("batched transcode accepts valid options");

        assert_eq!(batch.tiles.len(), inputs.len());
        assert_eq!(batch.report.tile_count, inputs.len());
        assert_eq!(batch.report.successful_tiles, inputs.len());
        assert_eq!(batch.report.failed_tiles, 0);
        assert_eq!(
            accelerator.reversible_dwt53_batch_sizes, fixture.expected_batch_sizes,
            "unexpected cross-tile component grouping for {}",
            fixture.name
        );
        assert_eq!(accelerator.reversible_dwt53_calls, 0);
        assert_eq!(
            batch.report.timings.batch_jobs,
            fixture.expected_batch_sizes.iter().sum::<usize>(),
            "batch report should count accelerated component jobs for {}",
            fixture.name
        );
        assert_eq!(
            batch.report.timings.accelerator_dispatches,
            fixture.expected_batch_sizes.len(),
            "batch report should count accelerator batch dispatches for {}",
            fixture.name
        );
        for tile in batch.tiles {
            let tile = tile.expect("valid tile transcodes");
            assert_eq!(
                tile.codestream, expected.codestream,
                "batch tile must match scalar output for {}",
                fixture.name
            );
            assert_eq!(
                tile.report.integer_reference_classification,
                Some(TranscodeValidationClassification::Exact)
            );
            assert_eq!(
                tile.report.timings.batch_jobs, 0,
                "tile report must not duplicate shared batch timing context for {}",
                fixture.name
            );
            assert_eq!(tile.report.transform_us, 0);
        }
    }
}

#[test]
fn float97_batch_transcode_groups_components_across_tiles() {
    for fixture in [
        BatchFixture {
            name: "grayscale",
            jpeg: JPEG_GRAYSCALE_8X8,
            expected_batch_sizes: &[4],
        },
        BatchFixture {
            name: "ycbcr_444",
            jpeg: JPEG_BASELINE_444_8X8,
            expected_batch_sizes: &[12],
        },
        BatchFixture {
            name: "ycbcr_422",
            jpeg: JPEG_BASELINE_422_16X8,
            expected_batch_sizes: &[4, 8],
        },
        BatchFixture {
            name: "ycbcr_420",
            jpeg: JPEG_BASELINE_420_16X16,
            expected_batch_sizes: &[4, 8],
        },
    ] {
        let options = JpegToHtj2kOptions::lossy_97();
        let inputs = vec![
            JpegTileBatchInput {
                bytes: fixture.jpeg
            };
            4
        ];
        let expected =
            jpeg_to_htj2k(fixture.jpeg, &options).expect("scalar 9/7 transcode succeeds");
        let mut transcoder = JpegToHtj2kTranscoder::default();
        let mut accelerator = CountingAccelerator::default();

        let batch = transcoder
            .transcode_batch_with_accelerator(&inputs, &options, &mut accelerator)
            .expect("batched 9/7 transcode accepts valid options");

        assert_eq!(batch.tiles.len(), inputs.len());
        assert_eq!(batch.report.tile_count, inputs.len());
        assert_eq!(batch.report.successful_tiles, inputs.len());
        assert_eq!(batch.report.failed_tiles, 0);
        assert_eq!(
            accelerator.dwt97_batch_sizes, fixture.expected_batch_sizes,
            "unexpected cross-tile 9/7 component grouping for {}",
            fixture.name
        );
        assert_eq!(accelerator.dwt97_calls, 0);
        assert_eq!(
            batch.report.timings.batch_jobs,
            fixture.expected_batch_sizes.iter().sum::<usize>(),
            "batch report should count 9/7 component jobs for {}",
            fixture.name
        );
        assert_eq!(
            batch.report.timings.accelerator_dispatches,
            fixture.expected_batch_sizes.len(),
            "batch report should count 9/7 accelerator batch dispatches for {}",
            fixture.name
        );
        for tile in batch.tiles {
            let tile = tile.expect("valid 9/7 tile transcodes");
            assert_eq!(
                tile.codestream, expected.codestream,
                "batch 9/7 tile must match scalar output for {}",
                fixture.name
            );
            assert_eq!(
                tile.report.timings.batch_jobs, 0,
                "tile report must not duplicate shared 9/7 batch timing context for {}",
                fixture.name
            );
            assert_eq!(tile.report.transform_us, 0);
        }
    }
}

#[test]
fn float97_batch_transcode_prefers_prequantized_codeblock_batches() {
    let jpeg = JPEG_GRAYSCALE_8X8;
    let options = JpegToHtj2kOptions::lossy_97();
    let inputs = vec![JpegTileBatchInput { bytes: jpeg }; 4];
    let mut transcoder = JpegToHtj2kTranscoder::default();
    let mut accelerator = Prequantized97Accelerator::default();

    let batch = transcoder
        .transcode_batch_with_accelerator(&inputs, &options, &mut accelerator)
        .expect("prequantized 9/7 batch transcode accepts valid options");

    assert_eq!(batch.tiles.len(), inputs.len());
    assert_eq!(batch.report.successful_tiles, inputs.len());
    assert_eq!(accelerator.prequantized_batch_sizes, vec![4]);
    assert_eq!(accelerator.dwt97_batch_calls, 0);
    assert_eq!(batch.report.timings.accelerator_dispatches, 1);
    assert_eq!(batch.report.timings.accelerator_dispatched_jobs, 4);
    for tile in batch.tiles {
        let tile = tile.expect("valid prequantized tile transcodes");
        assert!(tile.codestream.starts_with(&[0xFF, 0x4F]));
    }
}

#[test]
fn float97_prequantized_batch_receives_lossy_subband_profile() {
    let jpeg = JPEG_GRAYSCALE_8X8;
    let mut options = JpegToHtj2kOptions::lossy_97();
    options
        .encode_options
        .irreversible_quantization_subband_scales
        .high_high = 1.5;
    let inputs = vec![JpegTileBatchInput { bytes: jpeg }; 2];
    let mut transcoder = JpegToHtj2kTranscoder::default();
    let mut accelerator = Prequantized97Accelerator::default();

    let batch = transcoder
        .transcode_batch_with_accelerator(&inputs, &options, &mut accelerator)
        .expect("prequantized 9/7 batch transcode accepts subband profile");

    assert_eq!(batch.report.successful_tiles, inputs.len());
    assert_eq!(
        accelerator
            .last_options
            .expect("accelerator received code-block options")
            .irreversible_quantization_subband_scales
            .high_high
            .to_bits(),
        1.5f32.to_bits()
    );
}

#[test]
fn integer_direct_batch_transcode_offers_ht_blocks_to_encode_accelerator() {
    let jpeg = JPEG_GRAYSCALE_8X8;
    let options = JpegToHtj2kOptions::lossless_53();
    let inputs = vec![JpegTileBatchInput { bytes: jpeg }; 4];
    let mut transcoder = JpegToHtj2kTranscoder::default();
    let mut transform_accelerator = RayonReversibleDwt53Accelerator::default();
    let mut encode_accelerator = CountingHtEncodeAccelerator::default();

    let batch = transcoder
        .transcode_batch_with_accelerators(
            &inputs,
            &options,
            &mut transform_accelerator,
            &mut encode_accelerator,
        )
        .expect("5/3 batch transcode accepts separate encode accelerator");

    assert_eq!(batch.report.successful_tiles, inputs.len());
    assert_eq!(encode_accelerator.batches, inputs.len());
    assert!(encode_accelerator.jobs > 0);
    assert_eq!(encode_accelerator.single_blocks, encode_accelerator.jobs);
}

#[test]
fn batch_transcode_preserves_encode_hooks_when_parallel_cpu_fallback_requested() {
    let jpeg = JPEG_GRAYSCALE_8X8;
    let options = JpegToHtj2kOptions::lossless_53();
    let inputs = vec![JpegTileBatchInput { bytes: jpeg }; 4];
    let mut transcoder = JpegToHtj2kTranscoder::default();
    let mut transform_accelerator = RayonReversibleDwt53Accelerator::default();
    let mut encode_accelerator = CountingHtEncodeAccelerator {
        parallel_cpu_code_block_fallback: true,
        ..CountingHtEncodeAccelerator::default()
    };

    let batch = transcoder
        .transcode_batch_with_accelerators(
            &inputs,
            &options,
            &mut transform_accelerator,
            &mut encode_accelerator,
        )
        .expect("batch transcode preserves encode accelerator hooks");

    assert_eq!(batch.report.successful_tiles, inputs.len());
    assert_eq!(encode_accelerator.batches, inputs.len());
    assert!(encode_accelerator.jobs > 0);
    assert_eq!(encode_accelerator.single_blocks, 0);
}

#[test]
fn float97_batch_transcode_offers_prequantized_ht_blocks_to_encode_accelerator() {
    let jpeg = JPEG_GRAYSCALE_8X8;
    let options = JpegToHtj2kOptions::lossy_97();
    let inputs = vec![JpegTileBatchInput { bytes: jpeg }; 4];
    let mut transcoder = JpegToHtj2kTranscoder::default();
    let mut transform_accelerator = Prequantized97Accelerator::default();
    let mut encode_accelerator = CountingHtEncodeAccelerator::default();

    let batch = transcoder
        .transcode_batch_with_accelerators(
            &inputs,
            &options,
            &mut transform_accelerator,
            &mut encode_accelerator,
        )
        .expect("9/7 batch transcode accepts separate encode accelerator");

    assert_eq!(batch.report.successful_tiles, inputs.len());
    assert_eq!(transform_accelerator.prequantized_batch_sizes, vec![4]);
    assert_eq!(encode_accelerator.batches, inputs.len());
    assert!(encode_accelerator.jobs > 0);
    assert_eq!(encode_accelerator.single_blocks, encode_accelerator.jobs);
}

#[test]
fn float97_preencoded_batch_skips_encode_codeblock_hooks() {
    let jpeg = JPEG_GRAYSCALE_8X8;
    let options = JpegToHtj2kOptions::lossy_97();
    let inputs = vec![JpegTileBatchInput { bytes: jpeg }; 4];
    let mut transcoder = JpegToHtj2kTranscoder::default();
    let mut transform_accelerator = Preencoded97Accelerator::default();
    let mut encode_accelerator = CountingHtEncodeAccelerator::default();

    let batch = transcoder
        .transcode_batch_with_accelerators(
            &inputs,
            &options,
            &mut transform_accelerator,
            &mut encode_accelerator,
        )
        .expect("9/7 preencoded batch transcode accepts separate encode accelerator");

    assert_eq!(batch.report.successful_tiles, inputs.len());
    assert_eq!(transform_accelerator.preencoded_batch_sizes, vec![4]);
    assert_eq!(encode_accelerator.batches, 0);
    assert_eq!(encode_accelerator.jobs, 0);
    assert_eq!(batch.report.timings.dwt97_batch_ht_codeblock_dispatches, 1);
}

#[test]
fn float97_preencoded_batch_groups_compatible_color_components() {
    let jpeg = JPEG_BASELINE_444_8X8;
    let options = JpegToHtj2kOptions::lossy_97();
    let inputs = vec![JpegTileBatchInput { bytes: jpeg }; 4];
    let mut transcoder = JpegToHtj2kTranscoder::default();
    let mut transform_accelerator = Preencoded97Accelerator::default();
    let mut encode_accelerator = CountingHtEncodeAccelerator::default();

    let batch = transcoder
        .transcode_batch_with_accelerators(
            &inputs,
            &options,
            &mut transform_accelerator,
            &mut encode_accelerator,
        )
        .expect("9/7 preencoded batch groups compatible color components");

    assert_eq!(batch.report.successful_tiles, inputs.len());
    assert_eq!(transform_accelerator.preencoded_batch_sizes, vec![12]);
    assert_eq!(encode_accelerator.batches, 0);
    assert_eq!(encode_accelerator.jobs, 0);
}

#[test]
fn batch_transcode_reports_bad_tiles_without_aborting_valid_tiles() {
    let good = JPEG_GRAYSCALE_8X8;
    let inputs = [
        JpegTileBatchInput { bytes: good },
        JpegTileBatchInput {
            bytes: b"not a jpeg",
        },
        JpegTileBatchInput { bytes: good },
    ];
    let mut transcoder = JpegToHtj2kTranscoder::default();
    let mut accelerator = CountingAccelerator::default();

    let batch = transcoder
        .transcode_batch_with_accelerator(&inputs, &JpegToHtj2kOptions::default(), &mut accelerator)
        .expect("valid batch options do not fail globally");

    assert_eq!(batch.report.tile_count, 3);
    assert_eq!(batch.report.successful_tiles, 2);
    assert_eq!(batch.report.failed_tiles, 1);
    assert!(batch.tiles[0].is_ok());
    assert!(batch.tiles[1].is_err());
    assert!(batch.tiles[2].is_ok());
    assert_eq!(accelerator.reversible_dwt53_batch_sizes, vec![2]);
}

#[test]
fn stateful_transcoder_reuses_integer_idct_block_scratch_across_tiles() {
    let larger_jpeg = JPEG_BASELINE_420_16X16;
    let smaller_jpeg = JPEG_GRAYSCALE_8X8;
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
    reversible_dwt53_batch_calls: usize,
    reversible_dwt53_batch_sizes: Vec<usize>,
    dwt53_calls: usize,
    dwt97_calls: usize,
    dwt97_batch_calls: usize,
    dwt97_batch_sizes: Vec<usize>,
    dwt53_scratch: Dct53GridScratch,
    dwt97_scratch: Dct97GridScratch,
}

impl DctToWaveletStageAccelerator for CountingAccelerator {
    fn supports_dwt97_batch(&self) -> bool {
        true
    }

    fn dct_grid_to_reversible_dwt53(
        &mut self,
        job: DctGridToReversibleDwt53Job<'_>,
    ) -> Result<Option<ReversibleDwt53FirstLevel>, TranscodeStageError> {
        self.reversible_dwt53_calls += 1;
        Ok(Some(
            RayonReversibleDwt53Accelerator::default()
                .dct_grid_to_reversible_dwt53(job)?
                .expect("rayon accelerator handles test job"),
        ))
    }

    fn dct_grid_to_reversible_dwt53_batch(
        &mut self,
        jobs: &[DctGridToReversibleDwt53Job<'_>],
    ) -> Result<Option<Vec<ReversibleDwt53FirstLevel>>, TranscodeStageError> {
        self.reversible_dwt53_batch_calls += 1;
        self.reversible_dwt53_batch_sizes.push(jobs.len());
        let mut output = Vec::with_capacity(jobs.len());
        let mut rayon = RayonReversibleDwt53Accelerator::default();
        for job in jobs {
            output.push(
                rayon
                    .dct_grid_to_reversible_dwt53(*job)?
                    .expect("rayon accelerator handles batched test job"),
            );
        }
        Ok(Some(output))
    }

    fn dct_grid_to_dwt53(
        &mut self,
        job: DctGridToDwt53Job<'_>,
    ) -> Result<Option<Dwt53TwoDimensional<f64>>, TranscodeStageError> {
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
    ) -> Result<Option<Dwt97TwoDimensional<f64>>, TranscodeStageError> {
        self.dwt97_calls += 1;
        let dwt = dct8x8_blocks_then_dwt97_float_with_scratch(
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

    fn dct_grid_to_dwt97_batch(
        &mut self,
        jobs: &[DctGridToDwt97Job<'_>],
    ) -> Result<Option<Vec<Dwt97TwoDimensional<f64>>>, TranscodeStageError> {
        self.dwt97_batch_calls += 1;
        self.dwt97_batch_sizes.push(jobs.len());
        let mut output = Vec::with_capacity(jobs.len());
        for job in jobs {
            output.push(
                dct8x8_blocks_then_dwt97_float_with_scratch(
                    job.blocks,
                    job.block_cols,
                    job.block_rows,
                    job.width,
                    job.height,
                    &mut self.dwt97_scratch,
                )
                .map_err(|_| "test batched DCT 9/7 grid failed")?,
            );
        }
        Ok(Some(output))
    }
}

struct BatchFixture {
    name: &'static str,
    jpeg: &'static [u8],
    expected_batch_sizes: &'static [usize],
}

#[derive(Default)]
struct Prequantized97Accelerator {
    prequantized_batch_sizes: Vec<usize>,
    dwt97_batch_calls: usize,
    last_options: Option<Htj2k97CodeBlockOptions>,
}

#[derive(Default)]
struct Preencoded97Accelerator {
    preencoded_batch_sizes: Vec<usize>,
    last_timings: Option<Dwt97BatchStageTimings>,
}

#[derive(Default)]
struct CountingHtEncodeAccelerator {
    batches: usize,
    jobs: usize,
    single_blocks: usize,
    parallel_cpu_code_block_fallback: bool,
}

impl J2kEncodeStageAccelerator for CountingHtEncodeAccelerator {
    fn encode_ht_code_blocks(
        &mut self,
        jobs: &[J2kHtCodeBlockEncodeJob<'_>],
    ) -> Result<Option<Vec<EncodedHtJ2kCodeBlock>>, &'static str> {
        self.batches += 1;
        self.jobs += jobs.len();
        Ok(None)
    }

    fn encode_ht_code_block(
        &mut self,
        _job: J2kHtCodeBlockEncodeJob<'_>,
    ) -> Result<Option<EncodedHtJ2kCodeBlock>, &'static str> {
        self.single_blocks += 1;
        Ok(None)
    }

    fn prefer_parallel_cpu_code_block_fallback(&self) -> bool {
        self.parallel_cpu_code_block_fallback
    }
}

impl DctToWaveletStageAccelerator for Prequantized97Accelerator {
    fn supports_htj2k97_codeblock_batch(&self) -> bool {
        true
    }

    fn dct_grid_to_htj2k97_codeblock_batch(
        &mut self,
        jobs: &[DctGridToHtj2k97CodeBlockJob<'_>],
        options: Htj2k97CodeBlockOptions,
    ) -> Result<Option<Vec<PrequantizedHtj2k97Component>>, TranscodeStageError> {
        self.prequantized_batch_sizes.push(jobs.len());
        self.last_options = Some(options);
        Ok(Some(
            jobs.iter()
                .map(|job| zero_prequantized_component(job, options))
                .collect(),
        ))
    }

    fn dct_grid_to_dwt97_batch(
        &mut self,
        _jobs: &[DctGridToDwt97Job<'_>],
    ) -> Result<Option<Vec<Dwt97TwoDimensional<f64>>>, TranscodeStageError> {
        self.dwt97_batch_calls += 1;
        Ok(None)
    }
}

impl DctToWaveletStageAccelerator for Preencoded97Accelerator {
    fn supports_htj2k97_codeblock_batch(&self) -> bool {
        true
    }

    fn dct_grid_to_htj2k97_preencoded_batch(
        &mut self,
        jobs: &[DctGridToHtj2k97CodeBlockJob<'_>],
        options: Htj2k97CodeBlockOptions,
    ) -> Result<Option<Vec<PreencodedHtj2k97Component>>, TranscodeStageError> {
        self.preencoded_batch_sizes.push(jobs.len());
        self.last_timings = Some(Dwt97BatchStageTimings {
            ht_encode_us: 7,
            ht_codeblock_dispatches: 1,
            ..Dwt97BatchStageTimings::default()
        });
        Ok(Some(
            jobs.iter()
                .map(|job| zero_preencoded_component(job, options))
                .collect(),
        ))
    }

    fn dct_grid_to_htj2k97_codeblock_batch(
        &mut self,
        _jobs: &[DctGridToHtj2k97CodeBlockJob<'_>],
        _options: Htj2k97CodeBlockOptions,
    ) -> Result<Option<Vec<PrequantizedHtj2k97Component>>, TranscodeStageError> {
        panic!("preencoded accelerator should be offered before prequantized fallback")
    }

    fn last_dwt97_batch_stage_timings(&self) -> Option<Dwt97BatchStageTimings> {
        self.last_timings
    }
}

fn zero_prequantized_component(
    job: &DctGridToHtj2k97CodeBlockJob<'_>,
    options: Htj2k97CodeBlockOptions,
) -> PrequantizedHtj2k97Component {
    let low_width = job.width.div_ceil(2);
    let low_height = job.height.div_ceil(2);
    let high_width = job.width / 2;
    let high_height = job.height / 2;
    PrequantizedHtj2k97Component {
        x_rsiz: job.x_rsiz,
        y_rsiz: job.y_rsiz,
        resolutions: vec![
            PrequantizedHtj2k97Resolution {
                subbands: vec![zero_prequantized_subband(
                    low_width,
                    low_height,
                    J2kSubBandType::LowLow,
                    zero_prequantized_total_bitplanes(options, J2kSubBandType::LowLow),
                    options,
                )],
            },
            PrequantizedHtj2k97Resolution {
                subbands: vec![
                    zero_prequantized_subband(
                        high_width,
                        low_height,
                        J2kSubBandType::HighLow,
                        zero_prequantized_total_bitplanes(options, J2kSubBandType::HighLow),
                        options,
                    ),
                    zero_prequantized_subband(
                        low_width,
                        high_height,
                        J2kSubBandType::LowHigh,
                        zero_prequantized_total_bitplanes(options, J2kSubBandType::LowHigh),
                        options,
                    ),
                    zero_prequantized_subband(
                        high_width,
                        high_height,
                        J2kSubBandType::HighHigh,
                        zero_prequantized_total_bitplanes(options, J2kSubBandType::HighHigh),
                        options,
                    ),
                ],
            },
        ],
    }
}

fn zero_preencoded_component(
    job: &DctGridToHtj2k97CodeBlockJob<'_>,
    options: Htj2k97CodeBlockOptions,
) -> PreencodedHtj2k97Component {
    let low_width = job.width.div_ceil(2);
    let low_height = job.height.div_ceil(2);
    let high_width = job.width / 2;
    let high_height = job.height / 2;
    PreencodedHtj2k97Component {
        x_rsiz: job.x_rsiz,
        y_rsiz: job.y_rsiz,
        resolutions: vec![
            PreencodedHtj2k97Resolution {
                subbands: vec![zero_preencoded_subband(
                    low_width,
                    low_height,
                    J2kSubBandType::LowLow,
                    zero_prequantized_total_bitplanes(options, J2kSubBandType::LowLow),
                    options,
                )],
            },
            PreencodedHtj2k97Resolution {
                subbands: vec![
                    zero_preencoded_subband(
                        high_width,
                        low_height,
                        J2kSubBandType::HighLow,
                        zero_prequantized_total_bitplanes(options, J2kSubBandType::HighLow),
                        options,
                    ),
                    zero_preencoded_subband(
                        low_width,
                        high_height,
                        J2kSubBandType::LowHigh,
                        zero_prequantized_total_bitplanes(options, J2kSubBandType::LowHigh),
                        options,
                    ),
                    zero_preencoded_subband(
                        high_width,
                        high_height,
                        J2kSubBandType::HighHigh,
                        zero_prequantized_total_bitplanes(options, J2kSubBandType::HighHigh),
                        options,
                    ),
                ],
            },
        ],
    }
}

fn zero_prequantized_total_bitplanes(
    options: Htj2k97CodeBlockOptions,
    sub_band_type: J2kSubBandType,
) -> u8 {
    let base_delta = pow2i_f32_for_test(-i32::from(options.guard_bits))
        * options.irreversible_quantization_scale
        * match sub_band_type {
            J2kSubBandType::LowLow => options.irreversible_quantization_subband_scales.low_low,
            J2kSubBandType::HighLow => options.irreversible_quantization_subband_scales.high_low,
            J2kSubBandType::LowHigh => options.irreversible_quantization_subband_scales.low_high,
            J2kSubBandType::HighHigh => options.irreversible_quantization_subband_scales.high_high,
        };
    let floor_log2 = base_delta.log2().floor() as i32;
    let mut exponent = i32::from(options.bit_depth) - floor_log2;
    let normalized = base_delta / pow2i_f32_for_test(floor_log2);
    let mantissa = ((normalized - 1.0) * 2048.0).round() as i32;

    if mantissa >= 2048 {
        exponent -= 1;
    }

    options
        .guard_bits
        .saturating_add(u8::try_from(exponent.clamp(0, 31)).expect("clamped exponent fits u8"))
        .saturating_sub(1)
}

fn pow2i_f32_for_test(exp: i32) -> f32 {
    if exp >= 0 {
        (1u32 << exp.cast_unsigned()) as f32
    } else {
        1.0 / (1u32 << (-exp).cast_unsigned()) as f32
    }
}

fn zero_preencoded_subband(
    width: usize,
    height: usize,
    sub_band_type: J2kSubBandType,
    total_bitplanes: u8,
    options: Htj2k97CodeBlockOptions,
) -> PreencodedHtj2k97Subband {
    let cb_width = 1usize << (options.code_block_width_exp + 2);
    let cb_height = 1usize << (options.code_block_height_exp + 2);
    let num_cbs_x = width.div_ceil(cb_width);
    let num_cbs_y = height.div_ceil(cb_height);
    let mut code_blocks = Vec::with_capacity(num_cbs_x * num_cbs_y);
    for cby in 0..num_cbs_y {
        for cbx in 0..num_cbs_x {
            let x0 = cbx * cb_width;
            let y0 = cby * cb_height;
            let block_width = (width - x0).min(cb_width);
            let block_height = (height - y0).min(cb_height);
            code_blocks.push(PreencodedHtj2k97CodeBlock {
                width: block_width as u32,
                height: block_height as u32,
                encoded: EncodedHtJ2kCodeBlock {
                    data: Vec::new(),
                    cleanup_length: 0,
                    refinement_length: 0,
                    num_coding_passes: 0,
                    num_zero_bitplanes: total_bitplanes,
                },
            });
        }
    }

    PreencodedHtj2k97Subband {
        sub_band_type,
        num_cbs_x: num_cbs_x as u32,
        num_cbs_y: num_cbs_y as u32,
        total_bitplanes,
        code_blocks,
    }
}

fn zero_prequantized_subband(
    width: usize,
    height: usize,
    sub_band_type: J2kSubBandType,
    total_bitplanes: u8,
    options: Htj2k97CodeBlockOptions,
) -> PrequantizedHtj2k97Subband {
    let cb_width = 1usize << (options.code_block_width_exp + 2);
    let cb_height = 1usize << (options.code_block_height_exp + 2);
    let num_cbs_x = width.div_ceil(cb_width);
    let num_cbs_y = height.div_ceil(cb_height);
    let mut code_blocks = Vec::with_capacity(num_cbs_x * num_cbs_y);
    for cby in 0..num_cbs_y {
        for cbx in 0..num_cbs_x {
            let x0 = cbx * cb_width;
            let y0 = cby * cb_height;
            let block_width = (width - x0).min(cb_width);
            let block_height = (height - y0).min(cb_height);
            code_blocks.push(PrequantizedHtj2k97CodeBlock {
                coefficients: vec![0; block_width * block_height],
                width: block_width as u32,
                height: block_height as u32,
            });
        }
    }

    PrequantizedHtj2k97Subband {
        sub_band_type,
        num_cbs_x: num_cbs_x as u32,
        num_cbs_y: num_cbs_y as u32,
        total_bitplanes,
        code_blocks,
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
    let stem = format!("j2k-transcode-{}-{unique}", std::process::id());
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
