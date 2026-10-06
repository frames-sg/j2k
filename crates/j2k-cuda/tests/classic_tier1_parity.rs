// SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg(feature = "cuda-runtime")]

use j2k_cuda_j2k_engine::{
    CudaClassicCodeBlockJob, CudaClassicDecodeTarget, CudaClassicSegment, J2kCudaEngine,
};
use j2k_cuda_runtime::{CudaBufferPool, CudaContext, CudaError};
use j2k_native::{
    decode_j2k_code_block_scalar, encode_j2k_code_block_scalar_with_style, J2kCodeBlockDecodeJob,
    J2kCodeBlockSegment, J2kCodeBlockStyle, J2kSubBandType,
};
use j2k_test_support::cuda_runtime_and_strict_oxide_gate;

struct Tier1Case {
    name: &'static str,
    width: u32,
    height: u32,
    total_bitplanes: u8,
    subband: J2kSubBandType,
    style: J2kCodeBlockStyle,
    seed: u32,
}

fn generated_coefficients(case: &Tier1Case) -> Vec<i32> {
    if case.name == "normal_ll_1x1_31bit" {
        return vec![i32::MAX];
    }
    let mut coefficients = Vec::with_capacity(case.width as usize * case.height as usize);
    let mut state = case.seed ^ 0x9e37_79b9;
    for index in 0..case.width * case.height {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let value = i32::try_from((state >> 16) & 0x01ff).expect("masked coefficient") - 255;
        coefficients.push(if (index + case.seed).is_multiple_of(11) {
            0
        } else {
            value
        });
    }
    coefficients
}

fn style_flags(style: J2kCodeBlockStyle) -> u32 {
    u32::from(style.reset_context_probabilities)
        | (u32::from(style.termination_on_each_pass) << 1)
        | (u32::from(style.vertically_causal_context) << 2)
        | (u32::from(style.segmentation_symbols) << 3)
        | (u32::from(style.selective_arithmetic_coding_bypass) << 4)
}

fn subband_tag(subband: J2kSubBandType) -> u32 {
    match subband {
        J2kSubBandType::LowLow => 0,
        J2kSubBandType::HighLow => 1,
        J2kSubBandType::LowHigh => 2,
        J2kSubBandType::HighHigh => 3,
    }
}

#[test]
fn classic_tier1_cuda_matches_native_style_and_dimension_matrix() {
    if !cuda_runtime_and_strict_oxide_gate(module_path!()) {
        return;
    }
    let cases = tier1_cases();
    let context = CudaContext::system_default().expect("CUDA context");
    let pool = context.buffer_pool();
    for case in &cases {
        run_case(&context, &pool, case);
    }
    run_case_batch(&context, &pool, &cases);
}

fn tier1_cases() -> [Tier1Case; 10] {
    use J2kSubBandType::{HighHigh, HighLow, LowHigh, LowLow};
    let default_style = J2kCodeBlockStyle {
        selective_arithmetic_coding_bypass: false,
        reset_context_probabilities: false,
        termination_on_each_pass: false,
        vertically_causal_context: false,
        segmentation_symbols: false,
    };
    let bypass = J2kCodeBlockStyle {
        selective_arithmetic_coding_bypass: true,
        ..default_style
    };
    let term_reset = J2kCodeBlockStyle {
        reset_context_probabilities: true,
        termination_on_each_pass: true,
        ..default_style
    };
    let segmentation = J2kCodeBlockStyle {
        segmentation_symbols: true,
        ..default_style
    };
    let vcausal = J2kCodeBlockStyle {
        vertically_causal_context: true,
        ..default_style
    };
    let all_modes = J2kCodeBlockStyle {
        selective_arithmetic_coding_bypass: true,
        reset_context_probabilities: true,
        termination_on_each_pass: true,
        vertically_causal_context: true,
        segmentation_symbols: true,
    };
    [
        Tier1Case {
            total_bitplanes: 31,
            ..tier1_case("normal_ll_1x1_31bit", (1, 1), LowLow, default_style, 0x5100)
        },
        tier1_case("bypass_lh", (13, 9), LowHigh, bypass, 0x5200),
        tier1_case("term_reset_hl", (13, 9), HighLow, term_reset, 0x5300),
        tier1_case("segmentation_hh", (13, 9), HighHigh, segmentation, 0x5400),
        tier1_case("vcausal_ll", (13, 9), LowLow, vcausal, 0x5500),
        tier1_case("partial_height_2", (7, 2), LowLow, default_style, 0x5510),
        tier1_case("partial_height_3_vcausal", (7, 3), HighLow, vcausal, 0x5520),
        tier1_case("partial_height_5", (7, 5), LowHigh, default_style, 0x5530),
        tier1_case(
            "partial_height_63_vcausal",
            (7, 63),
            HighHigh,
            vcausal,
            0x5540,
        ),
        tier1_case("combined_64x64", (64, 64), HighHigh, all_modes, 0x5600),
    ]
}

/// A 10-bitplane case; override `total_bitplanes` with struct update syntax.
fn tier1_case(
    name: &'static str,
    (width, height): (u32, u32),
    subband: J2kSubBandType,
    style: J2kCodeBlockStyle,
    seed: u32,
) -> Tier1Case {
    Tier1Case {
        name,
        width,
        height,
        total_bitplanes: 10,
        subband,
        style,
        seed,
    }
}

fn run_case_batch(context: &CudaContext, pool: &CudaBufferPool, cases: &[Tier1Case]) {
    let mut payload = Vec::new();
    let mut jobs = Vec::new();
    let mut segments = Vec::new();
    let mut expected = Vec::new();
    let mut malformed_ranges = Vec::new();
    for case in cases {
        let encoded = encode_j2k_code_block_scalar_with_style(
            &generated_coefficients(case),
            case.width,
            case.height,
            case.subband,
            case.total_bitplanes,
            case.style,
        )
        .unwrap_or_else(|error| panic!("{} batch encode: {error}", case.name));
        let mut job = cuda_job(case, &encoded, encoded.data.len(), true);
        job.payload_offset = u64::try_from(payload.len()).expect("batch payload offset");
        job.segment_start = u32::try_from(segments.len()).expect("batch segment offset");
        let stride = case.width as usize;
        expected.resize(expected.len().div_ceil(stride) * stride, 0.0);
        job.output_offset = u32::try_from(expected.len()).expect("batch output offset");
        if case.name == "bypass_lh" {
            let mut malformed = encoded.data.clone();
            for segment in encoded
                .segments
                .iter()
                .filter(|segment| !segment.use_arithmetic)
            {
                let start = segment.data_offset as usize;
                let end = start + segment.data_length as usize;
                malformed[start..end].fill(0xff);
                malformed_ranges.push(payload.len() + start..payload.len() + end);
            }
            assert!(native_decode(case, &encoded, &malformed, &encoded.segments, true).is_err());
        }
        expected.extend(
            native_decode(case, &encoded, &encoded.data, &encoded.segments, true)
                .unwrap_or_else(|error| panic!("{} batch native decode: {error}", case.name)),
        );
        payload.extend_from_slice(&encoded.data);
        segments.extend(cuda_segments(&encoded.segments));
        jobs.push(job);
    }
    for queued in [false, true] {
        let actual = cuda_decode_jobs(
            context,
            pool,
            &payload,
            &jobs,
            &segments,
            expected.len(),
            queued,
        )
        .expect("heterogeneous classic batch decode");
        assert_eq!(actual, expected, "classic batch queued={queued}");
    }

    for range in malformed_ranges {
        payload[range].fill(0xff);
    }
    for queued in [false, true] {
        let error = cuda_decode_jobs(
            context,
            pool,
            &payload,
            &jobs,
            &segments,
            expected.len(),
            queued,
        )
        .expect_err("malformed bypass in the second input job must fail");
        let index = error
            .kernel_job_index()
            .unwrap_or_else(|| panic!("unexpected classic batch failure: {error}"));
        let expected = cases
            .iter()
            .position(|case| case.name == "bypass_lh")
            .expect("malformed bypass case");
        assert_eq!(
            index, expected,
            "failure keeps the input job index, queued={queued}"
        );
    }
}

fn run_case(context: &CudaContext, pool: &CudaBufferPool, case: &Tier1Case) {
    let coefficients = generated_coefficients(case);
    let encoded = encode_j2k_code_block_scalar_with_style(
        &coefficients,
        case.width,
        case.height,
        case.subband,
        case.total_bitplanes,
        case.style,
    )
    .unwrap_or_else(|error| panic!("{} encode: {error}", case.name));
    if case.name == "normal_ll_1x1_31bit" {
        assert_eq!(encoded.missing_bit_planes, 0);
        assert_eq!(encoded.number_of_coding_passes, 91);
    }
    let expected = native_decode(case, &encoded, &encoded.data, &encoded.segments, true)
        .unwrap_or_else(|error| panic!("{} native decode: {error}", case.name));
    let job = cuda_job(case, &encoded, encoded.data.len(), true);
    let segments = cuda_segments(&encoded.segments);
    let actual = cuda_decode(
        context,
        pool,
        &encoded.data,
        job,
        &segments,
        coefficients.len(),
    )
    .unwrap_or_else(|error| panic!("{} CUDA decode: {error}", case.name));
    assert_eq!(actual, expected, "{} coefficient parity", case.name);

    if case.name == "normal_ll_1x1_31bit" {
        check_empty_mq(context, pool, case, &encoded, coefficients.len());
    }
    if case.name == "bypass_lh" {
        check_truncated_bypass(context, pool, case, &encoded, coefficients.len());
    }
}

fn native_decode(
    case: &Tier1Case,
    encoded: &j2k_native::EncodedJ2kCodeBlock,
    data: &[u8],
    segments: &[J2kCodeBlockSegment],
    strict: bool,
) -> Result<Vec<f32>, String> {
    let mut output = vec![0.0; case.width as usize * case.height as usize];
    decode_j2k_code_block_scalar(
        J2kCodeBlockDecodeJob {
            data,
            segments,
            width: case.width,
            height: case.height,
            output_stride: case.width as usize,
            missing_bit_planes: encoded.missing_bit_planes,
            number_of_coding_passes: encoded.number_of_coding_passes,
            total_bitplanes: case.total_bitplanes,
            roi_shift: 0,
            sub_band_type: case.subband,
            style: case.style,
            strict,
            dequantization_step: 1.0,
        },
        &mut output,
    )
    .map_err(|error| error.to_string())?;
    Ok(output)
}

fn cuda_job(
    case: &Tier1Case,
    encoded: &j2k_native::EncodedJ2kCodeBlock,
    payload_len: usize,
    strict: bool,
) -> CudaClassicCodeBlockJob {
    CudaClassicCodeBlockJob {
        payload_offset: 0,
        payload_len: u32::try_from(payload_len).expect("payload length"),
        segment_start: 0,
        segment_count: u32::try_from(encoded.segments.len()).expect("segment count"),
        width: case.width,
        height: case.height,
        output_stride: case.width,
        output_offset: 0,
        missing_bitplanes: u32::from(encoded.missing_bit_planes),
        total_bitplanes: u32::from(case.total_bitplanes),
        number_of_coding_passes: u32::from(encoded.number_of_coding_passes),
        sub_band_type: subband_tag(case.subband),
        style_flags: style_flags(case.style),
        strict,
        irreversible_midpoint: false,
        roi_shift: 0,
        dequantization_step: 1.0,
    }
}

fn cuda_segments(segments: &[J2kCodeBlockSegment]) -> Vec<CudaClassicSegment> {
    segments
        .iter()
        .map(|segment| CudaClassicSegment {
            data_offset: segment.data_offset,
            data_length: segment.data_length,
            start_coding_pass: u32::from(segment.start_coding_pass),
            end_coding_pass: u32::from(segment.end_coding_pass),
            use_arithmetic: segment.use_arithmetic,
        })
        .collect()
}

fn cuda_decode(
    context: &CudaContext,
    pool: &CudaBufferPool,
    payload: &[u8],
    job: CudaClassicCodeBlockJob,
    segments: &[CudaClassicSegment],
    output_words: usize,
) -> Result<Vec<f32>, String> {
    cuda_decode_jobs(
        context,
        pool,
        payload,
        &[job],
        segments,
        output_words,
        false,
    )
    .map_err(|error| error.to_string())
}

fn cuda_decode_jobs(
    context: &CudaContext,
    pool: &CudaBufferPool,
    payload: &[u8],
    jobs: &[CudaClassicCodeBlockJob],
    segments: &[CudaClassicSegment],
    output_words: usize,
    queued: bool,
) -> Result<Vec<f32>, CudaError> {
    let engine = J2kCudaEngine::new(context);
    let resources = engine.upload_j2k_decode_payload(payload)?;
    let output = engine.allocate_classic_coefficients_with_pool(output_words, pool)?;
    let targets = [CudaClassicDecodeTarget {
        coefficients: output
            .as_device_buffer()
            .expect("device-resident classic output"),
        jobs,
        segments,
        output_words,
    }];
    if queued {
        let tables = engine.upload_classic_decode_table_resources()?;
        // SAFETY: payload, tables, output, and pool remain live and unchanged
        // until this submission is finished immediately below.
        unsafe {
            engine.decode_classic_codeblocks_multi_enqueue_with_resources_and_pool(
                &resources, &tables, &targets, pool, 0,
            )?
        }
        .finish()?;
    } else {
        engine.decode_classic_codeblocks_multi_with_resources_and_pool(
            &resources, &targets, pool, 0,
        )?;
    }
    let mut bytes = vec![0; output.byte_len()];
    output.copy_to_host(&mut bytes)?;
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| f32::from_ne_bytes(*word))
        .collect())
}

fn check_empty_mq(
    context: &CudaContext,
    pool: &CudaBufferPool,
    case: &Tier1Case,
    encoded: &j2k_native::EncodedJ2kCodeBlock,
    output_words: usize,
) {
    let segments = encoded
        .segments
        .iter()
        .copied()
        .map(|mut segment| {
            segment.data_offset = 0;
            segment.data_length = 0;
            segment
        })
        .collect::<Vec<_>>();
    for strict in [false, true] {
        let native = native_decode(case, encoded, &[], &segments, strict);
        let cuda = cuda_decode(
            context,
            pool,
            &[],
            cuda_job(case, encoded, 0, strict),
            &cuda_segments(&segments),
            output_words,
        );
        match (native, cuda) {
            (Ok(expected), Ok(actual)) => assert_eq!(actual, expected, "empty MQ strict={strict}"),
            (Err(_), Err(_)) => {}
            (native, cuda) => {
                panic!("empty MQ strict={strict} result mismatch: native={native:?} cuda={cuda:?}")
            }
        }
    }
}

fn check_truncated_bypass(
    context: &CudaContext,
    pool: &CudaBufferPool,
    case: &Tier1Case,
    encoded: &j2k_native::EncodedJ2kCodeBlock,
    output_words: usize,
) {
    let first_raw = encoded
        .segments
        .iter()
        .position(|segment| !segment.use_arithmetic)
        .expect("bypass fixture raw segment");
    let truncated_len = encoded.segments[first_raw].data_offset as usize;
    let data = &encoded.data[..truncated_len];
    let mut segments = encoded.segments.clone();
    for segment in &mut segments[first_raw..] {
        segment.data_offset = u32::try_from(truncated_len).expect("truncated offset");
        segment.data_length = 0;
    }
    let expected = native_decode(case, encoded, data, &segments, false)
        .expect("native lenient truncated bypass decode");
    let expected_strict = native_decode(case, encoded, data, &segments, true)
        .expect("native strict truncated bypass decode extends a clean segment end");
    assert_eq!(expected_strict, expected, "native strict truncated parity");
    let actual = cuda_decode(
        context,
        pool,
        data,
        cuda_job(case, encoded, truncated_len, false),
        &cuda_segments(&segments),
        output_words,
    )
    .expect("CUDA lenient truncated bypass decode");
    assert_eq!(actual, expected, "lenient truncated parity");
    let actual_strict = cuda_decode(
        context,
        pool,
        data,
        cuda_job(case, encoded, truncated_len, true),
        &cuda_segments(&segments),
        output_words,
    )
    .expect("CUDA strict truncated bypass decode extends a clean segment end");
    assert_eq!(actual_strict, expected_strict, "strict truncated parity");
}
