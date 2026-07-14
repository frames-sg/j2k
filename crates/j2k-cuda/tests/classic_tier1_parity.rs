// SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg(feature = "cuda-runtime")]

use j2k_cuda_runtime::{
    CudaBufferPool, CudaClassicCodeBlockJob, CudaClassicDecodeTarget, CudaClassicSegment,
    CudaContext,
};
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
    let default_style = J2kCodeBlockStyle {
        selective_arithmetic_coding_bypass: false,
        reset_context_probabilities: false,
        termination_on_each_pass: false,
        vertically_causal_context: false,
        segmentation_symbols: false,
    };
    let cases = [
        Tier1Case {
            name: "normal_ll_1x1_31bit",
            width: 1,
            height: 1,
            total_bitplanes: 31,
            subband: J2kSubBandType::LowLow,
            style: default_style,
            seed: 0x5100,
        },
        Tier1Case {
            name: "bypass_lh",
            width: 13,
            height: 9,
            total_bitplanes: 10,
            subband: J2kSubBandType::LowHigh,
            style: J2kCodeBlockStyle {
                selective_arithmetic_coding_bypass: true,
                ..default_style
            },
            seed: 0x5200,
        },
        Tier1Case {
            name: "term_reset_hl",
            width: 13,
            height: 9,
            total_bitplanes: 10,
            subband: J2kSubBandType::HighLow,
            style: J2kCodeBlockStyle {
                reset_context_probabilities: true,
                termination_on_each_pass: true,
                ..default_style
            },
            seed: 0x5300,
        },
        Tier1Case {
            name: "segmentation_hh",
            width: 13,
            height: 9,
            total_bitplanes: 10,
            subband: J2kSubBandType::HighHigh,
            style: J2kCodeBlockStyle {
                segmentation_symbols: true,
                ..default_style
            },
            seed: 0x5400,
        },
        Tier1Case {
            name: "vcausal_ll",
            width: 13,
            height: 9,
            total_bitplanes: 10,
            subband: J2kSubBandType::LowLow,
            style: J2kCodeBlockStyle {
                vertically_causal_context: true,
                ..default_style
            },
            seed: 0x5500,
        },
        Tier1Case {
            name: "combined_64x64",
            width: 64,
            height: 64,
            total_bitplanes: 10,
            subband: J2kSubBandType::HighHigh,
            style: J2kCodeBlockStyle {
                selective_arithmetic_coding_bypass: true,
                reset_context_probabilities: true,
                termination_on_each_pass: true,
                vertically_causal_context: true,
                segmentation_symbols: true,
            },
            seed: 0x5600,
        },
    ];

    let context = CudaContext::system_default().expect("CUDA context");
    let pool = context.buffer_pool();
    for case in cases {
        run_case(&context, &pool, &case);
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
    let resources = context
        .upload_j2k_decode_payload(payload)
        .map_err(|error| error.to_string())?;
    let output = context
        .allocate_classic_coefficients_with_pool(output_words, pool)
        .map_err(|error| error.to_string())?;
    context
        .decode_classic_codeblocks_multi_with_resources_and_pool(
            &resources,
            &[CudaClassicDecodeTarget {
                coefficients: output
                    .as_device_buffer()
                    .ok_or_else(|| "classic output is not device-resident".to_string())?,
                jobs: &[job],
                segments,
                output_words,
            }],
            pool,
            0,
        )
        .map_err(|error| error.to_string())?;
    let mut bytes = vec![0; output.byte_len()];
    output
        .copy_to_host(&mut bytes)
        .map_err(|error| error.to_string())?;
    Ok(bytes
        .chunks_exact(4)
        .map(|word| f32::from_ne_bytes(word.try_into().expect("f32 word")))
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
    assert!(native_decode(case, encoded, data, &segments, true).is_err());
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
    assert!(
        cuda_decode(
            context,
            pool,
            data,
            cuda_job(case, encoded, truncated_len, true),
            &cuda_segments(&segments),
            output_words,
        )
        .is_err(),
        "CUDA strict truncated bypass decode must fail"
    );
}
