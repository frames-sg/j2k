// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{abi::*, prepare::*};

fn max_job() -> CudaClassicCodeBlockJob {
    CudaClassicCodeBlockJob {
        payload_offset: 0,
        payload_len: 7,
        segment_start: 0,
        segment_count: 1,
        width: 64,
        height: 64,
        output_stride: 64,
        output_offset: 0,
        missing_bitplanes: 0,
        total_bitplanes: 31,
        number_of_coding_passes: 91,
        sub_band_type: 3,
        style_flags: 0,
        strict: true,
        dequantization_step: 1.0,
    }
}

#[test]
fn classic_preflight_accepts_maximum_contract() {
    let segments = [CudaClassicSegment {
        data_offset: 0,
        data_length: 7,
        start_coding_pass: 0,
        end_coding_pass: 91,
        use_arithmetic: true,
    }];
    validate_classic_job(7, &segments, 64 * 64, &max_job())
        .expect("maximum classic Tier-1 contract");
}

#[test]
fn classic_preflight_accepts_zero_length_prefix_segment() {
    let mut job = max_job();
    job.number_of_coding_passes = 1;
    job.style_flags = STYLE_TERMALL;
    job.segment_count = 2;
    let segments = [
        CudaClassicSegment {
            data_offset: 0,
            data_length: 0,
            start_coding_pass: 0,
            end_coding_pass: 0,
            use_arithmetic: true,
        },
        CudaClassicSegment {
            data_offset: 0,
            data_length: 7,
            start_coding_pass: 0,
            end_coding_pass: 1,
            use_arithmetic: true,
        },
    ];
    validate_classic_job(7, &segments, 64 * 64, &job).expect("zero-length classic prefix");
}

#[test]
fn classic_preflight_rejects_noncontiguous_segments_and_output_overrun() {
    let mut job = max_job();
    let segments = [CudaClassicSegment {
        data_offset: 1,
        data_length: 6,
        start_coding_pass: 0,
        end_coding_pass: 91,
        use_arithmetic: true,
    }];
    assert!(validate_classic_job(7, &segments, 64 * 64, &job).is_err());

    job.output_offset = 1;
    let segments = [CudaClassicSegment {
        data_offset: 0,
        data_length: 7,
        start_coding_pass: 0,
        end_coding_pass: 91,
        use_arithmetic: true,
    }];
    assert!(validate_classic_job(7, &segments, 64 * 64, &job).is_err());

    job.output_offset = 0;
    job.payload_len = 8;
    assert!(validate_classic_job(8, &segments, 64 * 64, &job).is_err());
}

#[cfg(target_pointer_width = "64")]
#[test]
fn classic_preflight_accepts_output_ranges_beyond_u32_words() {
    let mut job = max_job();
    job.width = 2;
    job.height = 2;
    job.output_stride = u32::MAX;
    job.number_of_coding_passes = 1;
    let segments = [CudaClassicSegment {
        data_offset: 0,
        data_length: 7,
        start_coding_pass: 0,
        end_coding_pass: 1,
        use_arithmetic: true,
    }];

    validate_classic_job(
        7,
        &segments,
        usize::try_from(u64::from(u32::MAX) + 2).expect("64-bit output words"),
        &job,
    )
    .expect("device output addressing must support the host-validated range");
}

#[test]
fn classic_runtime_validates_empty_work_and_times_only_status_copy() {
    let source = include_str!("launch.rs");
    let method = source
        .split("pub fn decode_classic_codeblocks_multi_with_resources_and_pool_timed")
        .nth(1)
        .expect("timed classic decode method");
    let owner_validation = method
        .find("validate_classic_launch_owners")
        .expect("owner validation");
    let prepare = method.find("prepare_classic_decode").expect("preparation");
    let empty_return = method
        .find("if prepared.jobs.is_empty()")
        .expect("validated empty fast path");
    assert!(owner_validation < empty_return && prepare < empty_return);

    let status_copy = method.find("statuses.copy_to_host").expect("status copy");
    let status_timing = method
        .find("let status_d2h_us")
        .expect("status timing result");
    let pool_release = method
        .find("let release_result = pool_reuse_guard.release()")
        .expect("pool release");
    assert!(status_copy < status_timing && status_timing < pool_release);
}
