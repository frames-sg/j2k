use cuda_device::{kernel, thread};
use cuda_host::cuda_module;

include!("../../../cuda_oxide_simt_prelude.rs");

const HT_ROI_SHIFT_MASK: u32 = 0xff;
const HT_IRREVERSIBLE_MIDPOINT_FLAG: u32 = 1 << 8;

#[repr(C)]
#[derive(Clone, Copy)]
struct J2kHtCleanupBatchJob {
    coded_offset: u32,
    width: u32,
    height: u32,
    coded_len: u32,
    cleanup_length: u32,
    refinement_length: u32,
    missing_msbs: u32,
    num_bitplanes: u32,
    reconstruction: u32,
    number_of_coding_passes: u32,
    output_stride: u32,
    output_offset: u32,
    dequantization_step: f32,
    stripe_causal: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct J2kHtDequantizeJob {
    output_ptr: u64,
    width: u32,
    height: u32,
    output_stride: u32,
    output_offset: u32,
    num_bitplanes: u32,
    reconstruction: u32,
    dequantization_step: f32,
    reserved_tail: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct J2kHtCleanupMultiBatchJob {
    output_ptr: u64,
    coded_offset: u32,
    width: u32,
    height: u32,
    coded_len: u32,
    cleanup_length: u32,
    refinement_length: u32,
    missing_msbs: u32,
    num_bitplanes: u32,
    number_of_coding_passes: u32,
    output_stride: u32,
    output_offset: u32,
    dequantization_step: f32,
    stripe_causal: u32,
    reconstruction: u32,
}

#[inline(always)]
fn load_u32(ptr: *const u32, index: u32) -> u32 {
    simt_load(ptr, index as usize)
}

#[inline(always)]
fn store_u32(ptr: *mut u32, index: u32, value: u32) {
    simt_store(ptr, index as usize, value);
}

#[inline(always)]
fn load_job<T: Copy>(ptr: *const T, index: u32) -> T {
    simt_load(ptr, index as usize)
}

#[inline(always)]
fn coefficient_to_i32(value: u32, k_max: u32) -> i32 {
    let shift = 31 - k_max;
    let magnitude = ((value & 0x7fff_ffff) >> shift) as i32;
    if (value & 0x8000_0000) != 0 {
        -magnitude
    } else {
        magnitude
    }
}

#[inline(always)]
fn coefficient_to_float_bits(value: u32, k_max: u32, scale: f32, reconstruction: u32) -> u32 {
    let roi_shift = reconstruction & HT_ROI_SHIFT_MASK;
    let irreversible_midpoint = reconstruction & HT_IRREVERSIBLE_MIDPOINT_FLAG != 0;
    if !irreversible_midpoint {
        let coefficient = coefficient_to_i32(value, k_max);
        let magnitude = if coefficient < 0 {
            (-coefficient) as u32
        } else {
            coefficient as u32
        };
        let shifted = if roi_shift != 0 && magnitude >= 1 << roi_shift {
            (magnitude >> roi_shift) as i32
        } else {
            magnitude as i32
        };
        return ((if coefficient < 0 { -shifted } else { shifted }) as f32 * scale).to_bits();
    }

    let fixed_scale = f32::from_bits((k_max + 96) << 23);
    let magnitude = (value & 0x7fff_ffff) as f32 * fixed_scale;
    let roi_scale = f32::from_bits((127 - roi_shift) << 23);
    let roi_threshold = f32::from_bits((127 + roi_shift) << 23);
    let reconstructed = if roi_shift != 0 && magnitude >= roi_threshold {
        magnitude * roi_scale
    } else {
        magnitude
    };
    let coefficient = if value & 0x8000_0000 != 0 {
        -reconstructed
    } else {
        reconstructed
    };
    (coefficient * scale).to_bits()
}

#[inline(always)]
fn dequantize_codeblock(
    decoded_data: *mut u32,
    width: u32,
    height: u32,
    output_stride: u32,
    output_offset: u32,
    num_bitplanes: u32,
    reconstruction: u32,
    dequantization_step: f32,
) {
    let sample_count = width * height;
    let mut sample = thread::threadIdx_x();
    let step = thread::blockDim_x();
    while sample < sample_count {
        let y = sample / width;
        let x = sample - y * width;
        let idx = output_offset + y * output_stride + x;
        store_u32(
            decoded_data,
            idx,
            coefficient_to_float_bits(
                load_u32(decoded_data.cast_const(), idx),
                num_bitplanes + (reconstruction & HT_ROI_SHIFT_MASK),
                dequantization_step,
                reconstruction,
            ),
        );
        sample += step;
    }
}

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub unsafe fn j2k_dequantize_htj2k_codeblocks(
        decoded_data: *mut u32,
        jobs: *const J2kHtCleanupBatchJob,
    ) {
        let job = load_job(jobs, thread::blockIdx_x());
        dequantize_codeblock(
            decoded_data,
            job.width,
            job.height,
            job.output_stride,
            job.output_offset,
            job.num_bitplanes,
            job.reconstruction,
            job.dequantization_step,
        );
    }

    #[kernel]
    pub unsafe fn j2k_dequantize_htj2k_codeblocks_multi(jobs: *const J2kHtDequantizeJob) {
        let job = load_job(jobs, thread::blockIdx_x());
        dequantize_codeblock(
            job.output_ptr as usize as *mut u32,
            job.width,
            job.height,
            job.output_stride,
            job.output_offset,
            job.num_bitplanes,
            job.reconstruction,
            job.dequantization_step,
        );
    }

    #[kernel]
    pub unsafe fn j2k_dequantize_htj2k_cleanup_jobs_multi(jobs: *const J2kHtCleanupMultiBatchJob) {
        let job = load_job(jobs, thread::blockIdx_x());
        dequantize_codeblock(
            job.output_ptr as usize as *mut u32,
            job.width,
            job.height,
            job.output_stride,
            job.output_offset,
            job.num_bitplanes,
            job.reconstruction,
            job.dequantization_step,
        );
    }
}

fn main() {}
