use cuda_device::{kernel, thread};
use cuda_host::cuda_module;

include!("../../../cuda_oxide_simt_prelude.rs");

#[repr(C)]
#[derive(Clone, Copy)]
struct CudaJ2kStoreGray8Job {
    input_width: u32,
    source_x: u32,
    source_y: u32,
    copy_width: u32,
    copy_height: u32,
    output_width: u32,
    output_height: u32,
    output_x: u32,
    output_y: u32,
    addend: f32,
    bit_depth: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CudaJ2kStoreGray16Job {
    input_width: u32,
    source_x: u32,
    source_y: u32,
    copy_width: u32,
    copy_height: u32,
    output_width: u32,
    output_height: u32,
    output_x: u32,
    output_y: u32,
    addend: f32,
    bit_depth: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CudaJ2kInverseMctJob {
    len: u32,
    irreversible97: u32,
    addend0: f32,
    addend1: f32,
    addend2: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CudaJ2kStoreRgb8Job {
    input_width0: u32,
    input_width1: u32,
    input_width2: u32,
    source_x0: u32,
    source_y0: u32,
    source_x1: u32,
    source_y1: u32,
    source_x2: u32,
    source_y2: u32,
    copy_width: u32,
    copy_height: u32,
    output_width: u32,
    output_height: u32,
    output_x: u32,
    output_y: u32,
    addend0: f32,
    addend1: f32,
    addend2: f32,
    bit_depth0: u32,
    bit_depth1: u32,
    bit_depth2: u32,
    rgba: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CudaJ2kStoreRgb16Job {
    input_width0: u32,
    input_width1: u32,
    input_width2: u32,
    source_x0: u32,
    source_y0: u32,
    source_x1: u32,
    source_y1: u32,
    source_x2: u32,
    source_y2: u32,
    copy_width: u32,
    copy_height: u32,
    output_width: u32,
    output_height: u32,
    output_x: u32,
    output_y: u32,
    addend0: f32,
    addend1: f32,
    addend2: f32,
    bit_depth0: u32,
    bit_depth1: u32,
    bit_depth2: u32,
    rgba: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CudaJ2kStoreRgb8MctJob {
    store: CudaJ2kStoreRgb8Job,
    irreversible97: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CudaJ2kStoreRgb8MctBatchJob {
    plane0_ptr: u64,
    plane1_ptr: u64,
    plane2_ptr: u64,
    output_ptr: u64,
    job: CudaJ2kStoreRgb8MctJob,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CudaJ2kStoreRgb16MctJob {
    store: CudaJ2kStoreRgb16Job,
    irreversible97: u32,
}

#[inline(always)]
fn load_f32(ptr: *const f32, index: u32) -> f32 {
    simt_load(ptr, index as usize)
}

#[inline(always)]
fn load_job<T: Copy>(ptr: *const T) -> T {
    simt_load(ptr, 0)
}

#[inline(always)]
fn store_f32(ptr: *mut f32, index: u32, value: f32) {
    simt_store(ptr, index as usize, value);
}

#[inline(always)]
fn store_u8(ptr: *mut u8, index: u32, value: u8) {
    simt_store(ptr, index as usize, value);
}

#[inline(always)]
fn store_u16(ptr: *mut u16, index: u32, value: u16) {
    simt_store(ptr, index as usize, value);
}

#[inline(always)]
fn floor_f32(value: f32) -> f32 {
    // f32::floor routes through libdevice in cuda-oxide, which emits NVVM IR
    // instead of the PTX loaded by this runtime path.
    let truncated = value as i32 as f32;
    if truncated > value {
        truncated - 1.0
    } else {
        truncated
    }
}

#[inline(always)]
fn round_f32(value: f32) -> f32 {
    if value >= 0.0 {
        floor_f32(value + 0.5)
    } else {
        -floor_f32(-value + 0.5)
    }
}

#[inline(always)]
fn clamp_f32(value: f32, min: f32, max: f32) -> f32 {
    if value < min {
        min
    } else if value > max {
        max
    } else {
        value
    }
}

#[inline(always)]
fn max_int_for_bit_depth(bit_depth: u32) -> u32 {
    if bit_depth == 0 {
        1
    } else {
        (1_u32 << bit_depth) - 1
    }
}

#[inline(always)]
fn sample_as_u8(sample: f32, bit_depth: u32) -> u8 {
    let rounded = round_f32(sample);
    if bit_depth >= 8 {
        return clamp_f32(rounded, 0.0, 255.0) as u8;
    }

    let max_int = (1_u32 << bit_depth) - 1;
    let max_value = if max_int > 1 { max_int as f32 } else { 1.0 };
    round_f32((clamp_f32(rounded, 0.0, max_value) / max_value) * 255.0) as u8
}

#[inline(always)]
fn sample_as_u16(sample: f32, bit_depth: u32) -> u16 {
    let rounded = round_f32(sample);
    if bit_depth >= 16 {
        return clamp_f32(rounded, 0.0, 65535.0) as u16;
    }

    let max_int = max_int_for_bit_depth(bit_depth);
    let max_value = if max_int > 1 { max_int as f32 } else { 1.0 };
    round_f32((clamp_f32(rounded, 0.0, max_value) / max_value) * 65535.0) as u16
}

#[inline(always)]
fn inverse_mct_sample(src0: f32, src1: f32, src2: f32, irreversible97: u32) -> (f32, f32, f32) {
    if irreversible97 != 0 {
        (
            src0 + 1.402 * src2,
            src0 - 0.34413 * src1 - 0.71414 * src2,
            src0 + 1.772 * src1,
        )
    } else {
        let green = src0 - floor_f32((src2 + src1) * 0.25);
        (src2 + green, green, src1 + green)
    }
}

#[inline(always)]
fn source_index(input_width: u32, source_x: u32, source_y: u32, row: u32, col: u32) -> u32 {
    (source_y + row) * input_width + source_x + col
}

#[inline(always)]
fn output_pixel_index(output_width: u32, output_x: u32, output_y: u32, row: u32, col: u32) -> u32 {
    (output_y + row) * output_width + output_x + col
}

#[inline(always)]
fn pixel_coords(gid: u32, copy_width: u32) -> (u32, u32) {
    let row = gid / copy_width;
    (row, gid - row * copy_width)
}

#[inline(always)]
fn store_rgb8_sample(
    plane0: *const f32,
    plane1: *const f32,
    plane2: *const f32,
    output: *mut u8,
    job: CudaJ2kStoreRgb8Job,
    gid: u32,
) {
    let (row, col) = pixel_coords(gid, job.copy_width);
    let src0 = source_index(job.input_width0, job.source_x0, job.source_y0, row, col);
    let src1 = source_index(job.input_width1, job.source_x1, job.source_y1, row, col);
    let src2 = source_index(job.input_width2, job.source_x2, job.source_y2, row, col);
    let channels = if job.rgba != 0 { 4 } else { 3 };
    let dst = output_pixel_index(job.output_width, job.output_x, job.output_y, row, col) * channels;

    store_u8(
        output,
        dst,
        sample_as_u8(load_f32(plane0, src0) + job.addend0, job.bit_depth0),
    );
    store_u8(
        output,
        dst + 1,
        sample_as_u8(load_f32(plane1, src1) + job.addend1, job.bit_depth1),
    );
    store_u8(
        output,
        dst + 2,
        sample_as_u8(load_f32(plane2, src2) + job.addend2, job.bit_depth2),
    );
    if job.rgba != 0 {
        store_u8(output, dst + 3, 255);
    }
}

#[inline(always)]
fn store_rgb16_sample(
    plane0: *const f32,
    plane1: *const f32,
    plane2: *const f32,
    output: *mut u16,
    job: CudaJ2kStoreRgb16Job,
    gid: u32,
) {
    let (row, col) = pixel_coords(gid, job.copy_width);
    let src0 = source_index(job.input_width0, job.source_x0, job.source_y0, row, col);
    let src1 = source_index(job.input_width1, job.source_x1, job.source_y1, row, col);
    let src2 = source_index(job.input_width2, job.source_x2, job.source_y2, row, col);
    let channels = if job.rgba != 0 { 4 } else { 3 };
    let dst = output_pixel_index(job.output_width, job.output_x, job.output_y, row, col) * channels;

    store_u16(
        output,
        dst,
        sample_as_u16(load_f32(plane0, src0) + job.addend0, job.bit_depth0),
    );
    store_u16(
        output,
        dst + 1,
        sample_as_u16(load_f32(plane1, src1) + job.addend1, job.bit_depth1),
    );
    store_u16(
        output,
        dst + 2,
        sample_as_u16(load_f32(plane2, src2) + job.addend2, job.bit_depth2),
    );
    if job.rgba != 0 {
        store_u16(output, dst + 3, 65535);
    }
}

#[inline(always)]
fn store_rgb8_mct_sample(
    plane0: *const f32,
    plane1: *const f32,
    plane2: *const f32,
    output: *mut u8,
    mct_job: CudaJ2kStoreRgb8MctJob,
    gid: u32,
) {
    let job = mct_job.store;
    let (row, col) = pixel_coords(gid, job.copy_width);
    let src0 = source_index(job.input_width0, job.source_x0, job.source_y0, row, col);
    let src1 = source_index(job.input_width1, job.source_x1, job.source_y1, row, col);
    let src2 = source_index(job.input_width2, job.source_x2, job.source_y2, row, col);
    let channels = if job.rgba != 0 { 4 } else { 3 };
    let dst = output_pixel_index(job.output_width, job.output_x, job.output_y, row, col) * channels;
    let (out0, out1, out2) = inverse_mct_sample(
        load_f32(plane0, src0),
        load_f32(plane1, src1),
        load_f32(plane2, src2),
        mct_job.irreversible97,
    );

    store_u8(
        output,
        dst,
        sample_as_u8(out0 + job.addend0, job.bit_depth0),
    );
    store_u8(
        output,
        dst + 1,
        sample_as_u8(out1 + job.addend1, job.bit_depth1),
    );
    store_u8(
        output,
        dst + 2,
        sample_as_u8(out2 + job.addend2, job.bit_depth2),
    );
    if job.rgba != 0 {
        store_u8(output, dst + 3, 255);
    }
}

#[inline(always)]
fn store_rgb16_mct_sample(
    plane0: *const f32,
    plane1: *const f32,
    plane2: *const f32,
    output: *mut u16,
    mct_job: CudaJ2kStoreRgb16MctJob,
    gid: u32,
) {
    let job = mct_job.store;
    let (row, col) = pixel_coords(gid, job.copy_width);
    let src0 = source_index(job.input_width0, job.source_x0, job.source_y0, row, col);
    let src1 = source_index(job.input_width1, job.source_x1, job.source_y1, row, col);
    let src2 = source_index(job.input_width2, job.source_x2, job.source_y2, row, col);
    let channels = if job.rgba != 0 { 4 } else { 3 };
    let dst = output_pixel_index(job.output_width, job.output_x, job.output_y, row, col) * channels;
    let (out0, out1, out2) = inverse_mct_sample(
        load_f32(plane0, src0),
        load_f32(plane1, src1),
        load_f32(plane2, src2),
        mct_job.irreversible97,
    );

    store_u16(
        output,
        dst,
        sample_as_u16(out0 + job.addend0, job.bit_depth0),
    );
    store_u16(
        output,
        dst + 1,
        sample_as_u16(out1 + job.addend1, job.bit_depth1),
    );
    store_u16(
        output,
        dst + 2,
        sample_as_u16(out2 + job.addend2, job.bit_depth2),
    );
    if job.rgba != 0 {
        store_u16(output, dst + 3, 65535);
    }
}

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub unsafe fn j2k_store_gray8(
        input: *const f32,
        output: *mut u8,
        job_buffer: *const CudaJ2kStoreGray8Job,
    ) {
        let job = load_job(job_buffer);
        let pixels = job.copy_width * job.copy_height;
        let gid = thread::index_1d().get() as u32;
        if gid >= pixels {
            return;
        }

        let (row, col) = pixel_coords(gid, job.copy_width);
        let src = source_index(job.input_width, job.source_x, job.source_y, row, col);
        let dst = output_pixel_index(job.output_width, job.output_x, job.output_y, row, col);
        store_u8(
            output,
            dst,
            sample_as_u8(load_f32(input, src) + job.addend, job.bit_depth),
        );
    }

    #[kernel]
    pub unsafe fn j2k_store_gray16(
        input: *const f32,
        output: *mut u16,
        job_buffer: *const CudaJ2kStoreGray16Job,
    ) {
        let job = load_job(job_buffer);
        let pixels = job.copy_width * job.copy_height;
        let gid = thread::index_1d().get() as u32;
        if gid >= pixels {
            return;
        }

        let (row, col) = pixel_coords(gid, job.copy_width);
        let src = source_index(job.input_width, job.source_x, job.source_y, row, col);
        let dst = output_pixel_index(job.output_width, job.output_x, job.output_y, row, col);
        store_u16(
            output,
            dst,
            sample_as_u16(load_f32(input, src) + job.addend, job.bit_depth),
        );
    }

    #[kernel]
    pub unsafe fn j2k_inverse_mct(
        plane0: *mut f32,
        plane1: *mut f32,
        plane2: *mut f32,
        job_buffer: *const CudaJ2kInverseMctJob,
    ) {
        let job = load_job(job_buffer);
        let gid = thread::index_1d().get() as u32;
        if gid >= job.len {
            return;
        }

        let (out0, out1, out2) = inverse_mct_sample(
            load_f32(plane0.cast_const(), gid),
            load_f32(plane1.cast_const(), gid),
            load_f32(plane2.cast_const(), gid),
            job.irreversible97,
        );
        store_f32(plane0, gid, out0 + job.addend0);
        store_f32(plane1, gid, out1 + job.addend1);
        store_f32(plane2, gid, out2 + job.addend2);
    }

    #[kernel]
    pub unsafe fn j2k_store_rgb8(
        plane0: *const f32,
        plane1: *const f32,
        plane2: *const f32,
        output: *mut u8,
        job_buffer: *const CudaJ2kStoreRgb8Job,
    ) {
        let job = load_job(job_buffer);
        let pixels = job.copy_width * job.copy_height;
        let gid = thread::index_1d().get() as u32;
        if gid >= pixels {
            return;
        }

        store_rgb8_sample(plane0, plane1, plane2, output, job, gid);
    }

    #[kernel]
    pub unsafe fn j2k_store_rgb16(
        plane0: *const f32,
        plane1: *const f32,
        plane2: *const f32,
        output: *mut u16,
        job_buffer: *const CudaJ2kStoreRgb16Job,
    ) {
        let job = load_job(job_buffer);
        let pixels = job.copy_width * job.copy_height;
        let gid = thread::index_1d().get() as u32;
        if gid >= pixels {
            return;
        }

        store_rgb16_sample(plane0, plane1, plane2, output, job, gid);
    }

    #[kernel]
    pub unsafe fn j2k_store_rgb8_mct_batch(jobs: *const CudaJ2kStoreRgb8MctBatchJob) {
        let item = load_job(unsafe { jobs.add(thread::index_2d_row() as usize) });
        let plane0 = item.plane0_ptr as usize as *const f32;
        let plane1 = item.plane1_ptr as usize as *const f32;
        let plane2 = item.plane2_ptr as usize as *const f32;
        let output = item.output_ptr as usize as *mut u8;
        let pixels = item.job.store.copy_width * item.job.store.copy_height;
        let gid = thread::index_2d_col() as u32;
        if gid >= pixels {
            return;
        }

        store_rgb8_mct_sample(plane0, plane1, plane2, output, item.job, gid);
    }

    #[kernel]
    pub unsafe fn j2k_store_rgb16_mct(
        plane0: *const f32,
        plane1: *const f32,
        plane2: *const f32,
        output: *mut u16,
        job_buffer: *const CudaJ2kStoreRgb16MctJob,
    ) {
        let mct_job = load_job(job_buffer);
        let pixels = mct_job.store.copy_width * mct_job.store.copy_height;
        let gid = thread::index_1d().get() as u32;
        if gid >= pixels {
            return;
        }

        store_rgb16_mct_sample(plane0, plane1, plane2, output, mct_job, gid);
    }
}

fn main() {}
