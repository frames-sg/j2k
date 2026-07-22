// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    abi::{
        CudaJ2kStoreRgb8Job, CudaJ2kStoreRgb8MctJob, CudaJ2kStoreRgb16Job, CudaJ2kStoreRgb16MctJob,
    },
    layout::{output_pixel_index, pixel_coords, source_index},
    memory::{load_f32, store_u8, store_u16},
    sample::{sample_as_u8, sample_as_u16},
    transform::inverse_mct_sample,
};

#[inline(always)]
pub(crate) fn store_rgb8_sample(
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
pub(crate) fn store_rgb16_sample(
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
pub(crate) fn store_rgb8_mct_sample(
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
pub(crate) fn store_rgb16_mct_sample(
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
