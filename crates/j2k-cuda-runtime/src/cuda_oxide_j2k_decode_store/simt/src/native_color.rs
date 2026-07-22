// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    abi::{CudaJ2kStoreRgbNativeJob, CudaJ2kStoreRgbaNativeJob},
    layout::{output_pixel_index, pixel_coords, source_index},
    memory::{load_f32, store_i16, store_u8, store_u16},
    sample::{sample_as_native_i16, sample_as_native_u8, sample_as_native_u16},
    transform::inverse_mct_sample,
};

#[inline(always)]
pub(crate) fn exact_rgb_samples(
    plane0: *const f32,
    plane1: *const f32,
    plane2: *const f32,
    job: CudaJ2kStoreRgbNativeJob,
    row: u32,
    col: u32,
) -> (f32, f32, f32) {
    let src0 = source_index(job.input_width0, job.source_x0, job.source_y0, row, col);
    let src1 = source_index(job.input_width1, job.source_x1, job.source_y1, row, col);
    let src2 = source_index(job.input_width2, job.source_x2, job.source_y2, row, col);
    let samples = (
        load_f32(plane0, src0),
        load_f32(plane1, src1),
        load_f32(plane2, src2),
    );
    let (out0, out1, out2) = if job.transform == 0 {
        samples
    } else {
        inverse_mct_sample(
            samples.0,
            samples.1,
            samples.2,
            u32::from(job.transform == 2),
        )
    };
    (out0 + job.addend0, out1 + job.addend1, out2 + job.addend2)
}

#[inline(always)]
pub(crate) fn exact_rgb_destination(
    job: CudaJ2kStoreRgbNativeJob,
    pixel: u32,
    channel: u32,
) -> u32 {
    if job.layout == 0 {
        pixel * 3 + channel
    } else {
        channel * job.output_width * job.output_height + pixel
    }
}

#[inline(always)]
pub(crate) fn store_rgb8_native_sample(
    plane0: *const f32,
    plane1: *const f32,
    plane2: *const f32,
    output: *mut u8,
    job: CudaJ2kStoreRgbNativeJob,
    gid: u32,
) {
    let (row, col) = pixel_coords(gid, job.copy_width);
    let pixel = output_pixel_index(job.output_width, job.output_x, job.output_y, row, col);
    let samples = exact_rgb_samples(plane0, plane1, plane2, job, row, col);
    store_u8(
        output,
        exact_rgb_destination(job, pixel, 0),
        sample_as_native_u8(samples.0, job.bit_depth0),
    );
    store_u8(
        output,
        exact_rgb_destination(job, pixel, 1),
        sample_as_native_u8(samples.1, job.bit_depth1),
    );
    store_u8(
        output,
        exact_rgb_destination(job, pixel, 2),
        sample_as_native_u8(samples.2, job.bit_depth2),
    );
}

#[inline(always)]
pub(crate) fn store_rgb16_native_sample(
    plane0: *const f32,
    plane1: *const f32,
    plane2: *const f32,
    output: *mut u16,
    job: CudaJ2kStoreRgbNativeJob,
    gid: u32,
) {
    let (row, col) = pixel_coords(gid, job.copy_width);
    let pixel = output_pixel_index(job.output_width, job.output_x, job.output_y, row, col);
    let samples = exact_rgb_samples(plane0, plane1, plane2, job, row, col);
    store_u16(
        output,
        exact_rgb_destination(job, pixel, 0),
        sample_as_native_u16(samples.0, job.bit_depth0),
    );
    store_u16(
        output,
        exact_rgb_destination(job, pixel, 1),
        sample_as_native_u16(samples.1, job.bit_depth1),
    );
    store_u16(
        output,
        exact_rgb_destination(job, pixel, 2),
        sample_as_native_u16(samples.2, job.bit_depth2),
    );
}

#[inline(always)]
pub(crate) fn store_rgbi16_native_sample(
    plane0: *const f32,
    plane1: *const f32,
    plane2: *const f32,
    output: *mut i16,
    job: CudaJ2kStoreRgbNativeJob,
    gid: u32,
) {
    let (row, col) = pixel_coords(gid, job.copy_width);
    let pixel = output_pixel_index(job.output_width, job.output_x, job.output_y, row, col);
    let samples = exact_rgb_samples(plane0, plane1, plane2, job, row, col);
    store_i16(
        output,
        exact_rgb_destination(job, pixel, 0),
        sample_as_native_i16(samples.0, job.bit_depth0),
    );
    store_i16(
        output,
        exact_rgb_destination(job, pixel, 1),
        sample_as_native_i16(samples.1, job.bit_depth1),
    );
    store_i16(
        output,
        exact_rgb_destination(job, pixel, 2),
        sample_as_native_i16(samples.2, job.bit_depth2),
    );
}

#[inline(always)]
pub(crate) fn exact_rgba_samples(
    plane0: *const f32,
    plane1: *const f32,
    plane2: *const f32,
    plane3: *const f32,
    job: CudaJ2kStoreRgbaNativeJob,
    row: u32,
    col: u32,
) -> (f32, f32, f32, f32) {
    let src0 = source_index(job.input_width0, job.source_x0, job.source_y0, row, col);
    let src1 = source_index(job.input_width1, job.source_x1, job.source_y1, row, col);
    let src2 = source_index(job.input_width2, job.source_x2, job.source_y2, row, col);
    let src3 = source_index(job.input_width3, job.source_x3, job.source_y3, row, col);
    let rgb = (
        load_f32(plane0, src0),
        load_f32(plane1, src1),
        load_f32(plane2, src2),
    );
    let (out0, out1, out2) = if job.transform == 0 {
        rgb
    } else {
        inverse_mct_sample(rgb.0, rgb.1, rgb.2, u32::from(job.transform == 2))
    };
    (
        out0 + job.addend0,
        out1 + job.addend1,
        out2 + job.addend2,
        load_f32(plane3, src3) + job.addend3,
    )
}

#[inline(always)]
pub(crate) fn exact_rgba_destination(
    job: CudaJ2kStoreRgbaNativeJob,
    pixel: u32,
    channel: u32,
) -> u32 {
    if job.layout == 0 {
        pixel * 4 + channel
    } else {
        channel * job.output_width * job.output_height + pixel
    }
}

#[inline(always)]
pub(crate) fn store_rgba8_native_sample(
    plane0: *const f32,
    plane1: *const f32,
    plane2: *const f32,
    plane3: *const f32,
    output: *mut u8,
    job: CudaJ2kStoreRgbaNativeJob,
    gid: u32,
) {
    let (row, col) = pixel_coords(gid, job.copy_width);
    let pixel = output_pixel_index(job.output_width, job.output_x, job.output_y, row, col);
    let samples = exact_rgba_samples(plane0, plane1, plane2, plane3, job, row, col);
    store_u8(
        output,
        exact_rgba_destination(job, pixel, 0),
        sample_as_native_u8(samples.0, job.bit_depth0),
    );
    store_u8(
        output,
        exact_rgba_destination(job, pixel, 1),
        sample_as_native_u8(samples.1, job.bit_depth1),
    );
    store_u8(
        output,
        exact_rgba_destination(job, pixel, 2),
        sample_as_native_u8(samples.2, job.bit_depth2),
    );
    store_u8(
        output,
        exact_rgba_destination(job, pixel, 3),
        sample_as_native_u8(samples.3, job.bit_depth3),
    );
}

#[inline(always)]
pub(crate) fn store_rgba16_native_sample(
    plane0: *const f32,
    plane1: *const f32,
    plane2: *const f32,
    plane3: *const f32,
    output: *mut u16,
    job: CudaJ2kStoreRgbaNativeJob,
    gid: u32,
) {
    let (row, col) = pixel_coords(gid, job.copy_width);
    let pixel = output_pixel_index(job.output_width, job.output_x, job.output_y, row, col);
    let samples = exact_rgba_samples(plane0, plane1, plane2, plane3, job, row, col);
    store_u16(
        output,
        exact_rgba_destination(job, pixel, 0),
        sample_as_native_u16(samples.0, job.bit_depth0),
    );
    store_u16(
        output,
        exact_rgba_destination(job, pixel, 1),
        sample_as_native_u16(samples.1, job.bit_depth1),
    );
    store_u16(
        output,
        exact_rgba_destination(job, pixel, 2),
        sample_as_native_u16(samples.2, job.bit_depth2),
    );
    store_u16(
        output,
        exact_rgba_destination(job, pixel, 3),
        sample_as_native_u16(samples.3, job.bit_depth3),
    );
}

#[inline(always)]
pub(crate) fn store_rgbai16_native_sample(
    plane0: *const f32,
    plane1: *const f32,
    plane2: *const f32,
    plane3: *const f32,
    output: *mut i16,
    job: CudaJ2kStoreRgbaNativeJob,
    gid: u32,
) {
    let (row, col) = pixel_coords(gid, job.copy_width);
    let pixel = output_pixel_index(job.output_width, job.output_x, job.output_y, row, col);
    let samples = exact_rgba_samples(plane0, plane1, plane2, plane3, job, row, col);
    store_i16(
        output,
        exact_rgba_destination(job, pixel, 0),
        sample_as_native_i16(samples.0, job.bit_depth0),
    );
    store_i16(
        output,
        exact_rgba_destination(job, pixel, 1),
        sample_as_native_i16(samples.1, job.bit_depth1),
    );
    store_i16(
        output,
        exact_rgba_destination(job, pixel, 2),
        sample_as_native_i16(samples.2, job.bit_depth2),
    );
    store_i16(
        output,
        exact_rgba_destination(job, pixel, 3),
        sample_as_native_i16(samples.3, job.bit_depth3),
    );
}
