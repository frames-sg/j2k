// SPDX-License-Identifier: MIT OR Apache-2.0

use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use crate::{
        abi::{
            CudaJ2kInverseMctJob, CudaJ2kStoreGray8BatchJob, CudaJ2kStoreGray8Job,
            CudaJ2kStoreGray16BatchJob, CudaJ2kStoreGray16Job, CudaJ2kStoreGrayI16BatchJob,
            CudaJ2kStoreRgb8Job, CudaJ2kStoreRgb8MctBatchJob, CudaJ2kStoreRgb16Job,
            CudaJ2kStoreRgb16MctJob, CudaJ2kStoreRgbNativeBatchJob, CudaJ2kStoreRgbaNativeBatchJob,
        },
        color::{
            store_rgb8_mct_sample, store_rgb8_sample, store_rgb16_mct_sample, store_rgb16_sample,
        },
        layout::{output_pixel_index, pixel_coords, source_index},
        memory::{load_f32, load_job, store_f32, store_i16, store_u8, store_u16},
        sample::{
            sample_as_native_i16, sample_as_native_u8, sample_as_native_u16, sample_as_u8,
            sample_as_u16,
        },
        native_color::{
            store_rgb8_native_sample, store_rgb16_native_sample, store_rgba8_native_sample,
            store_rgba16_native_sample, store_rgbai16_native_sample, store_rgbi16_native_sample,
        },
        transform::inverse_mct_sample,
    };
    use cuda_device::{kernel, thread};

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
    pub unsafe fn j2k_store_grayi16_batch(jobs: *const CudaJ2kStoreGrayI16BatchJob) {
        let item = load_job(unsafe { jobs.add(thread::index_2d_row() as usize) });
        let input = item.input_ptr as usize as *const f32;
        let output = item.output_ptr as usize as *mut i16;
        let job = item.job;
        let pixels = job.copy_width * job.copy_height;
        let gid = thread::index_2d_col() as u32;
        if gid >= pixels {
            return;
        }

        let (row, col) = pixel_coords(gid, job.copy_width);
        let src = source_index(job.input_width, job.source_x, job.source_y, row, col);
        let dst = output_pixel_index(job.output_width, job.output_x, job.output_y, row, col);
        store_i16(
            output,
            dst,
            sample_as_native_i16(load_f32(input, src) + job.addend, job.bit_depth),
        );
    }

    #[kernel]
    pub unsafe fn j2k_store_gray8_batch(jobs: *const CudaJ2kStoreGray8BatchJob) {
        let item = load_job(unsafe { jobs.add(thread::index_2d_row() as usize) });
        let input = item.input_ptr as usize as *const f32;
        let output = item.output_ptr as usize as *mut u8;
        let job = item.job;
        let pixels = job.copy_width * job.copy_height;
        let gid = thread::index_2d_col() as u32;
        if gid >= pixels {
            return;
        }

        let (row, col) = pixel_coords(gid, job.copy_width);
        let src = source_index(job.input_width, job.source_x, job.source_y, row, col);
        let dst = output_pixel_index(job.output_width, job.output_x, job.output_y, row, col);
        store_u8(
            output,
            dst,
            sample_as_native_u8(load_f32(input, src) + job.addend, job.bit_depth),
        );
    }

    #[kernel]
    pub unsafe fn j2k_store_gray16_batch(jobs: *const CudaJ2kStoreGray16BatchJob) {
        let item = load_job(unsafe { jobs.add(thread::index_2d_row() as usize) });
        let input = item.input_ptr as usize as *const f32;
        let output = item.output_ptr as usize as *mut u16;
        let job = item.job;
        let pixels = job.copy_width * job.copy_height;
        let gid = thread::index_2d_col() as u32;
        if gid >= pixels {
            return;
        }

        let (row, col) = pixel_coords(gid, job.copy_width);
        let src = source_index(job.input_width, job.source_x, job.source_y, row, col);
        let dst = output_pixel_index(job.output_width, job.output_x, job.output_y, row, col);
        store_u16(
            output,
            dst,
            sample_as_native_u16(load_f32(input, src) + job.addend, job.bit_depth),
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

    #[kernel]
    pub unsafe fn j2k_store_rgb8_native_batch(jobs: *const CudaJ2kStoreRgbNativeBatchJob) {
        let item = load_job(unsafe { jobs.add(thread::index_2d_row() as usize) });
        let pixels = item.job.copy_width * item.job.copy_height;
        let gid = thread::index_2d_col() as u32;
        if gid >= pixels {
            return;
        }
        store_rgb8_native_sample(
            item.plane0_ptr as usize as *const f32,
            item.plane1_ptr as usize as *const f32,
            item.plane2_ptr as usize as *const f32,
            item.output_ptr as usize as *mut u8,
            item.job,
            gid,
        );
    }

    #[kernel]
    pub unsafe fn j2k_store_rgb16_native_batch(jobs: *const CudaJ2kStoreRgbNativeBatchJob) {
        let item = load_job(unsafe { jobs.add(thread::index_2d_row() as usize) });
        let pixels = item.job.copy_width * item.job.copy_height;
        let gid = thread::index_2d_col() as u32;
        if gid >= pixels {
            return;
        }
        store_rgb16_native_sample(
            item.plane0_ptr as usize as *const f32,
            item.plane1_ptr as usize as *const f32,
            item.plane2_ptr as usize as *const f32,
            item.output_ptr as usize as *mut u16,
            item.job,
            gid,
        );
    }

    #[kernel]
    pub unsafe fn j2k_store_rgbi16_native_batch(jobs: *const CudaJ2kStoreRgbNativeBatchJob) {
        let item = load_job(unsafe { jobs.add(thread::index_2d_row() as usize) });
        let pixels = item.job.copy_width * item.job.copy_height;
        let gid = thread::index_2d_col() as u32;
        if gid >= pixels {
            return;
        }
        store_rgbi16_native_sample(
            item.plane0_ptr as usize as *const f32,
            item.plane1_ptr as usize as *const f32,
            item.plane2_ptr as usize as *const f32,
            item.output_ptr as usize as *mut i16,
            item.job,
            gid,
        );
    }

    #[kernel]
    pub unsafe fn j2k_store_rgba8_native_batch(jobs: *const CudaJ2kStoreRgbaNativeBatchJob) {
        let item = load_job(unsafe { jobs.add(thread::index_2d_row() as usize) });
        let pixels = item.job.copy_width * item.job.copy_height;
        let gid = thread::index_2d_col() as u32;
        if gid >= pixels {
            return;
        }
        store_rgba8_native_sample(
            item.plane0_ptr as usize as *const f32,
            item.plane1_ptr as usize as *const f32,
            item.plane2_ptr as usize as *const f32,
            item.plane3_ptr as usize as *const f32,
            item.output_ptr as usize as *mut u8,
            item.job,
            gid,
        );
    }

    #[kernel]
    pub unsafe fn j2k_store_rgba16_native_batch(jobs: *const CudaJ2kStoreRgbaNativeBatchJob) {
        let item = load_job(unsafe { jobs.add(thread::index_2d_row() as usize) });
        let pixels = item.job.copy_width * item.job.copy_height;
        let gid = thread::index_2d_col() as u32;
        if gid >= pixels {
            return;
        }
        store_rgba16_native_sample(
            item.plane0_ptr as usize as *const f32,
            item.plane1_ptr as usize as *const f32,
            item.plane2_ptr as usize as *const f32,
            item.plane3_ptr as usize as *const f32,
            item.output_ptr as usize as *mut u16,
            item.job,
            gid,
        );
    }

    #[kernel]
    pub unsafe fn j2k_store_rgbai16_native_batch(jobs: *const CudaJ2kStoreRgbaNativeBatchJob) {
        let item = load_job(unsafe { jobs.add(thread::index_2d_row() as usize) });
        let pixels = item.job.copy_width * item.job.copy_height;
        let gid = thread::index_2d_col() as u32;
        if gid >= pixels {
            return;
        }
        store_rgbai16_native_sample(
            item.plane0_ptr as usize as *const f32,
            item.plane1_ptr as usize as *const f32,
            item.plane2_ptr as usize as *const f32,
            item.plane3_ptr as usize as *const f32,
            item.output_ptr as usize as *mut i16,
            item.job,
            gid,
        );
    }
}
