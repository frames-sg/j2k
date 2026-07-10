// SPDX-License-Identifier: MIT OR Apache-2.0

use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use crate::{
        abi::{
            J2kHtEncodeCompactJob, J2kHtPacketBlock, J2kHtPacketJob, J2kHtPacketStatus,
            J2kHtPacketSubband, J2kHtPacketSubbandTagState, J2kHtPacketTagNodeState,
        },
        constants::{
            J2K_ENCODE_STATUS_OK, J2K_FDWT97_INV_KAPPA, J2K_FDWT97_KAPPA,
            J2K_HT_COMPACT_ASSEMBLE_FLAG, J2K_HT_COMPACT_LENGTH_MASK, J2K_HT_MEL_OFFSET,
            J2K_HT_VLC_OFFSET, J2K_HT_VLC_SIZE,
        },
        dwt53::{j2k_fdwt53_predict_col, j2k_fdwt53_predict_row},
        dwt97::{
            j2k_fdwt97_high2_col, j2k_fdwt97_high2_row, j2k_fdwt97_low2_col, j2k_fdwt97_low2_row,
        },
        helpers::{
            floor_f32, load_f32, load_f32_u64, load_job, load_u8, load_u32, store_f32,
            store_f32_u64, store_i32, store_u8, store_u32,
        },
        packetization::{
            j2k_packet_build_header_serial, j2k_packet_copy_body_cooperative, j2k_packet_status,
        },
        quantization::j2k_quantize_sample,
    };
    use cuda_device::{SharedArray, kernel, thread};

    #[kernel]
    pub unsafe fn j2k_deinterleave_to_f32(
        pixels: *const u8,
        components: *mut f32,
        num_pixels: u64,
        num_components: u32,
        bit_depth: u32,
        is_signed: u32,
    ) {
        let idx = thread::index_1d().get() as u64;
        if idx >= num_pixels || num_components == 0 || num_components > 4 {
            return;
        }

        let bytes_per_sample = if bit_depth <= 8 { 1_u32 } else { 2_u32 };
        let unsigned_offset = if is_signed != 0 {
            0.0
        } else {
            (1_u32 << (bit_depth - 1)) as f32
        };
        let pixel_base = idx * num_components as u64 * bytes_per_sample as u64;
        let mut component = 0_u32;
        while component < num_components {
            let sample_base = pixel_base + component as u64 * bytes_per_sample as u64;
            let sample = if bit_depth <= 8 {
                let raw = load_u8(pixels, sample_base);
                if is_signed != 0 {
                    (raw as i8) as f32
                } else {
                    raw as f32 - unsigned_offset
                }
            } else {
                let raw = load_u8(pixels, sample_base) as u16
                    | ((load_u8(pixels, sample_base + 1) as u16) << 8);
                if is_signed != 0 {
                    (raw as i16) as f32
                } else {
                    raw as f32 - unsigned_offset
                }
            };
            store_f32_u64(components, component as u64 * num_pixels + idx, sample);
            component += 1;
        }
    }

    #[kernel]
    pub unsafe fn j2k_deinterleave_strided_to_f32(
        pixels: *const u8,
        components: *mut f32,
        width: u64,
        height: u64,
        byte_offset: u64,
        pitch_bytes: u64,
        num_components: u32,
        bit_depth: u32,
        is_signed: u32,
    ) {
        let idx = thread::index_1d().get() as u64;
        let num_pixels = width * height;
        if idx >= num_pixels || num_components == 0 || num_components > 4 {
            return;
        }

        let bytes_per_sample = if bit_depth <= 8 { 1_u32 } else { 2_u32 };
        let unsigned_offset = if is_signed != 0 {
            0.0
        } else {
            (1_u32 << (bit_depth - 1)) as f32
        };
        let y = idx / width;
        let x = idx - y * width;
        let pixel_base =
            byte_offset + y * pitch_bytes + x * num_components as u64 * bytes_per_sample as u64;
        let mut component = 0_u32;
        while component < num_components {
            let sample_base = pixel_base + component as u64 * bytes_per_sample as u64;
            let sample = if bit_depth <= 8 {
                let raw = load_u8(pixels, sample_base);
                if is_signed != 0 {
                    (raw as i8) as f32
                } else {
                    raw as f32 - unsigned_offset
                }
            } else {
                let raw = load_u8(pixels, sample_base) as u16
                    | ((load_u8(pixels, sample_base + 1) as u16) << 8);
                if is_signed != 0 {
                    (raw as i16) as f32
                } else {
                    raw as f32 - unsigned_offset
                }
            };
            store_f32_u64(components, component as u64 * num_pixels + idx, sample);
            component += 1;
        }
    }

    #[kernel]
    pub unsafe fn j2k_forward_rct(plane0: *mut f32, plane1: *mut f32, plane2: *mut f32, len: u64) {
        let idx = thread::index_1d().get() as u64;
        if idx >= len {
            return;
        }

        let r = load_f32_u64(plane0.cast_const(), idx);
        let g = load_f32_u64(plane1.cast_const(), idx);
        let b = load_f32_u64(plane2.cast_const(), idx);
        store_f32_u64(plane0, idx, floor_f32((r + 2.0 * g + b) * 0.25));
        store_f32_u64(plane1, idx, b - g);
        store_f32_u64(plane2, idx, r - g);
    }

    #[kernel]
    pub unsafe fn j2k_forward_ict(plane0: *mut f32, plane1: *mut f32, plane2: *mut f32, len: u64) {
        let idx = thread::index_1d().get() as u64;
        if idx >= len {
            return;
        }

        let r = load_f32_u64(plane0.cast_const(), idx);
        let g = load_f32_u64(plane1.cast_const(), idx);
        let b = load_f32_u64(plane2.cast_const(), idx);
        store_f32_u64(plane0, idx, 0.299 * r + 0.587 * g + 0.114 * b);
        store_f32_u64(plane1, idx, -0.16875 * r - 0.33126 * g + 0.5 * b);
        store_f32_u64(plane2, idx, 0.5 * r - 0.41869 * g - 0.08131 * b);
    }

    #[kernel]
    pub unsafe fn j2k_forward_dwt53_horizontal(
        src: *const f32,
        dst: *mut f32,
        full_width: u32,
        current_width: u32,
        current_height: u32,
        low_width: u32,
    ) {
        let x = thread::index_2d_col() as u32;
        let y = thread::index_2d_row() as u32;
        if x >= current_width || y >= current_height {
            return;
        }

        let row_base = y * full_width;
        if x < low_width {
            let even = x * 2;
            let left = if x > 0 {
                j2k_fdwt53_predict_row(src, row_base, current_width, x - 1)
            } else {
                j2k_fdwt53_predict_row(src, row_base, current_width, 0)
            };
            let right = if even + 1 < current_width {
                j2k_fdwt53_predict_row(src, row_base, current_width, x)
            } else {
                left
            };
            let value = load_f32(src, row_base + even) + floor_f32((left + right) * 0.25 + 0.5);
            store_f32(dst, row_base + x, value);
            return;
        }

        let value = j2k_fdwt53_predict_row(src, row_base, current_width, x - low_width);
        store_f32(dst, row_base + x, value);
    }

    #[kernel]
    pub unsafe fn j2k_forward_dwt53_vertical(
        src: *const f32,
        dst: *mut f32,
        full_width: u32,
        current_width: u32,
        current_height: u32,
        low_height: u32,
    ) {
        let x = thread::index_2d_col() as u32;
        let y = thread::index_2d_row() as u32;
        if x >= current_width || y >= current_height {
            return;
        }

        if y < low_height {
            let even = y * 2;
            let top = if y > 0 {
                j2k_fdwt53_predict_col(src, x, full_width, current_height, y - 1)
            } else {
                j2k_fdwt53_predict_col(src, x, full_width, current_height, 0)
            };
            let bottom = if even + 1 < current_height {
                j2k_fdwt53_predict_col(src, x, full_width, current_height, y)
            } else {
                top
            };
            let value =
                load_f32(src, even * full_width + x) + floor_f32((top + bottom) * 0.25 + 0.5);
            store_f32(dst, y * full_width + x, value);
            return;
        }

        let value = j2k_fdwt53_predict_col(src, x, full_width, current_height, y - low_height);
        store_f32(dst, y * full_width + x, value);
    }

    #[kernel]
    pub unsafe fn j2k_forward_dwt97_horizontal(
        src: *const f32,
        dst: *mut f32,
        full_width: u32,
        current_width: u32,
        current_height: u32,
        low_width: u32,
    ) {
        let x = thread::index_2d_col() as u32;
        let y = thread::index_2d_row() as u32;
        if x >= current_width || y >= current_height {
            return;
        }

        let row_base = y * full_width;
        let value = if x < low_width {
            j2k_fdwt97_low2_row(src, row_base, current_width, x) * J2K_FDWT97_INV_KAPPA
        } else {
            j2k_fdwt97_high2_row(src, row_base, current_width, x - low_width) * J2K_FDWT97_KAPPA
        };
        store_f32(dst, row_base + x, value);
    }

    #[kernel]
    pub unsafe fn j2k_forward_dwt97_vertical(
        src: *const f32,
        dst: *mut f32,
        full_width: u32,
        current_width: u32,
        current_height: u32,
        low_height: u32,
    ) {
        let x = thread::index_2d_col() as u32;
        let y = thread::index_2d_row() as u32;
        if x >= current_width || y >= current_height {
            return;
        }

        let value = if y < low_height {
            j2k_fdwt97_low2_col(src, x, full_width, current_height, y) * J2K_FDWT97_INV_KAPPA
        } else {
            j2k_fdwt97_high2_col(src, x, full_width, current_height, y - low_height)
                * J2K_FDWT97_KAPPA
        };
        store_f32(dst, y * full_width + x, value);
    }

    #[kernel]
    pub unsafe fn j2k_quantize_subband(
        samples: *const f32,
        coefficients: *mut i32,
        len: u64,
        step_exponent: u32,
        step_mantissa: u32,
        range_bits: u32,
        reversible: u32,
    ) {
        let idx = thread::index_1d().get() as u64;
        if idx >= len {
            return;
        }

        let coefficient = j2k_quantize_sample(
            load_f32_u64(samples, idx),
            step_exponent,
            step_mantissa,
            range_bits,
            reversible,
        );
        store_i32(coefficients, idx, coefficient);
    }

    #[kernel]
    pub unsafe fn j2k_quantize_subband_strided(
        samples: *const f32,
        coefficients: *mut i32,
        x0: u32,
        y0: u32,
        width: u32,
        height: u32,
        stride: u32,
        step_exponent: u32,
        step_mantissa: u32,
        range_bits: u32,
        reversible: u32,
    ) {
        let x = thread::index_2d_col() as u32;
        let y = thread::index_2d_row() as u32;
        if x >= width || y >= height {
            return;
        }

        let source_index = (y0 + y) as u64 * stride as u64 + (x0 + x) as u64;
        let output_index = y as u64 * width as u64 + x as u64;
        let coefficient = j2k_quantize_sample(
            load_f32_u64(samples, source_index),
            step_exponent,
            step_mantissa,
            range_bits,
            reversible,
        );
        store_i32(coefficients, output_index, coefficient);
    }

    #[kernel]
    pub unsafe fn j2k_htj2k_compact_codeblocks(
        scratch: *const u8,
        compact: *mut u8,
        jobs: *const J2kHtEncodeCompactJob,
        job_count: u64,
    ) {
        let job_idx = thread::blockIdx_x();
        if job_idx as u64 >= job_count {
            return;
        }

        let job = load_job(jobs, job_idx);
        let mut idx = thread::threadIdx_x();
        let step = thread::blockDim_x();
        if (job.reserved & J2K_HT_COMPACT_ASSEMBLE_FLAG) != 0 {
            let mel_len = job.reserved & J2K_HT_COMPACT_LENGTH_MASK;
            let vlc_len = (job.reserved >> 15) & J2K_HT_COMPACT_LENGTH_MASK;
            let locator_bytes = mel_len + vlc_len;
            if locator_bytes > job.data_len {
                return;
            }
            let ms_len = job.data_len - locator_bytes;
            let vlc_start = J2K_HT_VLC_SIZE - vlc_len;
            while idx < job.data_len {
                let mut value = if idx < ms_len {
                    load_u8(scratch, (job.source_offset + idx) as u64)
                } else if idx < ms_len + mel_len {
                    load_u8(
                        scratch,
                        (job.source_offset + J2K_HT_MEL_OFFSET + idx - ms_len) as u64,
                    )
                } else {
                    load_u8(
                        scratch,
                        (job.source_offset + J2K_HT_VLC_OFFSET + vlc_start + idx - ms_len - mel_len)
                            as u64,
                    )
                };
                if job.data_len >= 2 {
                    if idx == job.data_len - 1 {
                        value = (locator_bytes >> 4) as u8;
                    } else if idx == job.data_len - 2 {
                        value = ((u32::from(value) & 0xf0) | (locator_bytes & 0x0f)) as u8;
                    }
                }
                store_u8(compact, (job.compact_offset + idx) as u64, value);
                idx += step;
            }
            return;
        }

        while idx < job.data_len {
            store_u8(
                compact,
                (job.compact_offset + idx) as u64,
                load_u8(scratch, (job.source_offset + idx) as u64),
            );
            idx += step;
        }
    }

    #[kernel]
    pub unsafe fn j2k_htj2k_packetize_cleanup(
        payload: *const u8,
        payload_len: u64,
        packets: *const J2kHtPacketJob,
        subbands: *const J2kHtPacketSubband,
        blocks: *const J2kHtPacketBlock,
        tag_states: *const J2kHtPacketSubbandTagState,
        tag_nodes: *const J2kHtPacketTagNodeState,
        tag_state_count: u64,
        tag_node_count: u64,
        out: *mut u8,
        statuses: *mut J2kHtPacketStatus,
        packet_count: u64,
    ) {
        static mut HEADER_RESULT: SharedArray<u32, 3> = SharedArray::UNINIT;

        let packet_idx = thread::blockIdx_x() as u64;
        if packet_idx >= packet_count {
            return;
        }

        let packet = load_job(packets, packet_idx as u32);
        let status = unsafe { statuses.add(packet_idx as usize) };
        let packet_out = unsafe { out.add(packet.output_offset as usize) };
        let header_result = unsafe { HEADER_RESULT.as_mut_ptr() };

        if thread::threadIdx_x() == 0 {
            let result = j2k_packet_build_header_serial(
                payload_len,
                packet,
                subbands,
                blocks,
                tag_states,
                tag_nodes,
                tag_state_count,
                tag_node_count,
                packet_out,
            );
            store_u32(header_result, 0, result.code);
            store_u32(header_result, 1, result.header_len);
            store_u32(header_result, 2, result.body_len);
            j2k_packet_status(status, result.code, result.detail, result.output_len);
        }
        thread::sync_threads();

        let shared_code = load_u32(header_result.cast_const(), 0);
        let shared_body_len = load_u32(header_result.cast_const(), 2);
        if shared_code != J2K_ENCODE_STATUS_OK || shared_body_len == 0 {
            return;
        }
        j2k_packet_copy_body_cooperative(
            payload,
            packet,
            subbands,
            blocks,
            packet_out,
            load_u32(header_result.cast_const(), 1),
            shared_body_len,
        );
    }
}
