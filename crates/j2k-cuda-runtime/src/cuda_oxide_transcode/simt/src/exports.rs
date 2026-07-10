// SPDX-License-Identifier: MIT OR Apache-2.0

use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use crate::{
        abi::Dwt97ColumnLiftQuantizeCodeblocksParams,
        constants::{
            DWT97_ALPHA, DWT97_BETA, DWT97_DELTA, DWT97_GAMMA, DWT97_INV_KAPPA, DWT97_KAPPA,
            DWT97_ROW_LIFT_MAX_WIDTH, DWT97_ROW_LIFT_ROWS_PER_BLOCK, DWT97_ROW_LIFT_SHARED_SAMPLES,
        },
        dwt97::{forward_lift_97, idct8x8_sample, idct8x8_sample_i16, shared_row_index},
        helpers::{load_f32, load_i32, offset_f32_mut, offset_i32_mut, store_f32, store_i32},
        quantization::{dwt97_codeblock_major_offset, quantize_dwt97_deadzone},
        reversible53::{idct_islow_signed, reversible_lift_row, vertical_high, vertical_low},
    };
    use cuda_device::{SharedArray, kernel, thread};

    #[kernel]
    pub unsafe fn transcode_reversible53_idct(
        blocks: *const i16,
        samples: *mut i32,
        block_count: u32,
    ) {
        let idx = thread::index_1d().get() as u32;
        if idx >= block_count {
            return;
        }
        idct_islow_signed(unsafe { blocks.add(idx as usize * 64) }, unsafe {
            samples.add(idx as usize * 64)
        });
    }

    #[kernel]
    pub unsafe fn transcode_reversible53_vertical_low(
        samples: *const i32,
        block_cols: i32,
        width: i32,
        height: i32,
        v_low: *mut i32,
        low_height: i32,
    ) {
        let x = thread::index_2d_col() as i32;
        let yl = thread::index_2d_row() as i32;
        if x >= width || yl >= low_height {
            return;
        }
        store_i32(
            v_low,
            (yl * width + x) as u64,
            vertical_low(samples, block_cols, height, x, yl),
        );
    }

    #[kernel]
    pub unsafe fn transcode_reversible53_vertical_high(
        samples: *const i32,
        block_cols: i32,
        width: i32,
        height: i32,
        v_high: *mut i32,
        high_height: i32,
    ) {
        let x = thread::index_2d_col() as i32;
        let yh = thread::index_2d_row() as i32;
        if x >= width || yh >= high_height {
            return;
        }
        store_i32(
            v_high,
            (yh * width + x) as u64,
            vertical_high(samples, block_cols, height, x, yh),
        );
    }

    #[kernel]
    pub unsafe fn transcode_reversible53_horizontal_low(
        v_low: *mut i32,
        width: i32,
        low_height: i32,
        low_width: i32,
        high_width: i32,
        ll: *mut i32,
        hl: *mut i32,
    ) {
        let yl = thread::index_1d().get() as i32;
        if yl >= low_height {
            return;
        }
        let row = offset_i32_mut(v_low, (yl * width) as u64);
        reversible_lift_row(row, width);
        let mut i = 0_i32;
        while i < low_width {
            store_i32(
                ll,
                (yl * low_width + i) as u64,
                load_i32(row.cast_const(), (i * 2) as u64),
            );
            i += 1;
        }
        let mut i = 0_i32;
        while i < high_width {
            store_i32(
                hl,
                (yl * high_width + i) as u64,
                load_i32(row.cast_const(), (i * 2 + 1) as u64),
            );
            i += 1;
        }
    }

    #[kernel]
    pub unsafe fn transcode_reversible53_horizontal_high(
        v_high: *mut i32,
        width: i32,
        high_height: i32,
        low_width: i32,
        high_width: i32,
        lh: *mut i32,
        hh: *mut i32,
    ) {
        let yh = thread::index_1d().get() as i32;
        if yh >= high_height {
            return;
        }
        let row = offset_i32_mut(v_high, (yh * width) as u64);
        reversible_lift_row(row, width);
        let mut i = 0_i32;
        while i < low_width {
            store_i32(
                lh,
                (yh * low_width + i) as u64,
                load_i32(row.cast_const(), (i * 2) as u64),
            );
            i += 1;
        }
        let mut i = 0_i32;
        while i < high_width {
            store_i32(
                hh,
                (yh * high_width + i) as u64,
                load_i32(row.cast_const(), (i * 2 + 1) as u64),
            );
            i += 1;
        }
    }

    #[kernel]
    pub unsafe fn transcode_dwt97_idct(
        blocks: *const f32,
        block_cols: i32,
        width: i32,
        height: i32,
        spatial: *mut f32,
    ) {
        let x = thread::index_2d_col() as i32;
        let y = thread::index_2d_row() as i32;
        if x >= width || y >= height {
            return;
        }
        let block_idx = (y >> 3) * block_cols + (x >> 3);
        let block = unsafe { blocks.add(block_idx as usize * 64) };
        store_f32(
            spatial,
            (y * width + x) as u64,
            idct8x8_sample(block, x & 7, y & 7),
        );
    }

    #[kernel]
    pub unsafe fn transcode_dwt97_row_lift(
        spatial: *mut f32,
        width: i32,
        height: i32,
        low_width: i32,
        high_width: i32,
        row_low: *mut f32,
        row_high: *mut f32,
    ) {
        let y = thread::index_1d().get() as i32;
        if y >= height {
            return;
        }
        let row = offset_f32_mut(spatial, (y * width) as u64);
        forward_lift_97(row, width, 1);
        let mut i = 0_i32;
        while i < low_width {
            store_f32(
                row_low,
                (y * low_width + i) as u64,
                load_f32(row.cast_const(), (i * 2) as u64),
            );
            i += 1;
        }
        let mut i = 0_i32;
        while i < high_width {
            store_f32(
                row_high,
                (y * high_width + i) as u64,
                load_f32(row.cast_const(), (i * 2 + 1) as u64),
            );
            i += 1;
        }
    }

    #[kernel]
    pub unsafe fn transcode_dwt97_column_lift(
        rows: *mut f32,
        band_width: i32,
        height: i32,
        low_out: *mut f32,
        high_out: *mut f32,
    ) {
        let x = thread::index_1d().get() as i32;
        if x >= band_width {
            return;
        }
        forward_lift_97(offset_f32_mut(rows, x as u64), height, band_width);
        let mut i = 0_i32;
        while i < height {
            let value = load_f32(rows.cast_const(), (i * band_width + x) as u64);
            if i & 1 == 0 {
                store_f32(low_out, ((i / 2) * band_width + x) as u64, value);
            } else {
                store_f32(high_out, ((i / 2) * band_width + x) as u64, value);
            }
            i += 1;
        }
    }

    #[kernel]
    pub unsafe fn transcode_dwt97_idct_batch(
        blocks: *const f32,
        block_cols: i32,
        width: i32,
        height: i32,
        blocks_per_item: i32,
        spatial: *mut f32,
    ) {
        let x = thread::index_2d_col() as i32;
        let y = thread::index_2d_row() as i32;
        let item = thread::blockIdx_z() as u64;
        if x >= width || y >= height {
            return;
        }
        let item_blocks = unsafe { blocks.add((item * blocks_per_item as u64 * 64) as usize) };
        let block_idx = (y >> 3) * block_cols + (x >> 3);
        let block = unsafe { item_blocks.add(block_idx as usize * 64) };
        store_f32(
            spatial,
            (item * height as u64 + y as u64) * width as u64 + x as u64,
            idct8x8_sample(block, x & 7, y & 7),
        );
    }

    #[kernel]
    pub unsafe fn transcode_dwt97_idct_i16_batch(
        blocks: *const i16,
        block_cols: i32,
        width: i32,
        height: i32,
        blocks_per_item: i32,
        spatial: *mut f32,
    ) {
        let x = thread::index_2d_col() as i32;
        let y = thread::index_2d_row() as i32;
        let item = thread::blockIdx_z() as u64;
        if x >= width || y >= height {
            return;
        }
        let item_blocks = unsafe { blocks.add((item * blocks_per_item as u64 * 64) as usize) };
        let block_idx = (y >> 3) * block_cols + (x >> 3);
        let block = unsafe { item_blocks.add(block_idx as usize * 64) };
        store_f32(
            spatial,
            (item * height as u64 + y as u64) * width as u64 + x as u64,
            idct8x8_sample_i16(block, x & 7, y & 7),
        );
    }

    #[kernel]
    pub unsafe fn transcode_dwt97_row_lift_batch(
        spatial: *mut f32,
        width: i32,
        height: i32,
        low_width: i32,
        high_width: i32,
        row_low: *mut f32,
        row_high: *mut f32,
    ) {
        let y = thread::blockIdx_x() as i32 * thread::blockDim_x() as i32
            + thread::threadIdx_x() as i32;
        let item = thread::blockIdx_y() as u64;
        if y >= height {
            return;
        }
        let item_spatial = offset_f32_mut(spatial, item * width as u64 * height as u64);
        let item_row_low = offset_f32_mut(row_low, item * height as u64 * low_width as u64);
        let item_row_high = offset_f32_mut(row_high, item * height as u64 * high_width as u64);
        let row = offset_f32_mut(item_spatial, y as u64 * width as u64);
        forward_lift_97(row, width, 1);
        let mut i = 0_i32;
        while i < low_width {
            store_f32(
                item_row_low,
                (y * low_width + i) as u64,
                load_f32(row.cast_const(), (i * 2) as u64),
            );
            i += 1;
        }
        let mut i = 0_i32;
        while i < high_width {
            store_f32(
                item_row_high,
                (y * high_width + i) as u64,
                load_f32(row.cast_const(), (i * 2 + 1) as u64),
            );
            i += 1;
        }
    }

    #[expect(
        static_mut_refs,
        reason = "CUDA block-shared storage is exposed by SharedArray through this kernel only"
    )]
    #[kernel]
    pub unsafe fn transcode_dwt97_row_lift_batch_coop(
        spatial: *const f32,
        width: i32,
        height: i32,
        low_width: i32,
        high_width: i32,
        row_low: *mut f32,
        row_high: *mut f32,
    ) {
        static mut ROWS: SharedArray<f32, DWT97_ROW_LIFT_SHARED_SAMPLES> = SharedArray::UNINIT;

        let rows = unsafe { ROWS.as_mut_ptr() };
        let row_lane = thread::threadIdx_y() as i32;
        let tid = thread::threadIdx_x() as i32;
        let block_step = thread::blockDim_x() as i32;
        let y = thread::blockIdx_x() as i32 * DWT97_ROW_LIFT_ROWS_PER_BLOCK as i32 + row_lane;
        let item = thread::blockIdx_y() as u64;
        let valid = y < height && width <= DWT97_ROW_LIFT_MAX_WIDTH as i32;

        if valid {
            let item_spatial =
                unsafe { spatial.add((item * width as u64 * height as u64) as usize) };
            let source = unsafe { item_spatial.add((y as u64 * width as u64) as usize) };
            let mut i = tid;
            while i < width {
                store_f32(
                    rows,
                    shared_row_index(row_lane, i),
                    load_f32(source, i as u64),
                );
                i += block_step;
            }
        }
        thread::sync_threads();

        if width >= 2 && width <= DWT97_ROW_LIFT_MAX_WIDTH as i32 {
            if valid {
                let last_even = if width % 2 == 0 { width - 2 } else { width - 1 };
                let mut i = tid * 2 + 1;
                while i < width {
                    let left = load_f32(rows.cast_const(), shared_row_index(row_lane, i - 1));
                    let right = if i + 1 < width {
                        load_f32(rows.cast_const(), shared_row_index(row_lane, i + 1))
                    } else {
                        load_f32(rows.cast_const(), shared_row_index(row_lane, last_even))
                    };
                    let value = load_f32(rows.cast_const(), shared_row_index(row_lane, i))
                        + DWT97_ALPHA * (left + right);
                    store_f32(rows, shared_row_index(row_lane, i), value);
                    i += block_step * 2;
                }
            }
            thread::sync_threads();

            if valid {
                let mut i = tid * 2;
                while i < width {
                    let left = if i > 0 {
                        load_f32(rows.cast_const(), shared_row_index(row_lane, i - 1))
                    } else {
                        load_f32(rows.cast_const(), shared_row_index(row_lane, 1))
                    };
                    let right = if i + 1 < width {
                        load_f32(rows.cast_const(), shared_row_index(row_lane, i + 1))
                    } else {
                        left
                    };
                    let value = load_f32(rows.cast_const(), shared_row_index(row_lane, i))
                        + DWT97_BETA * (left + right);
                    store_f32(rows, shared_row_index(row_lane, i), value);
                    i += block_step * 2;
                }
            }
            thread::sync_threads();

            if valid {
                let last_even = if width % 2 == 0 { width - 2 } else { width - 1 };
                let mut i = tid * 2 + 1;
                while i < width {
                    let left = load_f32(rows.cast_const(), shared_row_index(row_lane, i - 1));
                    let right = if i + 1 < width {
                        load_f32(rows.cast_const(), shared_row_index(row_lane, i + 1))
                    } else {
                        load_f32(rows.cast_const(), shared_row_index(row_lane, last_even))
                    };
                    let value = load_f32(rows.cast_const(), shared_row_index(row_lane, i))
                        + DWT97_GAMMA * (left + right);
                    store_f32(rows, shared_row_index(row_lane, i), value);
                    i += block_step * 2;
                }
            }
            thread::sync_threads();

            if valid {
                let mut i = tid * 2;
                while i < width {
                    let left = if i > 0 {
                        load_f32(rows.cast_const(), shared_row_index(row_lane, i - 1))
                    } else {
                        load_f32(rows.cast_const(), shared_row_index(row_lane, 1))
                    };
                    let right = if i + 1 < width {
                        load_f32(rows.cast_const(), shared_row_index(row_lane, i + 1))
                    } else {
                        left
                    };
                    let value = load_f32(rows.cast_const(), shared_row_index(row_lane, i))
                        + DWT97_DELTA * (left + right);
                    store_f32(rows, shared_row_index(row_lane, i), value);
                    i += block_step * 2;
                }
            }
            thread::sync_threads();

            if valid {
                let mut i = tid * 2;
                while i < width {
                    let value = load_f32(rows.cast_const(), shared_row_index(row_lane, i))
                        * DWT97_INV_KAPPA;
                    store_f32(rows, shared_row_index(row_lane, i), value);
                    i += block_step * 2;
                }
                let mut i = tid * 2 + 1;
                while i < width {
                    let value =
                        load_f32(rows.cast_const(), shared_row_index(row_lane, i)) * DWT97_KAPPA;
                    store_f32(rows, shared_row_index(row_lane, i), value);
                    i += block_step * 2;
                }
            }
            thread::sync_threads();
        }

        if valid {
            let item_row_low = offset_f32_mut(row_low, item * height as u64 * low_width as u64);
            let item_row_high = offset_f32_mut(row_high, item * height as u64 * high_width as u64);
            let mut i = tid;
            while i < low_width {
                store_f32(
                    item_row_low,
                    (y * low_width + i) as u64,
                    load_f32(rows.cast_const(), shared_row_index(row_lane, i * 2)),
                );
                i += block_step;
            }
            let mut i = tid;
            while i < high_width {
                store_f32(
                    item_row_high,
                    (y * high_width + i) as u64,
                    load_f32(rows.cast_const(), shared_row_index(row_lane, i * 2 + 1)),
                );
                i += block_step;
            }
        }
    }

    #[kernel]
    pub unsafe fn transcode_dwt97_column_lift_batch(
        rows: *mut f32,
        band_width: i32,
        height: i32,
        low_height: i32,
        high_height: i32,
        low_out: *mut f32,
        high_out: *mut f32,
    ) {
        let x = thread::blockIdx_x() as i32 * thread::blockDim_x() as i32
            + thread::threadIdx_x() as i32;
        let item = thread::blockIdx_y() as u64;
        if x >= band_width {
            return;
        }
        let item_rows = offset_f32_mut(rows, item * height as u64 * band_width as u64);
        let item_low = offset_f32_mut(low_out, item * low_height as u64 * band_width as u64);
        let item_high = offset_f32_mut(high_out, item * high_height as u64 * band_width as u64);
        forward_lift_97(offset_f32_mut(item_rows, x as u64), height, band_width);
        let mut i = 0_i32;
        while i < height {
            let value = load_f32(item_rows.cast_const(), (i * band_width + x) as u64);
            if i & 1 == 0 {
                store_f32(item_low, ((i / 2) * band_width + x) as u64, value);
            } else {
                store_f32(item_high, ((i / 2) * band_width + x) as u64, value);
            }
            i += 1;
        }
    }

    #[kernel]
    pub unsafe fn transcode_dwt97_quantize_codeblocks(
        band: *const f32,
        output: *mut i32,
        width: i32,
        height: i32,
        cb_width: i32,
        cb_height: i32,
        inv_delta: f32,
    ) {
        let x = thread::index_2d_col() as i32;
        let y = thread::index_2d_row() as i32;
        let item = thread::blockIdx_z() as u64;
        if x >= width || y >= height {
            return;
        }
        let item_stride = width as u64 * height as u64;
        let value = load_f32(
            band,
            item * item_stride + y as u64 * width as u64 + x as u64,
        );
        let offset = dwt97_codeblock_major_offset(x, y, width, height, cb_width, cb_height);
        store_i32(
            output,
            item * item_stride + offset,
            quantize_dwt97_deadzone(value, inv_delta),
        );
    }

    #[kernel]
    pub unsafe fn transcode_dwt97_column_lift_quantize_codeblocks_batch(
        rows: *mut f32,
        band_width: i32,
        height: i32,
        low_height: i32,
        high_height: i32,
        low_out: *mut i32,
        high_out: *mut i32,
        params: Dwt97ColumnLiftQuantizeCodeblocksParams,
    ) {
        let x = thread::blockIdx_x() as i32 * thread::blockDim_x() as i32
            + thread::threadIdx_x() as i32;
        let item = thread::blockIdx_y() as u64;
        if x >= band_width {
            return;
        }
        let item_rows = offset_f32_mut(rows, item * height as u64 * band_width as u64);
        let item_low = offset_i32_mut(low_out, item * low_height as u64 * band_width as u64);
        let item_high = offset_i32_mut(high_out, item * high_height as u64 * band_width as u64);

        forward_lift_97(offset_f32_mut(item_rows, x as u64), height, band_width);
        let mut i = 0_i32;
        while i < height {
            let value = load_f32(item_rows.cast_const(), (i * band_width + x) as u64);
            if i & 1 == 0 {
                let y = i / 2;
                let offset = dwt97_codeblock_major_offset(
                    x,
                    y,
                    band_width,
                    low_height,
                    params.cb_width,
                    params.cb_height,
                );
                store_i32(
                    item_low,
                    offset,
                    quantize_dwt97_deadzone(value, params.inv_delta_low),
                );
            } else {
                let y = i / 2;
                let offset = dwt97_codeblock_major_offset(
                    x,
                    y,
                    band_width,
                    high_height,
                    params.cb_width,
                    params.cb_height,
                );
                store_i32(
                    item_high,
                    offset,
                    quantize_dwt97_deadzone(value, params.inv_delta_high),
                );
            }
            i += 1;
        }
    }
}
