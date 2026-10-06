#![allow(
    static_mut_refs,
    reason = "CUDA shared-memory statics are accessed through device-scoped references"
)]

use cuda_device::{SharedArray, kernel, ptx_asm, thread};
use cuda_host::cuda_module;

include!("../../../cuda_oxide_simt_prelude.rs");

const IDWT_COOP_SAMPLES: usize = 512;
const IDWT_VERTICAL_COLUMNS: u32 = j2k_codec_math::dwt::IDWT_VERTICAL_STRIP_COLUMNS;
const IDWT_VERTICAL_SAMPLES: usize = IDWT_COOP_SAMPLES * IDWT_VERTICAL_COLUMNS as usize;
const IDWT_HORIZONTAL_TILE: u32 = j2k_codec_math::dwt::IDWT_HORIZONTAL_TILE_COLUMNS;
const IDWT_HORIZONTAL_HALO: u32 = j2k_codec_math::dwt::IDWT_HORIZONTAL_TILE_HALO;
const IDWT_HORIZONTAL_ROWS: u32 = j2k_codec_math::dwt::IDWT_HORIZONTAL_TILE_ROWS;
const IDWT_HORIZONTAL_ROW_SAMPLES: usize =
    j2k_codec_math::dwt::IDWT_HORIZONTAL_TILE_THREADS as usize;
const IDWT_HORIZONTAL_SAMPLES: usize = IDWT_HORIZONTAL_ROW_SAMPLES * IDWT_HORIZONTAL_ROWS as usize;
const IDWT_COLS4_COLUMNS: u32 = 4;
const IDWT_COLS4_SAMPLES: usize = 256 * IDWT_COLS4_COLUMNS as usize;
const IDWT_NEG_ALPHA: f32 = j2k_codec_math::dwt::IDWT97_NEG_ALPHA_F32;
const IDWT_NEG_BETA: f32 = j2k_codec_math::dwt::IDWT97_NEG_BETA_F32;
const IDWT_NEG_GAMMA: f32 = j2k_codec_math::dwt::IDWT97_NEG_GAMMA_F32;
const IDWT_NEG_DELTA: f32 = j2k_codec_math::dwt::IDWT97_NEG_DELTA_F32;
const IDWT_KAPPA: f32 = j2k_codec_math::dwt::DWT97_KAPPA_F32;
const IDWT_STANDARD_HIGH_PASS: f32 = j2k_codec_math::dwt::DWT97_INV_KAPPA_F32;
const IDWT_CODESTREAM_97_MODE: u32 = 2;
const IDWT_CODESTREAM_HIGH_PASS: f32 = j2k_codec_math::dwt::IDWT97_OPENJPEG_TWO_INV_KAPPA_F32 * 0.5;

#[repr(C)]
#[derive(Clone, Copy)]
struct CudaJ2kRect {
    x0: u32,
    y0: u32,
    x1: u32,
    y1: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CudaJ2kIdwtJob {
    rect: CudaJ2kRect,
    ll_rect: CudaJ2kRect,
    hl_rect: CudaJ2kRect,
    lh_rect: CudaJ2kRect,
    hh_rect: CudaJ2kRect,
    irreversible97: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CudaJ2kIdwtMultiJob {
    ll_ptr: u64,
    hl_ptr: u64,
    lh_ptr: u64,
    hh_ptr: u64,
    output_ptr: u64,
    job: CudaJ2kIdwtJob,
    reserved_tail: u32,
}

#[derive(Clone, Copy)]
struct IdwtPointers {
    ll: *const f32,
    hl: *const f32,
    lh: *const f32,
    hh: *const f32,
    output: *mut f32,
}

#[derive(Clone, Copy)]
struct SharedLine {
    samples: *mut f32,
    lane: u32,
    stride: u32,
    active: bool,
}

impl SharedLine {
    #[inline(always)]
    fn offset(self, index: u32) -> u32 {
        index * self.stride + self.lane
    }

    #[inline(always)]
    fn load(self, index: u32) -> f32 {
        load_f32(self.samples.cast_const(), self.offset(index))
    }

    #[inline(always)]
    fn store(self, index: u32, value: f32) {
        store_f32(self.samples, self.offset(index), value);
    }
}

#[inline(always)]
fn load_f32(ptr: *const f32, index: u32) -> f32 {
    simt_load(ptr, index as usize)
}

#[inline(always)]
fn store_f32(ptr: *mut f32, index: u32, value: f32) {
    simt_store(ptr, index as usize, value);
}

#[inline(always)]
fn load_job<T: Copy>(ptr: *const T, index: u32) -> T {
    simt_load(ptr, index as usize)
}

#[inline(always)]
fn floor_f32(value: f32) -> f32 {
    let truncated = value as i32 as f32;
    if truncated > value {
        truncated - 1.0
    } else {
        truncated
    }
}

#[inline(always)]
fn fused_mul_add_f32(multiplicand: f32, multiplier: f32, addend: f32) -> f32 {
    let output: f32;
    // SAFETY: this is a pure register-only IEEE binary32 FMA with no memory,
    // control-flow, or lane-participation contract.
    unsafe {
        ptx_asm!(
            "fma.rn.f32 %0, %1, %2, %3;",
            out("=f") output,
            in("f") multiplicand,
            in("f") multiplier,
            in("f") addend,
            options(register_only),
        );
    }
    output
}

#[inline(always)]
fn rect_width(rect: CudaJ2kRect) -> u32 {
    rect.x1 - rect.x0
}

#[inline(always)]
fn rect_height(rect: CudaJ2kRect) -> u32 {
    rect.y1 - rect.y0
}

#[inline(always)]
fn div_ceil_2(value: u32) -> u32 {
    (value + 1) >> 1
}

#[inline(always)]
fn idwt_band_coord(output_origin: u32, output_coord: u32, band_origin: u32, low: bool) -> u32 {
    let index = if low {
        div_ceil_2(output_coord) - div_ceil_2(output_origin)
    } else {
        (output_coord >> 1) - (output_origin >> 1)
    };
    band_origin + index
}

#[inline(always)]
fn source_get(source: *const f32, rect: CudaJ2kRect, x: u32, y: u32) -> f32 {
    if x < rect.x0 || x >= rect.x1 || y < rect.y0 || y >= rect.y1 {
        return 0.0;
    }
    let local_x = x - rect.x0;
    let local_y = y - rect.y0;
    load_f32(source, local_y * rect_width(rect) + local_x)
}

#[inline(always)]
fn pse_left(idx: u32, offset: u32) -> u32 {
    idx.abs_diff(offset)
}

#[inline(always)]
fn pse_right(idx: u32, offset: u32, length: u32) -> u32 {
    let new_idx = idx + offset;
    if new_idx >= length {
        let overshoot = new_idx - length;
        length - 2 - overshoot
    } else {
        new_idx
    }
}

#[inline(always)]
fn lift_53_sample(sample: f32, left: f32, right: f32, update_even: bool) -> f32 {
    if update_even {
        sample - floor_f32((left + right) * 0.25 + 0.5)
    } else {
        sample + floor_f32((left + right) * 0.5)
    }
}

#[inline(always)]
fn filter_step_horizontal_53(scanline: *mut f32, width: u32, first: u32, update_even: bool) {
    if first == 0 {
        let left = pse_left(0, 1);
        let right = pse_right(0, 1, width);
        let sample = load_f32(scanline.cast_const(), 0);
        let left_sample = load_f32(scanline.cast_const(), left);
        let right_sample = load_f32(scanline.cast_const(), right);
        store_f32(
            scanline,
            0,
            lift_53_sample(sample, left_sample, right_sample, update_even),
        );
    }

    let mut i = if first == 0 { 2 } else { 1 };
    while i + 1 < width {
        let sample = load_f32(scanline.cast_const(), i);
        let left = load_f32(scanline.cast_const(), i - 1);
        let right = load_f32(scanline.cast_const(), i + 1);
        store_f32(
            scanline,
            i,
            lift_53_sample(sample, left, right, update_even),
        );
        i += 2;
    }

    if width > 1 && ((width - 1) & 1) == first {
        let last = width - 1;
        let left = pse_left(last, 1);
        let right = pse_right(last, 1, width);
        let sample = load_f32(scanline.cast_const(), last);
        let left_sample = load_f32(scanline.cast_const(), left);
        let right_sample = load_f32(scanline.cast_const(), right);
        store_f32(
            scanline,
            last,
            lift_53_sample(sample, left_sample, right_sample, update_even),
        );
    }
}

#[inline(always)]
fn filter_step_horizontal_97(scanline: *mut f32, width: u32, first: u32, coefficient: f32) {
    if first == 0 {
        let left = pse_left(0, 1);
        let right = pse_right(0, 1, width);
        let sample = load_f32(scanline.cast_const(), 0);
        let left_sample = load_f32(scanline.cast_const(), left);
        let right_sample = load_f32(scanline.cast_const(), right);
        store_f32(
            scanline,
            0,
            fused_mul_add_f32(left_sample + right_sample, coefficient, sample),
        );
    }

    let mut i = if first == 0 { 2 } else { 1 };
    while i + 1 < width {
        let sample = load_f32(scanline.cast_const(), i);
        let left = load_f32(scanline.cast_const(), i - 1);
        let right = load_f32(scanline.cast_const(), i + 1);
        store_f32(
            scanline,
            i,
            fused_mul_add_f32(left + right, coefficient, sample),
        );
        i += 2;
    }

    if width > 1 && ((width - 1) & 1) == first {
        let last = width - 1;
        let left = pse_left(last, 1);
        let right = pse_right(last, 1, width);
        let sample = load_f32(scanline.cast_const(), last);
        let left_sample = load_f32(scanline.cast_const(), left);
        let right_sample = load_f32(scanline.cast_const(), right);
        store_f32(
            scanline,
            last,
            fused_mul_add_f32(left_sample + right_sample, coefficient, sample),
        );
    }
}

#[inline(always)]
fn idwt_high_pass(transform_mode: u32) -> f32 {
    if transform_mode == IDWT_CODESTREAM_97_MODE {
        IDWT_CODESTREAM_HIGH_PASS
    } else {
        IDWT_STANDARD_HIGH_PASS
    }
}

#[inline(always)]
fn filter_horizontal_scanline(scanline: *mut f32, width: u32, rect_x0: u32, transform_mode: u32) {
    if width == 1 {
        if (rect_x0 & 1) != 0 {
            store_f32(scanline, 0, load_f32(scanline.cast_const(), 0) * 0.5);
        }
        return;
    }

    let first_even = rect_x0 & 1;
    let first_odd = 1 - first_even;
    if transform_mode == 0 {
        filter_step_horizontal_53(scanline, width, first_even, true);
        filter_step_horizontal_53(scanline, width, first_odd, false);
    } else {
        let k0 = if first_even == 0 {
            IDWT_KAPPA
        } else {
            idwt_high_pass(transform_mode)
        };
        let k1 = if first_even == 0 {
            idwt_high_pass(transform_mode)
        } else {
            IDWT_KAPPA
        };
        let mut i = 0;
        while i + 1 < width {
            store_f32(scanline, i, load_f32(scanline.cast_const(), i) * k0);
            store_f32(scanline, i + 1, load_f32(scanline.cast_const(), i + 1) * k1);
            i += 2;
        }
        if (width & 1) != 0 {
            let last = width - 1;
            store_f32(scanline, last, load_f32(scanline.cast_const(), last) * k0);
        }
        filter_step_horizontal_97(scanline, width, first_even, IDWT_NEG_DELTA);
        filter_step_horizontal_97(scanline, width, first_odd, IDWT_NEG_GAMMA);
        filter_step_horizontal_97(scanline, width, first_even, IDWT_NEG_BETA);
        filter_step_horizontal_97(scanline, width, first_odd, IDWT_NEG_ALPHA);
    }
}

#[inline(always)]
fn filter_step_vertical_53_column(
    output: *mut f32,
    width: u32,
    height: u32,
    col: u32,
    first: u32,
    update_even: bool,
) {
    let mut row = first;
    while row < height {
        let row_above = pse_left(row, 1);
        let row_below = pse_right(row, 1, height);
        let idx = row * width + col;
        let sample = load_f32(output.cast_const(), idx);
        let above = load_f32(output.cast_const(), row_above * width + col);
        let below = load_f32(output.cast_const(), row_below * width + col);
        store_f32(
            output,
            idx,
            lift_53_sample(sample, above, below, update_even),
        );
        row += 2;
    }
}

#[inline(always)]
fn filter_step_vertical_97_column(
    output: *mut f32,
    width: u32,
    height: u32,
    col: u32,
    first: u32,
    coefficient: f32,
) {
    let mut row = first;
    while row < height {
        let row_above = pse_left(row, 1);
        let row_below = pse_right(row, 1, height);
        let idx = row * width + col;
        let sample = load_f32(output.cast_const(), idx);
        let above = load_f32(output.cast_const(), row_above * width + col);
        let below = load_f32(output.cast_const(), row_below * width + col);
        store_f32(
            output,
            idx,
            fused_mul_add_f32(above + below, coefficient, sample),
        );
        row += 2;
    }
}

#[inline(always)]
fn filter_vertical_column(
    output: *mut f32,
    width: u32,
    height: u32,
    rect_y0: u32,
    col: u32,
    transform_mode: u32,
) {
    if height == 1 {
        if (rect_y0 & 1) != 0 {
            store_f32(output, col, load_f32(output.cast_const(), col) * 0.5);
        }
        return;
    }

    let first_even = rect_y0 & 1;
    let first_odd = 1 - first_even;
    if transform_mode == 0 {
        filter_step_vertical_53_column(output, width, height, col, first_even, true);
        filter_step_vertical_53_column(output, width, height, col, first_odd, false);
    } else {
        let k0 = if first_even == 0 {
            IDWT_KAPPA
        } else {
            idwt_high_pass(transform_mode)
        };
        let k1 = if first_even == 0 {
            idwt_high_pass(transform_mode)
        } else {
            IDWT_KAPPA
        };
        let mut row = 0;
        while row + 1 < height {
            let idx0 = row * width + col;
            let idx1 = (row + 1) * width + col;
            store_f32(output, idx0, load_f32(output.cast_const(), idx0) * k0);
            store_f32(output, idx1, load_f32(output.cast_const(), idx1) * k1);
            row += 2;
        }
        if (height & 1) != 0 {
            let idx = (height - 1) * width + col;
            store_f32(output, idx, load_f32(output.cast_const(), idx) * k0);
        }
        filter_step_vertical_97_column(output, width, height, col, first_even, IDWT_NEG_DELTA);
        filter_step_vertical_97_column(output, width, height, col, first_odd, IDWT_NEG_GAMMA);
        filter_step_vertical_97_column(output, width, height, col, first_even, IDWT_NEG_BETA);
        filter_step_vertical_97_column(output, width, height, col, first_odd, IDWT_NEG_ALPHA);
    }
}

#[inline(always)]
fn idwt_interleave_sample(
    ll: *const f32,
    hl: *const f32,
    lh: *const f32,
    hh: *const f32,
    job: CudaJ2kIdwtJob,
    local_x: u32,
    local_y: u32,
) -> f32 {
    let x = job.rect.x0 + local_x;
    let y = job.rect.y0 + local_y;
    let low_x = (x & 1) == 0;
    let low_y = (y & 1) == 0;
    let (source, source_rect, band_x, band_y) = if low_x && low_y {
        (
            ll,
            job.ll_rect,
            idwt_band_coord(job.rect.x0, x, job.ll_rect.x0, true),
            idwt_band_coord(job.rect.y0, y, job.ll_rect.y0, true),
        )
    } else if !low_x && low_y {
        (
            hl,
            job.hl_rect,
            idwt_band_coord(job.rect.x0, x, job.hl_rect.x0, false),
            idwt_band_coord(job.rect.y0, y, job.hl_rect.y0, true),
        )
    } else if low_x && !low_y {
        (
            lh,
            job.lh_rect,
            idwt_band_coord(job.rect.x0, x, job.lh_rect.x0, true),
            idwt_band_coord(job.rect.y0, y, job.lh_rect.y0, false),
        )
    } else {
        (
            hh,
            job.hh_rect,
            idwt_band_coord(job.rect.x0, x, job.hh_rect.x0, false),
            idwt_band_coord(job.rect.y0, y, job.hh_rect.y0, false),
        )
    };
    source_get(source, source_rect, band_x, band_y)
}

#[inline(always)]
fn run_interleave(pointers: IdwtPointers, job: CudaJ2kIdwtJob, local_x: u32, local_y: u32) {
    let width = rect_width(job.rect);
    let height = rect_height(job.rect);
    if local_x >= width || local_y >= height {
        return;
    }
    store_f32(
        pointers.output,
        local_y * width + local_x,
        idwt_interleave_sample(
            pointers.ll,
            pointers.hl,
            pointers.lh,
            pointers.hh,
            job,
            local_x,
            local_y,
        ),
    );
}

#[inline(always)]
fn multi_job_pointers(item: CudaJ2kIdwtMultiJob) -> IdwtPointers {
    IdwtPointers {
        ll: item.ll_ptr as usize as *const f32,
        hl: item.hl_ptr as usize as *const f32,
        lh: item.lh_ptr as usize as *const f32,
        hh: item.hh_ptr as usize as *const f32,
        output: item.output_ptr as usize as *mut f32,
    }
}

#[inline(always)]
fn filter_shared_single_sample(line: SharedLine, index: u32, len: u32, origin: u32) -> bool {
    if len != 1 {
        return false;
    }
    if line.active && index == 0 && (origin & 1) != 0 {
        line.store(0, line.load(0) * 0.5);
    }
    thread::sync_threads();
    true
}

#[inline(always)]
fn filter_shared_53_step(line: SharedLine, index: u32, len: u32, first: u32, update_even: bool) {
    if !line.active || index >= len || (index & 1) != first {
        return;
    }

    let left = pse_left(index, 1);
    let right = pse_right(index, 1, len);
    line.store(
        index,
        lift_53_sample(
            line.load(index),
            line.load(left),
            line.load(right),
            update_even,
        ),
    );
}

#[inline(always)]
fn filter_shared_53(line: SharedLine, index: u32, len: u32, origin: u32) {
    if filter_shared_single_sample(line, index, len, origin) {
        return;
    }

    let first_even = origin & 1;
    let first_odd = 1 - first_even;
    filter_shared_53_step(line, index, len, first_even, true);
    thread::sync_threads();
    filter_shared_53_step(line, index, len, first_odd, false);
    thread::sync_threads();
}

#[inline(always)]
fn horizontal_tiled_line(samples: *mut f32, row: u32) -> SharedLine {
    SharedLine {
        samples: simt_mut_ptr_at(samples, row as usize * IDWT_HORIZONTAL_ROW_SAMPLES),
        lane: 0,
        stride: 1,
        active: true,
    }
}

#[inline(always)]
fn filter_shared_53_rows(samples: *mut f32, index: u32, len: u32, origin: u32, rows: u32) {
    let first_even = origin & 1;
    let first_odd = 1 - first_even;
    let mut row = 0;
    while row < rows {
        filter_shared_53_step(
            horizontal_tiled_line(samples, row),
            index,
            len,
            first_even,
            true,
        );
        row += 1;
    }
    thread::sync_threads();
    row = 0;
    while row < rows {
        filter_shared_53_step(
            horizontal_tiled_line(samples, row),
            index,
            len,
            first_odd,
            false,
        );
        row += 1;
    }
    thread::sync_threads();
}

#[inline(always)]
fn scale_shared_97(line: SharedLine, index: u32, len: u32, first_even: u32, high_pass: f32) {
    if !line.active || index >= len {
        return;
    }
    let k0 = if first_even == 0 {
        IDWT_KAPPA
    } else {
        high_pass
    };
    let k1 = if first_even == 0 {
        high_pass
    } else {
        IDWT_KAPPA
    };
    let scale = if (index & 1) == 0 { k0 } else { k1 };
    line.store(index, line.load(index) * scale);
}

#[inline(always)]
fn filter_shared_97_step(line: SharedLine, index: u32, len: u32, first: u32, coefficient: f32) {
    if !line.active || index >= len || (index & 1) != first {
        return;
    }

    let left = pse_left(index, 1);
    let right = pse_right(index, 1, len);
    line.store(
        index,
        fused_mul_add_f32(
            line.load(left) + line.load(right),
            coefficient,
            line.load(index),
        ),
    );
}

#[inline(always)]
fn filter_shared_97(line: SharedLine, index: u32, len: u32, origin: u32, high_pass: f32) {
    if filter_shared_single_sample(line, index, len, origin) {
        return;
    }

    let first_even = origin & 1;
    let first_odd = 1 - first_even;
    scale_shared_97(line, index, len, first_even, high_pass);
    thread::sync_threads();
    filter_shared_97_step(line, index, len, first_even, IDWT_NEG_DELTA);
    thread::sync_threads();
    filter_shared_97_step(line, index, len, first_odd, IDWT_NEG_GAMMA);
    thread::sync_threads();
    filter_shared_97_step(line, index, len, first_even, IDWT_NEG_BETA);
    thread::sync_threads();
    filter_shared_97_step(line, index, len, first_odd, IDWT_NEG_ALPHA);
    thread::sync_threads();
}

#[inline(always)]
fn filter_shared_97_rows(
    samples: *mut f32,
    index: u32,
    len: u32,
    origin: u32,
    high_pass: f32,
    rows: u32,
) {
    let first_even = origin & 1;
    let first_odd = 1 - first_even;
    let mut row = 0;
    while row < rows {
        scale_shared_97(
            horizontal_tiled_line(samples, row),
            index,
            len,
            first_even,
            high_pass,
        );
        row += 1;
    }
    thread::sync_threads();
    filter_shared_97_rows_step(samples, index, len, first_even, IDWT_NEG_DELTA, rows);
    filter_shared_97_rows_step(samples, index, len, first_odd, IDWT_NEG_GAMMA, rows);
    filter_shared_97_rows_step(samples, index, len, first_even, IDWT_NEG_BETA, rows);
    filter_shared_97_rows_step(samples, index, len, first_odd, IDWT_NEG_ALPHA, rows);
}

#[inline(always)]
fn filter_shared_97_rows_step(
    samples: *mut f32,
    index: u32,
    len: u32,
    first: u32,
    coefficient: f32,
    rows: u32,
) {
    let mut row = 0;
    while row < rows {
        filter_shared_97_step(
            horizontal_tiled_line(samples, row),
            index,
            len,
            first,
            coefficient,
        );
        row += 1;
    }
    thread::sync_threads();
}

#[inline(always)]
fn filter_column_strip_step(
    line: SharedLine,
    len: u32,
    first: u32,
    coefficient: f32,
    irreversible: bool,
    update_even: bool,
) {
    let mut row = 2 * thread::threadIdx_y() + first;
    while row < len {
        if irreversible {
            filter_shared_97_step(line, row, len, first, coefficient);
        } else {
            filter_shared_53_step(line, row, len, first, update_even);
        }
        row += 2 * thread::blockDim_y();
    }
    thread::sync_threads();
}

#[inline(always)]
fn run_vertical_strip<const MAX_ROWS: u32>(
    jobs: *const CudaJ2kIdwtMultiJob,
    samples: *mut f32,
    irreversible: bool,
) {
    let item = load_job(jobs, thread::blockIdx_y());
    let job = item.job;
    let width = rect_width(job.rect);
    let height = rect_height(job.rect);
    // The host launches strip kernels only for heights within `MAX_ROWS`. This
    // kernel has no status word, so a routing mistake leaves the job
    // untransformed here rather than overrunning shared memory.
    if height > MAX_ROWS {
        return;
    }
    let col = thread::blockIdx_x() * IDWT_VERTICAL_COLUMNS + thread::threadIdx_x();
    let output = item.output_ptr as usize as *mut f32;
    let line = SharedLine {
        samples,
        lane: thread::threadIdx_x(),
        stride: IDWT_VERTICAL_COLUMNS,
        active: col < width,
    };

    // Each block owns complete columns, so the in-place transform has no
    // cross-block read/write overlap. Eight adjacent columns fill a 32-byte
    // memory sector; threads stride over rows to keep the block at 256 threads.
    let mut row = thread::threadIdx_y();
    while row < height {
        if line.active {
            line.store(row, load_f32(output.cast_const(), row * width + col));
        }
        row += thread::blockDim_y();
    }
    thread::sync_threads();

    if !filter_shared_single_sample(line, thread::threadIdx_y(), height, job.rect.y0) {
        let even = job.rect.y0 & 1;
        let odd = 1 - even;
        if irreversible {
            let mut row = thread::threadIdx_y();
            while row < height {
                scale_shared_97(line, row, height, even, idwt_high_pass(job.irreversible97));
                row += thread::blockDim_y();
            }
            thread::sync_threads();
            filter_column_strip_step(line, height, even, IDWT_NEG_DELTA, true, true);
            filter_column_strip_step(line, height, odd, IDWT_NEG_GAMMA, true, false);
            filter_column_strip_step(line, height, even, IDWT_NEG_BETA, true, true);
            filter_column_strip_step(line, height, odd, IDWT_NEG_ALPHA, true, false);
        } else {
            filter_column_strip_step(line, height, even, 0.0, false, true);
            filter_column_strip_step(line, height, odd, 0.0, false, false);
        }
    }

    let mut row = thread::threadIdx_y();
    while row < height {
        if line.active {
            store_f32(output, row * width + col, line.load(row));
        }
        row += thread::blockDim_y();
    }
}

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub unsafe fn j2k_idwt_interleave(
        ll: *const f32,
        hl: *const f32,
        lh: *const f32,
        hh: *const f32,
        output: *mut f32,
        job_buffer: *const CudaJ2kIdwtJob,
    ) {
        let job = load_job(job_buffer, 0);
        let local_x = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        let local_y = thread::blockIdx_y() * thread::blockDim_y() + thread::threadIdx_y();
        run_interleave(
            IdwtPointers {
                ll,
                hl,
                lh,
                hh,
                output,
            },
            job,
            local_x,
            local_y,
        );
    }

    #[kernel]
    pub unsafe fn j2k_idwt_interleave_horizontal_multi(jobs: *const CudaJ2kIdwtMultiJob) {
        let job_idx = thread::blockIdx_y();
        let item = load_job(jobs, job_idx);
        let job = item.job;
        let pointers = multi_job_pointers(item);
        let width = rect_width(job.rect);
        let height = rect_height(job.rect);
        let local_y = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        if local_y >= height {
            return;
        }

        let mut local_x = 0;
        while local_x < width {
            store_f32(
                pointers.output,
                local_y * width + local_x,
                idwt_interleave_sample(
                    pointers.ll,
                    pointers.hl,
                    pointers.lh,
                    pointers.hh,
                    job,
                    local_x,
                    local_y,
                ),
            );
            local_x += 1;
        }
        filter_horizontal_scanline(
            unsafe { pointers.output.add((local_y * width) as usize) },
            width,
            job.rect.x0,
            job.irreversible97,
        );
    }

    #[kernel]
    pub unsafe fn j2k_idwt_interleave_horizontal_tiled_multi(jobs: *const CudaJ2kIdwtMultiJob) {
        static mut ROW_SAMPLES: SharedArray<f32, IDWT_HORIZONTAL_SAMPLES> = SharedArray::UNINIT;
        let item = load_job(jobs, thread::blockIdx_y());
        let job = item.job;
        let pointers = multi_job_pointers(item);
        let width = rect_width(job.rect);
        if width == 0 {
            return;
        }
        let tiles = (width - 1) / IDWT_HORIZONTAL_TILE + 1;
        let row = thread::blockIdx_x() / tiles * IDWT_HORIZONTAL_ROWS;
        let start = (thread::blockIdx_x() % tiles) * IDWT_HORIZONTAL_TILE;
        if row >= rect_height(job.rect) {
            return;
        }
        let lane = thread::threadIdx_x();
        let rows = (rect_height(job.rect) - row).min(IDWT_HORIZONTAL_ROWS);
        if width == 1 {
            if lane == 0 {
                let mut row_offset = 0;
                while row_offset < rows {
                    let value = idwt_interleave_sample(
                        pointers.ll,
                        pointers.hl,
                        pointers.lh,
                        pointers.hh,
                        job,
                        0,
                        row + row_offset,
                    );
                    store_f32(
                        pointers.output,
                        row + row_offset,
                        if job.rect.x0 & 1 == 0 {
                            value
                        } else {
                            value * 0.5
                        },
                    );
                    row_offset += 1;
                }
            }
            return;
        }
        let count = (width - start).min(IDWT_HORIZONTAL_TILE);
        let len = count + 2 * IDWT_HORIZONTAL_HALO;
        let samples = unsafe { ROW_SAMPLES.as_mut_ptr() };
        // Input subbands are immutable. Neighboring tiles can therefore load
        // overlapping halos while writing disjoint output spans. Four halo
        // samples cover all four lifting steps without changing their order.
        let x = if lane < len {
            j2k_codec_math::dwt::reflect_index(
                start as i64 + lane as i64 - IDWT_HORIZONTAL_HALO as i64,
                width,
            )
        } else {
            0
        };
        let mut row_offset = 0;
        while row_offset < rows {
            if lane < len {
                horizontal_tiled_line(samples, row_offset).store(
                    lane,
                    idwt_interleave_sample(
                        pointers.ll,
                        pointers.hl,
                        pointers.lh,
                        pointers.hh,
                        job,
                        x,
                        row + row_offset,
                    ),
                );
            }
            row_offset += 1;
        }
        thread::sync_threads();
        // Tile width and halo are both even, preserving the global parity.
        if job.irreversible97 == 0 {
            filter_shared_53_rows(samples, lane, len, job.rect.x0, rows);
        } else {
            filter_shared_97_rows(
                samples,
                lane,
                len,
                job.rect.x0,
                idwt_high_pass(job.irreversible97),
                rows,
            );
        }
        row_offset = 0;
        while row_offset < rows {
            if lane < count {
                store_f32(
                    pointers.output,
                    (row + row_offset) * width + start + lane,
                    horizontal_tiled_line(samples, row_offset).load(lane + IDWT_HORIZONTAL_HALO),
                );
            }
            row_offset += 1;
        }
    }

    #[kernel]
    pub unsafe fn j2k_idwt_interleave_horizontal_53_multi(jobs: *const CudaJ2kIdwtMultiJob) {
        static mut ROW_SAMPLES: SharedArray<f32, IDWT_COOP_SAMPLES> = SharedArray::UNINIT;

        let row_samples = unsafe { ROW_SAMPLES.as_mut_ptr() };
        let shared = SharedLine {
            samples: row_samples,
            lane: 0,
            stride: 1,
            active: true,
        };
        let local_x = thread::threadIdx_x();
        let local_y = thread::blockIdx_x();
        let item = load_job(jobs, thread::blockIdx_y());
        let job = item.job;
        let pointers = multi_job_pointers(item);
        let width = rect_width(job.rect);
        let height = rect_height(job.rect);
        if local_y >= height {
            return;
        }

        if local_x < width {
            shared.store(
                local_x,
                idwt_interleave_sample(
                    pointers.ll,
                    pointers.hl,
                    pointers.lh,
                    pointers.hh,
                    job,
                    local_x,
                    local_y,
                ),
            );
        }
        thread::sync_threads();

        filter_shared_53(shared, local_x, width, job.rect.x0);
        if local_x < width {
            store_f32(
                pointers.output,
                local_y * width + local_x,
                shared.load(local_x),
            );
        }
    }

    #[kernel]
    pub unsafe fn j2k_idwt_interleave_horizontal_97_multi(jobs: *const CudaJ2kIdwtMultiJob) {
        static mut ROW_SAMPLES: SharedArray<f32, IDWT_COOP_SAMPLES> = SharedArray::UNINIT;

        let row_samples = unsafe { ROW_SAMPLES.as_mut_ptr() };
        let shared = SharedLine {
            samples: row_samples,
            lane: 0,
            stride: 1,
            active: true,
        };
        let local_x = thread::threadIdx_x();
        let local_y = thread::blockIdx_x();
        let item = load_job(jobs, thread::blockIdx_y());
        let job = item.job;
        let pointers = multi_job_pointers(item);
        let width = rect_width(job.rect);
        let height = rect_height(job.rect);
        if local_y >= height {
            return;
        }

        if local_x < width {
            shared.store(
                local_x,
                idwt_interleave_sample(
                    pointers.ll,
                    pointers.hl,
                    pointers.lh,
                    pointers.hh,
                    job,
                    local_x,
                    local_y,
                ),
            );
        }
        thread::sync_threads();

        filter_shared_97(
            shared,
            local_x,
            width,
            job.rect.x0,
            idwt_high_pass(job.irreversible97),
        );
        if local_x < width {
            store_f32(
                pointers.output,
                local_y * width + local_x,
                shared.load(local_x),
            );
        }
    }

    #[kernel]
    pub unsafe fn j2k_idwt_horizontal_53(output: *mut f32, job_buffer: *const CudaJ2kIdwtJob) {
        let job = load_job(job_buffer, 0);
        let width = rect_width(job.rect);
        let height = rect_height(job.rect);
        let row = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        if row >= height {
            return;
        }
        filter_horizontal_scanline(
            unsafe { output.add((row * width) as usize) },
            width,
            job.rect.x0,
            0,
        );
    }

    #[kernel]
    pub unsafe fn j2k_idwt_horizontal_97(output: *mut f32, job_buffer: *const CudaJ2kIdwtJob) {
        let job = load_job(job_buffer, 0);
        let width = rect_width(job.rect);
        let height = rect_height(job.rect);
        let row = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        if row >= height {
            return;
        }
        filter_horizontal_scanline(
            unsafe { output.add((row * width) as usize) },
            width,
            job.rect.x0,
            job.irreversible97,
        );
    }

    #[kernel]
    pub unsafe fn j2k_idwt_vertical_53(output: *mut f32, job_buffer: *const CudaJ2kIdwtJob) {
        let job = load_job(job_buffer, 0);
        let width = rect_width(job.rect);
        let height = rect_height(job.rect);
        let col = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        if col >= width {
            return;
        }
        filter_vertical_column(output, width, height, job.rect.y0, col, 0);
    }

    #[kernel]
    pub unsafe fn j2k_idwt_vertical_97(output: *mut f32, job_buffer: *const CudaJ2kIdwtJob) {
        let job = load_job(job_buffer, 0);
        let width = rect_width(job.rect);
        let height = rect_height(job.rect);
        let col = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        if col >= width {
            return;
        }
        filter_vertical_column(output, width, height, job.rect.y0, col, job.irreversible97);
    }

    #[kernel]
    pub unsafe fn j2k_idwt_vertical_multi(jobs: *const CudaJ2kIdwtMultiJob) {
        let job_idx = thread::blockIdx_y();
        let item = load_job(jobs, job_idx);
        let job = item.job;
        let output = item.output_ptr as usize as *mut f32;
        let width = rect_width(job.rect);
        let height = rect_height(job.rect);
        let col = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        if col >= width {
            return;
        }
        filter_vertical_column(output, width, height, job.rect.y0, col, job.irreversible97);
    }

    #[kernel]
    pub unsafe fn j2k_idwt_vertical_tall_strip_multi(jobs: *const CudaJ2kIdwtMultiJob) {
        static mut COLUMN_SAMPLES: SharedArray<f32, { 1024 * IDWT_VERTICAL_COLUMNS as usize }> =
            SharedArray::UNINIT;
        let item = load_job(jobs, thread::blockIdx_y());
        run_vertical_strip::<1024>(
            jobs,
            unsafe { COLUMN_SAMPLES.as_mut_ptr() },
            item.job.irreversible97 != 0,
        );
    }

    #[kernel]
    pub unsafe fn j2k_idwt_vertical_strip_multi(jobs: *const CudaJ2kIdwtMultiJob) {
        static mut COLUMN_SAMPLES: SharedArray<f32, IDWT_VERTICAL_SAMPLES> = SharedArray::UNINIT;
        let item = load_job(jobs, thread::blockIdx_y());
        run_vertical_strip::<512>(
            jobs,
            unsafe { COLUMN_SAMPLES.as_mut_ptr() },
            item.job.irreversible97 != 0,
        );
    }

    #[kernel]
    pub unsafe fn j2k_idwt_vertical_53_multi(jobs: *const CudaJ2kIdwtMultiJob) {
        static mut COLUMN_SAMPLES: SharedArray<f32, IDWT_VERTICAL_SAMPLES> = SharedArray::UNINIT;
        run_vertical_strip::<512>(jobs, unsafe { COLUMN_SAMPLES.as_mut_ptr() }, false);
    }

    #[kernel]
    pub unsafe fn j2k_idwt_vertical_97_multi(jobs: *const CudaJ2kIdwtMultiJob) {
        static mut COLUMN_SAMPLES: SharedArray<f32, IDWT_VERTICAL_SAMPLES> = SharedArray::UNINIT;
        run_vertical_strip::<512>(jobs, unsafe { COLUMN_SAMPLES.as_mut_ptr() }, true);
    }

    #[kernel]
    pub unsafe fn j2k_idwt_vertical_97_multi_cols4(jobs: *const CudaJ2kIdwtMultiJob) {
        static mut COLUMN_SAMPLES: SharedArray<f32, IDWT_COLS4_SAMPLES> = SharedArray::UNINIT;

        let column_samples = unsafe { COLUMN_SAMPLES.as_mut_ptr() };
        let local_col = thread::threadIdx_x();
        let row = thread::threadIdx_y();
        let col = thread::blockIdx_x() * IDWT_COLS4_COLUMNS + local_col;
        let item = load_job(jobs, thread::blockIdx_y());
        let job = item.job;
        let output = item.output_ptr as usize as *mut f32;
        let width = rect_width(job.rect);
        let height = rect_height(job.rect);
        if height > 256 {
            return;
        }

        let shared = SharedLine {
            samples: column_samples,
            lane: local_col,
            stride: IDWT_COLS4_COLUMNS,
            active: col < width,
        };
        let valid = shared.active && row < height;
        if valid {
            shared.store(row, load_f32(output.cast_const(), row * width + col));
        }
        thread::sync_threads();

        filter_shared_97(
            shared,
            row,
            height,
            job.rect.y0,
            idwt_high_pass(job.irreversible97),
        );
        if valid {
            store_f32(output, row * width + col, shared.load(row));
        }
    }
}

fn main() {}
