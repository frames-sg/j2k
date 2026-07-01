#![allow(static_mut_refs)]

use cuda_device::{SharedArray, kernel, thread};
use cuda_host::cuda_module;

const IDWT_COOP_SAMPLES: usize = 512;
const IDWT_COLS4_COLUMNS: u32 = 4;
const IDWT_COLS4_SAMPLES: usize = 256 * IDWT_COLS4_COLUMNS as usize;
const IDWT_NEG_ALPHA: f32 = j2k_codec_math::dwt::IDWT97_NEG_ALPHA_F32;
const IDWT_NEG_BETA: f32 = j2k_codec_math::dwt::IDWT97_NEG_BETA_F32;
const IDWT_NEG_GAMMA: f32 = j2k_codec_math::dwt::IDWT97_NEG_GAMMA_F32;
const IDWT_NEG_DELTA: f32 = j2k_codec_math::dwt::IDWT97_NEG_DELTA_F32;
const IDWT_KAPPA: f32 = j2k_codec_math::dwt::DWT97_KAPPA_F32;
const IDWT_INV_KAPPA: f32 = j2k_codec_math::dwt::DWT97_INV_KAPPA_F32;

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
    unsafe { *ptr.add(index as usize) }
}

#[inline(always)]
fn store_f32(ptr: *mut f32, index: u32, value: f32) {
    unsafe {
        *ptr.add(index as usize) = value;
    }
}

#[inline(always)]
fn load_job<T: Copy>(ptr: *const T, index: u32) -> T {
    unsafe { *ptr.add(index as usize) }
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
            sample + (left_sample + right_sample) * coefficient,
        );
    }

    let mut i = if first == 0 { 2 } else { 1 };
    while i + 1 < width {
        let sample = load_f32(scanline.cast_const(), i);
        let left = load_f32(scanline.cast_const(), i - 1);
        let right = load_f32(scanline.cast_const(), i + 1);
        store_f32(scanline, i, sample + (left + right) * coefficient);
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
            sample + (left_sample + right_sample) * coefficient,
        );
    }
}

#[inline(always)]
fn filter_horizontal_scanline(scanline: *mut f32, width: u32, rect_x0: u32, irreversible97: bool) {
    if width == 1 {
        if (rect_x0 & 1) != 0 {
            store_f32(scanline, 0, load_f32(scanline.cast_const(), 0) * 0.5);
        }
        return;
    }

    let first_even = rect_x0 & 1;
    let first_odd = 1 - first_even;
    if !irreversible97 {
        filter_step_horizontal_53(scanline, width, first_even, true);
        filter_step_horizontal_53(scanline, width, first_odd, false);
    } else {
        let k0 = if first_even == 0 {
            IDWT_KAPPA
        } else {
            IDWT_INV_KAPPA
        };
        let k1 = if first_even == 0 {
            IDWT_INV_KAPPA
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
fn filter_horizontal(output: *mut f32, rect: CudaJ2kRect, irreversible97: bool) {
    let width = rect_width(rect);
    let height = rect_height(rect);
    let mut y = 0;
    while y < height {
        filter_horizontal_scanline(
            unsafe { output.add((y * width) as usize) },
            width,
            rect.x0,
            irreversible97,
        );
        y += 1;
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
        store_f32(output, idx, sample + (above + below) * coefficient);
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
    irreversible97: bool,
) {
    if height == 1 {
        if (rect_y0 & 1) != 0 {
            store_f32(output, col, load_f32(output.cast_const(), col) * 0.5);
        }
        return;
    }

    let first_even = rect_y0 & 1;
    let first_odd = 1 - first_even;
    if !irreversible97 {
        filter_step_vertical_53_column(output, width, height, col, first_even, true);
        filter_step_vertical_53_column(output, width, height, col, first_odd, false);
    } else {
        let k0 = if first_even == 0 {
            IDWT_KAPPA
        } else {
            IDWT_INV_KAPPA
        };
        let k1 = if first_even == 0 {
            IDWT_INV_KAPPA
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
fn filter_vertical(output: *mut f32, rect: CudaJ2kRect, irreversible97: bool) {
    let width = rect_width(rect);
    let height = rect_height(rect);
    let mut col = 0;
    while col < width {
        filter_vertical_column(output, width, height, rect.y0, col, irreversible97);
        col += 1;
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
fn scale_shared_97(line: SharedLine, index: u32, len: u32, first_even: u32) {
    if !line.active || index >= len {
        return;
    }
    let k0 = if first_even == 0 {
        IDWT_KAPPA
    } else {
        IDWT_INV_KAPPA
    };
    let k1 = if first_even == 0 {
        IDWT_INV_KAPPA
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
        line.load(index) + (line.load(left) + line.load(right)) * coefficient,
    );
}

#[inline(always)]
fn filter_shared_97(line: SharedLine, index: u32, len: u32, origin: u32) {
    if filter_shared_single_sample(line, index, len, origin) {
        return;
    }

    let first_even = origin & 1;
    let first_odd = 1 - first_even;
    scale_shared_97(line, index, len, first_even);
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

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub unsafe fn j2k_inverse_dwt_single(
        ll: *const f32,
        hl: *const f32,
        lh: *const f32,
        hh: *const f32,
        output: *mut f32,
        job_buffer: *const CudaJ2kIdwtJob,
    ) {
        if thread::blockIdx_x() != 0 || thread::threadIdx_x() != 0 {
            return;
        }

        let job = load_job(job_buffer, 0);
        let width = rect_width(job.rect);
        let height = rect_height(job.rect);
        let mut local_y = 0;
        while local_y < height {
            let mut local_x = 0;
            while local_x < width {
                store_f32(
                    output,
                    local_y * width + local_x,
                    idwt_interleave_sample(ll, hl, lh, hh, job, local_x, local_y),
                );
                local_x += 1;
            }
            local_y += 1;
        }

        if width > 0 && height > 0 {
            let irreversible97 = job.irreversible97 != 0;
            filter_horizontal(output, job.rect, irreversible97);
            filter_vertical(output, job.rect, irreversible97);
        }
    }

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
            job.irreversible97 != 0,
        );
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

        filter_shared_97(shared, local_x, width, job.rect.x0);
        if local_x < width {
            store_f32(
                pointers.output,
                local_y * width + local_x,
                shared.load(local_x),
            );
        }
    }

    #[kernel]
    pub unsafe fn j2k_idwt_horizontal(output: *mut f32, job_buffer: *const CudaJ2kIdwtJob) {
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
            job.irreversible97 != 0,
        );
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
            false,
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
            true,
        );
    }

    #[kernel]
    pub unsafe fn j2k_idwt_vertical(output: *mut f32, job_buffer: *const CudaJ2kIdwtJob) {
        let job = load_job(job_buffer, 0);
        let width = rect_width(job.rect);
        let height = rect_height(job.rect);
        let col = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        if col >= width {
            return;
        }
        filter_vertical_column(
            output,
            width,
            height,
            job.rect.y0,
            col,
            job.irreversible97 != 0,
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
        filter_vertical_column(output, width, height, job.rect.y0, col, false);
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
        filter_vertical_column(output, width, height, job.rect.y0, col, true);
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
        filter_vertical_column(
            output,
            width,
            height,
            job.rect.y0,
            col,
            job.irreversible97 != 0,
        );
    }

    #[kernel]
    pub unsafe fn j2k_idwt_vertical_53_multi(jobs: *const CudaJ2kIdwtMultiJob) {
        static mut COLUMN_SAMPLES: SharedArray<f32, IDWT_COOP_SAMPLES> = SharedArray::UNINIT;

        let column_samples = unsafe { COLUMN_SAMPLES.as_mut_ptr() };
        let shared = SharedLine {
            samples: column_samples,
            lane: 0,
            stride: 1,
            active: true,
        };
        let row = thread::threadIdx_x();
        let col = thread::blockIdx_x();
        let item = load_job(jobs, thread::blockIdx_y());
        let job = item.job;
        let output = item.output_ptr as usize as *mut f32;
        let width = rect_width(job.rect);
        let height = rect_height(job.rect);
        if col >= width {
            return;
        }

        if row < height {
            shared.store(row, load_f32(output.cast_const(), row * width + col));
        }
        thread::sync_threads();

        filter_shared_53(shared, row, height, job.rect.y0);
        if row < height {
            store_f32(output, row * width + col, shared.load(row));
        }
    }

    #[kernel]
    pub unsafe fn j2k_idwt_vertical_97_multi(jobs: *const CudaJ2kIdwtMultiJob) {
        static mut COLUMN_SAMPLES: SharedArray<f32, IDWT_COOP_SAMPLES> = SharedArray::UNINIT;

        let column_samples = unsafe { COLUMN_SAMPLES.as_mut_ptr() };
        let shared = SharedLine {
            samples: column_samples,
            lane: 0,
            stride: 1,
            active: true,
        };
        let row = thread::threadIdx_x();
        let col = thread::blockIdx_x();
        let item = load_job(jobs, thread::blockIdx_y());
        let job = item.job;
        let output = item.output_ptr as usize as *mut f32;
        let width = rect_width(job.rect);
        let height = rect_height(job.rect);
        if col >= width {
            return;
        }

        if row < height {
            shared.store(row, load_f32(output.cast_const(), row * width + col));
        }
        thread::sync_threads();

        filter_shared_97(shared, row, height, job.rect.y0);
        if row < height {
            store_f32(output, row * width + col, shared.load(row));
        }
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

        filter_shared_97(shared, row, height, job.rect.y0);
        if valid {
            store_f32(output, row * width + col, shared.load(row));
        }
    }
}

fn main() {}
