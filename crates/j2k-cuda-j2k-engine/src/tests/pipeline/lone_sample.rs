// SPDX-License-Identifier: MIT OR Apache-2.0

//! Reversible 5/3 inverse DWT on windows one sample wide or tall
//! (frames-sg/j2k#333). A lone sample at an odd coordinate is high-pass, and
//! the inverse transform divides it by two with floor division, as the CPU
//! does. Each kernel route is compared with the integer reference in
//! `j2k-test-support`.

use super::super::cuda_runtime_gate;
use crate::j2k_decode::{
    idwt_batch_kernel_mode, CudaJ2kIdwtBatchKernelMode, CudaJ2kIdwtMultiKernelJob,
};
use crate::{
    f32_slice_as_bytes, f32_slice_as_bytes_mut, CudaContext, CudaJ2kIdwtJob, CudaJ2kIdwtTarget,
    CudaJ2kRect,
};
use j2k_cuda_runtime::{CudaBufferPool, CudaDeviceBuffer};
use j2k_test_support::{
    dwt53_bands, dwt53_interleave, dwt53_samples_f32, inverse_dwt53, Dwt53Window,
    DWT53_LONE_SAMPLE_WINDOWS,
};

/// Widening a batch to 128 samples or more routes its horizontal pass to the
/// tiled kernel, and 128 to 1024 rows route its vertical pass to the strip
/// kernel. Both together make the batch cooperative.
const WIDE: Dwt53Window = Dwt53Window::new(200, 1, 1, 1);
const TALL: Dwt53Window = Dwt53Window::new(1, 200, 1, 1);

struct Case {
    window: Dwt53Window,
    job: CudaJ2kIdwtJob,
    bands: [CudaDeviceBuffer; 4],
    expected: Vec<f32>,
}

fn case(context: &CudaContext, window: Dwt53Window) -> Case {
    let bands = dwt53_bands(window, 0);
    let mut plane = dwt53_interleave(window, &bands);
    inverse_dwt53(window, &mut plane, true, true);
    // Absolute band rectangles: low bands start at ceil(origin / 2), high
    // bands at floor(origin / 2).
    let low = (window.x0.div_ceil(2), window.y0.div_ceil(2));
    let high = (window.x0 / 2, window.y0 / 2);
    let origins = [low, (high.0, low.1), (low.0, high.1), high];
    let dims = window.band_dims();
    let rects: [CudaJ2kRect; 4] = std::array::from_fn(|band| {
        let ((x0, y0), (width, height)) = (origins[band], dims[band]);
        CudaJ2kRect {
            x0,
            y0,
            x1: x0 + width,
            y1: y0 + height,
        }
    });
    Case {
        window,
        job: CudaJ2kIdwtJob {
            rect: CudaJ2kRect {
                x0: window.x0,
                y0: window.y0,
                x1: window.x0 + window.width,
                y1: window.y0 + window.height,
            },
            ll_rect: rects[0],
            hl_rect: rects[1],
            lh_rect: rects[2],
            hh_rect: rects[3],
            irreversible97: 0,
        },
        bands: bands.map(|samples| {
            context
                .upload(f32_slice_as_bytes(&dwt53_samples_f32(&samples)))
                .expect("upload band")
        }),
        expected: dwt53_samples_f32(&plane),
    }
}

fn single_mismatch(
    engine: &crate::J2kCudaEngine<'_>,
    pool: &CudaBufferPool,
    case: &Case,
) -> Option<String> {
    let [ll, hl, lh, hh] = &case.bands;
    let output = engine
        .j2k_inverse_dwt_single_device_with_pool(ll, hl, lh, hh, case.job, pool)
        .expect("single-job IDWT");
    let mut actual = vec![0.0_f32; case.expected.len()];
    output
        .buffer()
        .expect("single-job output")
        .copy_range_to_host(0, f32_slice_as_bytes_mut(&mut actual))
        .expect("read single-job output");
    (actual != case.expected).then(|| {
        format!(
            "single {:?}: {actual:?}, expected {:?}",
            case.window, case.expected
        )
    })
}

fn batch_mismatches(
    engine: &crate::J2kCudaEngine<'_>,
    pool: &CudaBufferPool,
    label: &str,
    cases: &[&Case],
    mode: CudaJ2kIdwtBatchKernelMode,
) -> Vec<String> {
    let kernel_jobs = cases
        .iter()
        .map(|case| CudaJ2kIdwtMultiKernelJob {
            ll_ptr: 0,
            hl_ptr: 0,
            lh_ptr: 0,
            hh_ptr: 0,
            output_ptr: 0,
            job: case.job,
            reserved_tail: 0,
        })
        .collect::<Vec<_>>();
    let max_width = cases.iter().map(|case| case.window.width).max();
    let max_height = cases.iter().map(|case| case.window.height).max();
    assert_eq!(
        idwt_batch_kernel_mode(
            &kernel_jobs,
            max_width.unwrap_or(0),
            max_height.unwrap_or(0)
        ),
        mode,
        "{label} batch kernel mode"
    );

    let outputs = cases
        .iter()
        .map(|case| {
            pool.take(case.expected.len() * size_of::<f32>())
                .expect("batch output")
        })
        .collect::<Vec<_>>();
    let targets = cases
        .iter()
        .zip(&outputs)
        .map(|(case, output)| CudaJ2kIdwtTarget {
            ll: &case.bands[0],
            hl: &case.bands[1],
            lh: &case.bands[2],
            hh: &case.bands[3],
            output: output.as_device_buffer().expect("batch output buffer"),
            job: case.job,
        })
        .collect::<Vec<_>>();
    engine
        .j2k_inverse_dwt_batch_device_with_pool(&targets, pool)
        .expect("batched IDWT");
    cases
        .iter()
        .zip(&outputs)
        .filter_map(|(case, output)| {
            let mut actual = vec![0.0_f32; case.expected.len()];
            output
                .copy_to_host(f32_slice_as_bytes_mut(&mut actual))
                .expect("read batch output");
            (actual != case.expected).then(|| {
                format!(
                    "{label} batch {:?}: {actual:?}, expected {:?}",
                    case.window, case.expected
                )
            })
        })
        .collect()
}

#[test]
fn j2k_inverse_dwt_53_lone_samples_match_integer_reference_when_runtime_required() {
    if !cuda_runtime_gate() {
        return;
    }
    let context = CudaContext::system_default().expect("CUDA context");
    let pool = context.buffer_pool();
    let engine = crate::J2kCudaEngine::new(&context);
    let cases = DWT53_LONE_SAMPLE_WINDOWS.map(|window| case(&context, window));
    let wide = case(&context, WIDE);
    let tall = case(&context, TALL);

    let mut found = cases
        .iter()
        .chain([&wide, &tall])
        .filter_map(|case| single_mismatch(&engine, &pool, case))
        .collect::<Vec<_>>();
    let base = cases.iter().collect::<Vec<_>>();
    let batches = [
        ("generic", base.clone(), CudaJ2kIdwtBatchKernelMode::Generic),
        (
            "generic wide",
            [base.clone(), vec![&wide]].concat(),
            CudaJ2kIdwtBatchKernelMode::Generic,
        ),
        (
            "generic tall",
            [base.clone(), vec![&tall]].concat(),
            CudaJ2kIdwtBatchKernelMode::Generic,
        ),
        (
            "cooperative",
            [base, vec![&wide, &tall]].concat(),
            CudaJ2kIdwtBatchKernelMode::Cooperative53,
        ),
    ];
    for (label, batch, mode) in batches {
        found.extend(batch_mismatches(&engine, &pool, label, &batch, mode));
    }
    assert!(found.is_empty(), "{}", found.join("\n"));
}
