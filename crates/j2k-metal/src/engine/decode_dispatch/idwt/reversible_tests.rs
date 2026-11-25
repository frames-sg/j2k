// SPDX-License-Identifier: MIT OR Apache-2.0

//! Reversible 5/3 kernels on windows one sample wide or tall
//! (frames-sg/j2k#333). A lone sample at an odd coordinate is high-pass, and
//! the inverse transform divides it by two with floor division, as the CPU
//! does. Each kernel is compared with the integer reference in
//! `j2k-test-support`.

use super::super::checked_buffer_slice;
use super::*;
use crate::engine::runtime::MetalRuntime;
use j2k_test_support::{
    dwt53_bands, dwt53_interleave, dwt53_samples_f32, inverse_dwt53, Dwt53Window,
    DWT53_LONE_SAMPLE_WINDOWS,
};

/// 200 samples spans more than one 128-sample tile of the fused kernels.
const LONG: [Dwt53Window; 2] = [
    Dwt53Window::new(1, 200, 1, 1),
    Dwt53Window::new(200, 1, 1, 1),
];

const BATCH: usize = 3;

fn single_params(window: Dwt53Window) -> J2kIdwtSingleDecompositionParams {
    let [ll, hl, lh, hh] = window.band_dims();
    J2kIdwtSingleDecompositionParams {
        x0: window.x0,
        y0: window.y0,
        output_x: 0,
        output_y: 0,
        width: window.width,
        height: window.height,
        ll_x: 0,
        ll_y: 0,
        ll_width: ll.0,
        ll_height: ll.1,
        hl_x: 0,
        hl_y: 0,
        hl_width: hl.0,
        hl_height: hl.1,
        lh_x: 0,
        lh_y: 0,
        lh_width: lh.0,
        lh_height: lh.1,
        hh_x: 0,
        hh_y: 0,
        hh_width: hh.0,
        hh_height: hh.1,
    }
}

fn repeated_params(window: Dwt53Window, batch: usize) -> J2kRepeatedIdwtSingleDecompositionParams {
    let single = single_params(window);
    let strides = window
        .band_lens()
        .map(|len| u32::try_from(len).expect("band stride"));
    J2kRepeatedIdwtSingleDecompositionParams {
        x0: single.x0,
        y0: single.y0,
        output_x: 0,
        output_y: 0,
        width: single.width,
        height: single.height,
        ll_x: 0,
        ll_y: 0,
        ll_width: single.ll_width,
        ll_height: single.ll_height,
        hl_x: 0,
        hl_y: 0,
        hl_width: single.hl_width,
        hl_height: single.hl_height,
        lh_x: 0,
        lh_y: 0,
        lh_width: single.lh_width,
        lh_height: single.lh_height,
        hh_x: 0,
        hh_y: 0,
        hh_width: single.hh_width,
        hh_height: single.hh_height,
        ll_instance_stride: strides[0],
        hl_instance_stride: strides[1],
        lh_instance_stride: strides[2],
        hh_instance_stride: strides[3],
        batch_count: u32::try_from(batch).expect("batch count"),
    }
}

/// A buffer of `values`, or of one unread sample when a band is empty.
fn buffer_of(runtime: &MetalRuntime, values: &[f32]) -> Buffer {
    let values = if values.is_empty() {
        &[0.0][..]
    } else {
        values
    };
    copied_slice_buffer(&runtime.device, values).expect("Metal buffer")
}

/// The kernels a test runs, encoded on one compute encoder.
#[derive(Clone, Copy, Debug)]
enum Route {
    /// Interleave, horizontal, and vertical passes as decode dispatches them.
    Production,
    /// The fused interleave and horizontal kernel alone.
    InterleaveHorizontalFused,
    /// The fused vertical kernel alone, on an interleaved plane.
    VerticalFused,
}

impl Route {
    /// Whether the route lifts rows and columns.
    const fn axes(self) -> (bool, bool) {
        match self {
            Self::Production => (true, true),
            Self::InterleaveHorizontalFused => (true, false),
            Self::VerticalFused => (false, true),
        }
    }
}

/// Encodes the fused vertical kernel on `decoded` without the height gate
/// that keeps decode from routing a one-row window to it.
fn encode_vertical_fused(
    encoder: &ComputeCommandEncoderRef,
    kernels: &crate::engine::runtime::DecodeKernels,
    decoded: &Buffer,
    window: Dwt53Window,
    batch: usize,
) {
    if batch == 1 {
        encoder.setComputePipelineState(&kernels.idwt_reversible53_vertical_fused);
        encoder.set_bytes::<J2kIdwtSingleDecompositionParams>(1, &single_params(window));
    } else {
        encoder.setComputePipelineState(&kernels.idwt_reversible53_vertical_fused_batched);
        encoder.set_bytes::<J2kRepeatedIdwtSingleDecompositionParams>(
            1,
            &repeated_params(window, batch),
        );
    }
    encoder.set_buffer(0, Some(decoded), 0);
    encoder.dispatchThreadgroups_threadsPerThreadgroup(
        j2k_metal_support::mtl_size(
            u64::from(window.width.div_ceil(IDWT53_COLS_PER_GROUP)),
            1,
            batch as u64,
        ),
        j2k_metal_support::mtl_size(
            u64::from(IDWT53_COLS_PER_GROUP),
            u64::from(IDWT53_COL_ROW_THREADS),
            1,
        ),
    );
}

/// Runs `route` on `window` for each item of `items`, through the
/// single-image kernels when there is one item and the batched kernels
/// otherwise, and returns the decoded planes.
fn run(
    runtime: &MetalRuntime,
    window: Dwt53Window,
    items: &[[Vec<i32>; 4]],
    route: Route,
) -> Vec<f32> {
    let kernels = runtime.decode().expect("decode kernels");
    let band_buffers: [Buffer; 4] = std::array::from_fn(|band| {
        let samples = items.iter().flat_map(|bands| bands[band].clone());
        buffer_of(runtime, &dwt53_samples_f32(&samples.collect::<Vec<_>>()))
    });
    let plane_len = (window.width * window.height) as usize;
    let initial = match route {
        Route::VerticalFused => items
            .iter()
            .flat_map(|bands| dwt53_samples_f32(&dwt53_interleave(window, bands)))
            .collect(),
        _ => vec![f32::NAN; plane_len * items.len()],
    };
    let decoded = buffer_of(runtime, &initial);
    let sub_bands = IdwtSubBandBuffers {
        ll: &band_buffers[0],
        ll_offset: 0,
        hl: &band_buffers[1],
        hl_offset: 0,
        lh: &band_buffers[2],
        lh_offset: 0,
        hh: &band_buffers[3],
        hh_offset: 0,
    };
    let single = SingleIdwtDispatch {
        kernels,
        sub_bands,
        params: single_params(window),
        decoded: &decoded,
        decoded_offset: 0,
    };
    let repeated = RepeatedIdwtDispatch {
        kernels,
        sub_bands,
        params: repeated_params(window, items.len()),
        decoded: &decoded,
    };

    let command = new_command_buffer(&runtime.queue).expect("command buffer");
    let encoder = new_compute_command_encoder(&command).expect("compute encoder");
    match (route, items.len()) {
        (Route::Production, 1) => {
            dispatch_reversible53_single_decomposition_buffers_in_encoder_with_offsets(
                &encoder, single,
            );
        }
        (Route::Production, _) => {
            dispatch_reversible53_repeated_buffers_in_encoder_with_offsets(&encoder, repeated);
        }
        (Route::InterleaveHorizontalFused, 1) => {
            dispatch_reversible53_interleave_horizontal_fused_single(&encoder, single);
        }
        (Route::InterleaveHorizontalFused, _) => {
            dispatch_reversible53_interleave_horizontal_fused_repeated(&encoder, repeated);
        }
        (Route::VerticalFused, batch) => {
            encode_vertical_fused(&encoder, kernels, &decoded, window, batch);
        }
    }
    encoder.endEncoding();
    commit_and_wait_metal(&command).expect("completed IDWT");
    checked_buffer_slice::<f32>(&decoded, plane_len * items.len(), "IDWT output")
        .expect("IDWT readback")
}

/// The mismatches of `route` on `window`, one line per batch size.
fn mismatches(runtime: &MetalRuntime, window: Dwt53Window, route: Route) -> Vec<String> {
    let (horizontal, vertical) = route.axes();
    let mut found = Vec::new();
    for batch in [1, BATCH] {
        let items = (0..batch)
            .map(|instance| dwt53_bands(window, instance))
            .collect::<Vec<_>>();
        let expected = items
            .iter()
            .flat_map(|bands| {
                let mut plane = dwt53_interleave(window, bands);
                inverse_dwt53(window, &mut plane, horizontal, vertical);
                dwt53_samples_f32(&plane)
            })
            .collect::<Vec<_>>();
        let actual = run(runtime, window, &items, route);
        if actual != expected {
            found.push(format!(
                "{route:?} {window:?} batch {batch}: {actual:?}, expected {expected:?}"
            ));
        }
    }
    found
}

#[test]
fn reversible53_lone_samples_match_integer_reference() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }
    let found = with_runtime(|runtime| {
        let mut found = Vec::new();
        for window in DWT53_LONE_SAMPLE_WINDOWS.into_iter().chain(LONG) {
            found.extend(mismatches(runtime, window, Route::Production));
            // Decode routes the fused kernels only to windows wider or taller
            // than one tile, but each kernel handles a lone sample itself.
            if window.width == 1 {
                found.extend(mismatches(
                    runtime,
                    window,
                    Route::InterleaveHorizontalFused,
                ));
            }
            if window.height == 1 {
                found.extend(mismatches(runtime, window, Route::VerticalFused));
            }
        }
        Ok(found)
    })
    .expect("Metal runtime");
    assert!(found.is_empty(), "{}", found.join("\n"));
}
