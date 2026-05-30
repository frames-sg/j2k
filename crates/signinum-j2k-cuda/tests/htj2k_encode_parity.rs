// Stage-level byte/value-exact parity tests: CUDA forward-encode stages vs.
// native CPU reference.
//
// Every test gates on `runtime_required()` and returns early when
// `SIGNINUM_REQUIRE_CUDA_RUNTIME` is absent.  They run for real only on the
// CI CUDA runner.

#[cfg(feature = "cuda-runtime")]
use signinum_cuda_runtime::{CudaContext, CudaJ2kQuantizeJob};
#[cfg(feature = "cuda-runtime")]
use signinum_j2k_native::{
    deinterleave_reference, forward_dwt53_reference, forward_rct_reference,
    quantize_reversible_reference,
};

// ---------------------------------------------------------------------------
// Imports needed only by the facade parity matrix test.
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda-runtime")]
use signinum_j2k::{
    encode_j2k_lossless, EncodeBackendPreference, J2kBlockCodingMode, J2kEncodeValidation,
    J2kLosslessEncodeOptions, J2kLosslessSamples,
};
#[cfg(feature = "cuda-runtime")]
use signinum_j2k_cuda::encode_j2k_lossless_with_cuda;
#[cfg(feature = "cuda-runtime")]
use signinum_j2k_native::{DecodeSettings, Image};

// ---------------------------------------------------------------------------
// Gating helper — mirrors the pattern in htj2k_cuda_kernels.rs (line 24–26).
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda-runtime")]
fn runtime_required() -> bool {
    std::env::var_os("SIGNINUM_REQUIRE_CUDA_RUNTIME").is_some()
}

// ---------------------------------------------------------------------------
// DWT sub-band extraction helper
//
// The CUDA `j2k_forward_dwt53` output stores coefficients in the same
// "polyphase-flat" layout used by the internal CPU DWT: the full image stride
// is preserved, low-frequency rows come first (rows 0..low_height), then
// high-frequency rows (rows low_height..height), and within each row the
// low-frequency columns come first (cols 0..low_width) followed by the
// high-frequency columns (cols low_width..width).
//
// The native `forward_dwt53_reference` returns separately allocated sub-band
// vecs (ll, hl, lh, hh) with their own row-major storage.  This helper
// extracts those four sub-bands from the flat CUDA buffer so the comparison
// is apples-to-apples.
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda-runtime")]
fn extract_subbands_from_flat(
    flat: &[f32],
    full_width: u32,
    low_width: u32,
    low_height: u32,
    high_width: u32,
    high_height: u32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let fw = full_width as usize;
    let lw = low_width as usize;
    let lh = low_height as usize;
    let hw = high_width as usize;
    let hh = high_height as usize;

    let mut ll = Vec::with_capacity(lw * lh);
    let mut hl = Vec::with_capacity(hw * lh);
    let mut lh_out = Vec::with_capacity(lw * hh);
    let mut hh_out = Vec::with_capacity(hw * hh);

    // Low-frequency rows (rows 0..lh) → LL (cols 0..lw) and HL (cols lw..lw+hw)
    for row in 0..lh {
        let row_start = row * fw;
        ll.extend_from_slice(&flat[row_start..row_start + lw]);
        hl.extend_from_slice(&flat[row_start + lw..row_start + lw + hw]);
    }

    // High-frequency rows (rows lh..lh+hh) → LH (cols 0..lw) and HH (cols lw..lw+hw)
    for row in 0..hh {
        let row_start = (lh + row) * fw;
        lh_out.extend_from_slice(&flat[row_start..row_start + lw]);
        hh_out.extend_from_slice(&flat[row_start + lw..row_start + lw + hw]);
    }

    (ll, hl, lh_out, hh_out)
}

// ---------------------------------------------------------------------------
// Test 1: forward DWT 5/3
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda-runtime")]
#[test]
fn cuda_forward_dwt53_matches_native_reference_when_required() {
    if !runtime_required() {
        return;
    }

    // 40×24, 2 decomposition levels; deterministic signed-ish integer samples.
    let width: u32 = 40;
    let height: u32 = 24;
    let num_levels: u8 = 2;
    let samples: Vec<f32> = (0u32..width * height)
        .map(|i| {
            // Produce values in [-128, 127] so the lossless path keeps integer coefficients.
            // The modulus bounds the value to [0, 255] which fits in i16, so f32::from is lossless.
            let v = i16::try_from((i * 7 + 3) % 256).expect("sample fits in i16") - 128;
            f32::from(v)
        })
        .collect();

    // Native CPU reference
    let native = forward_dwt53_reference(&samples, width, height, num_levels);

    // CUDA
    let context = CudaContext::system_default().expect("CUDA context");
    let cuda_out = context
        .j2k_forward_dwt53(&samples, width, height, num_levels)
        .expect("CUDA forward DWT 5/3");

    assert_eq!(cuda_out.levels().len(), native.levels.len(), "level count");
    assert_eq!(
        cuda_out.ll_dimensions(),
        (native.ll_width, native.ll_height),
        "LL dimensions"
    );

    // Walk each decomposition level from coarsest to finest and compare sub-bands.
    // After each level the active region shrinks to (low_width × low_height).
    // We iterate in finest-first order (index 0 is the outermost DWT level).
    //
    // Reshape note: the CUDA flat plane stores all coefficients in the original
    // image stride; subbands are addressed by quadrant (rows × cols).  The
    // native reference keeps HL/LH/HH as separate vecs per level, with the LL
    // band only available for the deepest level in `native.ll`.  We compare
    // HL/LH/HH at every level and LL only at the final (deepest) level.
    let mut current_width = width;

    for (level_idx, (cuda_level, native_level)) in cuda_out
        .levels()
        .iter()
        .zip(native.levels.iter())
        .enumerate()
    {
        let low_width = cuda_level.low_width;
        let low_height = cuda_level.low_height;
        let high_width = cuda_level.high_width;
        let high_height = cuda_level.high_height;

        // Extract HL/LH/HH from the flat CUDA plane; LL is skipped here because
        // the native reference only exposes LL at the deepest level.
        let (_, cuda_horiz, cuda_vert, cuda_diag) = extract_subbands_from_flat(
            cuda_out.transformed(),
            current_width,
            low_width,
            low_height,
            high_width,
            high_height,
        );

        // HL = horizontal high-pass, LH = vertical high-pass, HH = diagonal high-pass.
        assert_eq!(cuda_horiz, native_level.hl, "level {level_idx} HL mismatch");
        assert_eq!(cuda_vert, native_level.lh, "level {level_idx} LH mismatch");
        assert_eq!(cuda_diag, native_level.hh, "level {level_idx} HH mismatch");

        // Advance current region to the low-pass output for the next level.
        current_width = low_width;
    }

    // Compare the deepest LL sub-band (native.ll_width × native.ll_height).
    let (cuda_ll_final, _, _, _) = extract_subbands_from_flat(
        cuda_out.transformed(),
        current_width,
        native.ll_width,
        native.ll_height,
        0,
        0,
    );
    assert_eq!(cuda_ll_final, native.ll, "final LL mismatch");
}

// ---------------------------------------------------------------------------
// Test 2: forward RCT
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda-runtime")]
#[test]
fn cuda_forward_rct_matches_native_reference_when_required() {
    if !runtime_required() {
        return;
    }

    // Three 4×3 = 12-sample planes with RGB-like integer values.
    let plane0: Vec<f32> = (0u8..12).map(|i| f32::from(i) * 10.0).collect();
    let plane1: Vec<f32> = (0u8..12).map(|i| f32::from(i) * 5.0 + 20.0).collect();
    let plane2: Vec<f32> = (0u8..12).map(|i| 120.0 - f32::from(i) * 8.0).collect();

    // Native CPU reference — takes ownership, returns transformed planes.
    let native = forward_rct_reference(vec![plane0.clone(), plane1.clone(), plane2.clone()]);

    // CUDA — j2k_forward_rct modifies slices in-place (lib.rs line 2532).
    let mut cuda_plane0 = plane0.clone();
    let mut cuda_plane1 = plane1.clone();
    let mut cuda_plane2 = plane2.clone();
    let context = CudaContext::system_default().expect("CUDA context");
    let stats = context
        .j2k_forward_rct(&mut cuda_plane0, &mut cuda_plane1, &mut cuda_plane2)
        .expect("CUDA forward RCT");

    assert_eq!(stats.kernel_dispatches(), 1);
    assert_eq!(cuda_plane0, native[0], "plane 0 (Y) mismatch");
    assert_eq!(cuda_plane1, native[1], "plane 1 (Cb) mismatch");
    assert_eq!(cuda_plane2, native[2], "plane 2 (Cr) mismatch");
}

// ---------------------------------------------------------------------------
// Test 3: reversible sub-band quantization
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda-runtime")]
#[test]
fn cuda_quantize_reversible_matches_native_reference_when_required() {
    if !runtime_required() {
        return;
    }

    // Small sub-band with a mix of positive, negative, and near-zero values.
    let coefficients: Vec<f32> = vec![
        0.0, 3.7, -8.2, 1.0, -0.5, 10.0, -1.5, 2.5, -3.0, 7.0, -4.4, 0.9, 5.6, -6.1, 1.2, -2.3,
    ];
    let step_exponent: u16 = 8;
    let step_mantissa: u16 = 0;
    let range_bits: u8 = 8;

    // Native CPU reference (reversible = true → integer rounding).
    let native = quantize_reversible_reference(
        &coefficients,
        step_exponent,
        step_mantissa,
        range_bits,
        true,
    );

    // CUDA (lib.rs line 2936).
    let context = CudaContext::system_default().expect("CUDA context");
    let cuda_out = context
        .j2k_quantize_subband(
            &coefficients,
            CudaJ2kQuantizeJob {
                step_exponent,
                step_mantissa,
                range_bits,
                reversible: true,
            },
        )
        .expect("CUDA reversible quantize");

    assert_eq!(cuda_out.execution().kernel_dispatches(), 1);
    assert_eq!(cuda_out.coefficients(), native.as_slice());
}

// ---------------------------------------------------------------------------
// Test 4: pixel deinterleave (covers 8-bit unsigned, 8-bit signed, 16-bit unsigned)
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda-runtime")]
#[test]
fn cuda_deinterleave_matches_native_reference_when_required() {
    if !runtime_required() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");

    // --- 4a: 8-bit unsigned RGB, 4 pixels ---
    {
        let pixels: Vec<u8> = vec![
            10, 20, 30, // pixel 0
            40, 50, 60, // pixel 1
            70, 80, 90, // pixel 2
            100, 110, 120, // pixel 3
        ];
        let num_pixels = 4usize;
        let num_components = 3u8;
        let bit_depth = 8u8;
        let signed = false;

        let native = deinterleave_reference(&pixels, num_pixels, num_components, bit_depth, signed);
        let cuda_out = context
            .j2k_deinterleave_to_f32(&pixels, num_pixels, num_components, bit_depth, signed)
            .expect("CUDA deinterleave 8-bit unsigned RGB");

        assert_eq!(
            cuda_out.execution().kernel_dispatches(),
            1,
            "8-bit unsigned kernel_dispatches"
        );
        // components() returns one Vec<f32> per component (lib.rs line 4550).
        assert_eq!(
            cuda_out.components(),
            native.as_slice(),
            "8-bit unsigned deinterleave mismatch"
        );
    }

    // --- 4b: 8-bit signed single-component grayscale, 6 pixels ---
    {
        let pixels: Vec<u8> = (0u8..6).collect();
        let num_pixels = 6usize;
        let num_components = 1u8;
        let bit_depth = 8u8;
        let signed = true;

        let native = deinterleave_reference(&pixels, num_pixels, num_components, bit_depth, signed);
        let cuda_out = context
            .j2k_deinterleave_to_f32(&pixels, num_pixels, num_components, bit_depth, signed)
            .expect("CUDA deinterleave 8-bit signed gray");

        assert_eq!(
            cuda_out.components(),
            native.as_slice(),
            "8-bit signed grayscale deinterleave mismatch"
        );
    }

    // --- 4c: 16-bit unsigned RGB, 2 pixels (little-endian pairs) ---
    {
        // Pack 2 pixels × 3 components × 2 bytes = 12 bytes.
        let values: &[u16] = &[0x0010, 0x0080, 0x00FF, 0x0100, 0x0800, 0x0FFF];
        let mut pixels: Vec<u8> = Vec::with_capacity(values.len() * 2);
        for v in values {
            pixels.extend_from_slice(&v.to_ne_bytes());
        }
        let num_pixels = 2usize;
        let num_components = 3u8;
        let bit_depth = 16u8;
        let signed = false;

        let native = deinterleave_reference(&pixels, num_pixels, num_components, bit_depth, signed);
        let cuda_out = context
            .j2k_deinterleave_to_f32(&pixels, num_pixels, num_components, bit_depth, signed)
            .expect("CUDA deinterleave 16-bit unsigned RGB");

        assert_eq!(
            cuda_out.components(),
            native.as_slice(),
            "16-bit unsigned deinterleave mismatch"
        );
    }

    // --- 4d: 16-bit signed single-component, 4 pixels ---
    {
        let values: &[i16] = &[-32768, -1, 0, 32767];
        let mut pixels: Vec<u8> = Vec::with_capacity(values.len() * 2);
        for v in values {
            pixels.extend_from_slice(&v.to_ne_bytes());
        }
        let num_pixels = 4usize;
        let num_components = 1u8;
        let bit_depth = 16u8;
        let signed = true;

        let native = deinterleave_reference(&pixels, num_pixels, num_components, bit_depth, signed);
        let cuda_out = context
            .j2k_deinterleave_to_f32(&pixels, num_pixels, num_components, bit_depth, signed)
            .expect("CUDA deinterleave 16-bit signed gray");

        assert_eq!(
            cuda_out.components(),
            native.as_slice(),
            "16-bit signed deinterleave mismatch"
        );
    }
}

// ---------------------------------------------------------------------------
// Tripwire (ungated — runs on every host, no GPU required)
// ---------------------------------------------------------------------------
//
// If CI sets SIGNINUM_REQUIRE_CUDA_RUNTIME but the `cuda-runtime` feature was
// not compiled in, every gated parity test early-returns and false-greens.
// This test is NOT inside any `#[cfg(feature = "cuda-runtime")]` block so it
// is always compiled and always executed regardless of feature flags.

#[test]
fn cuda_runtime_required_implies_feature_compiled() {
    // Fail-closed: if CI asserts the CUDA runtime is required but the cuda-runtime
    // feature was not compiled in, every gated parity test would silently early-return
    // and masquerade as green. Make that misconfiguration a hard failure instead.
    if std::env::var_os("SIGNINUM_REQUIRE_CUDA_RUNTIME").is_some() {
        assert!(
            cfg!(feature = "cuda-runtime"),
            "SIGNINUM_REQUIRE_CUDA_RUNTIME is set but the cuda-runtime feature is not compiled — \
             gated CUDA parity tests would silently skip and false-green"
        );
    }
}

// ---------------------------------------------------------------------------
// Phase-1 ACCEPTANCE GATE
// Test 5: CUDA facade byte-exact parity matrix vs. CPU reference
// ---------------------------------------------------------------------------
//
// CPU reference rationale
// -----------------------
// `encode_j2k_lossless_with_cuda` calls `strict_cuda_encode_options(*options)`
// (crates/signinum-j2k-cuda/src/encode.rs:88–91), which sets
// `backend = RequireDevice` and otherwise leaves all other fields untouched.
// It then delegates to `signinum_j2k::encode_j2k_lossless_with_accelerator`
// (encode.rs:40–45), which internally calls `native_lossless_options` to build
// the `EncodeOptions` for the native encoder (signinum-j2k/src/encode.rs:377–392).
//
// The CPU reference here is `signinum_j2k::encode_j2k_lossless(samples, &opts)`
// with `backend = CpuOnly` and all other option fields identical to what the
// caller passed to the CUDA facade.  `encode_j2k_lossless` calls `encode_cpu`
// (encode.rs:254–270) which calls the same `native_lossless_options` helper,
// producing the same `EncodeOptions` (same reversible transform, block coding
// mode, progression order, decomposition levels, write_tlm flag, use_mct flag).
// The only difference is the dispatch path: CUDA uses the GPU accelerator for
// compute stages; CPU uses the scalar fallback.  When the GPU produces a
// bit-identical codestream for supported configurations, `bytes_cuda ==
// bytes_native` must hold.
//
// Validation policy for the CPU call is `External` to avoid double round-trip
// overhead in CI; the CUDA facade runs its own validation internally.

// Cell descriptor used for matrix iteration and assertion messages.
#[cfg(feature = "cuda-runtime")]
#[derive(Debug, Clone, Copy)]
struct ParityCell {
    w: u32,
    h: u32,
    comps: u8,
    depth: u8,
    signed: bool,
    levels: u8,
}

/// Build a deterministic byte buffer for the given cell geometry.
///
/// For 8-bit samples each byte is a single sample.
/// For 16-bit samples each sample is two bytes (native-endian u16), packed
/// interleaved (e.g., `lo_byte`, `hi_byte` per sample per component).
#[cfg(feature = "cuda-runtime")]
fn synthesize_pixels(cell: &ParityCell) -> Vec<u8> {
    let npixels = cell.w as usize * cell.h as usize;
    let nsamples = npixels * cell.comps as usize;

    if cell.depth <= 8 {
        // 8-bit: one byte per sample; deterministic from index.
        // The modulus is exactly 256, so truncation is intentional.
        #[allow(clippy::cast_possible_truncation)]
        (0..nsamples).map(|i| ((i * 31 + 17) % 256) as u8).collect()
    } else {
        // 16-bit: two bytes per sample (native-endian u16).
        let max_val: u32 = (1u32 << cell.depth) - 1;
        let mut buf = Vec::with_capacity(nsamples * 2);
        for i in 0..nsamples {
            // i < w*h*comps ≤ 64*48*4 = 12288, safely fits in u32.
            #[allow(clippy::cast_possible_truncation)]
            let i32 = i as u32;
            // Result is bounded by max_val < 2^16, so truncation to u16 is safe.
            #[allow(clippy::cast_possible_truncation)]
            let v = ((i32 * 1_000_003 + 7) % (max_val + 1)) as u16;
            buf.extend_from_slice(&v.to_ne_bytes());
        }
        buf
    }
}

/// Build the `J2kLosslessEncodeOptions` for a given cell.
///
/// We use HTJ2K block coding and LRCP progression (matching the strict CUDA
/// facade's default contract), and pass the cell's `levels` as an explicit
/// `max_decomposition_levels` request.  The reversible transform is left at
/// the facade default (`Rct53` for 3-component; the native encoder also applies
/// it to 1-component via a no-op path that does not change the codestream bytes).
#[cfg(feature = "cuda-runtime")]
fn cell_encode_options(cell: &ParityCell) -> J2kLosslessEncodeOptions {
    J2kLosslessEncodeOptions::default()
        .with_block_coding_mode(J2kBlockCodingMode::HighThroughput)
        .with_max_decomposition_levels(Some(cell.levels))
        .with_validation(J2kEncodeValidation::External)
}

// Matrix constants live at module scope to avoid the items_after_statements lint.
// comps ∈ {1, 3, 4} (2-component is OUT OF SCOPE — native decoder rejects it).
// 4-component resident MCT is implemented (RCT on planes 0-2, passthrough 4);
// these cells are now asserted byte-exact like every other supported cell.
#[cfg(feature = "cuda-runtime")]
const MATRIX_COMPS: &[u8] = &[1, 3, 4];
#[cfg(feature = "cuda-runtime")]
const MATRIX_DEPTHS: &[u8] = &[8, 16];
#[cfg(feature = "cuda-runtime")]
const MATRIX_SIGNED: &[bool] = &[false, true];
#[cfg(feature = "cuda-runtime")]
const MATRIX_LEVELS: &[u8] = &[0, 1, 3];

#[cfg(feature = "cuda-runtime")]
#[test]
fn cuda_facade_byte_matches_native_across_matrix_when_required() {
    if !runtime_required() {
        return;
    }

    let w: u32 = 64;
    let h: u32 = 48;

    let mut failures: Vec<String> = Vec::new();

    for &comps in MATRIX_COMPS {
        for &depth in MATRIX_DEPTHS {
            for &signed in MATRIX_SIGNED {
                for &levels in MATRIX_LEVELS {
                    let cell = ParityCell {
                        w,
                        h,
                        comps,
                        depth,
                        signed,
                        levels,
                    };
                    let pixels = synthesize_pixels(&cell);
                    let opts = cell_encode_options(&cell);

                    // Build J2kLosslessSamples through the real public
                    // constructor for every cell, including 4-component:
                    // J2kLosslessSamples::new now accepts comps ∈ {1, 3, 4}
                    // (2-component stays rejected). The CUDA facade supports
                    // 4-component resident MCT end-to-end.
                    let samples = J2kLosslessSamples::new(
                        pixels.as_slice(),
                        cell.w,
                        cell.h,
                        cell.comps,
                        cell.depth,
                        cell.signed,
                    )
                    .unwrap_or_else(|err| {
                        panic!("cell={cell:?}: J2kLosslessSamples::new rejected an in-scope cell: {err}")
                    });

                    // --- CUDA encode ---
                    let cuda_result = encode_j2k_lossless_with_cuda(samples, &opts);

                    // --- CPU reference encode (configuration-identical) ---
                    // Uses encode_j2k_lossless with CpuOnly backend so it takes
                    // the identical native_lossless_options path as the CUDA facade
                    // (signinum-j2k/src/encode.rs:341–355 via encode_cpu).
                    let cpu_opts = opts.with_backend(EncodeBackendPreference::CpuOnly);
                    let cpu_result = encode_j2k_lossless(samples, &cpu_opts);

                    match (cuda_result, cpu_result) {
                        (Err(cuda_err), Err(cpu_err)) => {
                            // Every matrix cell (comps ∈ {1, 3, 4}) is in scope and
                            // must encode successfully on both paths.
                            failures.push(format!(
                                "cell={cell:?}: both encoders rejected an in-scope cell: \
                                 cuda_err={cuda_err} cpu_err={cpu_err}"
                            ));
                        }
                        (Err(cuda_err), Ok(_)) => {
                            // CUDA must dispatch for every in-scope cell, including
                            // 4-component (resident MCT lands in P1-T7).
                            failures.push(format!(
                                "cell={cell:?}: CUDA encode failed but CPU succeeded: \
                                 {cuda_err}"
                            ));
                        }
                        (Ok(_), Err(cpu_err)) => {
                            failures.push(format!(
                                "cell={cell:?}: CPU encode failed but CUDA succeeded: \
                                 {cpu_err}"
                            ));
                        }
                        (Ok(bytes_cuda), Ok(bytes_native)) => {
                            // --- Byte-exact codestream parity assertion ---
                            if bytes_cuda.codestream != bytes_native.codestream {
                                failures.push(format!(
                                    "cell={cell:?}: codestream byte mismatch \
                                     (cuda={} bytes, cpu={} bytes)",
                                    bytes_cuda.codestream.len(),
                                    bytes_native.codestream.len()
                                ));
                                continue;
                            }

                            // --- Round-trip decode of the CUDA codestream ---
                            // Decode with signinum_j2k_native::Image::decode_native
                            // and compare the raw pixel bytes to the original input.
                            let image = match Image::new(
                                &bytes_cuda.codestream,
                                &DecodeSettings::default(),
                            ) {
                                Ok(img) => img,
                                Err(e) => {
                                    failures.push(format!(
                                        "cell={cell:?}: CUDA codestream parse failed: {e}"
                                    ));
                                    continue;
                                }
                            };

                            let decoded = match image.decode_native() {
                                Ok(bmp) => bmp,
                                Err(e) => {
                                    failures.push(format!(
                                        "cell={cell:?}: CUDA codestream decode failed: {e}"
                                    ));
                                    continue;
                                }
                            };

                            if decoded.data != pixels {
                                failures.push(format!(
                                    "cell={cell:?}: round-trip pixel mismatch \
                                     (decoded={} bytes, original={} bytes)",
                                    decoded.data.len(),
                                    pixels.len()
                                ));
                            }
                        }
                    }
                }
            }
        }
    }

    assert!(
        failures.is_empty(),
        "facade parity matrix failures:\n{}",
        failures.join("\n")
    );
}

// ---------------------------------------------------------------------------
// Test 6: subsampling != (1,1) rejection via the accelerator hook (runner-gated)
//
// `cuda_encode_htj2k_tile_body` now returns a typed Err — not Ok(None) — for
// inputs with component subsampling factors != (1, 1).  This test verifies the
// rejection at the hook level via `J2kEncodeStageAccelerator::encode_htj2k_tile`
// on the CUDA runner where the subsampling check actually executes.
//
// NOTE: Subsampling != (1,1) is NOT expressible through the public lossless
// facade (`encode_j2k_lossless_with_cuda` / `J2kLosslessEncodeOptions`):
// `native_lossless_options` never sets `EncodeOptions::component_sampling`,
// which defaults to `None` → all (1,1).  The rejection here is a defense-in-
// depth contract within `cuda_encode_htj2k_tile_body`, exercised via the
// lower-level hook to confirm the guard exists.
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda-runtime")]
#[test]
fn cuda_htj2k_tile_encode_hook_rejects_subsampling_with_typed_err_when_runtime_required() {
    use signinum_j2k_cuda::CudaEncodeStageAccelerator;
    use signinum_j2k_native::{
        J2kEncodeStageAccelerator as _, J2kHtj2kTileEncodeJob, J2kPacketizationProgressionOrder,
    };

    if !runtime_required() {
        return;
    }

    // Minimal 4x4 grayscale HTJ2K tile job with subsampling (2,1) on the single
    // component — this is the only path that triggers the subsampling guard in
    // cuda_encode_htj2k_tile_body.
    let pixels: Vec<u8> = (0u8..16).collect();
    let quantization_steps = [(8u16, 0u16)]; // num_decomposition_levels=0 → 1 step
    let sampling = [(2u8, 1u8)]; // non-unit subsampling for component 0

    let mut accelerator = CudaEncodeStageAccelerator::default();
    let result = accelerator.encode_htj2k_tile(J2kHtj2kTileEncodeJob {
        pixels: &pixels,
        width: 4,
        height: 4,
        num_components: 1,
        bit_depth: 8,
        signed: false,
        reversible: true,
        use_mct: false,
        num_decomposition_levels: 0,
        guard_bits: 1,
        code_block_width: 4,
        code_block_height: 4,
        quantization_steps: &quantization_steps,
        component_sampling: &sampling,
        progression_order: J2kPacketizationProgressionOrder::Lrcp,
    });

    let err = result.expect_err(
        "encode_htj2k_tile must return Err for subsampling != (1,1) when CUDA is available",
    );
    assert!(
        err.contains("subsampling"),
        "typed rejection must mention subsampling, got: {err}"
    );
}

// ---------------------------------------------------------------------------
// Test 7 (CPU-only, no runtime gate): assert that the lossless facade's public
// types cannot express subsampling != (1,1).
//
// `J2kLosslessEncodeOptions` has no subsampling field.  `native_lossless_options`
// always leaves `EncodeOptions::component_sampling` as `None`, which the native
// encoder resolves to all-(1,1).  This structural invariant means the subsampling
// rejection in `cuda_encode_htj2k_tile_body` is unreachable through the facade.
//
// This test encodes a 1-component in-scope cell and asserts the call succeeds —
// proving no silent Ok(None) fallback occurs for in-scope inputs.
// ---------------------------------------------------------------------------

#[test]
fn lossless_facade_in_scope_input_never_hits_ok_none_fallback() {
    use signinum_core::{BackendKind, CodecError as _};
    use signinum_j2k::{
        J2kBlockCodingMode, J2kEncodeValidation, J2kLosslessEncodeOptions, J2kLosslessSamples,
    };
    use signinum_j2k_cuda::encode_j2k_lossless_with_cuda;

    // Small 8x8 single-component 8-bit input — minimal in-scope cell.
    let pixels: Vec<u8> = (0u8..64).collect();
    let samples =
        J2kLosslessSamples::new(&pixels, 8, 8, 1, 8, false).expect("valid in-scope samples");
    let options = J2kLosslessEncodeOptions::default()
        .with_block_coding_mode(J2kBlockCodingMode::HighThroughput)
        .with_max_decomposition_levels(Some(0))
        .with_validation(J2kEncodeValidation::External);

    let result = encode_j2k_lossless_with_cuda(samples, &options);

    // Without CUDA, the facade must return a typed Err (not Ok with CPU backend),
    // because strict_cuda_encode_options forces RequireDevice and the missing
    // deinterleave dispatch triggers the unsupported-backend error.
    // With CUDA, the call succeeds.
    // Either way, the result must NOT be Ok with backend == Cpu (silent fallback).
    match result {
        Ok(encoded) => {
            assert_eq!(
                encoded.backend,
                BackendKind::Cuda,
                "encode_j2k_lossless_with_cuda must not silently fall back to CPU backend for in-scope inputs"
            );
        }
        Err(err) => {
            // Strict encode error is expected when CUDA is unavailable.
            assert!(
                err.is_unsupported(),
                "rejection for an in-scope input must be a typed unsupported error, not an internal panic or silent fallback: {err}"
            );
        }
    }
}
