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
