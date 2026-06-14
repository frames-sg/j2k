// Stage-level byte/value-exact parity tests: CUDA forward-encode stages vs.
// native CPU reference.
//
// Every test gates on `cuda_runtime_required()` and returns early when
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
use signinum_j2k_cuda::{cuda_dwt53_output_to_j2k_for_test, encode_j2k_lossless_with_cuda};
#[cfg(feature = "cuda-runtime")]
use signinum_j2k_native::{DecodeSettings, Image};

// ---------------------------------------------------------------------------
// Gating helper — mirrors the pattern in htj2k_cuda_kernels.rs (line 24–26).
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda-runtime")]
use signinum_test_support::cuda_runtime_required;

// ---------------------------------------------------------------------------
// DWT sub-band reshape
//
// The CUDA `j2k_forward_dwt53` output stores coefficients in a single
// "polyphase-flat" plane that preserves the original image stride.  For a
// MULTI-level DWT the deeper levels are nested inside the LL quadrant of the
// previous level, so each sub-band must be addressed with an explicit (x0, y0)
// origin — not anchored at the buffer origin.  Additionally, CUDA emits its
// `levels()` finest→coarsest, while native `forward_dwt53_reference` returns
// them coarsest→finest (it calls `levels.reverse()`), so a naive same-index
// zip compares mismatched levels.
//
// Rather than re-derive this geometry in the test (the source of the original
// bug), we reuse the EXACT production conversion `cuda_dwt53_output_to_j2k`
// (re-exported as `cuda_dwt53_output_to_j2k_for_test`).  It extracts each band
// with explicit offsets (HL at (low_width, 0), LH at (0, low_height), HH at
// (low_width, low_height)) and reverses the level order, producing a
// `J2kForwardDwt53Output` directly comparable to `forward_dwt53_reference`.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Test 1: forward DWT 5/3
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda-runtime")]
fn assert_cuda_forward_dwt53_matches_native(width: u32, height: u32, num_levels: u8) {
    // Deterministic signed-ish integer samples in [-128, 127] so the lossless
    // path keeps integer coefficients (f32::from is lossless for this range).
    let samples: Vec<f32> = (0u32..width * height)
        .map(|i| {
            let v = i16::try_from((i * 7 + 3) % 256).expect("sample fits in i16") - 128;
            f32::from(v)
        })
        .collect();

    // Native CPU reference (levels ordered coarsest→finest; LL at deepest level).
    let native = forward_dwt53_reference(&samples, width, height, num_levels);

    // CUDA forward DWT (levels ordered finest→coarsest in the flat plane).
    let context = CudaContext::system_default().expect("CUDA context");
    let cuda_out = context
        .j2k_forward_dwt53(&samples, width, height, num_levels)
        .expect("CUDA forward DWT 5/3");

    assert_eq!(
        cuda_out.levels().len(),
        native.levels.len(),
        "level count (levels={num_levels})"
    );
    assert_eq!(
        cuda_out.ll_dimensions(),
        (native.ll_width, native.ll_height),
        "LL dimensions (levels={num_levels})"
    );

    // Convert the flat CUDA plane to the native sub-band representation using
    // the SAME production reshape the encoder uses.  This handles the nested
    // per-level band offsets AND the finest→coarsest to coarsest→finest level
    // reversal, so the result lines up index-for-index with `native`.
    let cuda_as_native = cuda_dwt53_output_to_j2k_for_test(&cuda_out)
        .expect("CUDA DWT output -> native subband reshape");

    assert_eq!(
        cuda_as_native.levels.len(),
        native.levels.len(),
        "reshaped level count (levels={num_levels})"
    );
    assert_eq!(
        (cuda_as_native.ll_width, cuda_as_native.ll_height),
        (native.ll_width, native.ll_height),
        "reshaped LL dimensions (levels={num_levels})"
    );

    // Per-level HL/LH/HH parity (both now coarsest→finest at the same index).
    for (level_idx, (cuda_level, native_level)) in cuda_as_native
        .levels
        .iter()
        .zip(native.levels.iter())
        .enumerate()
    {
        assert_eq!(
            cuda_level.hl, native_level.hl,
            "levels={num_levels} level {level_idx} HL mismatch"
        );
        assert_eq!(
            cuda_level.lh, native_level.lh,
            "levels={num_levels} level {level_idx} LH mismatch"
        );
        assert_eq!(
            cuda_level.hh, native_level.hh,
            "levels={num_levels} level {level_idx} HH mismatch"
        );
    }

    // Deepest LL sub-band parity.
    assert_eq!(
        cuda_as_native.ll, native.ll,
        "levels={num_levels} final LL mismatch"
    );
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn cuda_forward_dwt53_matches_native_reference_when_required() {
    if !cuda_runtime_required() {
        return;
    }

    // num_levels = 2 exercises the multi-level (nested-band) path that the
    // original buggy reshape mishandled.  num_levels = 1 and 3 lock the level
    // ordering for the single-level and deeper-nesting cases respectively.
    // 40×24 stays divisible enough that every level keeps a non-degenerate
    // high-pass quadrant.
    assert_cuda_forward_dwt53_matches_native(40, 24, 1);
    assert_cuda_forward_dwt53_matches_native(40, 24, 2);
    assert_cuda_forward_dwt53_matches_native(40, 24, 3);
}

// ---------------------------------------------------------------------------
// Test 2: forward RCT
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda-runtime")]
#[test]
fn cuda_forward_rct_matches_native_reference_when_required() {
    if !cuda_runtime_required() {
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
    if !cuda_runtime_required() {
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
    if !cuda_runtime_required() {
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
            // Native deinterleave reads u16::from_le_bytes and the CUDA kernel
            // reads little-endian explicitly, so pack little-endian regardless
            // of host endianness.
            pixels.extend_from_slice(&v.to_le_bytes());
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
            // Little-endian to match native deinterleave and the CUDA kernel.
            pixels.extend_from_slice(&v.to_le_bytes());
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
    let cuda_runtime_required = std::env::var_os("SIGNINUM_REQUIRE_CUDA_RUNTIME").is_some();
    let cuda_runtime_feature_enabled = cfg!(feature = "cuda-runtime");
    assert!(
        !cuda_runtime_required || cuda_runtime_feature_enabled,
        "SIGNINUM_REQUIRE_CUDA_RUNTIME is set but the cuda-runtime feature is not compiled — \
         gated CUDA parity tests would silently skip and false-green"
    );
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
//
// Round-trip decode scope (signed vs unsigned)
// --------------------------------------------
// Every cell asserts CUDA-vs-native CODESTREAM byte parity (the Phase-1
// deliverable) and that the codestream parses and decodes. The additional
// byte-exact pixel round-trip — decode the codestream, compare to the original
// input — is asserted only for UNSIGNED components. The native DECODER does not
// reconstruct signed samples: it reads the SIZ Ssiz signed bit and ignores it
// (signinum-j2k-native/src/j2c/codestream.rs) and unconditionally re-applies the
// unsigned inverse DC level-shift, then clamps negatives
// (decode_native_with_context in signinum-j2k-native/src/lib.rs), so signed
// output is offset by +2^(depth-1). This is a SHARED native-decoder limitation —
// the CPU and Metal decode paths are affected identically because decode
// reconstruction is shared — and is independent of the CUDA ENCODER, whose
// codestream is byte-identical to native for signed and unsigned alike (asserted
// for every cell). Signed-source decode round-trip is not a supported contract
// in this repo (e.g. the recode path rejects signed sources outright). Tracked
// as a native-decoder non-goal in the public support policy.

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
/// For 16-bit samples each sample is two bytes (little-endian u16), packed
/// interleaved (e.g., `lo_byte`, `hi_byte` per sample per component).  The
/// native encoder reads 16-bit samples with `u16::from_le_bytes`, so packing
/// little-endian keeps the input interpretation host-endianness independent.
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
        // 16-bit: two bytes per sample (little-endian u16, matching the
        // native encoder's u16::from_le_bytes read).
        //
        // `modulus == 2^depth`; for the in-scope depths (≤ 16) this is ≤ 65536,
        // so every `value % modulus` lands in 0..2^16 and the u16 cast is exact.
        let modulus: u64 = 1u64 << cell.depth;
        let mut buf = Vec::with_capacity(nsamples * 2);
        for i in 0..nsamples {
            // Mix in u64: the intermediate product `idx * 1_000_003` exceeds
            // u32::MAX once idx ≥ 4295 (reached at comps ≥ 3, since
            // nsamples = w*h*comps), which previously panicked in debug builds
            // with "attempt to multiply with overflow". idx ≤ 12288 here, so
            // idx * 1_000_003 + 7 < 2^34 and cannot overflow u64.
            let idx = i as u64;
            // The modulo keeps the result < modulus ≤ 2^16, so the cast is exact.
            #[allow(clippy::cast_possible_truncation)]
            let v = ((idx * 1_000_003 + 7) % modulus) as u16;
            buf.extend_from_slice(&v.to_le_bytes());
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
    if !cuda_runtime_required() {
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
                            // Parse + decode for EVERY cell to prove the (byte-identical)
                            // codestream is well-formed and decodable. The byte-exact pixel
                            // comparison is asserted only for UNSIGNED components; see the
                            // "Round-trip decode scope" note in the acceptance-gate header
                            // above. In brief: the native decoder does not reconstruct
                            // signed samples — it reads the SIZ Ssiz signed bit but ignores
                            // it (signinum-j2k-native/src/j2c/codestream.rs) and
                            // unconditionally applies the unsigned inverse level-shift +
                            // clamp (decode_native_with_context in
                            // signinum-j2k-native/src/lib.rs), so signed output is offset by
                            // +2^(depth-1). That is a shared native-decoder limitation
                            // (CPU and Metal decode are affected identically) and is
                            // independent of the CUDA *encoder*, whose codestream byte-parity
                            // is asserted above for signed and unsigned alike.
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

                            if cell.signed {
                                // Signed values cannot round-trip through the native decoder
                                // (see note). Still require the decoded buffer to have the
                                // correct shape — confirms dimensions, component count, and
                                // bit depth survived encode + decode.
                                let expected_len = pixels.len();
                                if decoded.data.len() != expected_len {
                                    failures.push(format!(
                                        "cell={cell:?}: signed decode produced wrong-sized \
                                         buffer (decoded={} bytes, expected={expected_len})",
                                        decoded.data.len()
                                    ));
                                }
                            } else if decoded.data != pixels {
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
fn cuda_htj2k_tile_encode_hook_rejects_subsampling_with_typed_err_when_cuda_runtime_required() {
    use signinum_j2k::{
        J2kEncodeStageAccelerator as _, J2kHtj2kTileEncodeJob, J2kPacketizationProgressionOrder,
    };
    use signinum_j2k_cuda::CudaEncodeStageAccelerator;

    if !cuda_runtime_required() {
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
