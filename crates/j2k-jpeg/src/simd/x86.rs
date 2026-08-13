// SPDX-License-Identifier: MIT OR Apache-2.0

//! Exact-AVX2 capability token and safe kernel bridge.

use j2k_core::CpuFeatures;

/// Proof that AVX2 and the operating-system AVX register state are available.
///
/// This deliberately represents AVX2 alone, rather than
/// `fearless_simd::Avx2`, whose v0.7 contract is the wider x86-64-v3 feature
/// set. Keeping the token private prevents callers from forging capability
/// state while preserving the decoder's existing AVX2 acceleration envelope.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ExactAvx2 {
    _private: (),
}

impl ExactAvx2 {
    pub(crate) fn detect() -> Option<Self> {
        CpuFeatures::detect().avx2.then_some(Self { _private: () })
    }
}

/// Define a safe AVX2 entry point whose first argument is an unforgeable
/// [`ExactAvx2`] capability token.
macro_rules! exact_avx2_kernel {
    (
        $(#[$meta:meta])*
        $vis:vis fn $name:ident(
            $token:ident : ExactAvx2 $(, $arg:ident : $arg_ty:ty)* $(,)?
        ) $(-> $ret:ty)? {
            $($body:tt)*
        }
    ) => {
        $(#[$meta])*
        #[inline(always)]
        $vis fn $name(
            $token: $crate::simd::x86::ExactAvx2 $(, $arg: $arg_ty)*
        ) $(-> $ret)? {
            #[inline]
            #[target_feature(enable = "avx2")]
            fn kernel(
                $token: $crate::simd::x86::ExactAvx2 $(, $arg: $arg_ty)*
            ) $(-> $ret)? {
                let _ = $token;
                $($body)*
            }

            // SAFETY:
            // - Feature availability: `ExactAvx2` can only be constructed by
            //   successful runtime AVX2 plus OS-state detection.
            // - Bounds: kernel arguments retain their safe Rust slice/array
            //   bounds; raw memory operations live in fixed-size leaves.
            // - Alignment: fixed-size leaves use explicitly unaligned-capable
            //   loads and stores and never strengthen reference alignment.
            // - Aliasing: the generated signature preserves Rust shared and
            //   exclusive reference rules across the call.
            // - Initialization: inputs are initialized references and outputs
            //   remain initialized byte arrays or slices throughout the call.
            unsafe { kernel($token $(, $arg)*) }
        }
    };
}

pub(crate) use exact_avx2_kernel;
