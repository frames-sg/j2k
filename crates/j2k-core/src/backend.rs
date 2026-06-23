// SPDX-License-Identifier: MIT OR Apache-2.0

use core::sync::atomic::{AtomicU8, Ordering};

/// Runtime backend that executes codec work.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendKind {
    /// Portable CPU implementation.
    Cpu,
    /// Apple Metal implementation.
    Metal,
    /// NVIDIA CUDA implementation.
    Cuda,
}

/// Caller preference for backend selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum BackendRequest {
    /// Let the codec choose the best available backend.
    #[default]
    Auto,
    /// Force the portable CPU backend.
    Cpu,
    /// Force Metal and fail if unavailable.
    Metal,
    /// Force CUDA and fail if unavailable.
    Cuda,
}

impl BackendRequest {
    /// Adaptive accelerated route: let the codec choose CPU and device stages
    /// for benchmark-approved workload shapes.
    pub const ACCELERATED: Self = Self::Auto;
    /// Explicit portable CPU route.
    pub const CPU_ONLY: Self = Self::Cpu;
    /// Strict Metal route; fail when Metal is unavailable or unsupported.
    pub const STRICT_METAL: Self = Self::Metal;
    /// Strict CUDA route; fail when CUDA is unavailable or unsupported.
    pub const STRICT_CUDA: Self = Self::Cuda;
}

/// CPU SIMD feature flags detected for the current host.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct CpuFeatures {
    /// True when AVX2 is available and enabled by the OS.
    pub avx2: bool,
    /// True when SSE4.1 is available.
    pub sse41: bool,
    /// True when NEON is available.
    pub neon: bool,
}

impl CpuFeatures {
    /// Detect CPU SIMD features once and reuse the cached result.
    pub fn detect() -> Self {
        static DETECTED: AtomicU8 = AtomicU8::new(0);

        let cached = DETECTED.load(Ordering::Acquire);
        if cached != 0 {
            return Self::from_cache_byte(cached);
        }

        let detected = Self::detect_uncached();
        let encoded = detected.to_cache_byte();
        let _ = DETECTED.compare_exchange(0, encoded, Ordering::AcqRel, Ordering::Acquire);
        Self::from_cache_byte(DETECTED.load(Ordering::Acquire))
    }

    fn detect_uncached() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Self {
                avx2: detect_x86_avx2(),
                sse41: detect_x86_sse41(),
                neon: false,
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            Self {
                avx2: false,
                sse41: false,
                neon: true,
            }
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self::default()
        }
    }

    const fn to_cache_byte(self) -> u8 {
        let mut encoded = 1_u8;
        if self.avx2 {
            encoded |= 1 << 1;
        }
        if self.sse41 {
            encoded |= 1 << 2;
        }
        if self.neon {
            encoded |= 1 << 3;
        }
        encoded
    }

    const fn from_cache_byte(encoded: u8) -> Self {
        let bits = encoded.saturating_sub(1);
        Self {
            avx2: (bits & (1 << 1)) != 0,
            sse41: (bits & (1 << 2)) != 0,
            neon: (bits & (1 << 3)) != 0,
        }
    }
}

#[cfg(target_arch = "x86_64")]
fn detect_x86_sse41() -> bool {
    let features = core::arch::x86_64::__cpuid(1);
    (features.ecx & (1 << 19)) != 0
}

#[cfg(target_arch = "x86_64")]
fn detect_x86_avx2() -> bool {
    let leaf1 = core::arch::x86_64::__cpuid(1);
    let osxsave = (leaf1.ecx & (1 << 27)) != 0;
    let avx = (leaf1.ecx & (1 << 28)) != 0;
    if !(osxsave && avx) {
        return false;
    }

    // SAFETY: XGETBV is only executed after CPUID reports OSXSAVE support.
    let xcr0 = unsafe { core::arch::x86_64::_xgetbv(0) };
    let xmm_enabled = (xcr0 & 0b10) != 0;
    let ymm_enabled = (xcr0 & 0b100) != 0;
    if !(xmm_enabled && ymm_enabled) {
        return false;
    }

    let max_leaf = core::arch::x86_64::__cpuid(0).eax;
    if max_leaf < 7 {
        return false;
    }

    let leaf7 = core::arch::x86_64::__cpuid_count(7, 0);
    (leaf7.ebx & (1 << 5)) != 0
}

/// Backend availability for a codec/runtime combination.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackendCapabilities {
    /// Host CPU feature set.
    pub cpu: CpuFeatures,
    /// True when Metal is available to this crate.
    pub metal: bool,
    /// True when CUDA is available to this crate.
    pub cuda: bool,
}

impl BackendCapabilities {
    /// Return default capabilities implied by the current build target.
    ///
    /// This does not probe GPU devices or runtime libraries. Codec facades and
    /// adapters must further gate the returned device flags by their compiled
    /// features and runtime availability.
    #[must_use]
    pub fn compile_time_defaults() -> Self {
        Self {
            cpu: CpuFeatures::detect(),
            metal: cfg!(target_os = "macos"),
            cuda: false,
        }
    }

    /// Return whether a backend request can be satisfied.
    #[must_use]
    pub const fn supports(self, request: BackendRequest) -> bool {
        match request {
            BackendRequest::Auto | BackendRequest::Cpu => true,
            BackendRequest::Metal => self.metal,
            BackendRequest::Cuda => self.cuda,
        }
    }

    /// Resolve a backend request to the concrete backend that should run.
    ///
    /// `Auto` resolves to CPU here. Workload-aware device promotion belongs in
    /// codec-specific route planners that have benchmark evidence for the
    /// requested operation.
    #[must_use]
    pub fn resolve(self, request: BackendRequest) -> Option<BackendKind> {
        match request {
            BackendRequest::Auto | BackendRequest::Cpu => Some(BackendKind::Cpu),
            BackendRequest::Metal if self.metal => Some(BackendKind::Metal),
            BackendRequest::Cuda if self.cuda => Some(BackendKind::Cuda),
            BackendRequest::Metal | BackendRequest::Cuda => None,
        }
    }

    /// Return an available accelerator backend without implying it should be
    /// selected for a workload.
    #[must_use]
    pub const fn first_available_accelerator(self) -> Option<BackendKind> {
        if self.metal {
            Some(BackendKind::Metal)
        } else if self.cuda {
            Some(BackendKind::Cuda)
        } else {
            None
        }
    }
}
