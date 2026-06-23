// SPDX-License-Identifier: MIT OR Apache-2.0

/// CPU parallelism policy for JPEG 2000 decode.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum CpuDecodeParallelism {
    /// Allow a single tile decode to use internal code-block parallelism.
    #[default]
    Auto,
    /// Keep code-block decode serial for callers that already parallelize tiles.
    Serial,
}

impl CpuDecodeParallelism {
    pub(crate) const fn to_native(self) -> j2k_native::CpuDecodeParallelism {
        match self {
            Self::Auto => j2k_native::CpuDecodeParallelism::Auto,
            Self::Serial => j2k_native::CpuDecodeParallelism::Serial,
        }
    }

    pub(crate) const fn from_native(parallelism: j2k_native::CpuDecodeParallelism) -> Self {
        match parallelism {
            j2k_native::CpuDecodeParallelism::Auto => Self::Auto,
            j2k_native::CpuDecodeParallelism::Serial => Self::Serial,
        }
    }
}
