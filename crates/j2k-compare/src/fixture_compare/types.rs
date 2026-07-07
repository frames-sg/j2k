// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::{Downscale, Rect};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum BenchmarkMode {
    PortableNative,
    PortableEmulated,
    Capability,
}

impl BenchmarkMode {
    pub(super) const fn label(self) -> &'static str {
        match self {
            Self::PortableNative => "portable-native",
            Self::PortableEmulated => "portable-emulated",
            Self::Capability => "capability",
        }
    }

    pub(super) const fn comparable_scope(self) -> &'static str {
        match self {
            Self::PortableNative => "native-operations-only",
            Self::PortableEmulated => "task-equivalent-with-method-labels",
            Self::Capability => "feature-coverage-with-explicit-noncomparable-skips",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Codec {
    Classic,
    Htj2k,
    Unknown,
}

impl Codec {
    pub(super) const fn label(self) -> &'static str {
        match self {
            Self::Classic => "j2k",
            Self::Htj2k => "htj2k",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Container {
    RawCodestream,
    Jp2,
    Jph,
    Jhc,
}

impl Container {
    pub(super) const fn label(self) -> &'static str {
        match self {
            Self::RawCodestream => "raw-codestream",
            Self::Jp2 => "jp2",
            Self::Jph => "jph",
            Self::Jhc => "jhc",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Operation {
    Full,
    Region(Rect),
    Scaled(Downscale),
    RegionScaled { roi: Rect, scale: Downscale },
}

impl Operation {
    pub(super) const fn label(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::Region(_) => "roi",
            Self::Scaled(_) => "scaled",
            Self::RegionScaled { .. } => "roi-scaled",
        }
    }

    pub(super) const fn roi(self) -> Option<Rect> {
        match self {
            Self::Full | Self::Scaled(_) => None,
            Self::Region(roi) | Self::RegionScaled { roi, .. } => Some(roi),
        }
    }

    pub(super) const fn scale(self) -> Downscale {
        match self {
            Self::Full | Self::Region(_) => Downscale::None,
            Self::Scaled(scale) | Self::RegionScaled { scale, .. } => scale,
        }
    }

    pub(super) fn output_rect(self, dimensions: (u32, u32)) -> Rect {
        let source = self.roi().unwrap_or_else(|| Rect::full(dimensions));
        source.scaled_covering(self.scale())
    }

    pub(super) const fn class(self) -> OperationClass {
        match self {
            Self::Full => OperationClass::Full,
            Self::Region(_) => OperationClass::Region,
            Self::Scaled(_) => OperationClass::Scaled,
            Self::RegionScaled { .. } => OperationClass::RegionScaled,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum OperationClass {
    Full,
    Region,
    Scaled,
    RegionScaled,
}

impl OperationClass {
    pub(super) const fn label(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::Region => "roi",
            Self::Scaled => "scaled",
            Self::RegionScaled => "roi-scaled",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum DecoderKind {
    J2k,
    OpenJpeg,
    Grok,
    OpenJph,
    Kakadu,
}

impl DecoderKind {
    pub(super) const fn label(self) -> &'static str {
        match self {
            Self::J2k => "j2k",
            Self::OpenJpeg => "openjpeg",
            Self::Grok => "grok",
            Self::OpenJph => "openjph",
            Self::Kakadu => "kakadu",
        }
    }
}

pub(super) struct BatchInputs {
    buffers: Vec<Vec<u8>>,
    batch_size: usize,
}

impl BatchInputs {
    pub(super) fn new(input: &[u8], batch_size: usize, copy_count: usize) -> Self {
        let buffers = (0..copy_count).map(|_| input.to_vec()).collect::<Vec<_>>();
        Self {
            buffers,
            batch_size,
        }
    }

    pub(super) const fn len(&self) -> usize {
        self.batch_size
    }

    pub(super) fn input(&self, index: usize) -> &[u8] {
        self.buffers[index % self.buffers.len()].as_slice()
    }
}
