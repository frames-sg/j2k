// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{collections::HashMap, num::NonZeroUsize, path::PathBuf};

use j2k_core::{Downscale, PixelFormat, Rect};
use j2k_test_support::fnv1a64_hex;

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

pub(super) const DEFAULT_REPEATS: usize = 5;
pub(super) const DEFAULT_CASE_BATCH_SIZES: &[usize] = &[1];
pub(super) const DEFAULT_MIXED_BATCH_SIZES: &[usize] = &[1, 16, 256, 1024];
pub(super) const BATCH_INPUT_COPY_LIMIT: usize = 32;
pub(super) const MIN_PUBLICATION_EXTERNAL_CASES: usize = 24;
pub(super) const MIN_PUBLICATION_EXTERNAL_INPUTS: usize = 24;
pub(super) const MIN_PUBLICATION_MIXED_DISTINCT_INPUTS: usize = 2;
pub(super) const SMALL_SIDE: u32 = 128;
pub(super) const LARGE_SIDE: u32 = 512;
pub(super) const DEFAULT_BENCHMARK_MODE: BenchmarkMode = BenchmarkMode::PortableNative;

#[derive(Clone)]
pub(super) struct FixtureCase {
    pub(super) name: String,
    pub(super) input_source: String,
    pub(super) corpus_category: String,
    pub(super) corpus_name: String,
    pub(super) license_status: String,
    pub(super) encode_command: String,
    pub(super) manifest_status: String,
    pub(super) source_fnv1a64: Option<String>,
    pub(super) codec: Codec,
    pub(super) container: Container,
    pub(super) bytes: Vec<u8>,
    pub(super) dimensions: (u32, u32),
    pub(super) format: PixelFormat,
    pub(super) operation: Operation,
}

impl FixtureCase {
    pub(super) fn input_len(&self) -> usize {
        self.bytes.len()
    }

    pub(super) fn input_digest(&self) -> String {
        fnv1a64_hex(&self.bytes)
    }

    pub(super) fn source_digest(&self) -> String {
        self.source_fnv1a64
            .clone()
            .unwrap_or_else(|| self.input_digest())
    }

    pub(super) fn output_rect(&self) -> Rect {
        self.operation.output_rect(self.dimensions)
    }

    pub(super) fn output_stride(&self) -> usize {
        self.output_rect().w as usize * self.format.bytes_per_pixel()
    }

    pub(super) fn output_len(&self) -> usize {
        self.output_stride() * self.output_rect().h as usize
    }
}

#[derive(Clone)]
pub(super) struct FixtureMetadata {
    pub(super) input_source: String,
    pub(super) corpus_category: String,
    pub(super) corpus_name: String,
    pub(super) license_status: String,
    pub(super) encode_command: String,
    pub(super) manifest_status: String,
    pub(super) source_fnv1a64: Option<String>,
}

pub(super) struct FixtureManifest {
    pub(super) entries: HashMap<PathBuf, ManifestEntry>,
}

pub(super) struct ManifestEntry {
    pub(super) corpus_category: String,
    pub(super) corpus_name: String,
    pub(super) license_status: String,
    pub(super) encode_command: String,
    pub(super) input_fnv1a64: Option<String>,
    pub(super) source_fnv1a64: Option<String>,
    pub(super) codec: Option<Codec>,
    pub(super) container: Option<Container>,
}

pub(super) struct Measurement {
    pub(super) decoder: DecoderKind,
    pub(super) repeats: usize,
    pub(super) batch_size: usize,
    pub(super) median_us: f64,
    pub(super) mean_us: f64,
    pub(super) tiles_per_second_median: f64,
    pub(super) decoded_bytes_per_repeat: usize,
    pub(super) samples_us: Vec<f64>,
}

pub(super) struct ActiveMeasurement {
    pub(super) decoder: DecoderKind,
    pub(super) batch_inputs: BatchInputs,
    pub(super) samples_us: Vec<f64>,
    pub(super) decoded_bytes_per_repeat: Option<usize>,
}

pub(super) struct MixedFixtureBatch {
    pub(super) name: String,
    pub(super) cases: Vec<FixtureCase>,
    pub(super) format: PixelFormat,
    pub(super) operation_class: OperationClass,
}

pub(super) struct ActiveMixedMeasurement {
    pub(super) decoder: DecoderKind,
    pub(super) samples_us: Vec<f64>,
    pub(super) decoded_bytes_per_repeat: Option<usize>,
}

#[derive(Clone, Copy)]
pub(super) struct MetadataContext<'a> {
    pub(super) args: &'a [String],
    pub(super) benchmark_mode: BenchmarkMode,
    pub(super) repeats: usize,
    pub(super) batch_sizes: &'a [usize],
    pub(super) case_batch_sizes: &'a [usize],
    pub(super) mixed_batch_sizes: &'a [usize],
    pub(super) workers: Option<NonZeroUsize>,
    pub(super) cases: &'a [FixtureCase],
    pub(super) mixed_batches: &'a [MixedFixtureBatch],
    pub(super) mode_excluded_cases: &'a [String],
    pub(super) filters_empty: bool,
}
