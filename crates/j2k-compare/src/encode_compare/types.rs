// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{fnv1a64_hex, HashMap, PathBuf, PixelFormat};

pub(super) const DEFAULT_REPEATS: usize = 5;
pub(super) const DEFAULT_CASE_BATCH_SIZES: &[usize] = &[1];
pub(super) const DEFAULT_MIXED_BATCH_SIZES: &[usize] = &[1, 16, 256, 1024];
pub(super) const MIN_PUBLICATION_EXTERNAL_IMAGES: usize = 24;
pub(super) const MIN_PUBLICATION_MIXED_DISTINCT_INPUTS: usize = 2;
pub(super) const MIN_PUBLICATION_EXTERNAL_DIMENSIONS: usize = 3;
pub(super) const MIN_PUBLICATION_EXTERNAL_SOURCE_FORMATS: usize = 2;

#[derive(Clone)]
pub(super) struct ImageCase {
    pub(super) name: String,
    pub(super) input_source: String,
    pub(super) corpus_category: String,
    pub(super) corpus_name: String,
    pub(super) license_status: String,
    pub(super) source_command: String,
    pub(super) manifest_status: String,
    pub(super) source_format: String,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) components: u8,
    pub(super) pixels: Vec<u8>,
    pub(super) pnm_path: PathBuf,
}

impl ImageCase {
    pub(super) fn format_label(&self) -> &'static str {
        match self.components {
            1 => "gray8",
            3 => "rgb8",
            _ => "unsupported",
        }
    }

    pub(super) fn pixel_format(&self) -> Result<PixelFormat, String> {
        match self.components {
            1 => Ok(PixelFormat::Gray8),
            3 => Ok(PixelFormat::Rgb8),
            other => Err(format!(
                "{} has unsupported component count {other}",
                self.name
            )),
        }
    }

    pub(super) fn input_digest(&self) -> String {
        fnv1a64_hex(&self.pixels)
    }
}

pub(super) struct MixedImageBatch {
    pub(super) name: String,
    pub(super) cases: Vec<ImageCase>,
    pub(super) components: u8,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum EncoderKind {
    J2k,
    OpenJpeg,
    Grok,
    Kakadu,
}

impl EncoderKind {
    pub(super) const fn label(self) -> &'static str {
        match self {
            Self::J2k => "j2k",
            Self::OpenJpeg => "openjpeg",
            Self::Grok => "grok",
            Self::Kakadu => "kakadu",
        }
    }
}

#[derive(Clone)]
pub(super) struct EncoderTool {
    pub(super) kind: EncoderKind,
    pub(super) program: PathBuf,
    pub(super) available: bool,
}

pub(super) struct Measurement {
    pub(super) batch_size: usize,
    pub(super) repeats: usize,
    pub(super) median_us: f64,
    pub(super) mean_us: f64,
    pub(super) images_per_second_median: f64,
    pub(super) encoded_bytes_per_repeat: usize,
    pub(super) samples_us: Vec<f64>,
}

pub(super) struct EncodeMeasurementState<'a> {
    pub(super) tool: &'a EncoderTool,
    pub(super) encoded_bytes_per_repeat: Option<usize>,
    pub(super) samples_us: Vec<f64>,
}

pub(super) struct EncodeManifest {
    pub(super) entries: HashMap<PathBuf, EncodeManifestEntry>,
}

pub(super) struct EncodeManifestEntry {
    pub(super) corpus_category: String,
    pub(super) corpus_name: String,
    pub(super) license_status: String,
    pub(super) source_command: String,
    pub(super) input_fnv1a64: Option<String>,
}

pub(super) struct ExternalImageMetadata {
    pub(super) input_source: String,
    pub(super) corpus_category: String,
    pub(super) corpus_name: String,
    pub(super) license_status: String,
    pub(super) source_command: String,
    pub(super) manifest_status: String,
}

#[derive(Clone, Copy)]
pub(super) struct MetadataInput<'a> {
    pub(super) args: &'a [String],
    pub(super) repeats: usize,
    pub(super) batch_sizes: &'a [usize],
    pub(super) case_batch_sizes: &'a [usize],
    pub(super) mixed_batch_sizes: &'a [usize],
    pub(super) cases: &'a [ImageCase],
    pub(super) mixed_batches: &'a [MixedImageBatch],
    pub(super) selected_tools: &'a [EncoderTool],
    pub(super) all_tools: &'a [EncoderTool],
    pub(super) filters_empty: bool,
}

pub(super) struct PnmImage {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) components: u8,
    pub(super) pixels: Vec<u8>,
    pub(super) source_command: String,
}
