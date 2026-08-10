// SPDX-License-Identifier: MIT OR Apache-2.0

use serde::{Deserialize, Serialize};

/// Encoder implementation under test.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum EncoderIut {
    /// Portable `j2k` CPU encoder surfaces.
    Cpu,
    /// `j2k-cuda` adapter encoder surfaces.
    Cuda,
    /// `j2k-metal` adapter encoder surfaces.
    Metal,
}

/// Compression mode exercised by one encoder case.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum EncoderMode {
    /// Reversible Part 1 encode.
    Lossless,
    /// Irreversible Part 1 encode.
    Lossy,
}

/// Public codec operation exercised by one encoder-evidence case.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum EncoderOperation {
    /// Encode deterministic component samples.
    Encode,
    /// Recode a generated classic J2K or JP2 source to lossless HTJ2K.
    Recode,
}

/// Code-block coding family exercised by one encoder case.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum EncoderBlockCoding {
    /// JPEG 2000 Part 1 EBCOT block coding.
    Classic,
    /// JPEG 2000 Part 15 high-throughput block coding.
    HighThroughput,
}

/// Independent decoder selected for one informative encoder matrix case.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum EncoderReferenceDecoder {
    /// Pinned T.804 `OpenJPEG` reference software.
    #[default]
    OpenJpeg,
    /// Pinned `OpenHTJ2K` interoperability decoder for HT features `OpenJPEG` rejects.
    OpenHtj2k,
}

impl EncoderReferenceDecoder {
    #[expect(
        clippy::trivially_copy_pass_by_ref,
        reason = "serde skip_serializing_if requires a borrowed field predicate"
    )]
    fn is_openjpeg(value: &Self) -> bool {
        *value == Self::OpenJpeg
    }
}

/// Payload shape decoded by the T.804 reference implementation.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum EncoderPayload {
    /// Raw JPEG 2000-family codestream.
    Codestream,
    /// Classic JPEG 2000 codestream wrapped in a JP2 file.
    Jp2,
    /// HTJ2K codestream wrapped in a JPH file.
    Jph,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum EncoderPairwiseScope {
    Part1,
    Part15,
}

/// Part 1 packet progression selected in COD.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum EncoderProgression {
    /// Layer-resolution-component-position.
    Lrcp,
    /// Resolution-layer-component-position.
    Rlcp,
    /// Resolution-position-component-layer.
    Rpcl,
    /// Position-component-resolution-layer.
    Pcrl,
    /// Component-position-resolution-layer.
    Cprl,
}

/// Marker listed in T.803 Table F.1.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum EncoderMarker {
    Soc,
    Cap,
    Prf,
    Cpf,
    Sot,
    Sod,
    Eoc,
    Siz,
    Cod,
    Coc,
    Rgn,
    Qcd,
    Qcc,
    Poc,
    Tlm,
    Plm,
    Plt,
    Ppm,
    Ppt,
    Sop,
    Eph,
    Crg,
    Com,
}

pub(super) const TABLE_F1_MARKERS: [EncoderMarker; 23] = [
    EncoderMarker::Soc,
    EncoderMarker::Cap,
    EncoderMarker::Prf,
    EncoderMarker::Cpf,
    EncoderMarker::Sot,
    EncoderMarker::Sod,
    EncoderMarker::Eoc,
    EncoderMarker::Siz,
    EncoderMarker::Cod,
    EncoderMarker::Coc,
    EncoderMarker::Rgn,
    EncoderMarker::Qcd,
    EncoderMarker::Qcc,
    EncoderMarker::Poc,
    EncoderMarker::Tlm,
    EncoderMarker::Plm,
    EncoderMarker::Plt,
    EncoderMarker::Ppm,
    EncoderMarker::Ppt,
    EncoderMarker::Sop,
    EncoderMarker::Eph,
    EncoderMarker::Crg,
    EncoderMarker::Com,
];

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum EncoderInputKind {
    #[default]
    Interleaved,
    ComponentPlanes,
    TypedComponentPlanes,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum EncoderPattern {
    #[default]
    Gradient,
    Checkerboard,
    DeterministicNoise,
    Impulse,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "kind", content = "value", rename_all = "kebab-case")]
pub(crate) enum EncoderRateTarget {
    BitsPerPixel(f64),
    Bytes(u64),
    PsnrDb(f64),
}

/// One rectangular maxshift request.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct EncoderRoi {
    pub(crate) component: u16,
    pub(crate) x: u32,
    pub(crate) y: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) shift: u8,
}

/// One stable Annex D encoder test case.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EncoderCase {
    /// Stable case identifier.
    pub id: String,
    /// Adapter IUTs to which this case applies.
    pub iuts: Vec<EncoderIut>,
    /// Independent decoder used to validate this case.
    #[serde(default, skip_serializing_if = "EncoderReferenceDecoder::is_openjpeg")]
    pub reference_decoder: EncoderReferenceDecoder,
    /// Reversible or irreversible coding.
    pub mode: EncoderMode,
    /// Encode or coefficient-domain recode operation.
    pub operation: EncoderOperation,
    /// Classic or high-throughput code-block coding.
    pub block_coding: EncoderBlockCoding,
    /// Raw codestream or JPH output supplied to the reference decoder.
    pub payload: EncoderPayload,
    /// Raw J2K or JP2 input for a recode case; absent for encode cases.
    #[serde(default)]
    pub source_payload: Option<EncoderPayload>,
    #[serde(default)]
    pub(crate) input: EncoderInputKind,
    /// Reference-grid width.
    pub width: u32,
    /// Reference-grid height.
    pub height: u32,
    /// Component count.
    pub components: u16,
    /// Common sample precision for interleaved and homogeneous planar inputs.
    pub bit_depth: u8,
    /// Common signedness for interleaved and homogeneous planar inputs.
    pub signed: bool,
    #[serde(default)]
    pub(crate) pattern: EncoderPattern,
    #[serde(default)]
    pub(crate) sampling: Vec<[u8; 2]>,
    #[serde(default)]
    pub(crate) component_bit_depths: Vec<u8>,
    #[serde(default)]
    pub(crate) component_signedness: Vec<bool>,
    /// COD packet progression.
    pub progression: EncoderProgression,
    /// Requested wavelet decomposition levels.
    pub decomposition_levels: u8,
    #[serde(default = "one_quality_layer")]
    pub(crate) lossless_quality_layers: u8,
    #[serde(default)]
    pub(crate) lossy_rate_target: Option<EncoderRateTarget>,
    #[serde(default)]
    pub(crate) lossy_quality_layers: Vec<EncoderRateTarget>,
    #[serde(default)]
    pub(crate) minimum_psnr_db: Option<f64>,
    #[serde(default)]
    pub(crate) maximum_rate_overshoot_percent: Option<f64>,
    #[serde(default)]
    pub(crate) tile_size: Option<[u32; 2]>,
    #[serde(default)]
    pub(crate) tile_part_packet_limit: Option<u16>,
    #[serde(default)]
    pub(crate) precinct_exponents: Vec<[u8; 2]>,
    #[serde(default)]
    pub(crate) roi: Option<EncoderRoi>,
    /// Optional markers this case explicitly requests.
    #[serde(default)]
    pub markers: Vec<EncoderMarker>,
    /// Declared pairwise covering array containing this row.
    #[serde(default)]
    pub(crate) pairwise_scope: Option<EncoderPairwiseScope>,
}

const fn one_quality_layer() -> u8 {
    1
}
