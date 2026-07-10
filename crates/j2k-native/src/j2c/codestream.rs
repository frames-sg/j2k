//! Read and decode a JPEG2000 codestream, as described in Annex A.

mod auxiliary;
mod coding;
mod header;
pub(crate) mod markers;
mod model;
mod progression;
mod quantization;
mod size;
mod validation;

pub(crate) use auxiliary::{decode_packet_lengths, plt_marker, rgn_marker, skip_marker_segment};
pub(crate) use coding::{coc_marker, cod_marker};
pub(crate) use header::read_header;
pub(crate) use model::{
    CodeBlockStyle, CodingStyleComponent, CodingStyleDefault, CodingStyleFlags,
    CodingStyleParameters, ComponentInfo, ComponentSizeInfo, Header, PacketLengthMarker,
    PpmMarkerData, PpmPacket, ProgressionChange, ProgressionOrder, QuantizationInfo,
    QuantizationStyle, RgnMarkerData, SizeData, StepSize, WaveletTransform,
};
pub(crate) use progression::poc_marker;
pub(crate) use quantization::{qcc_marker, qcd_marker};

#[cfg(test)]
mod tests;
