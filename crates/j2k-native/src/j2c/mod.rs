mod arithmetic_decoder;
pub(crate) mod arithmetic_encoder;
pub(crate) mod bitplane;
pub(crate) mod bitplane_encode;
pub(crate) mod build;
pub(crate) mod codestream;
pub(crate) mod codestream_write;
mod decode;
pub(crate) mod encode;
pub(crate) mod fdwt;
pub(crate) mod forward_mct;
pub(crate) mod ht_block_decode;
pub(crate) mod ht_block_encode;
pub(crate) mod ht_encode_tables;
pub(crate) mod ht_tables;
pub(crate) mod idwt;
mod mct;
mod mq;
pub(crate) mod packet_encode;
mod progression;
pub(crate) mod quantize;
pub(crate) mod recode;
mod rect;
mod roi;
mod segment;
mod tag_tree;
pub(crate) mod tag_tree_encode;
mod tile;

use alloc::vec::Vec;

use super::jp2::colr::{ColorSpace, ColorSpecificationBox, EnumeratedColorspace};
use super::jp2::ImageBoxes;
use crate::error::{bail, FormatError, MarkerError, Result};
use crate::j2c::codestream::markers;
use crate::reader::BitReader;
use crate::{resolve_alpha_and_color_space, DecodeSettings, Image};

use crate::math::{SimdBuffer, SIMD_WIDTH};
pub(crate) use codestream::Header;
#[cfg(test)]
pub(crate) use decode::should_decode_classic_sub_band_in_parallel;
pub(crate) use decode::{build_direct_color_plan, build_direct_grayscale_plan, decode};
pub use decode::{CpuDecodeParallelism, DecoderContext};
pub use recode::Reversible53CoefficientImage;
pub(crate) use segment::MAX_BITPLANE_COUNT;

pub(crate) struct ParsedCodestream<'a> {
    pub(crate) header: Header<'a>,
    pub(crate) data: &'a [u8],
}

#[derive(Debug, Clone)]
pub(crate) struct ComponentData {
    pub(crate) container: SimdBuffer<{ SIMD_WIDTH }>,
    pub(crate) integer_container: Option<Vec<i64>>,
    pub(crate) bit_depth: u8,
    pub(crate) signed: bool,
}

pub(crate) fn parse<'a>(stream: &'a [u8], settings: &DecodeSettings) -> Result<Image<'a>> {
    let parsed_codestream = parse_raw(stream, settings)?;
    let header = &parsed_codestream.header;
    let mut boxes = ImageBoxes::default();

    // Raw codestreams do not carry JP2 channel definitions. Keep the
    // conventional grayscale/RGB assumptions for 1- and 3-component images,
    // but preserve two-component data as independent channels instead of
    // forcing it through grayscale validation.
    let (cs, enumerated_value) = match header.component_infos.len() {
        1 => (
            ColorSpace::Enumerated(EnumeratedColorspace::Greyscale),
            Some(17),
        ),
        2 => (ColorSpace::Unknown, None),
        _ => (ColorSpace::Enumerated(EnumeratedColorspace::Srgb), Some(16)),
    };

    let color_specification = ColorSpecificationBox {
        method: if enumerated_value.is_some() { 1 } else { 0 },
        enumerated_value,
        color_space: cs,
    };
    boxes.color_specifications.push(color_specification.clone());
    boxes.color_specification = Some(color_specification);

    let (color_space, has_alpha) =
        resolve_alpha_and_color_space(&boxes, &parsed_codestream.header, settings)?;
    Ok(Image {
        codestream: parsed_codestream.data,
        header: parsed_codestream.header,
        boxes,
        settings: *settings,
        color_space,
        has_alpha,
    })
}

pub(crate) fn parse_raw<'a>(
    stream: &'a [u8],
    settings: &DecodeSettings,
) -> Result<ParsedCodestream<'a>> {
    let mut reader = BitReader::new(stream);

    let marker = reader.read_marker()?;
    if marker != markers::SOC {
        bail!(MarkerError::Expected("SOC"));
    }

    let header = codestream::read_header(&mut reader, settings)?;
    let code_stream_data = reader.tail().ok_or(FormatError::MissingCodestream)?;

    Ok(ParsedCodestream {
        header,
        data: code_stream_data,
    })
}
