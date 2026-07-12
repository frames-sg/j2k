//! The color specification box (colr), defined in I.5.3.3.

use alloc::vec::Vec;

use crate::error::{FormatError, Result};
use crate::jp2::allocation::Jp2AllocationBudget;
use crate::jp2::ImageBoxes;
use crate::reader::BitReader;
use crate::DecodeError;

pub(super) fn parse(
    boxes: &mut ImageBoxes,
    data: &[u8],
    budget: &mut Jp2AllocationBudget,
) -> Result<()> {
    let mut reader = BitReader::new(data);

    let meth = reader.read_byte().ok_or(FormatError::InvalidBox)?;
    // We don't care about those.
    let _prec = reader.read_byte().ok_or(FormatError::InvalidBox)?;
    let _approx = reader.read_byte().ok_or(FormatError::InvalidBox)?;

    let mut enumerated_value = None;
    let method = match meth {
        1 => {
            let enumerated = reader.read_u32().ok_or(FormatError::InvalidBox)?;
            enumerated_value = Some(enumerated);
            ColorSpace::Enumerated(
                EnumeratedColorspace::from_raw(enumerated, &mut reader)
                    .ok_or(FormatError::InvalidBox)?,
            )
        }
        2 => {
            let profile_data = budget.try_copy_bytes(
                reader.tail().ok_or(FormatError::InvalidBox)?,
                "JP2 ICC profile",
            )?;
            ColorSpace::Icc(profile_data)
        }
        _ => ColorSpace::Unknown,
    };

    let parsed = ColorSpecificationBox {
        method: meth,
        enumerated_value,
        color_space: method,
    };
    if boxes.color_specifications.len() == boxes.color_specifications.capacity() {
        return Err(DecodeError::AllocationTooLarge {
            what: "JP2 COLR metadata plan",
            requested: usize::MAX,
            cap: crate::DEFAULT_MAX_DECODE_BYTES,
        });
    }
    boxes.color_specifications.push(parsed);

    Ok(())
}

#[derive(Debug)]
pub(crate) struct ColorSpecificationBox {
    pub(crate) method: u8,
    pub(crate) enumerated_value: Option<u32>,
    pub(crate) color_space: ColorSpace,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum ColorSpace {
    Enumerated(EnumeratedColorspace),
    Icc(Vec<u8>),
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EnumeratedColorspace {
    BiLevel1,
    YCbCr1,
    YCbCr2,
    YCbCr3,
    PhotoYcc,
    Cmy,
    Cmyk,
    Ycck,
    CieLab(CieLab),
    BiLevel2,
    Srgb,
    Greyscale,
    Sycc,
    CieJab,
    EsRgb,
    RommRgb,
    YPbPr112560,
    YPbPr125050,
    EsYcc,
    ScRgb,
    ScRgbGray,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CieLab {
    pub(crate) rl: Option<u32>,
    pub(crate) ol: Option<u32>,
    pub(crate) ra: Option<u32>,
    pub(crate) oa: Option<u32>,
    pub(crate) rb: Option<u32>,
    pub(crate) ob: Option<u32>,
}

impl EnumeratedColorspace {
    fn from_raw(value: u32, reader: &mut BitReader<'_>) -> Option<Self> {
        match value {
            0 => Some(Self::BiLevel1),
            1 => Some(Self::YCbCr1),
            3 => Some(Self::YCbCr2),
            4 => Some(Self::YCbCr3),
            9 => Some(Self::PhotoYcc),
            11 => Some(Self::Cmy),
            12 => Some(Self::Cmyk),
            13 => Some(Self::Ycck),
            14 => {
                // M.11.7.4.1 EP field format for the CIELab colourspace
                let rl = reader.read_u32();
                let ol = reader.read_u32();
                let ra = reader.read_u32();
                let oa = reader.read_u32();
                let rb = reader.read_u32();
                let ob = reader.read_u32();
                // Not supported for now.
                let _il = reader.read_u32();

                Some(Self::CieLab(CieLab {
                    rl,
                    ol,
                    ra,
                    oa,
                    rb,
                    ob,
                }))
            }
            15 => Some(Self::BiLevel2),
            16 => Some(Self::Srgb),
            17 => Some(Self::Greyscale),
            18 => Some(Self::Sycc),
            19 => Some(Self::CieJab),
            20 => Some(Self::EsRgb),
            21 => Some(Self::RommRgb),
            22 => Some(Self::YPbPr112560),
            23 => Some(Self::YPbPr125050),
            24 => Some(Self::EsYcc),
            25 => Some(Self::ScRgb),
            26 => Some(Self::ScRgbGray),
            _ => None,
        }
    }
}
