#[derive(Clone, Hash, Debug, Eq, PartialEq)]
pub(crate) enum ICCColorSpace {
    Xyz,
    Lab,
    Luv,
    Ycbr,
    Yxy,
    Lms,
    Rgb,
    Gray,
    Hsv,
    Hls,
    Cmyk,
    Cmy,
    OneClr,
    ThreeClr,
    FourClr,
    // There are more, but those should be the most important
    // ones.
}

impl ICCColorSpace {
    pub(crate) fn num_components(&self) -> u8 {
        match self {
            Self::Xyz
            | Self::Lab
            | Self::Luv
            | Self::Ycbr
            | Self::Yxy
            | Self::Lms
            | Self::Rgb
            | Self::Hsv
            | Self::Hls
            | Self::Cmy
            | Self::ThreeClr => 3,
            Self::Gray | Self::OneClr => 1,
            Self::Cmyk | Self::FourClr => 4,
        }
    }
}

impl TryFrom<u32> for ICCColorSpace {
    type Error = ();

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0x5859_5A20 => Ok(Self::Xyz),
            0x4C61_6220 => Ok(Self::Lab),
            0x4C75_7620 => Ok(Self::Luv),
            0x5943_6272 => Ok(Self::Ycbr),
            0x5978_7920 => Ok(Self::Yxy),
            0x4C4D_5320 => Ok(Self::Lms),
            0x5247_4220 => Ok(Self::Rgb),
            0x4752_4159 => Ok(Self::Gray),
            0x4853_5620 => Ok(Self::Hsv),
            0x484C_5320 => Ok(Self::Hls),
            0x434D_594B => Ok(Self::Cmyk),
            0x434D_5920 => Ok(Self::Cmy),
            0x3143_4C52 => Ok(Self::OneClr),
            0x3343_4C52 => Ok(Self::ThreeClr),
            0x3443_4C52 => Ok(Self::FourClr),
            _ => Err(()),
        }
    }
}

#[derive(Clone, Hash, Debug, Eq, PartialEq)]
pub(crate) struct ICCMetadata {
    pub(crate) color_space: ICCColorSpace,
}

impl ICCMetadata {
    pub(crate) fn from_data(data: &[u8]) -> Option<Self> {
        let color_space = {
            let marker = u32::from_be_bytes(data.get(16..20)?.try_into().ok()?);
            ICCColorSpace::try_from(marker).ok()?
        };

        Some(Self { color_space })
    }
}
