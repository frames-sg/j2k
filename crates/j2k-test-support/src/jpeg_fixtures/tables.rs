// SPDX-License-Identifier: MIT OR Apache-2.0

pub const LOSSLESS_GRAYSCALE_3X3_PIXELS: [u8; 9] = [130, 132, 136, 128, 135, 142, 125, 137, 150];

pub const LOSSLESS_GRAYSCALE_16BIT_3X3_PIXELS: [u16; 9] = [
    33000, 33012, 33025, 32990, 33020, 33044, 32970, 33030, 33080,
];

pub const LOSSLESS_RGB_3X3_PIXELS: [u8; 27] = [
    130, 50, 200, 132, 53, 198, 136, 55, 195, 128, 54, 202, 135, 56, 199, 142, 59, 196, 125, 57,
    204, 137, 60, 201, 150, 64, 198,
];

pub const LOSSLESS_RGB_16BIT_3X3_PIXELS: [u16; 27] = [
    33000, 16000, 50000, 33012, 16040, 49960, 33025, 16075, 49910, 32990, 16055, 50025, 33020,
    16090, 49980, 33044, 16120, 49930, 32970, 16080, 50050, 33030, 16130, 50000, 33080, 16190,
    49950,
];

pub const LOSSLESS_YCBCR_3X3_PIXELS: [u8; 27] = [
    100, 150, 200, 104, 145, 196, 110, 140, 190, 96, 155, 202, 102, 150, 198, 108, 144, 193, 92,
    160, 205, 101, 153, 199, 116, 148, 194,
];

pub const LOSSLESS_YCBCR_16BIT_3X3_PIXELS: [u16; 27] = [
    33000, 35000, 40000, 33120, 34800, 39700, 33280, 34600, 39250, 32940, 35200, 40250, 33080,
    35040, 39880, 33300, 34720, 39440, 32780, 35480, 40500, 33160, 35120, 39960, 33600, 34920,
    39680,
];

pub(super) const LOSSLESS_RGB_8BIT_422_4X2_C0: [u8; 8] = [130, 132, 136, 140, 128, 135, 142, 150];
pub(super) const LOSSLESS_RGB_8BIT_422_4X2_C1: [u8; 4] = [50, 55, 54, 60];
pub(super) const LOSSLESS_RGB_8BIT_422_4X2_C2: [u8; 4] = [200, 195, 202, 198];

pub(super) const LOSSLESS_YCBCR_8BIT_422_4X2_C0: [u8; 8] = [100, 104, 110, 116, 96, 102, 108, 114];
pub(super) const LOSSLESS_YCBCR_8BIT_422_4X2_C1: [u8; 4] = [150, 140, 155, 144];
pub(super) const LOSSLESS_YCBCR_8BIT_422_4X2_C2: [u8; 4] = [200, 190, 202, 193];

pub(super) const LOSSLESS_RGB_8BIT_420_4X4_C0: [u8; 16] = [
    130, 132, 136, 140, 128, 135, 142, 150, 126, 133, 139, 146, 124, 131, 137, 144,
];
pub(super) const LOSSLESS_RGB_8BIT_420_4X4_C1: [u8; 4] = [50, 55, 54, 60];
pub(super) const LOSSLESS_RGB_8BIT_420_4X4_C2: [u8; 4] = [200, 195, 202, 198];

pub(super) const LOSSLESS_YCBCR_8BIT_420_4X4_C0: [u8; 16] = [
    100, 104, 110, 116, 96, 102, 108, 114, 92, 101, 109, 117, 90, 99, 107, 115,
];
pub(super) const LOSSLESS_YCBCR_8BIT_420_4X4_C1: [u8; 4] = [150, 140, 155, 144];
pub(super) const LOSSLESS_YCBCR_8BIT_420_4X4_C2: [u8; 4] = [200, 190, 202, 193];

pub(super) const LOSSLESS_RGB_16BIT_422_4X2_C0: [u16; 8] =
    [33000, 33012, 33025, 33045, 32990, 33020, 33044, 33070];
pub(super) const LOSSLESS_RGB_16BIT_422_4X2_C1: [u16; 4] = [16000, 16100, 16055, 16140];
pub(super) const LOSSLESS_RGB_16BIT_422_4X2_C2: [u16; 4] = [50000, 49880, 50025, 49920];

pub(super) const LOSSLESS_YCBCR_16BIT_422_4X2_C0: [u16; 8] =
    [33000, 33120, 33280, 33420, 32940, 33080, 33300, 33460];
pub(super) const LOSSLESS_YCBCR_16BIT_422_4X2_C1: [u16; 4] = [35000, 34600, 35200, 34720];
pub(super) const LOSSLESS_YCBCR_16BIT_422_4X2_C2: [u16; 4] = [40000, 39250, 40250, 39440];

pub(super) const LOSSLESS_RGB_16BIT_420_4X4_C0: [u16; 16] = [
    33000, 33012, 33025, 33045, 32990, 33020, 33044, 33070, 33010, 33034, 33058, 33082, 32980,
    33016, 33052, 33088,
];
pub(super) const LOSSLESS_RGB_16BIT_420_4X4_C1: [u16; 4] = [16000, 16100, 16055, 16140];
pub(super) const LOSSLESS_RGB_16BIT_420_4X4_C2: [u16; 4] = [50000, 49880, 50025, 49920];

pub(super) const LOSSLESS_YCBCR_16BIT_420_4X4_C0: [u16; 16] = [
    33000, 33120, 33280, 33420, 32940, 33080, 33300, 33460, 33030, 33160, 33310, 33470, 32970,
    33110, 33340, 33490,
];
pub(super) const LOSSLESS_YCBCR_16BIT_420_4X4_C1: [u16; 4] = [35000, 34600, 35200, 34720];
pub(super) const LOSSLESS_YCBCR_16BIT_420_4X4_C2: [u16; 4] = [40000, 39250, 40250, 39440];

#[derive(Clone, Copy)]
pub(super) struct Lossless422Planes<'a> {
    pub(super) c0: &'a [u16],
    pub(super) c1: &'a [u16],
    pub(super) c2: &'a [u16],
}

#[derive(Clone, Copy)]
pub(super) struct Lossless420Planes<'a> {
    pub(super) c0: &'a [u16],
    pub(super) c1: &'a [u16],
    pub(super) c2: &'a [u16],
}

#[derive(Clone, Copy)]
pub(super) struct Lossless422Planes8<'a> {
    pub(super) c0: &'a [u8],
    pub(super) c1: &'a [u8],
    pub(super) c2: &'a [u8],
}

#[derive(Clone, Copy)]
pub(super) struct Lossless420Planes8<'a> {
    pub(super) c0: &'a [u8],
    pub(super) c1: &'a [u8],
    pub(super) c2: &'a [u8],
}
