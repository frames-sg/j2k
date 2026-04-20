//! Internal (local) parameter structures for JPEG 2000 codestream markers.
//!
//! Port of `ojph_params_local.h` and the implementation from `ojph_params.cpp`.

use std::f64::consts::LN_2;

use crate::arch::population_count;
use crate::error::{OjphError, Result};
use crate::file::{InfileBase, OutfileBase};
use crate::types::*;

// =========================================================================
// JPEG 2000 Marker Codes
// =========================================================================

#[allow(dead_code)]
pub(crate) mod markers {
    pub const SOC: u16 = 0xFF4F;
    pub const CAP: u16 = 0xFF50;
    pub const SIZ: u16 = 0xFF51;
    pub const COD: u16 = 0xFF52;
    pub const COC: u16 = 0xFF53;
    pub const TLM: u16 = 0xFF55;
    pub const PRF: u16 = 0xFF56;
    pub const PLM: u16 = 0xFF57;
    pub const PLT: u16 = 0xFF58;
    pub const CPF: u16 = 0xFF59;
    pub const QCD: u16 = 0xFF5C;
    pub const QCC: u16 = 0xFF5D;
    pub const RGN: u16 = 0xFF5E;
    pub const POC: u16 = 0xFF5F;
    pub const PPM: u16 = 0xFF60;
    pub const PPT: u16 = 0xFF61;
    pub const CRG: u16 = 0xFF63;
    pub const COM: u16 = 0xFF64;
    pub const DFS: u16 = 0xFF72;
    pub const ADS: u16 = 0xFF73;
    pub const NLT: u16 = 0xFF76;
    pub const ATK: u16 = 0xFF79;
    pub const SOT: u16 = 0xFF90;
    pub const SOP: u16 = 0xFF91;
    pub const EPH: u16 = 0xFF92;
    pub const SOD: u16 = 0xFF93;
    pub const EOC: u16 = 0xFFD9;
}

// =========================================================================
// Progression Orders
// =========================================================================

/// JPEG 2000 progression order for packet sequencing.
///
/// Determines the order in which packets are written into the codestream:
/// Layer (L), Resolution (R), Component (C), and Position/Precinct (P).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum ProgressionOrder {
    /// Layer–Resolution–Component–Position.
    LRCP = 0,
    /// Resolution–Layer–Component–Position.
    RLCP = 1,
    /// Resolution–Position–Component–Layer.
    RPCL = 2,
    /// Position–Component–Resolution–Layer.
    PCRL = 3,
    /// Component–Position–Resolution–Layer.
    CPRL = 4,
}

impl ProgressionOrder {
    /// Converts an integer to a progression order, if valid.
    pub fn from_i32(v: i32) -> Option<Self> {
        match v {
            0 => Some(Self::LRCP),
            1 => Some(Self::RLCP),
            2 => Some(Self::RPCL),
            3 => Some(Self::PCRL),
            4 => Some(Self::CPRL),
            _ => None,
        }
    }

    /// Returns the four-character string representation (e.g. `"LRCP"`).
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::LRCP => "LRCP",
            Self::RLCP => "RLCP",
            Self::RPCL => "RPCL",
            Self::PCRL => "PCRL",
            Self::CPRL => "CPRL",
        }
    }

    /// Parses a progression order from a case-insensitive string.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "LRCP" => Some(Self::LRCP),
            "RLCP" => Some(Self::RLCP),
            "RPCL" => Some(Self::RPCL),
            "PCRL" => Some(Self::PCRL),
            "CPRL" => Some(Self::CPRL),
            _ => None,
        }
    }
}

// =========================================================================
// Profile Numbers
// =========================================================================

/// JPEG 2000 codestream profile identifiers.
///
/// Profiles constrain certain codestream parameters to meet specific
/// application requirements (e.g. cinema, broadcast).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
#[allow(dead_code)]
pub enum ProfileNum {
    /// No profile specified.
    Undefined = 0,
    /// Profile 0.
    Profile0 = 1,
    /// Profile 1.
    Profile1 = 2,
    /// Digital Cinema 2K.
    Cinema2K = 3,
    /// Digital Cinema 4K.
    Cinema4K = 4,
    /// Scalable Digital Cinema 2K.
    CinemaS2K = 5,
    /// Scalable Digital Cinema 4K.
    CinemaS4K = 6,
    /// Broadcast profile.
    Broadcast = 7,
    /// Interoperable Master Format (IMF).
    Imf = 8,
}

impl ProfileNum {
    /// Parses a profile name from a case-insensitive string.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "PROFILE0" => Some(Self::Profile0),
            "PROFILE1" => Some(Self::Profile1),
            "CINEMA2K" => Some(Self::Cinema2K),
            "CINEMA4K" => Some(Self::Cinema4K),
            "CINEMAS2K" => Some(Self::CinemaS2K),
            "CINEMAS4K" => Some(Self::CinemaS4K),
            "BROADCAST" => Some(Self::Broadcast),
            "IMF" => Some(Self::Imf),
            _ => None,
        }
    }
}

// =========================================================================
// Tilepart Division flags
// =========================================================================

#[allow(dead_code)]
pub(crate) const TILEPART_NO_DIVISIONS: u32 = 0x0;
pub(crate) const TILEPART_RESOLUTIONS: u32 = 0x1;
pub(crate) const TILEPART_COMPONENTS: u32 = 0x2;
#[allow(dead_code)]
pub(crate) const TILEPART_LAYERS: u32 = 0x4;
#[allow(dead_code)]
pub(crate) const TILEPART_MASK: u32 = 0x3;

// =========================================================================
// Byte-swap helpers (big-endian I/O)
// =========================================================================

#[inline]
#[allow(dead_code)]
pub(crate) fn swap_byte_u16(v: u16) -> u16 {
    v.swap_bytes()
}

#[inline]
#[allow(dead_code)]
pub(crate) fn swap_byte_u32(v: u32) -> u32 {
    v.swap_bytes()
}

#[inline]
#[allow(dead_code)]
pub(crate) fn swap_byte_u64(v: u64) -> u64 {
    v.swap_bytes()
}

// Read/write big-endian helpers
fn read_u8(file: &mut dyn InfileBase) -> Result<u8> {
    let mut buf = [0u8; 1];
    if file.read(&mut buf)? != 1 {
        return Err(OjphError::Codec {
            code: 0,
            message: "unexpected EOF reading u8".into(),
        });
    }
    Ok(buf[0])
}

fn read_u16_be(file: &mut dyn InfileBase) -> Result<u16> {
    let mut buf = [0u8; 2];
    if file.read(&mut buf)? != 2 {
        return Err(OjphError::Codec {
            code: 0,
            message: "unexpected EOF reading u16".into(),
        });
    }
    Ok(u16::from_be_bytes(buf))
}

fn read_u32_be(file: &mut dyn InfileBase) -> Result<u32> {
    let mut buf = [0u8; 4];
    if file.read(&mut buf)? != 4 {
        return Err(OjphError::Codec {
            code: 0,
            message: "unexpected EOF reading u32".into(),
        });
    }
    Ok(u32::from_be_bytes(buf))
}

fn write_u8(file: &mut dyn OutfileBase, v: u8) -> Result<bool> {
    Ok(file.write(&[v])? == 1)
}

fn write_u16_be(file: &mut dyn OutfileBase, v: u16) -> Result<bool> {
    Ok(file.write(&v.to_be_bytes())? == 2)
}

fn write_u32_be(file: &mut dyn OutfileBase, v: u32) -> Result<bool> {
    Ok(file.write(&v.to_be_bytes())? == 4)
}

// =========================================================================
// SIZ component info
// =========================================================================

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct SizCompInfo {
    pub ssiz: u8,
    pub xr_siz: u8,
    pub yr_siz: u8,
}

// =========================================================================
// param_siz — Image and Tile Size marker segment
// =========================================================================

#[allow(dead_code)]
pub(crate) const RSIZ_NLT_FLAG: u16 = 0x200;
pub(crate) const RSIZ_HT_FLAG: u16 = 0x4000;
#[allow(dead_code)]
pub(crate) const RSIZ_EXT_FLAG: u16 = 0x8000;

/// SIZ marker segment — image and tile size parameters.
///
/// Contains the fundamental geometry of the image: reference grid size,
/// tile partitioning, image and tile offsets, and per-component
/// subsampling and bit-depth information.
///
/// # Examples
///
/// ```rust
/// use openjph_core::codestream::Codestream;
/// use openjph_core::types::{Point, Size};
///
/// let mut cs = Codestream::new();
/// let siz = cs.access_siz_mut();
/// siz.set_image_extent(Point::new(1920, 1080));
/// siz.set_tile_size(Size::new(1920, 1080));
/// siz.set_num_components(3);
/// for c in 0..3 {
///     siz.set_comp_info(c, Point::new(1, 1), 8, false);
/// }
/// assert_eq!(siz.get_num_components(), 3);
/// ```
#[derive(Debug)]
pub struct ParamSiz {
    pub(crate) lsiz: u16,
    pub(crate) rsiz: u16,
    pub(crate) xsiz: u32,
    pub(crate) ysiz: u32,
    pub(crate) xo_siz: u32,
    pub(crate) yo_siz: u32,
    pub(crate) xt_siz: u32,
    pub(crate) yt_siz: u32,
    pub(crate) xto_siz: u32,
    pub(crate) yto_siz: u32,
    pub(crate) csiz: u16,
    pub(crate) components: Vec<SizCompInfo>,
    pub(crate) skipped_resolutions: u32,
}

impl Default for ParamSiz {
    fn default() -> Self {
        Self {
            lsiz: 0,
            rsiz: RSIZ_HT_FLAG,
            xsiz: 0,
            ysiz: 0,
            xo_siz: 0,
            yo_siz: 0,
            xt_siz: 0,
            yt_siz: 0,
            xto_siz: 0,
            yto_siz: 0,
            csiz: 0,
            components: Vec::new(),
            skipped_resolutions: 0,
        }
    }
}

impl ParamSiz {
    /// Sets the image reference grid extent (Xsiz, Ysiz).
    pub fn set_image_extent(&mut self, extent: Point) {
        self.xsiz = extent.x;
        self.ysiz = extent.y;
    }

    /// Returns the image reference grid extent.
    pub fn get_image_extent(&self) -> Point {
        Point::new(self.xsiz, self.ysiz)
    }

    /// Sets the tile size (XTsiz, YTsiz).
    pub fn set_tile_size(&mut self, s: Size) {
        self.xt_siz = s.w;
        self.yt_siz = s.h;
    }

    /// Returns the tile size.
    pub fn get_tile_size(&self) -> Size {
        Size::new(self.xt_siz, self.yt_siz)
    }

    /// Sets the image origin offset (XOsiz, YOsiz).
    pub fn set_image_offset(&mut self, offset: Point) {
        self.xo_siz = offset.x;
        self.yo_siz = offset.y;
    }

    /// Returns the image origin offset.
    pub fn get_image_offset(&self) -> Point {
        Point::new(self.xo_siz, self.yo_siz)
    }

    /// Sets the tile grid origin offset (XTOsiz, YTOsiz).
    pub fn set_tile_offset(&mut self, offset: Point) {
        self.xto_siz = offset.x;
        self.yto_siz = offset.y;
    }

    /// Returns the tile grid origin offset.
    pub fn get_tile_offset(&self) -> Point {
        Point::new(self.xto_siz, self.yto_siz)
    }

    /// Sets the number of image components (Csiz) and allocates storage.
    pub fn set_num_components(&mut self, num_comps: u32) {
        self.csiz = num_comps as u16;
        self.components
            .resize(num_comps as usize, SizCompInfo::default());
    }

    /// Returns the number of image components.
    pub fn get_num_components(&self) -> u16 {
        self.csiz
    }

    /// Sets per-component information: subsampling factors, bit depth, and
    /// signedness.
    ///
    /// # Panics
    ///
    /// Debug-panics if `comp_num >= num_components` or if either
    /// downsampling factor is zero.
    pub fn set_comp_info(
        &mut self,
        comp_num: u32,
        downsampling: Point,
        bit_depth: u32,
        is_signed: bool,
    ) {
        debug_assert!(comp_num < self.csiz as u32);
        debug_assert!(downsampling.x != 0 && downsampling.y != 0);
        let c = &mut self.components[comp_num as usize];
        c.ssiz = (bit_depth - 1) as u8 + if is_signed { 0x80 } else { 0 };
        c.xr_siz = downsampling.x as u8;
        c.yr_siz = downsampling.y as u8;
    }

    /// Returns the bit depth (1–38) for the specified component.
    pub fn get_bit_depth(&self, comp_num: u32) -> u32 {
        debug_assert!(comp_num < self.csiz as u32);
        ((self.components[comp_num as usize].ssiz & 0x7F) + 1) as u32
    }

    /// Returns `true` if the specified component uses signed samples.
    pub fn is_signed(&self, comp_num: u32) -> bool {
        debug_assert!(comp_num < self.csiz as u32);
        (self.components[comp_num as usize].ssiz & 0x80) != 0
    }

    /// Returns the subsampling factors (XRsiz, YRsiz) for the specified component.
    pub fn get_downsampling(&self, comp_num: u32) -> Point {
        debug_assert!(comp_num < self.csiz as u32);
        let c = &self.components[comp_num as usize];
        Point::new(c.xr_siz as u32, c.yr_siz as u32)
    }

    /// Returns the width (in samples) of the specified component on the
    /// reference grid.
    pub fn get_width(&self, comp_num: u32) -> u32 {
        let ds = self.components[comp_num as usize].xr_siz as u32;
        div_ceil(self.xsiz, ds) - div_ceil(self.xo_siz, ds)
    }

    /// Returns the height (in samples) of the specified component on the
    /// reference grid.
    pub fn get_height(&self, comp_num: u32) -> u32 {
        let ds = self.components[comp_num as usize].yr_siz as u32;
        div_ceil(self.ysiz, ds) - div_ceil(self.yo_siz, ds)
    }

    /// Returns the reconstructed width accounting for skipped resolutions.
    pub fn get_recon_width(&self, comp_num: u32) -> u32 {
        let factor = self.get_recon_downsampling(comp_num);
        div_ceil(self.xsiz, factor.x) - div_ceil(self.xo_siz, factor.x)
    }

    /// Returns the reconstructed height accounting for skipped resolutions.
    pub fn get_recon_height(&self, comp_num: u32) -> u32 {
        let factor = self.get_recon_downsampling(comp_num);
        div_ceil(self.ysiz, factor.y) - div_ceil(self.yo_siz, factor.y)
    }

    /// Returns the effective downsampling factor for reconstruction,
    /// combining component subsampling with skipped resolutions.
    pub fn get_recon_downsampling(&self, comp_num: u32) -> Point {
        let sr = self.skipped_resolutions;
        let mut factor = Point::new(1u32 << sr, 1u32 << sr);
        factor.x *= self.components[comp_num as usize].xr_siz as u32;
        factor.y *= self.components[comp_num as usize].yr_siz as u32;
        factor
    }

    /// Sets a flag bit in the Rsiz field.
    pub fn set_rsiz_flag(&mut self, flag: u16) {
        self.rsiz |= flag;
    }

    /// Clears a flag bit in the Rsiz field.
    #[allow(dead_code)]
    pub fn reset_rsiz_flag(&mut self, flag: u16) {
        self.rsiz &= !flag;
    }

    /// Sets the number of resolution levels to skip during decoding.
    pub fn set_skipped_resolutions(&mut self, sr: u32) {
        self.skipped_resolutions = sr;
    }

    /// Validates the SIZ parameters.
    ///
    /// # Errors
    ///
    /// Returns [`OjphError::Codec`] if the image extent, tile size, or
    /// offsets are invalid (zero extent, bad offsets, etc.).
    pub fn check_validity(&self) -> Result<()> {
        if self.xsiz == 0 || self.ysiz == 0 || self.xt_siz == 0 || self.yt_siz == 0 {
            return Err(OjphError::Codec {
                code: 0x00040001,
                message: "Image extent and/or tile size cannot be zero".into(),
            });
        }
        if self.xto_siz > self.xo_siz || self.yto_siz > self.yo_siz {
            return Err(OjphError::Codec {
                code: 0x00040002,
                message: "Tile offset has to be smaller than the image offset".into(),
            });
        }
        if self.xt_siz + self.xto_siz <= self.xo_siz || self.yt_siz + self.yto_siz <= self.yo_siz {
            return Err(OjphError::Codec {
                code: 0x00040003,
                message: "The top left tile must intersect with the image".into(),
            });
        }
        if self.xsiz <= self.xo_siz || self.ysiz <= self.yo_siz {
            return Err(OjphError::Codec {
                code: 0x00040004,
                message: "Image extent must be larger than image offset".into(),
            });
        }
        Ok(())
    }

    pub fn write(&mut self, file: &mut dyn OutfileBase) -> Result<bool> {
        self.lsiz = 38 + 3 * self.csiz;
        let mut ok = true;
        ok &= write_u16_be(file, markers::SIZ)?;
        ok &= write_u16_be(file, self.lsiz)?;
        ok &= write_u16_be(file, self.rsiz)?;
        ok &= write_u32_be(file, self.xsiz)?;
        ok &= write_u32_be(file, self.ysiz)?;
        ok &= write_u32_be(file, self.xo_siz)?;
        ok &= write_u32_be(file, self.yo_siz)?;
        ok &= write_u32_be(file, self.xt_siz)?;
        ok &= write_u32_be(file, self.yt_siz)?;
        ok &= write_u32_be(file, self.xto_siz)?;
        ok &= write_u32_be(file, self.yto_siz)?;
        ok &= write_u16_be(file, self.csiz)?;
        for c in &self.components {
            ok &= write_u8(file, c.ssiz)?;
            ok &= write_u8(file, c.xr_siz)?;
            ok &= write_u8(file, c.yr_siz)?;
        }
        Ok(ok)
    }

    pub fn read(&mut self, file: &mut dyn InfileBase) -> Result<()> {
        self.lsiz = read_u16_be(file).map_err(|_| OjphError::Codec {
            code: 0x00050041,
            message: "error reading SIZ marker".into(),
        })?;
        if self.lsiz < 38 {
            return Err(OjphError::Codec {
                code: 0x00050042,
                message: "error in SIZ marker length".into(),
            });
        }
        let num_comps = ((self.lsiz - 38) / 3) as i32;
        if self.lsiz != 38 + 3 * num_comps as u16 {
            return Err(OjphError::Codec {
                code: 0x00050042,
                message: "error in SIZ marker length".into(),
            });
        }
        self.rsiz = read_u16_be(file).map_err(|_| OjphError::Codec {
            code: 0x00050043,
            message: "error reading SIZ marker".into(),
        })?;
        if (self.rsiz & 0x4000) == 0 {
            return Err(OjphError::Codec {
                code: 0x00050044,
                message: "Rsiz bit 14 is not set (this is not a JPH file)".into(),
            });
        }
        self.xsiz = read_u32_be(file).map_err(|_| OjphError::Codec {
            code: 0x00050045,
            message: "error reading SIZ marker".into(),
        })?;
        self.ysiz = read_u32_be(file).map_err(|_| OjphError::Codec {
            code: 0x00050046,
            message: "error reading SIZ marker".into(),
        })?;
        let xo = read_u32_be(file).map_err(|_| OjphError::Codec {
            code: 0x00050047,
            message: "error reading SIZ marker".into(),
        })?;
        let yo = read_u32_be(file).map_err(|_| OjphError::Codec {
            code: 0x00050048,
            message: "error reading SIZ marker".into(),
        })?;
        self.set_image_offset(Point::new(xo, yo));
        let xtw = read_u32_be(file).map_err(|_| OjphError::Codec {
            code: 0x00050049,
            message: "error reading SIZ marker".into(),
        })?;
        let yth = read_u32_be(file).map_err(|_| OjphError::Codec {
            code: 0x0005004A,
            message: "error reading SIZ marker".into(),
        })?;
        self.set_tile_size(Size::new(xtw, yth));
        let xto = read_u32_be(file).map_err(|_| OjphError::Codec {
            code: 0x0005004B,
            message: "error reading SIZ marker".into(),
        })?;
        let yto = read_u32_be(file).map_err(|_| OjphError::Codec {
            code: 0x0005004C,
            message: "error reading SIZ marker".into(),
        })?;
        self.set_tile_offset(Point::new(xto, yto));
        self.csiz = read_u16_be(file).map_err(|_| OjphError::Codec {
            code: 0x0005004D,
            message: "error reading SIZ marker".into(),
        })?;
        if self.csiz as i32 != num_comps {
            return Err(OjphError::Codec {
                code: 0x0005004E,
                message: "Csiz does not match the SIZ marker size".into(),
            });
        }
        self.set_num_components(self.csiz as u32);
        for c in 0..self.csiz as usize {
            self.components[c].ssiz = read_u8(file).map_err(|_| OjphError::Codec {
                code: 0x00050051,
                message: "error reading SIZ marker".into(),
            })?;
            self.components[c].xr_siz = read_u8(file).map_err(|_| OjphError::Codec {
                code: 0x00050052,
                message: "error reading SIZ marker".into(),
            })?;
            self.components[c].yr_siz = read_u8(file).map_err(|_| OjphError::Codec {
                code: 0x00050053,
                message: "error reading SIZ marker".into(),
            })?;
            if (self.components[c].ssiz & 0x7F) > 37 {
                return Err(OjphError::Codec {
                    code: 0x00050054,
                    message: format!("Wrong SIZ-SSiz value of {}", self.components[c].ssiz),
                });
            }
            if self.components[c].xr_siz == 0 {
                return Err(OjphError::Codec {
                    code: 0x00050055,
                    message: format!("Wrong SIZ-XRsiz value of {}", self.components[c].xr_siz),
                });
            }
            if self.components[c].yr_siz == 0 {
                return Err(OjphError::Codec {
                    code: 0x00050056,
                    message: format!("Wrong SIZ-YRsiz value of {}", self.components[c].yr_siz),
                });
            }
        }
        self.check_validity()?;
        Ok(())
    }
}

// =========================================================================
// COD/COC SPcod sub-structure
// =========================================================================

#[derive(Debug, Clone)]
pub(crate) struct CodSPcod {
    pub num_decomp: u8,
    pub block_width: u8,
    pub block_height: u8,
    pub block_style: u8,
    pub wavelet_trans: u8,
    pub precinct_size: [u8; 33],
}

impl Default for CodSPcod {
    fn default() -> Self {
        Self {
            num_decomp: 5,
            block_width: 4,    // 2^(4+2)=64
            block_height: 4,   // 2^(4+2)=64
            block_style: 0x40, // HT mode
            wavelet_trans: 0,  // reversible 5/3
            precinct_size: [0; 33],
        }
    }
}

impl CodSPcod {
    pub fn get_log_block_dims(&self) -> Size {
        Size::new(
            (self.block_width + 2) as u32,
            (self.block_height + 2) as u32,
        )
    }

    pub fn get_block_dims(&self) -> Size {
        let t = self.get_log_block_dims();
        Size::new(1 << t.w, 1 << t.h)
    }

    pub fn get_log_precinct_size(&self, res_num: u32) -> Size {
        let p = self.precinct_size[res_num as usize];
        Size::new((p & 0xF) as u32, (p >> 4) as u32)
    }
}

// COD SGcod sub-structure
#[derive(Debug, Clone)]
pub(crate) struct CodSGcod {
    pub prog_order: u8,
    pub num_layers: u16,
    pub mc_trans: u8,
}

impl Default for CodSGcod {
    fn default() -> Self {
        Self {
            prog_order: ProgressionOrder::RPCL as u8,
            num_layers: 1,
            mc_trans: 0,
        }
    }
}

// =========================================================================
// Block coding style constants
// =========================================================================

#[allow(dead_code)]
pub(crate) const VERT_CAUSAL_MODE: u8 = 0x8;
pub(crate) const HT_MODE: u8 = 0x40;

// DWT type
pub(crate) const DWT_IRV97: u8 = 0;
pub(crate) const DWT_REV53: u8 = 1;

// COD type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CodType {
    #[allow(dead_code)]
    Undefined,
    CodMain,
    CocMain,
}

// =========================================================================
// param_cod — Coding Style Default / Component
// =========================================================================

pub(crate) const COD_DEFAULT_COMP: u16 = 65535;
#[allow(dead_code)]
pub(crate) const COD_UNKNOWN_COMP: u16 = 65534;

/// COD/COC marker segment — coding style parameters.
///
/// Holds default coding parameters (COD) and optionally per-component
/// overrides (COC). Controls the wavelet transform, block coder settings,
/// precinct sizes, and progression order.
#[derive(Debug, Clone)]
pub struct ParamCod {
    pub(crate) cod_type: CodType,
    pub(crate) lcod: u16,
    pub(crate) scod: u8,
    pub(crate) sg_cod: CodSGcod,
    pub(crate) sp_cod: CodSPcod,
    pub(crate) comp_idx: u16,
    /// COC children chained here
    pub(crate) children: Vec<ParamCod>,
}

impl Default for ParamCod {
    fn default() -> Self {
        Self {
            cod_type: CodType::CodMain,
            lcod: 0,
            scod: 0,
            sg_cod: CodSGcod::default(),
            sp_cod: CodSPcod::default(),
            comp_idx: COD_DEFAULT_COMP,
            children: Vec::new(),
        }
    }
}

impl ParamCod {
    /// Creates a new COC (coding style component) instance for the given
    /// component index.
    pub fn new_coc(comp_idx: u16) -> Self {
        Self {
            cod_type: CodType::CocMain,
            lcod: 0,
            scod: 0,
            sg_cod: CodSGcod::default(),
            sp_cod: CodSPcod::default(),
            comp_idx,
            children: Vec::new(),
        }
    }

    /// Sets whether the wavelet transform is reversible (lossless 5/3) or
    /// irreversible (lossy 9/7).
    pub fn set_reversible(&mut self, reversible: bool) {
        self.sp_cod.wavelet_trans = if reversible { DWT_REV53 } else { DWT_IRV97 };
    }

    /// Enables or disables the multi-component (color) transform.
    ///
    /// When enabled with a reversible transform, RCT is used; with
    /// irreversible, ICT is used. Requires ≥ 3 components.
    pub fn set_color_transform(&mut self, ct: bool) {
        self.sg_cod.mc_trans = if ct { 1 } else { 0 };
    }

    /// Sets the number of DWT decomposition levels (0–32).
    pub fn set_num_decomposition(&mut self, num: u32) {
        self.sp_cod.num_decomp = num as u8;
    }

    /// Sets the code block dimensions. Both `width` and `height` must be
    /// powers of two in the range 4–1024 and their product ≤ 4096.
    pub fn set_block_dims(&mut self, width: u32, height: u32) {
        self.sp_cod.block_width = (width as f64).log2() as u8 - 2;
        self.sp_cod.block_height = (height as f64).log2() as u8 - 2;
    }

    /// Sets custom precinct sizes per resolution level.
    pub fn set_precinct_size(&mut self, num_levels: i32, sizes: &[Size]) {
        if num_levels > 0 && !sizes.is_empty() {
            self.scod |= 1;
            for i in 0..=self.sp_cod.num_decomp as usize {
                let idx = i.min(sizes.len() - 1);
                let w = (sizes[idx].w as f64).log2() as u8;
                let h = (sizes[idx].h as f64).log2() as u8;
                self.sp_cod.precinct_size[i] = w | (h << 4);
            }
        }
    }

    /// Sets the progression order by name (e.g. `"LRCP"`, `"RPCL"`).
    ///
    /// # Errors
    ///
    /// Returns [`OjphError::InvalidParam`] if `name` is not a recognised
    /// progression order string.
    pub fn set_progression_order(&mut self, name: &str) -> Result<()> {
        match ProgressionOrder::from_str(name) {
            Some(po) => {
                self.sg_cod.prog_order = po as u8;
                Ok(())
            }
            None => Err(OjphError::InvalidParam(format!(
                "unknown progression order: {}",
                name
            ))),
        }
    }

    /// Returns the number of DWT decomposition levels.
    pub fn get_num_decompositions(&self) -> u8 {
        if self.cod_type == CodType::CocMain && self.is_dfs_defined() {
            self.sp_cod.num_decomp & 0x7F
        } else {
            self.sp_cod.num_decomp
        }
    }

    /// Returns the code block dimensions.
    pub fn get_block_dims(&self) -> Size {
        self.sp_cod.get_block_dims()
    }

    /// Returns the log₂ code block dimensions.
    pub fn get_log_block_dims(&self) -> Size {
        self.sp_cod.get_log_block_dims()
    }

    /// Returns the wavelet kernel type (0 = irreversible 9/7, 1 = reversible 5/3).
    pub fn get_wavelet_kern(&self) -> u8 {
        self.sp_cod.wavelet_trans
    }

    /// Returns `true` if using the reversible (lossless) 5/3 wavelet.
    pub fn is_reversible(&self) -> bool {
        self.sp_cod.wavelet_trans == DWT_REV53
    }

    /// Returns `true` if the multi-component color transform is enabled.
    pub fn is_employing_color_transform(&self) -> bool {
        self.sg_cod.mc_trans == 1
    }

    /// Returns the precinct size at the given resolution level.
    pub fn get_precinct_size(&self, res_num: u32) -> Size {
        let t = self.get_log_precinct_size(res_num);
        Size::new(1 << t.w, 1 << t.h)
    }

    /// Returns the log₂ precinct size at the given resolution level.
    pub fn get_log_precinct_size(&self, res_num: u32) -> Size {
        if self.scod & 1 != 0 {
            self.sp_cod.get_log_precinct_size(res_num)
        } else {
            Size::new(15, 15)
        }
    }

    /// Returns `true` if SOP markers may be used in packets.
    pub fn packets_may_use_sop(&self) -> bool {
        if self.cod_type == CodType::CodMain {
            (self.scod & 2) == 2
        } else {
            false
        }
    }

    /// Returns `true` if EPH markers are used in packets.
    pub fn packets_use_eph(&self) -> bool {
        if self.cod_type == CodType::CodMain {
            (self.scod & 4) == 4
        } else {
            false
        }
    }

    /// Returns `true` if vertical causal context is enabled.
    pub fn get_block_vertical_causality(&self) -> bool {
        (self.sp_cod.block_style & VERT_CAUSAL_MODE) != 0
    }

    /// Returns the progression order as an integer.
    pub fn get_progression_order(&self) -> i32 {
        self.sg_cod.prog_order as i32
    }

    /// Returns the progression order as a four-character string.
    pub fn get_progression_order_as_string(&self) -> &'static str {
        ProgressionOrder::from_i32(self.sg_cod.prog_order as i32)
            .unwrap_or(ProgressionOrder::LRCP)
            .as_str()
    }

    /// Returns the number of quality layers.
    pub fn get_num_layers(&self) -> i32 {
        self.sg_cod.num_layers as i32
    }

    /// Returns `true` if a DFS marker is referenced.
    pub fn is_dfs_defined(&self) -> bool {
        (self.sp_cod.num_decomp & 0x80) != 0
    }

    /// Returns the DFS marker index.
    #[allow(dead_code)]
    pub fn get_dfs_index(&self) -> u16 {
        (self.sp_cod.num_decomp & 0xF) as u16
    }

    /// Returns the component index this COC applies to.
    pub fn get_comp_idx(&self) -> u16 {
        self.comp_idx
    }

    /// Returns the COC for a specific component, falling back to the COD
    /// defaults if no per-component override exists.
    pub fn get_coc(&self, comp_idx: u32) -> &ParamCod {
        for child in &self.children {
            if child.comp_idx == comp_idx as u16 {
                return child;
            }
        }
        self
    }

    /// Returns a mutable reference to the COC for a specific component.
    pub fn get_coc_mut(&mut self, comp_idx: u32) -> &mut ParamCod {
        for i in 0..self.children.len() {
            if self.children[i].comp_idx == comp_idx as u16 {
                return &mut self.children[i];
            }
        }
        self
    }

    /// Adds a new COC override for the specified component.
    pub fn add_coc(&mut self, comp_idx: u32) -> &mut ParamCod {
        let coc = ParamCod::new_coc(comp_idx as u16);
        self.children.push(coc);
        self.children.last_mut().unwrap()
    }

    pub fn check_validity(&self, siz: &ParamSiz) -> Result<()> {
        debug_assert!(self.cod_type == CodType::CodMain);
        let num_comps = siz.get_num_components();
        if self.sg_cod.mc_trans == 1 && num_comps < 3 {
            return Err(OjphError::Codec {
                code: 0x00040011,
                message: "color transform needs 3+ components".into(),
            });
        }
        if self.sg_cod.mc_trans == 1 {
            let p = siz.get_downsampling(0);
            let bd = siz.get_bit_depth(0);
            let s = siz.is_signed(0);
            for i in 1..3u32 {
                let pi = siz.get_downsampling(i);
                if p.x != pi.x || p.y != pi.y {
                    return Err(OjphError::Codec {
                        code: 0x00040012,
                        message:
                            "color transform requires same downsampling for first 3 components"
                                .into(),
                    });
                }
                if bd != siz.get_bit_depth(i) {
                    return Err(OjphError::Codec {
                        code: 0x00040014,
                        message: "color transform requires same bit depth for first 3 components"
                            .into(),
                    });
                }
                if s != siz.is_signed(i) {
                    return Err(OjphError::Codec {
                        code: 0x00040015,
                        message: "color transform requires same signedness for first 3 components"
                            .into(),
                    });
                }
            }
        }
        Ok(())
    }

    pub fn write(&mut self, file: &mut dyn OutfileBase) -> Result<bool> {
        debug_assert!(self.cod_type == CodType::CodMain);
        self.lcod = 12;
        if self.scod & 1 != 0 {
            self.lcod += 1 + self.sp_cod.num_decomp as u16;
        }
        let mut ok = true;
        ok &= write_u16_be(file, markers::COD)?;
        ok &= write_u16_be(file, self.lcod)?;
        ok &= write_u8(file, self.scod)?;
        ok &= write_u8(file, self.sg_cod.prog_order)?;
        ok &= write_u16_be(file, self.sg_cod.num_layers)?;
        ok &= write_u8(file, self.sg_cod.mc_trans)?;
        ok &= write_u8(file, self.sp_cod.num_decomp)?;
        ok &= write_u8(file, self.sp_cod.block_width)?;
        ok &= write_u8(file, self.sp_cod.block_height)?;
        ok &= write_u8(file, self.sp_cod.block_style)?;
        ok &= write_u8(file, self.sp_cod.wavelet_trans)?;
        if self.scod & 1 != 0 {
            for i in 0..=self.sp_cod.num_decomp as usize {
                ok &= write_u8(file, self.sp_cod.precinct_size[i])?;
            }
        }
        Ok(ok)
    }

    pub fn write_coc(&self, file: &mut dyn OutfileBase, num_comps: u32) -> Result<bool> {
        let mut ok = true;
        for child in &self.children {
            if (child.comp_idx as u32) < num_comps {
                ok &= child.internal_write_coc(file, num_comps)?;
            }
        }
        Ok(ok)
    }

    fn internal_write_coc(&self, file: &mut dyn OutfileBase, num_comps: u32) -> Result<bool> {
        let lcod: u16 = if num_comps < 257 { 9 } else { 10 }
            + if self.scod & 1 != 0 {
                1 + self.sp_cod.num_decomp as u16
            } else {
                0
            };
        let mut ok = true;
        ok &= write_u16_be(file, markers::COC)?;
        ok &= write_u16_be(file, lcod)?;
        if num_comps < 257 {
            ok &= write_u8(file, self.comp_idx as u8)?;
        } else {
            ok &= write_u16_be(file, self.comp_idx)?;
        }
        ok &= write_u8(file, self.scod)?;
        ok &= write_u8(file, self.sp_cod.num_decomp)?;
        ok &= write_u8(file, self.sp_cod.block_width)?;
        ok &= write_u8(file, self.sp_cod.block_height)?;
        ok &= write_u8(file, self.sp_cod.block_style)?;
        ok &= write_u8(file, self.sp_cod.wavelet_trans)?;
        if self.scod & 1 != 0 {
            for i in 0..=self.sp_cod.num_decomp as usize {
                ok &= write_u8(file, self.sp_cod.precinct_size[i])?;
            }
        }
        Ok(ok)
    }

    pub fn read(&mut self, file: &mut dyn InfileBase) -> Result<()> {
        self.lcod = read_u16_be(file).map_err(|_| OjphError::Codec {
            code: 0x00050071,
            message: "error reading COD segment".into(),
        })?;
        self.scod = read_u8(file).map_err(|_| OjphError::Codec {
            code: 0x00050072,
            message: "error reading COD segment".into(),
        })?;
        self.sg_cod.prog_order = read_u8(file).map_err(|_| OjphError::Codec {
            code: 0x00050073,
            message: "error reading COD segment".into(),
        })?;
        self.sg_cod.num_layers = read_u16_be(file).map_err(|_| OjphError::Codec {
            code: 0x00050074,
            message: "error reading COD segment".into(),
        })?;
        self.sg_cod.mc_trans = read_u8(file).map_err(|_| OjphError::Codec {
            code: 0x00050075,
            message: "error reading COD segment".into(),
        })?;
        self.sp_cod.num_decomp = read_u8(file).map_err(|_| OjphError::Codec {
            code: 0x00050076,
            message: "error reading COD segment".into(),
        })?;
        self.sp_cod.block_width = read_u8(file).map_err(|_| OjphError::Codec {
            code: 0x00050077,
            message: "error reading COD segment".into(),
        })?;
        self.sp_cod.block_height = read_u8(file).map_err(|_| OjphError::Codec {
            code: 0x00050078,
            message: "error reading COD segment".into(),
        })?;
        self.sp_cod.block_style = read_u8(file).map_err(|_| OjphError::Codec {
            code: 0x00050079,
            message: "error reading COD segment".into(),
        })?;
        self.sp_cod.wavelet_trans = read_u8(file).map_err(|_| OjphError::Codec {
            code: 0x0005007A,
            message: "error reading COD segment".into(),
        })?;

        if self.get_num_decompositions() > 32
            || self.sp_cod.block_width > 8
            || self.sp_cod.block_height > 8
            || self.sp_cod.block_width + self.sp_cod.block_height > 8
            || (self.sp_cod.block_style & HT_MODE) != HT_MODE
            || (self.sp_cod.block_style & 0xB7) != 0x00
        {
            return Err(OjphError::Codec {
                code: 0x0005007D,
                message: "wrong settings in COD-SPcod parameter".into(),
            });
        }

        let nd = self.get_num_decompositions();
        if self.scod & 1 != 0 {
            for i in 0..=nd as usize {
                self.sp_cod.precinct_size[i] = read_u8(file).map_err(|_| OjphError::Codec {
                    code: 0x0005007B,
                    message: "error reading COD segment".into(),
                })?;
            }
        }
        let expected = 12
            + if self.scod & 1 != 0 {
                1 + self.sp_cod.num_decomp as u16
            } else {
                0
            };
        if self.lcod != expected {
            return Err(OjphError::Codec {
                code: 0x0005007C,
                message: "error in COD segment length".into(),
            });
        }
        Ok(())
    }

    pub fn read_coc(&mut self, file: &mut dyn InfileBase, num_comps: u32) -> Result<()> {
        self.cod_type = CodType::CocMain;
        self.lcod = read_u16_be(file).map_err(|_| OjphError::Codec {
            code: 0x00050121,
            message: "error reading COC segment".into(),
        })?;
        if num_comps < 257 {
            self.comp_idx = read_u8(file).map_err(|_| OjphError::Codec {
                code: 0x00050122,
                message: "error reading COC segment".into(),
            })? as u16;
        } else {
            self.comp_idx = read_u16_be(file).map_err(|_| OjphError::Codec {
                code: 0x00050123,
                message: "error reading COC segment".into(),
            })?;
        }
        self.scod = read_u8(file).map_err(|_| OjphError::Codec {
            code: 0x00050124,
            message: "error reading COC segment".into(),
        })?;
        self.sp_cod.num_decomp = read_u8(file).map_err(|_| OjphError::Codec {
            code: 0x00050125,
            message: "error reading COC segment".into(),
        })?;
        self.sp_cod.block_width = read_u8(file).map_err(|_| OjphError::Codec {
            code: 0x00050126,
            message: "error reading COC segment".into(),
        })?;
        self.sp_cod.block_height = read_u8(file).map_err(|_| OjphError::Codec {
            code: 0x00050127,
            message: "error reading COC segment".into(),
        })?;
        self.sp_cod.block_style = read_u8(file).map_err(|_| OjphError::Codec {
            code: 0x00050128,
            message: "error reading COC segment".into(),
        })?;
        self.sp_cod.wavelet_trans = read_u8(file).map_err(|_| OjphError::Codec {
            code: 0x00050129,
            message: "error reading COC segment".into(),
        })?;

        if self.get_num_decompositions() > 32
            || self.sp_cod.block_width > 8
            || self.sp_cod.block_height > 8
            || (self.sp_cod.block_style & HT_MODE) != HT_MODE
            || (self.sp_cod.block_style & 0xB7) != 0x00
        {
            return Err(OjphError::Codec {
                code: 0x0005012C,
                message: "wrong settings in COC-SPcoc parameter".into(),
            });
        }

        let nd = self.get_num_decompositions();
        if self.scod & 1 != 0 {
            for i in 0..=nd as usize {
                self.sp_cod.precinct_size[i] = read_u8(file).map_err(|_| OjphError::Codec {
                    code: 0x0005012A,
                    message: "error reading COC segment".into(),
                })?;
            }
        }
        let mut expected: u32 = 9 + if num_comps < 257 { 0 } else { 1 };
        expected += if self.scod & 1 != 0 { 1 + nd as u32 } else { 0 };
        if self.lcod as u32 != expected {
            return Err(OjphError::Codec {
                code: 0x0005012B,
                message: "error in COC segment length".into(),
            });
        }
        Ok(())
    }
}

// =========================================================================
// param_qcd — Quantization Default / Component
// =========================================================================

pub(crate) const QCD_DEFAULT_COMP: u16 = 65535;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum QcdType {
    QcdMain,
    QccMain,
}

/// Quantization step data — reversible uses u8, irreversible uses u16.
#[derive(Debug, Clone)]
pub(crate) enum SpqcdData {
    Reversible(Vec<u8>),
    Irreversible(Vec<u16>),
}

impl Default for SpqcdData {
    fn default() -> Self {
        SpqcdData::Reversible(Vec::new())
    }
}

/// QCD/QCC marker segment — quantization parameters.
///
/// Defines the quantization step sizes for all subbands (QCD = default,
/// QCC = per-component override). For reversible transforms, the step
/// sizes encode exponent information; for irreversible transforms,
/// mantissa+exponent pairs are stored.
#[derive(Debug, Clone)]
pub struct ParamQcd {
    pub(crate) qcd_type: QcdType,
    pub(crate) lqcd: u16,
    pub(crate) sqcd: u8,
    pub(crate) sp_qcd: SpqcdData,
    pub(crate) num_subbands: u32,
    pub(crate) base_delta: f32,
    pub(crate) enabled: bool,
    pub(crate) comp_idx: u16,
    /// QCC children chained here (owned)
    pub(crate) children: Vec<ParamQcd>,
}

impl Default for ParamQcd {
    fn default() -> Self {
        Self {
            qcd_type: QcdType::QcdMain,
            lqcd: 0,
            sqcd: 0,
            sp_qcd: SpqcdData::default(),
            num_subbands: 0,
            base_delta: -1.0,
            enabled: true,
            comp_idx: QCD_DEFAULT_COMP,
            children: Vec::new(),
        }
    }
}

// BIBO gain tables for reversible quantization
mod bibo_gains {
    pub fn get_bibo_gain_l(num_decomps: u32, _rev: bool) -> f64 {
        static GAINS_L: [f64; 34] = [
            1.0,
            1.5,
            2.0,
            2.75,
            3.6875,
            4.96875,
            6.671875,
            8.953125,
            12.015625,
            16.125,
            21.65625,
            29.078125,
            39.046875,
            52.4375,
            70.421875,
            94.578125,
            127.015625,
            170.53125,
            229.015625,
            307.578125,
            413.015625,
            554.578125,
            744.734375,
            1000.234375,
            1343.015625,
            1803.234375,
            2420.734375,
            3250.734375,
            4365.234375,
            5862.234375,
            7871.234375,
            10571.234375,
            14198.734375,
            19066.734375,
        ];
        GAINS_L[num_decomps.min(33) as usize]
    }

    pub fn get_bibo_gain_h(num_decomps: u32, _rev: bool) -> f64 {
        static GAINS_H: [f64; 34] = [
            2.0,
            2.75,
            3.6875,
            4.96875,
            6.671875,
            8.953125,
            12.015625,
            16.125,
            21.65625,
            29.078125,
            39.046875,
            52.4375,
            70.421875,
            94.578125,
            127.015625,
            170.53125,
            229.015625,
            307.578125,
            413.015625,
            554.578125,
            744.734375,
            1000.234375,
            1343.015625,
            1803.234375,
            2420.734375,
            3250.734375,
            4365.234375,
            5862.234375,
            7871.234375,
            10571.234375,
            14198.734375,
            19066.734375,
            25606.234375,
            34394.234375,
        ];
        GAINS_H[num_decomps.min(33) as usize]
    }
}

// Sqrt energy gains for irreversible quantization
#[allow(clippy::excessive_precision)]
mod sqrt_energy_gains {
    pub fn get_gain_l(num_decomps: u32, _rev: bool) -> f32 {
        static GAINS_L: [f32; 34] = [
            1.0, 1.4021, 1.9692, 2.7665, 3.8873, 5.4645, 7.6816, 10.7968, 15.1781, 21.3348,
            29.9852, 42.1538, 59.2485, 83.2750, 117.0424, 164.4989, 231.2285, 325.0069, 456.8019,
            641.9960, 902.5009, 1268.6768, 1783.6345, 2506.8855, 3522.8044, 4952.2319, 6960.2544,
            9781.8203, 13748.4473, 19325.1445, 27167.4688, 38181.2344, 53668.6953, 75428.9531,
        ];
        GAINS_L[num_decomps.min(33) as usize]
    }

    pub fn get_gain_h(num_decomps: u32, _rev: bool) -> f32 {
        static GAINS_H: [f32; 34] = [
            1.0, 1.4425, 2.0286, 2.8525, 4.0104, 5.6381, 7.9270, 11.1440, 15.6658, 22.0236,
            30.9537, 43.5079, 61.1495, 85.9350, 120.8194, 169.8440, 238.7607, 335.6575, 471.8611,
            663.2765, 932.3842, 1310.5906, 1842.0957, 2589.3047, 3639.6855, 5116.5957, 7193.1211,
            10112.9805, 14213.5547, 19979.6094, 28089.3984, 39483.7109, 55517.6484, 78048.0000,
        ];
        GAINS_H[num_decomps.min(33) as usize]
    }
}

impl ParamQcd {
    /// Creates a new QCC (per-component quantization) instance.
    pub fn new_qcc(comp_idx: u16) -> Self {
        Self {
            qcd_type: QcdType::QccMain,
            comp_idx,
            ..Default::default()
        }
    }

    /// Sets the base quantization step size (Δ) for irreversible coding.
    ///
    /// Typical values are in the range 0.0001–1.0. Smaller values yield
    /// higher quality and larger files.
    pub fn set_delta(&mut self, delta: f32) {
        self.base_delta = delta;
    }

    /// Sets the base quantization step size for a specific component.
    pub fn set_delta_for_comp(&mut self, comp_idx: u32, delta: f32) {
        let qcc = self.get_or_add_qcc(comp_idx);
        qcc.base_delta = delta;
    }

    /// Returns the number of guard bits.
    pub fn get_num_guard_bits(&self) -> u32 {
        (self.sqcd >> 5) as u32
    }

    fn decode_spqcd(&self, v: u8) -> u8 {
        v >> 3
    }

    fn encode_spqcd(&self, v: u8) -> u8 {
        v << 3
    }

    pub fn get_magb(&self) -> u32 {
        let mut b = 0u32;
        self.compute_magb_for(&mut b);
        for child in &self.children {
            child.compute_magb_for(&mut b);
        }
        b
    }

    fn compute_magb_for(&self, b: &mut u32) {
        let num_decomps = (self.num_subbands.saturating_sub(1)) / 3;
        let irrev = self.sqcd & 0x1F;
        if irrev == 0 {
            if let SpqcdData::Reversible(ref data) = self.sp_qcd {
                for i in 0..self.num_subbands.min(data.len() as u32) {
                    let t =
                        self.decode_spqcd(data[i as usize]) as u32 + self.get_num_guard_bits() - 1;
                    *b = (*b).max(t);
                }
            }
        } else if irrev == 2 {
            if let SpqcdData::Irreversible(ref data) = self.sp_qcd {
                for i in 0..self.num_subbands.min(data.len() as u32) {
                    let nb = num_decomps - if i > 0 { (i - 1) / 3 } else { 0 };
                    let t = (data[i as usize] >> 11) as u32 + self.get_num_guard_bits() - nb;
                    *b = (*b).max(t);
                }
            }
        }
    }

    pub fn get_kmax(&self, _num_decompositions: u32, resolution: u32, subband: u32) -> u32 {
        let idx = if resolution > 0 {
            (resolution - 1) * 3 + subband
        } else {
            0
        };
        let idx = idx.min(self.num_subbands.saturating_sub(1));
        let irrev = self.sqcd & 0x1F;
        let num_bits = if irrev == 0 {
            if let SpqcdData::Reversible(ref data) = self.sp_qcd {
                let v = self.decode_spqcd(data[idx as usize]);
                if v == 0 {
                    0u32
                } else {
                    v as u32 - 1
                }
            } else {
                0
            }
        } else if irrev == 2 {
            if let SpqcdData::Irreversible(ref data) = self.sp_qcd {
                (data[idx as usize] >> 11) as u32 - 1
            } else {
                0
            }
        } else {
            0
        };
        num_bits + self.get_num_guard_bits()
    }

    pub fn get_irrev_delta(&self, _num_decompositions: u32, resolution: u32, subband: u32) -> f32 {
        let arr: [f32; 4] = [1.0, 2.0, 2.0, 4.0];
        let idx = if resolution > 0 {
            (resolution - 1) * 3 + subband
        } else {
            0
        };
        let idx = idx.min(self.num_subbands.saturating_sub(1));
        if let SpqcdData::Irreversible(ref data) = self.sp_qcd {
            let eps = data[idx as usize] >> 11;
            let mut mantissa =
                ((data[idx as usize] & 0x7FF) | 0x800) as f32 * arr[subband as usize];
            mantissa /= (1u32 << 11) as f32;
            mantissa /= (1u32 << eps) as f32;
            mantissa
        } else {
            1.0
        }
    }

    fn set_rev_quant(&mut self, num_decomps: u32, bit_depth: u32, employing_ct: bool) {
        // Keep one extra magnitude bit in the reversible quantization exponents so
        // the full centered sample range (e.g. unsigned 8-bit -> [-128, 127]) is
        // representable after the JPEG 2000 sign-magnitude conversion.
        let b = bit_depth + if employing_ct { 1 } else { 0 } + 1;
        let ns = 1 + 3 * num_decomps;
        let mut sp = vec![0u8; ns as usize];
        let mut s = 0usize;
        let bibo_l = bibo_gains::get_bibo_gain_l(num_decomps, true);
        let x = (bibo_l * bibo_l).ln() / LN_2;
        let x = x.ceil() as u32;
        sp[s] = (b + x) as u8;
        let mut max_bpx = b + x;
        s += 1;
        for d in (1..=num_decomps).rev() {
            let bl = bibo_gains::get_bibo_gain_l(d, true);
            let bh = bibo_gains::get_bibo_gain_h(d - 1, true);
            let x = ((bh * bl).ln() / LN_2).ceil() as u32;
            sp[s] = (b + x) as u8;
            max_bpx = max_bpx.max(b + x);
            s += 1;
            sp[s] = (b + x) as u8;
            max_bpx = max_bpx.max(b + x);
            s += 1;
            let x = ((bh * bh).ln() / LN_2).ceil() as u32;
            sp[s] = (b + x) as u8;
            max_bpx = max_bpx.max(b + x);
            s += 1;
        }
        let guard_bits = 1i32.max(max_bpx as i32 - 31);
        self.sqcd = (guard_bits as u8) << 5;
        for v in sp.iter_mut() {
            *v = self.encode_spqcd((*v as i32 - guard_bits) as u8);
        }
        self.sp_qcd = SpqcdData::Reversible(sp);
        self.num_subbands = ns;
    }

    fn set_irrev_quant(&mut self, num_decomps: u32) {
        let guard_bits = 1u8;
        self.sqcd = (guard_bits << 5) | 0x2;
        let ns = 1 + 3 * num_decomps;
        let mut sp = vec![0u16; ns as usize];
        let mut s = 0usize;

        let gain_l = sqrt_energy_gains::get_gain_l(num_decomps, false);
        let delta_b = self.base_delta / (gain_l * gain_l);
        let (exp, mantissa) = quantize_delta(delta_b);
        sp[s] = (exp << 11) | mantissa;
        s += 1;

        for d in (1..=num_decomps).rev() {
            let gl = sqrt_energy_gains::get_gain_l(d, false);
            let gh = sqrt_energy_gains::get_gain_h(d - 1, false);

            let delta_b = self.base_delta / (gl * gh);
            let (exp, mantissa) = quantize_delta(delta_b);
            sp[s] = (exp << 11) | mantissa;
            s += 1;
            sp[s] = (exp << 11) | mantissa;
            s += 1;

            let delta_b = self.base_delta / (gh * gh);
            let (exp, mantissa) = quantize_delta(delta_b);
            sp[s] = (exp << 11) | mantissa;
            s += 1;
        }
        self.sp_qcd = SpqcdData::Irreversible(sp);
        self.num_subbands = ns;
    }

    pub fn check_validity(&mut self, siz: &ParamSiz, cod: &ParamCod) -> Result<()> {
        let num_comps = siz.get_num_components() as u32;

        let qcd_num_decomps = cod.get_num_decompositions() as u32;
        let qcd_bit_depth = siz.get_bit_depth(0);
        let qcd_wavelet_kern = cod.get_wavelet_kern();
        let employing_ct = cod.is_employing_color_transform();

        self.num_subbands = 1 + 3 * qcd_num_decomps;
        if qcd_wavelet_kern == DWT_REV53 {
            self.set_rev_quant(qcd_num_decomps, qcd_bit_depth, employing_ct);
        } else {
            if self.base_delta < 0.0 {
                self.base_delta = 1.0 / (1u32 << qcd_bit_depth) as f32;
            }
            self.set_irrev_quant(qcd_num_decomps);
        }

        // Process QCC children
        for child in &mut self.children {
            if child.comp_idx >= num_comps as u16 {
                child.enabled = false;
                continue;
            }
            let c = child.comp_idx as u32;
            let cp = cod.get_coc(c);
            let nd = cp.get_num_decompositions() as u32;
            child.num_subbands = 1 + 3 * nd;
            let bd = siz.get_bit_depth(c);
            if cp.get_wavelet_kern() == DWT_REV53 {
                child.set_rev_quant(nd, bd, if c < 3 { employing_ct } else { false });
            } else {
                if child.base_delta < 0.0 {
                    child.base_delta = 1.0 / (1u32 << bd) as f32;
                }
                child.set_irrev_quant(nd);
            }
        }
        Ok(())
    }

    /// Get QCC for a component, or self if not found
    pub fn get_qcc(&self, comp_idx: u32) -> &ParamQcd {
        for child in &self.children {
            if child.comp_idx == comp_idx as u16 {
                return child;
            }
        }
        self
    }

    pub fn get_or_add_qcc(&mut self, comp_idx: u32) -> &mut ParamQcd {
        for i in 0..self.children.len() {
            if self.children[i].comp_idx == comp_idx as u16 {
                return &mut self.children[i];
            }
        }
        let qcc = ParamQcd::new_qcc(comp_idx as u16);
        self.children.push(qcc);
        self.children.last_mut().unwrap()
    }

    pub fn write(&mut self, file: &mut dyn OutfileBase) -> Result<bool> {
        let irrev = self.sqcd & 0x1F;
        self.lqcd = 3;
        if irrev == 0 {
            self.lqcd += self.num_subbands as u16;
        } else if irrev == 2 {
            self.lqcd += 2 * self.num_subbands as u16;
        }
        let mut ok = true;
        ok &= write_u16_be(file, markers::QCD)?;
        ok &= write_u16_be(file, self.lqcd)?;
        ok &= write_u8(file, self.sqcd)?;
        match &self.sp_qcd {
            SpqcdData::Reversible(data) => {
                for item in data.iter().take(self.num_subbands as usize) {
                    ok &= write_u8(file, *item)?;
                }
            }
            SpqcdData::Irreversible(data) => {
                for item in data.iter().take(self.num_subbands as usize) {
                    ok &= write_u16_be(file, *item)?;
                }
            }
        }
        Ok(ok)
    }

    pub fn write_qcc(&self, file: &mut dyn OutfileBase, num_comps: u32) -> Result<bool> {
        let mut ok = true;
        for child in &self.children {
            if child.enabled {
                ok &= child.internal_write_qcc(file, num_comps)?;
            }
        }
        Ok(ok)
    }

    fn internal_write_qcc(&self, file: &mut dyn OutfileBase, num_comps: u32) -> Result<bool> {
        let irrev = self.sqcd & 0x1F;
        let mut lqcd: u16 = 4 + if num_comps < 257 { 0 } else { 1 };
        if irrev == 0 {
            lqcd += self.num_subbands as u16;
        } else if irrev == 2 {
            lqcd += 2 * self.num_subbands as u16;
        }
        let mut ok = true;
        ok &= write_u16_be(file, markers::QCC)?;
        ok &= write_u16_be(file, lqcd)?;
        if num_comps < 257 {
            ok &= write_u8(file, self.comp_idx as u8)?;
        } else {
            ok &= write_u16_be(file, self.comp_idx)?;
        }
        ok &= write_u8(file, self.sqcd)?;
        match &self.sp_qcd {
            SpqcdData::Reversible(data) => {
                for item in data.iter().take(self.num_subbands as usize) {
                    ok &= write_u8(file, *item)?;
                }
            }
            SpqcdData::Irreversible(data) => {
                for item in data.iter().take(self.num_subbands as usize) {
                    ok &= write_u16_be(file, *item)?;
                }
            }
        }
        Ok(ok)
    }

    pub fn read(&mut self, file: &mut dyn InfileBase) -> Result<()> {
        self.lqcd = read_u16_be(file).map_err(|_| OjphError::Codec {
            code: 0x00050081,
            message: "error reading QCD marker".into(),
        })?;
        self.sqcd = read_u8(file).map_err(|_| OjphError::Codec {
            code: 0x00050082,
            message: "error reading QCD marker".into(),
        })?;
        let irrev = self.sqcd & 0x1F;
        if irrev == 0 {
            self.num_subbands = (self.lqcd - 3) as u32;
            if self.num_subbands > 97 || self.lqcd != 3 + self.num_subbands as u16 {
                return Err(OjphError::Codec {
                    code: 0x00050083,
                    message: format!("wrong Lqcd value of {} in QCD marker", self.lqcd),
                });
            }
            let mut data = vec![0u8; self.num_subbands as usize];
            for item in data.iter_mut().take(self.num_subbands as usize) {
                *item = read_u8(file).map_err(|_| OjphError::Codec {
                    code: 0x00050084,
                    message: "error reading QCD marker".into(),
                })?;
            }
            self.sp_qcd = SpqcdData::Reversible(data);
        } else if irrev == 2 {
            self.num_subbands = ((self.lqcd - 3) / 2) as u32;
            if self.num_subbands > 97 || self.lqcd != 3 + 2 * self.num_subbands as u16 {
                return Err(OjphError::Codec {
                    code: 0x00050086,
                    message: format!("wrong Lqcd value of {} in QCD marker", self.lqcd),
                });
            }
            let mut data = vec![0u16; self.num_subbands as usize];
            for item in data.iter_mut().take(self.num_subbands as usize) {
                *item = read_u16_be(file).map_err(|_| OjphError::Codec {
                    code: 0x00050087,
                    message: "error reading QCD marker".into(),
                })?;
            }
            self.sp_qcd = SpqcdData::Irreversible(data);
        } else {
            return Err(OjphError::Codec {
                code: 0x00050088,
                message: "wrong Sqcd value in QCD marker".into(),
            });
        }
        Ok(())
    }

    pub fn read_qcc(&mut self, file: &mut dyn InfileBase, num_comps: u32) -> Result<()> {
        self.qcd_type = QcdType::QccMain;
        self.lqcd = read_u16_be(file).map_err(|_| OjphError::Codec {
            code: 0x000500A1,
            message: "error reading QCC marker".into(),
        })?;
        if num_comps < 257 {
            self.comp_idx = read_u8(file).map_err(|_| OjphError::Codec {
                code: 0x000500A2,
                message: "error reading QCC marker".into(),
            })? as u16;
        } else {
            self.comp_idx = read_u16_be(file).map_err(|_| OjphError::Codec {
                code: 0x000500A3,
                message: "error reading QCC marker".into(),
            })?;
        }
        self.sqcd = read_u8(file).map_err(|_| OjphError::Codec {
            code: 0x000500A4,
            message: "error reading QCC marker".into(),
        })?;
        let offset: u32 = if num_comps < 257 { 4 } else { 5 };
        let irrev = self.sqcd & 0x1F;
        if irrev == 0 {
            self.num_subbands = (self.lqcd as u32).saturating_sub(offset);
            let mut data = vec![0u8; self.num_subbands as usize];
            for item in data.iter_mut().take(self.num_subbands as usize) {
                *item = read_u8(file).map_err(|_| OjphError::Codec {
                    code: 0x000500A6,
                    message: "error reading QCC marker".into(),
                })?;
            }
            self.sp_qcd = SpqcdData::Reversible(data);
        } else if irrev == 2 {
            self.num_subbands = ((self.lqcd as u32).saturating_sub(offset)) / 2;
            let mut data = vec![0u16; self.num_subbands as usize];
            for item in data.iter_mut().take(self.num_subbands as usize) {
                *item = read_u16_be(file).map_err(|_| OjphError::Codec {
                    code: 0x000500A9,
                    message: "error reading QCC marker".into(),
                })?;
            }
            self.sp_qcd = SpqcdData::Irreversible(data);
        } else {
            return Err(OjphError::Codec {
                code: 0x000500AA,
                message: "wrong Sqcc value in QCC marker".into(),
            });
        }
        Ok(())
    }
}

fn quantize_delta(mut delta_b: f32) -> (u16, u16) {
    let mut exp: u16 = 0;
    while delta_b < 1.0 {
        exp += 1;
        delta_b *= 2.0;
    }
    let mut mantissa = (delta_b * (1u32 << 11) as f32).round() as i32 - (1i32 << 11);
    mantissa = mantissa.min(0x7FF);
    (exp, mantissa as u16)
}

// =========================================================================
// param_cap — Extended Capability marker
// =========================================================================

/// CAP marker segment — extended capability descriptor.
///
/// Identifies HTJ2K (Part 15) capabilities and parameters such as the
/// Ccap value that encodes the wavelet type and magnitude bound.
#[derive(Debug, Clone)]
pub struct ParamCap {
    pub(crate) lcap: u16,
    pub(crate) pcap: u32,
    pub(crate) ccap: [u16; 32],
}

impl Default for ParamCap {
    fn default() -> Self {
        let mut cap = Self {
            lcap: 8,
            pcap: 0x00020000,
            ccap: [0u16; 32],
        };
        cap.ccap[0] = 0;
        cap
    }
}

impl ParamCap {
    pub fn check_validity(&mut self, cod: &ParamCod, qcd: &ParamQcd) {
        if cod.get_wavelet_kern() == DWT_REV53 {
            self.ccap[0] &= 0xFFDF;
        } else {
            self.ccap[0] |= 0x0020;
        }
        self.ccap[0] &= 0xFFE0;
        let b = qcd.get_magb();
        let bp = if b <= 8 {
            0
        } else if b < 28 {
            b - 8
        } else {
            13 + (b >> 2)
        };
        self.ccap[0] |= bp as u16;
    }

    pub fn write(&self, file: &mut dyn OutfileBase) -> Result<bool> {
        let mut ok = true;
        ok &= write_u16_be(file, markers::CAP)?;
        ok &= write_u16_be(file, self.lcap)?;
        ok &= write_u32_be(file, self.pcap)?;
        ok &= write_u16_be(file, self.ccap[0])?;
        Ok(ok)
    }

    pub fn read(&mut self, file: &mut dyn InfileBase) -> Result<()> {
        self.lcap = read_u16_be(file).map_err(|_| OjphError::Codec {
            code: 0x00050061,
            message: "error reading CAP marker".into(),
        })?;
        self.pcap = read_u32_be(file).map_err(|_| OjphError::Codec {
            code: 0x00050062,
            message: "error reading CAP marker".into(),
        })?;
        let count = population_count(self.pcap);
        if (self.pcap & 0x00020000) == 0 {
            return Err(OjphError::Codec {
                code: 0x00050064,
                message: "Pcap should have its 15th MSB set. Not a JPH file".into(),
            });
        }
        for i in 0..count as usize {
            self.ccap[i] = read_u16_be(file).map_err(|_| OjphError::Codec {
                code: 0x00050065,
                message: "error reading CAP marker".into(),
            })?;
        }
        if self.lcap != 6 + 2 * count as u16 {
            return Err(OjphError::Codec {
                code: 0x00050066,
                message: "error in CAP marker length".into(),
            });
        }
        Ok(())
    }
}

// =========================================================================
// param_sot — Start of Tile-Part
// =========================================================================

/// SOT marker segment — start of tile-part header.
///
/// Contains the tile index, tile-part length, tile-part index,
/// and total number of tile-parts.
#[derive(Debug, Clone, Default)]
pub struct ParamSot {
    pub(crate) isot: u16,
    pub(crate) psot: u32,
    pub(crate) tp_sot: u8,
    pub(crate) tn_sot: u8,
}

impl ParamSot {
    pub fn init(
        &mut self,
        payload_length: u32,
        tile_idx: u16,
        tile_part_index: u8,
        num_tile_parts: u8,
    ) {
        self.psot = payload_length + 12;
        self.isot = tile_idx;
        self.tp_sot = tile_part_index;
        self.tn_sot = num_tile_parts;
    }

    pub fn get_tile_index(&self) -> u16 {
        self.isot
    }
    pub fn get_payload_length(&self) -> u32 {
        if self.psot > 0 {
            self.psot - 12
        } else {
            0
        }
    }
    pub fn get_tile_part_index(&self) -> u8 {
        self.tp_sot
    }
    #[allow(dead_code)]
    pub fn get_num_tile_parts(&self) -> u8 {
        self.tn_sot
    }

    pub fn write(&mut self, file: &mut dyn OutfileBase, payload_len: u32) -> Result<bool> {
        self.psot = payload_len + 14;
        let mut ok = true;
        ok &= write_u16_be(file, markers::SOT)?;
        ok &= write_u16_be(file, 10)?; // Lsot is always 10
        ok &= write_u16_be(file, self.isot)?;
        ok &= write_u32_be(file, self.psot)?;
        ok &= write_u8(file, self.tp_sot)?;
        ok &= write_u8(file, self.tn_sot)?;
        Ok(ok)
    }

    pub fn read(&mut self, file: &mut dyn InfileBase, resilient: bool) -> Result<bool> {
        if resilient {
            let lsot = match read_u16_be(file) {
                Ok(v) => v,
                Err(_) => {
                    self.clear();
                    return Ok(false);
                }
            };
            if lsot != 10 {
                self.clear();
                return Ok(false);
            }
            self.isot = match read_u16_be(file) {
                Ok(v) => v,
                Err(_) => {
                    self.clear();
                    return Ok(false);
                }
            };
            if self.isot == 0xFFFF {
                self.clear();
                return Ok(false);
            }
            self.psot = match read_u32_be(file) {
                Ok(v) => v,
                Err(_) => {
                    self.clear();
                    return Ok(false);
                }
            };
            self.tp_sot = match read_u8(file) {
                Ok(v) => v,
                Err(_) => {
                    self.clear();
                    return Ok(false);
                }
            };
            self.tn_sot = match read_u8(file) {
                Ok(v) => v,
                Err(_) => {
                    self.clear();
                    return Ok(false);
                }
            };
        } else {
            let lsot = read_u16_be(file).map_err(|_| OjphError::Codec {
                code: 0x00050091,
                message: "error reading SOT marker".into(),
            })?;
            if lsot != 10 {
                return Err(OjphError::Codec {
                    code: 0x00050092,
                    message: "error in SOT length".into(),
                });
            }
            self.isot = read_u16_be(file).map_err(|_| OjphError::Codec {
                code: 0x00050093,
                message: "error reading SOT marker".into(),
            })?;
            if self.isot == 0xFFFF {
                return Err(OjphError::Codec {
                    code: 0x00050094,
                    message: "tile index in SOT marker cannot be 0xFFFF".into(),
                });
            }
            self.psot = read_u32_be(file).map_err(|_| OjphError::Codec {
                code: 0x00050095,
                message: "error reading SOT marker".into(),
            })?;
            self.tp_sot = read_u8(file).map_err(|_| OjphError::Codec {
                code: 0x00050096,
                message: "error reading SOT marker".into(),
            })?;
            self.tn_sot = read_u8(file).map_err(|_| OjphError::Codec {
                code: 0x00050097,
                message: "error reading SOT marker".into(),
            })?;
        }
        Ok(true)
    }

    fn clear(&mut self) {
        self.isot = 0;
        self.psot = 0;
        self.tp_sot = 0;
        self.tn_sot = 0;
    }
}

// =========================================================================
// param_tlm — Tile-part Length Marker
// =========================================================================

/// A single (tile-index, tile-part-length) pair in a TLM marker.
#[derive(Debug, Clone, Default)]
pub struct TtlmPtlmPair {
    /// Tile index.
    pub ttlm: u16,
    /// Tile-part length (including the SOT header).
    pub ptlm: u32,
}

/// TLM marker segment — tile-part length index.
///
/// Allows random access to tile-parts without sequentially parsing the
/// entire codestream.
#[derive(Debug, Clone, Default)]
pub struct ParamTlm {
    pub(crate) ltlm: u16,
    pub(crate) ztlm: u8,
    pub(crate) stlm: u8,
    pub(crate) pairs: Vec<TtlmPtlmPair>,
    pub(crate) next_pair_index: u32,
}

impl ParamTlm {
    pub fn init(&mut self, num_pairs: u32) {
        self.pairs
            .resize(num_pairs as usize, TtlmPtlmPair::default());
        self.ltlm = 4 + 6 * num_pairs as u16;
        self.ztlm = 0;
        self.stlm = 0x60;
        self.next_pair_index = 0;
    }

    pub fn set_next_pair(&mut self, ttlm: u16, ptlm: u32) {
        let idx = self.next_pair_index as usize;
        self.pairs[idx].ttlm = ttlm;
        self.pairs[idx].ptlm = ptlm + 14;
        self.next_pair_index += 1;
    }

    pub fn write(&self, file: &mut dyn OutfileBase) -> Result<bool> {
        let mut ok = true;
        ok &= write_u16_be(file, markers::TLM)?;
        ok &= write_u16_be(file, self.ltlm)?;
        ok &= write_u8(file, self.ztlm)?;
        ok &= write_u8(file, self.stlm)?;
        for pair in &self.pairs {
            ok &= write_u16_be(file, pair.ttlm)?;
            ok &= write_u32_be(file, pair.ptlm)?;
        }
        Ok(ok)
    }
}

// =========================================================================
// param_nlt — Non-Linearity Point Transformation
// =========================================================================

pub(crate) const NLT_ALL_COMPS: u16 = 65535;
pub(crate) const NLT_NO_NLT: u8 = 0;
#[allow(dead_code)]
pub(crate) const NLT_BINARY_COMPLEMENT: u8 = 3;
pub(crate) const NLT_UNDEFINED: u8 = 255;

/// NLT marker segment — non-linearity point transformation.
///
/// Provides per-component non-linear transforms (e.g. two's-complement
/// conversion for signed data).
#[derive(Debug, Clone)]
pub struct ParamNlt {
    pub(crate) lnlt: u16,
    pub(crate) cnlt: u16,
    pub(crate) bd_nlt: u8,
    pub(crate) tnlt: u8,
    pub(crate) enabled: bool,
    pub(crate) children: Vec<ParamNlt>,
}

impl Default for ParamNlt {
    fn default() -> Self {
        Self {
            lnlt: 6,
            cnlt: NLT_ALL_COMPS,
            bd_nlt: 0,
            tnlt: NLT_UNDEFINED,
            enabled: false,
            children: Vec::new(),
        }
    }
}

impl ParamNlt {
    /// Sets the non-linear transform type for a specific component.
    ///
    /// - `nl_type = 0` — no NLT
    /// - `nl_type = 3` — binary complement (two's-complement conversion)
    ///
    /// # Errors
    ///
    /// Returns [`OjphError::Unsupported`] if `nl_type` is not 0 or 3.
    pub fn set_nonlinear_transform(&mut self, comp_num: u32, nl_type: u8) -> Result<()> {
        if nl_type != NLT_NO_NLT && nl_type != NLT_BINARY_COMPLEMENT {
            return Err(OjphError::Unsupported(
                "only NLT types 0 and 3 are supported".into(),
            ));
        }
        let child = self.get_or_add_child(comp_num);
        child.tnlt = nl_type;
        child.enabled = true;
        Ok(())
    }

    /// Returns `(bit_depth, is_signed, nl_type)` for a component,
    /// or `None` if no NLT is configured.
    pub fn get_nonlinear_transform(&self, comp_num: u32) -> Option<(u8, bool, u8)> {
        for child in &self.children {
            if child.cnlt == comp_num as u16 && child.enabled {
                let bd = (child.bd_nlt & 0x7F) + 1;
                let is_signed = (child.bd_nlt & 0x80) == 0x80;
                return Some((bd.min(38), is_signed, child.tnlt));
            }
        }
        if self.enabled {
            let bd = (self.bd_nlt & 0x7F) + 1;
            let is_signed = (self.bd_nlt & 0x80) == 0x80;
            return Some((bd.min(38), is_signed, self.tnlt));
        }
        None
    }

    /// Returns `true` if any component has an NLT configured.
    pub fn is_any_enabled(&self) -> bool {
        if self.enabled {
            return true;
        }
        self.children.iter().any(|c| c.enabled)
    }

    fn get_or_add_child(&mut self, comp_num: u32) -> &mut ParamNlt {
        for i in 0..self.children.len() {
            if self.children[i].cnlt == comp_num as u16 {
                return &mut self.children[i];
            }
        }
        let child = ParamNlt {
            cnlt: comp_num as u16,
            ..Default::default()
        };
        self.children.push(child);
        self.children.last_mut().unwrap()
    }

    pub fn write(&self, file: &mut dyn OutfileBase) -> Result<bool> {
        if !self.is_any_enabled() {
            return Ok(true);
        }
        let mut ok = true;
        if self.enabled {
            ok &= write_u16_be(file, markers::NLT)?;
            ok &= write_u16_be(file, self.lnlt)?;
            ok &= write_u16_be(file, self.cnlt)?;
            ok &= write_u8(file, self.bd_nlt)?;
            ok &= write_u8(file, self.tnlt)?;
        }
        for child in &self.children {
            if child.enabled {
                ok &= write_u16_be(file, markers::NLT)?;
                ok &= write_u16_be(file, child.lnlt)?;
                ok &= write_u16_be(file, child.cnlt)?;
                ok &= write_u8(file, child.bd_nlt)?;
                ok &= write_u8(file, child.tnlt)?;
            }
        }
        Ok(ok)
    }

    pub fn read(&mut self, file: &mut dyn InfileBase) -> Result<()> {
        let mut buf = [0u8; 6];
        let mut offset = 0;
        while offset < 6 {
            let n = file.read(&mut buf[offset..])?;
            if n == 0 {
                return Err(OjphError::Codec {
                    code: 0x00050141,
                    message: "error reading NLT marker".into(),
                });
            }
            offset += n;
        }
        let length = u16::from_be_bytes([buf[0], buf[1]]);
        if length != 6 || (buf[5] != 3 && buf[5] != 0) {
            return Err(OjphError::Codec {
                code: 0x00050142,
                message: format!("Unsupported NLT type {}", buf[5]),
            });
        }
        let comp = u16::from_be_bytes([buf[2], buf[3]]);
        let child = self.get_or_add_child(comp as u32);
        child.enabled = true;
        child.cnlt = comp;
        child.bd_nlt = buf[4];
        child.tnlt = buf[5];
        Ok(())
    }
}

// =========================================================================
// param_dfs — Downsampling Factor Styles
// =========================================================================

#[allow(clippy::enum_variant_names)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum DfsDwtType {
    NoDwt = 0,
    BidirDwt = 1,
    HorzDwt = 2,
    VertDwt = 3,
}

/// DFS marker segment — downsampling factor styles.
///
/// Defines per-decomposition-level DWT type (bidirectional, horizontal-only,
/// vertical-only, or none).
#[derive(Debug, Clone, Default)]
pub struct ParamDfs {
    pub(crate) ldfs: u16,
    pub(crate) sdfs: u16,
    pub(crate) ids: u8,
    pub(crate) ddfs: [u8; 8],
    pub(crate) children: Vec<ParamDfs>,
}

impl ParamDfs {
    pub fn exists(&self) -> bool {
        self.ldfs != 0
    }

    #[allow(dead_code)]
    pub fn get_dfs(&self, index: i32) -> Option<&ParamDfs> {
        if self.sdfs == index as u16 {
            return Some(self);
        }
        self.children
            .iter()
            .find(|child| child.sdfs == index as u16)
    }

    pub fn get_dwt_type(&self, decomp_level: u32) -> DfsDwtType {
        let dl = decomp_level.min(self.ids as u32);
        let d = dl - 1;
        let idx = d >> 2;
        let bits = d & 0x3;
        let val = (self.ddfs[idx as usize] >> (6 - 2 * bits)) & 0x3;
        match val {
            0 => DfsDwtType::NoDwt,
            1 => DfsDwtType::BidirDwt,
            2 => DfsDwtType::HorzDwt,
            3 => DfsDwtType::VertDwt,
            _ => DfsDwtType::BidirDwt,
        }
    }

    pub fn read(&mut self, file: &mut dyn InfileBase) -> Result<bool> {
        if self.ldfs != 0 {
            let mut child = ParamDfs::default();
            let ok = child.read(file)?;
            self.children.push(child);
            return Ok(ok);
        }
        self.ldfs = read_u16_be(file).map_err(|_| OjphError::Codec {
            code: 0x000500D1,
            message: "error reading DFS-Ldfs".into(),
        })?;
        self.sdfs = read_u16_be(file).map_err(|_| OjphError::Codec {
            code: 0x000500D2,
            message: "error reading DFS-Sdfs".into(),
        })?;
        let l_ids = read_u8(file).map_err(|_| OjphError::Codec {
            code: 0x000500D4,
            message: "error reading DFS-Ids".into(),
        })?;
        let max_ddfs = (self.ddfs.len() * 4) as u8;
        self.ids = l_ids.min(max_ddfs);
        for i in (0..self.ids).step_by(4) {
            self.ddfs[(i / 4) as usize] = read_u8(file).map_err(|_| OjphError::Codec {
                code: 0x000500D6,
                message: "error reading DFS-Ddfs".into(),
            })?;
        }
        for _ in (self.ids..l_ids).step_by(4) {
            let _ = read_u8(file);
        }
        Ok(true)
    }
}

// =========================================================================
// Comment Exchange
// =========================================================================

/// COM marker data for exchange between caller and encoder.
///
/// Holds the comment body and registration value (Rcom):
/// - `rcom = 0` — binary data
/// - `rcom = 1` — Latin text (ISO 8859-15)
#[derive(Debug, Clone, Default)]
pub struct CommentExchange {
    /// Comment body bytes.
    pub data: Vec<u8>,
    /// Registration value (0 = binary, 1 = Latin text).
    pub rcom: u16,
}

impl CommentExchange {
    /// Sets the comment to a Latin text string (Rcom = 1).
    pub fn set_string(&mut self, s: &str) {
        self.data = s.as_bytes().to_vec();
        self.rcom = 1; // Latin (ISO 8859-15)
    }

    /// Sets the comment to binary data (Rcom = 0).
    pub fn set_data(&mut self, data: &[u8]) {
        self.data = data.to_vec();
        self.rcom = 0; // binary
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::file::{MemInfile, MemOutfile};
    use crate::types::{Point, Size};

    // -----------------------------------------------------------------
    // ParamSiz tests
    // -----------------------------------------------------------------

    #[test]
    fn param_siz_set_get() {
        let mut siz = ParamSiz::default();
        siz.set_image_extent(Point::new(1920, 1080));
        siz.set_tile_size(Size::new(512, 512));
        siz.set_image_offset(Point::new(10, 20));
        siz.set_tile_offset(Point::new(5, 10));

        assert_eq!(siz.get_image_extent(), Point::new(1920, 1080));
        assert_eq!(siz.get_tile_size(), Size::new(512, 512));
        assert_eq!(siz.get_image_offset(), Point::new(10, 20));
        assert_eq!(siz.get_tile_offset(), Point::new(5, 10));
    }

    #[test]
    fn param_siz_components() {
        let mut siz = ParamSiz::default();
        siz.set_num_components(3);
        assert_eq!(siz.get_num_components(), 3);

        // Component 0: 8-bit unsigned, no downsampling
        siz.set_comp_info(0, Point::new(1, 1), 8, false);
        // Component 1: 12-bit signed, 2x2 downsampling
        siz.set_comp_info(1, Point::new(2, 2), 12, true);
        // Component 2: 16-bit unsigned, 1x2 downsampling
        siz.set_comp_info(2, Point::new(1, 2), 16, false);

        assert_eq!(siz.get_bit_depth(0), 8);
        assert!(!siz.is_signed(0));
        assert_eq!(siz.get_downsampling(0), Point::new(1, 1));

        assert_eq!(siz.get_bit_depth(1), 12);
        assert!(siz.is_signed(1));
        assert_eq!(siz.get_downsampling(1), Point::new(2, 2));

        assert_eq!(siz.get_bit_depth(2), 16);
        assert!(!siz.is_signed(2));
        assert_eq!(siz.get_downsampling(2), Point::new(1, 2));
    }

    #[test]
    fn param_siz_validity_ok() {
        let mut siz = ParamSiz::default();
        siz.set_image_extent(Point::new(1920, 1080));
        siz.set_tile_size(Size::new(512, 512));
        siz.set_image_offset(Point::new(0, 0));
        siz.set_tile_offset(Point::new(0, 0));
        assert!(siz.check_validity().is_ok());
    }

    #[test]
    fn param_siz_validity_zero_extent() {
        let mut siz = ParamSiz::default();
        siz.set_image_extent(Point::new(0, 0));
        siz.set_tile_size(Size::new(512, 512));
        assert!(siz.check_validity().is_err());
    }

    #[test]
    fn param_siz_validity_bad_offset() {
        let mut siz = ParamSiz::default();
        siz.set_image_extent(Point::new(1920, 1080));
        siz.set_tile_size(Size::new(512, 512));
        // tile offset > image offset should fail
        siz.set_image_offset(Point::new(10, 10));
        siz.set_tile_offset(Point::new(20, 20));
        assert!(siz.check_validity().is_err());
    }

    #[test]
    fn param_siz_write_read_roundtrip() {
        let mut siz = ParamSiz::default();
        siz.set_image_extent(Point::new(1920, 1080));
        siz.set_tile_size(Size::new(512, 512));
        siz.set_num_components(3);
        for i in 0..3u32 {
            siz.set_comp_info(i, Point::new(1, 1), 8, false);
        }

        let mut out = MemOutfile::new();
        siz.write(&mut out).expect("write failed");

        // Skip the 2-byte marker code (0xFF51)
        let data = out.get_data();
        let mut inp = MemInfile::new(&data[2..]);
        let mut siz2 = ParamSiz::default();
        siz2.read(&mut inp).expect("read failed");

        assert_eq!(siz2.xsiz, 1920);
        assert_eq!(siz2.ysiz, 1080);
        assert_eq!(siz2.xt_siz, 512);
        assert_eq!(siz2.yt_siz, 512);
        assert_eq!(siz2.csiz, 3);
        assert_eq!(siz2.get_bit_depth(0), 8);
        assert_eq!(siz2.get_bit_depth(1), 8);
        assert_eq!(siz2.get_bit_depth(2), 8);
        assert!(!siz2.is_signed(0));
        assert_eq!(siz2.get_downsampling(0), Point::new(1, 1));
    }

    #[test]
    fn param_siz_get_width_height() {
        let mut siz = ParamSiz::default();
        siz.set_image_extent(Point::new(1920, 1080));
        siz.set_image_offset(Point::new(100, 50));
        siz.set_tile_size(Size::new(1920, 1080));
        siz.set_num_components(2);
        // Component 0: no downsampling
        siz.set_comp_info(0, Point::new(1, 1), 8, false);
        // Component 1: 2x2 downsampling
        siz.set_comp_info(1, Point::new(2, 2), 8, false);

        // width = ceil(1920/1) - ceil(100/1) = 1920 - 100 = 1820
        assert_eq!(siz.get_width(0), 1820);
        // height = ceil(1080/1) - ceil(50/1) = 1080 - 50 = 1030
        assert_eq!(siz.get_height(0), 1030);

        // width = ceil(1920/2) - ceil(100/2) = 960 - 50 = 910
        assert_eq!(siz.get_width(1), 910);
        // height = ceil(1080/2) - ceil(50/2) = 540 - 25 = 515
        assert_eq!(siz.get_height(1), 515);
    }

    // -----------------------------------------------------------------
    // ParamCod tests
    // -----------------------------------------------------------------

    #[test]
    fn param_cod_set_get() {
        let mut cod = ParamCod::default();
        cod.set_reversible(true);
        assert!(cod.is_reversible());
        assert_eq!(cod.get_wavelet_kern(), DWT_REV53);

        cod.set_reversible(false);
        assert!(!cod.is_reversible());
        assert_eq!(cod.get_wavelet_kern(), DWT_IRV97);

        cod.set_num_decomposition(5);
        assert_eq!(cod.get_num_decompositions(), 5);

        cod.set_color_transform(true);
        assert!(cod.is_employing_color_transform());
        cod.set_color_transform(false);
        assert!(!cod.is_employing_color_transform());
    }

    #[test]
    fn param_cod_progression_order() {
        let mut cod = ParamCod::default();
        cod.set_progression_order("CPRL")
            .expect("valid progression");
        assert_eq!(cod.get_progression_order_as_string(), "CPRL");

        cod.set_progression_order("lrcp").expect("case-insensitive");
        assert_eq!(cod.get_progression_order_as_string(), "LRCP");
    }

    #[test]
    fn param_cod_invalid_progression() {
        let mut cod = ParamCod::default();
        assert!(cod.set_progression_order("INVALID").is_err());
    }

    #[test]
    fn param_cod_block_dims() {
        let mut cod = ParamCod::default();
        cod.set_block_dims(32, 32);
        assert_eq!(cod.get_block_dims(), Size::new(32, 32));

        cod.set_block_dims(64, 64);
        assert_eq!(cod.get_block_dims(), Size::new(64, 64));

        cod.set_block_dims(32, 64);
        assert_eq!(cod.get_block_dims(), Size::new(32, 64));
    }

    #[test]
    fn param_cod_write_read_roundtrip() {
        let mut cod = ParamCod::default();
        cod.set_reversible(true);
        cod.set_num_decomposition(5);
        cod.set_block_dims(64, 64);

        let mut out = MemOutfile::new();
        cod.write(&mut out).expect("write failed");

        let data = out.get_data();
        let mut inp = MemInfile::new(&data[2..]); // skip marker
        let mut cod2 = ParamCod::default();
        cod2.read(&mut inp).expect("read failed");

        assert_eq!(cod2.get_num_decompositions(), 5);
        assert!(cod2.is_reversible());
        assert_eq!(cod2.get_block_dims(), Size::new(64, 64));
    }

    // -----------------------------------------------------------------
    // ParamQcd tests
    // -----------------------------------------------------------------

    #[test]
    fn param_qcd_reversible() {
        let mut siz = ParamSiz::default();
        siz.set_image_extent(Point::new(256, 256));
        siz.set_tile_size(Size::new(256, 256));
        siz.set_num_components(1);
        siz.set_comp_info(0, Point::new(1, 1), 8, false);

        let mut cod = ParamCod::default();
        cod.set_reversible(true);
        cod.set_num_decomposition(5);

        let mut qcd = ParamQcd::default();
        qcd.check_validity(&siz, &cod)
            .expect("check_validity failed");

        // After check_validity with reversible, sqcd lower 5 bits should be 0
        assert_eq!(qcd.sqcd & 0x1F, 0);
        assert!(qcd.get_num_guard_bits() >= 1);
        assert_eq!(qcd.num_subbands, 1 + 3 * 5);
    }

    #[test]
    fn param_qcd_reversible_no_dwt_8bit_uses_kmax_8() {
        let mut siz = ParamSiz::default();
        siz.set_image_extent(Point::new(33, 33));
        siz.set_tile_size(Size::new(33, 33));
        siz.set_num_components(1);
        siz.set_comp_info(0, Point::new(1, 1), 8, false);

        let mut cod = ParamCod::default();
        cod.set_reversible(true);
        cod.set_num_decomposition(0);

        let mut qcd = ParamQcd::default();
        qcd.check_validity(&siz, &cod)
            .expect("check_validity failed");

        assert_eq!(qcd.get_kmax(0, 0, 0), 8);
        assert_eq!(qcd.get_magb(), 8);
    }

    #[test]
    fn param_qcd_write_read_roundtrip() {
        let mut siz = ParamSiz::default();
        siz.set_image_extent(Point::new(256, 256));
        siz.set_tile_size(Size::new(256, 256));
        siz.set_num_components(1);
        siz.set_comp_info(0, Point::new(1, 1), 8, false);

        let mut cod = ParamCod::default();
        cod.set_reversible(true);
        cod.set_num_decomposition(5);

        let mut qcd = ParamQcd::default();
        qcd.check_validity(&siz, &cod)
            .expect("check_validity failed");

        let mut out = MemOutfile::new();
        qcd.write(&mut out).expect("write failed");

        let data = out.get_data();
        let mut inp = MemInfile::new(&data[2..]); // skip marker
        let mut qcd2 = ParamQcd::default();
        qcd2.read(&mut inp).expect("read failed");

        assert_eq!(qcd2.sqcd, qcd.sqcd);
        assert_eq!(qcd2.num_subbands, qcd.num_subbands);
        assert_eq!(qcd2.get_num_guard_bits(), qcd.get_num_guard_bits());

        // Compare subband data
        match (&qcd.sp_qcd, &qcd2.sp_qcd) {
            (SpqcdData::Reversible(a), SpqcdData::Reversible(b)) => {
                assert_eq!(a, b);
            }
            _ => panic!("expected reversible quantization data"),
        }
    }

    // -----------------------------------------------------------------
    // ProgressionOrder tests
    // -----------------------------------------------------------------

    #[test]
    fn progression_order_from_str() {
        assert_eq!(
            ProgressionOrder::from_str("LRCP"),
            Some(ProgressionOrder::LRCP)
        );
        assert_eq!(
            ProgressionOrder::from_str("RLCP"),
            Some(ProgressionOrder::RLCP)
        );
        assert_eq!(
            ProgressionOrder::from_str("RPCL"),
            Some(ProgressionOrder::RPCL)
        );
        assert_eq!(
            ProgressionOrder::from_str("PCRL"),
            Some(ProgressionOrder::PCRL)
        );
        assert_eq!(
            ProgressionOrder::from_str("CPRL"),
            Some(ProgressionOrder::CPRL)
        );
        // case-insensitive
        assert_eq!(
            ProgressionOrder::from_str("lrcp"),
            Some(ProgressionOrder::LRCP)
        );
        // invalid
        assert_eq!(ProgressionOrder::from_str("INVALID"), None);
    }

    #[test]
    fn progression_order_as_str() {
        assert_eq!(ProgressionOrder::LRCP.as_str(), "LRCP");
        assert_eq!(ProgressionOrder::RLCP.as_str(), "RLCP");
        assert_eq!(ProgressionOrder::RPCL.as_str(), "RPCL");
        assert_eq!(ProgressionOrder::PCRL.as_str(), "PCRL");
        assert_eq!(ProgressionOrder::CPRL.as_str(), "CPRL");
    }

    #[test]
    fn progression_order_from_i32() {
        assert_eq!(ProgressionOrder::from_i32(0), Some(ProgressionOrder::LRCP));
        assert_eq!(ProgressionOrder::from_i32(1), Some(ProgressionOrder::RLCP));
        assert_eq!(ProgressionOrder::from_i32(2), Some(ProgressionOrder::RPCL));
        assert_eq!(ProgressionOrder::from_i32(3), Some(ProgressionOrder::PCRL));
        assert_eq!(ProgressionOrder::from_i32(4), Some(ProgressionOrder::CPRL));
        assert_eq!(ProgressionOrder::from_i32(5), None);
        assert_eq!(ProgressionOrder::from_i32(-1), None);
    }
}
