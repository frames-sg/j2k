//! JPEG 2000 codestream parameter marker segments (SIZ, COD, QCD, etc.)
//!
//! Port of `ojph_params.h`, `ojph_params_local.h`, and `ojph_params.cpp`.
//!
//! These types represent the marker segments that form the main header of a
//! JPEG 2000 Part 15 (HTJ2K) codestream:
//!
//! | Type | Marker | Description |
//! |------|--------|-------------|
//! | [`ParamSiz`] | SIZ | Image and tile size, component info |
//! | [`ParamCod`] | COD/COC | Coding style defaults / per-component |
//! | [`ParamQcd`] | QCD/QCC | Quantization defaults / per-component |
//! | [`ParamCap`] | CAP | Extended capabilities |
//! | [`ParamSot`] | SOT | Start of tile-part header |
//! | [`ParamTlm`] | TLM | Tile-part length marker |
//! | [`ParamNlt`] | NLT | Non-linearity point transformation |
//! | [`ParamDfs`] | DFS | Downsampling factor styles |
//! | [`CommentExchange`] | COM | Comment marker data |
//!
//! # Configuration example
//!
//! ```rust
//! use openjph_core::params::{ParamSiz, ParamCod};
//! use openjph_core::codestream::Codestream;
//! use openjph_core::types::{Point, Size};
//!
//! let mut cs = Codestream::new();
//!
//! // Configure image geometry
//! cs.access_siz_mut().set_image_extent(Point::new(1920, 1080));
//! cs.access_siz_mut().set_num_components(3);
//! for c in 0..3 {
//!     cs.access_siz_mut().set_comp_info(c, Point::new(1, 1), 8, false);
//! }
//! cs.access_siz_mut().set_tile_size(Size::new(1920, 1080));
//!
//! // Configure coding style
//! cs.access_cod_mut().set_num_decomposition(5);
//! cs.access_cod_mut().set_reversible(true);
//! cs.access_cod_mut().set_color_transform(true);
//! ```

pub(crate) mod local;

// Re-export public types
pub use local::{
    CommentExchange, ParamCap, ParamCod, ParamDfs, ParamNlt, ParamQcd, ParamSiz, ParamSot,
    ParamTlm, ProfileNum, ProgressionOrder, TtlmPtlmPair,
};
