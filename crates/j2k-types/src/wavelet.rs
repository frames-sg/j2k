// SPDX-License-Identifier: MIT OR Apache-2.0

//! Framework-independent native wavelet output contracts.

use alloc::vec::Vec;

pub use crate::{
    J2kRect as WaveletRect, J2kSubBandType as WaveletBandKind,
    J2kWaveletTransform as WaveletTransform,
};

/// Requested tiles, components, resolution indices, and band orientations.
///
/// Empty lists select all values. Lists are sets: duplicates do not duplicate
/// output, which always follows codestream tile/component/resolution order.
/// Resolution 0 is the coarsest LL band; 1 through N contain detail bands in
/// order from coarsest to finest. A nonempty resolution list must be valid for
/// every selected tile component. Selections matching no band are errors.
///
/// Unselected bands are not entropy-decoded. Packet headers are still parsed,
/// and active-tile coefficient storage is still allocated. This is not a
/// promise of proportional I/O or memory savings, or incremental refinement.
#[derive(Debug, Clone, Default)]
pub struct WaveletSelection {
    /// Row-major tile indices. Empty selects every tile.
    pub tiles: Vec<u32>,
    /// Codestream component indices, before inverse color transformation.
    pub components: Vec<u16>,
    /// Resolution indices: 0 for LL, then coarsest-to-finest detail levels.
    pub resolutions: Vec<u8>,
    /// Band orientations; LL exists only at resolution 0.
    pub bands: Vec<WaveletBandKind>,
}

/// Owned, row-major coefficients in a band.
#[derive(Debug, PartialEq)]
pub enum WaveletSamples {
    /// Exact reversible 5/3 coefficients, after ROI maxshift inversion.
    I64(Vec<i64>),
    /// Dequantized irreversible 9/7 coefficients, including the decoder's
    /// midpoint reconstruction for truncated coding passes.
    F32(Vec<f32>),
}

/// One band in component coordinates, before inverse wavelet reconstruction.
#[derive(Debug)]
pub struct WaveletBand {
    /// Low/high filtering along x, followed by y.
    pub kind: WaveletBandKind,
    /// Codestream resolution index: 0 is LL; larger indices are finer detail.
    pub resolution: u8,
    /// Band coordinates, including nonzero origins and odd-size geometry.
    pub rect: WaveletRect,
    /// Dequantization multiplier already applied to the returned coefficients.
    /// This is 1 for reversible output. Values are not normalized for a model.
    pub quantization_step: f32,
    /// Samples in rows of `rect.width()` elements.
    pub samples: WaveletSamples,
}

/// Selected bands from one codestream component in one tile.
#[derive(Debug)]
pub struct WaveletComponent {
    /// Original codestream component index.
    pub index: u16,
    /// Component tile rectangle at full component resolution.
    pub rect: WaveletRect,
    /// Source component precision in bits.
    pub precision: u8,
    /// Whether source samples are signed, before unsigned level shifting.
    pub signed: bool,
    /// Horizontal and vertical sample spacing in the image reference grid.
    pub sampling: (u8, u8),
    /// Reversible 5/3 or irreversible 9/7 transformation.
    pub transform: WaveletTransform,
    /// Number of wavelet decomposition levels in this component tile.
    pub decomposition_levels: u8,
    /// Selected bands in resolution order, with HL, LH, HH within each level.
    pub bands: Vec<WaveletBand>,
}

/// Selected component coefficients belonging to one independently coded tile.
#[derive(Debug)]
pub struct WaveletTile {
    /// Original row-major tile index.
    pub index: u32,
    /// Tile rectangle in the image reference grid.
    pub rect: WaveletRect,
    /// Whether the first three components are in the codestream's transformed
    /// color domain (RCT for reversible, ICT for irreversible coding), not RGB.
    pub uses_color_transform: bool,
    /// Selected components. JP2 palette, channel, and ICC transformations have
    /// not been applied; indices refer to codestream components.
    pub components: Vec<WaveletComponent>,
}

/// Native coefficients extracted without inverse DWT, color conversion, or packing.
#[derive(Debug)]
pub struct WaveletImage {
    /// Full image rectangle in the reference grid.
    pub rect: WaveletRect,
    /// Selected tiles. Tile boundaries are preserved, never stitched silently.
    pub tiles: Vec<WaveletTile>,
}
