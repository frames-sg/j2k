// SPDX-License-Identifier: Apache-2.0

//! HTJ2K-ready 5/3 wavelet band descriptors.

use core::fmt;

/// JPEG 2000 SIZ component sampling factors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ComponentSampling {
    /// Horizontal sampling factor (`XRsiz`).
    pub x_rsiz: u16,
    /// Vertical sampling factor (`YRsiz`).
    pub y_rsiz: u16,
}

/// One row-major wavelet band.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WaveletBand53<T> {
    /// Band width in coefficients.
    pub width: usize,
    /// Band height in coefficients.
    pub height: usize,
    /// Row-major band coefficients.
    pub coefficients: Vec<T>,
}

/// One decomposition level's high-pass bands.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WaveletLevel53<T> {
    /// High-horizontal, low-vertical band.
    pub hl: WaveletBand53<T>,
    /// Low-horizontal, high-vertical band.
    pub lh: WaveletBand53<T>,
    /// High-horizontal, high-vertical band.
    pub hh: WaveletBand53<T>,
}

/// One component's 5/3 wavelet representation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WaveletComponent53<T> {
    /// Original component width in samples.
    pub width: usize,
    /// Original component height in samples.
    pub height: usize,
    /// Component precision in bits.
    pub bit_depth: u8,
    /// Whether coefficients represent a signed component.
    pub is_signed: bool,
    /// Component sampling relative to the reference grid.
    pub sampling: ComponentSampling,
    /// Lowest-resolution LL band after all levels.
    pub final_ll: WaveletBand53<T>,
    /// High-pass bands ordered from full resolution toward the final LL.
    pub levels: Vec<WaveletLevel53<T>>,
}

/// Multi-component 5/3 wavelet image.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WaveletImage53<T> {
    /// Components at their native resolutions.
    pub components: Vec<WaveletComponent53<T>>,
}

impl<T> WaveletImage53<T> {
    /// Validate component metadata, recursive 5/3 geometry, and band lengths.
    pub fn validate(&self) -> Result<(), WaveletValidationError> {
        if self.components.is_empty() {
            return Err(WaveletValidationError::NoComponents);
        }

        for (component_index, component) in self.components.iter().enumerate() {
            validate_component(component_index, component)?;
        }

        Ok(())
    }
}

/// Validation failure for an HTJ2K-ready wavelet image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaveletValidationError {
    /// Image has no components.
    NoComponents,
    /// Component has no decomposition levels.
    NoLevels { component: usize },
    /// Component bit depth is outside JPEG 2000's representable precision.
    InvalidBitDepth { component: usize, bit_depth: u8 },
    /// Component sampling factor is zero.
    InvalidSampling {
        component: usize,
        x_rsiz: u16,
        y_rsiz: u16,
    },
    /// A band length does not equal `width * height`.
    BandLength {
        component: usize,
        band: &'static str,
        expected: usize,
        actual: usize,
    },
    /// A band's declared geometry does not match recursive 5/3 expectations.
    BandGeometry {
        component: usize,
        band: &'static str,
        expected_width: usize,
        expected_height: usize,
        actual_width: usize,
        actual_height: usize,
    },
}

impl fmt::Display for WaveletValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoComponents => write!(f, "wavelet image has no components"),
            Self::NoLevels { component } => {
                write!(f, "component {component} has no decomposition levels")
            }
            Self::InvalidBitDepth {
                component,
                bit_depth,
            } => write!(f, "component {component} has invalid bit depth {bit_depth}"),
            Self::InvalidSampling {
                component,
                x_rsiz,
                y_rsiz,
            } => write!(
                f,
                "component {component} has invalid sampling XRsiz={x_rsiz}, YRsiz={y_rsiz}"
            ),
            Self::BandLength {
                component,
                band,
                expected,
                actual,
            } => write!(
                f,
                "component {component} band {band} has {actual} coefficients, expected {expected}"
            ),
            Self::BandGeometry {
                component,
                band,
                expected_width,
                expected_height,
                actual_width,
                actual_height,
            } => write!(
                f,
                "component {component} band {band} is {actual_width}x{actual_height}, expected {expected_width}x{expected_height}"
            ),
        }
    }
}

impl std::error::Error for WaveletValidationError {}

fn validate_component<T>(
    component_index: usize,
    component: &WaveletComponent53<T>,
) -> Result<(), WaveletValidationError> {
    if component.levels.is_empty() {
        return Err(WaveletValidationError::NoLevels {
            component: component_index,
        });
    }
    if !(1..=38).contains(&component.bit_depth) {
        return Err(WaveletValidationError::InvalidBitDepth {
            component: component_index,
            bit_depth: component.bit_depth,
        });
    }
    if component.sampling.x_rsiz == 0 || component.sampling.y_rsiz == 0 {
        return Err(WaveletValidationError::InvalidSampling {
            component: component_index,
            x_rsiz: component.sampling.x_rsiz,
            y_rsiz: component.sampling.y_rsiz,
        });
    }

    let mut width = component.width;
    let mut height = component.height;
    for level in &component.levels {
        let low_width = width.div_ceil(2);
        let low_height = height.div_ceil(2);
        let high_width = width / 2;
        let high_height = height / 2;

        validate_band(component_index, "HL", &level.hl, high_width, low_height)?;
        validate_band(component_index, "LH", &level.lh, low_width, high_height)?;
        validate_band(component_index, "HH", &level.hh, high_width, high_height)?;

        width = low_width;
        height = low_height;
    }

    validate_band(component_index, "LL", &component.final_ll, width, height)
}

fn validate_band<T>(
    component: usize,
    band_name: &'static str,
    band: &WaveletBand53<T>,
    expected_width: usize,
    expected_height: usize,
) -> Result<(), WaveletValidationError> {
    if band.width != expected_width || band.height != expected_height {
        return Err(WaveletValidationError::BandGeometry {
            component,
            band: band_name,
            expected_width,
            expected_height,
            actual_width: band.width,
            actual_height: band.height,
        });
    }

    let expected = expected_width * expected_height;
    if band.coefficients.len() != expected {
        return Err(WaveletValidationError::BandLength {
            component,
            band: band_name,
            expected,
            actual: band.coefficients.len(),
        });
    }

    Ok(())
}
