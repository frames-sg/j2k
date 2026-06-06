// SPDX-License-Identifier: Apache-2.0

//! HTJ2K-ready 5/3 and 9/7 wavelet band descriptors.

use core::fmt;

use signinum_j2k::{
    J2kForwardDwt53Level, J2kForwardDwt53Output, J2kForwardDwt97Level, J2kForwardDwt97Output,
    PrecomputedHtj2k53Component, PrecomputedHtj2k53Image, PrecomputedHtj2k97Component,
    PrecomputedHtj2k97Image,
};

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

/// One row-major irreversible 9/7 wavelet band.
pub type WaveletBand97<T> = WaveletBand53<T>;
/// One irreversible 9/7 decomposition level's high-pass bands.
pub type WaveletLevel97<T> = WaveletLevel53<T>;
/// One component's irreversible 9/7 wavelet representation.
pub type WaveletComponent97<T> = WaveletComponent53<T>;
/// Multi-component irreversible 9/7 wavelet image.
pub type WaveletImage97<T> = WaveletImage53<T>;

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

impl WaveletImage53<i32> {
    /// Convert validated integer 5/3 bands into the precomputed HTJ2K encoder
    /// representation.
    pub fn to_precomputed_htj2k_53(
        &self,
        reference_width: u32,
        reference_height: u32,
    ) -> Result<PrecomputedHtj2k53Image, WaveletToPrecomputedError> {
        if reference_width == 0 || reference_height == 0 {
            return Err(WaveletToPrecomputedError::InvalidReferenceDimensions {
                width: reference_width,
                height: reference_height,
            });
        }
        self.validate()
            .map_err(WaveletToPrecomputedError::Validation)?;

        let first = self
            .components
            .first()
            .ok_or(WaveletToPrecomputedError::Validation(
                WaveletValidationError::NoComponents,
            ))?;
        let bit_depth = first.bit_depth;
        let signed = first.is_signed;

        let mut components = Vec::with_capacity(self.components.len());
        for (component_index, component) in self.components.iter().enumerate() {
            if component.bit_depth != bit_depth {
                return Err(WaveletToPrecomputedError::MixedBitDepth {
                    component: component_index,
                    expected: bit_depth,
                    actual: component.bit_depth,
                });
            }
            if component.is_signed != signed {
                return Err(WaveletToPrecomputedError::MixedSignedness {
                    component: component_index,
                    expected: signed,
                    actual: component.is_signed,
                });
            }
            let x_rsiz = u8::try_from(component.sampling.x_rsiz).map_err(|_| {
                WaveletToPrecomputedError::SamplingTooLarge {
                    component: component_index,
                    x_rsiz: component.sampling.x_rsiz,
                    y_rsiz: component.sampling.y_rsiz,
                }
            })?;
            let y_rsiz = u8::try_from(component.sampling.y_rsiz).map_err(|_| {
                WaveletToPrecomputedError::SamplingTooLarge {
                    component: component_index,
                    x_rsiz: component.sampling.x_rsiz,
                    y_rsiz: component.sampling.y_rsiz,
                }
            })?;

            let expected_width = reference_width.div_ceil(u32::from(x_rsiz));
            let expected_height = reference_height.div_ceil(u32::from(y_rsiz));
            let actual_width = usize_to_u32(component.width, component_index, "component width")?;
            let actual_height =
                usize_to_u32(component.height, component_index, "component height")?;
            if actual_width != expected_width || actual_height != expected_height {
                return Err(WaveletToPrecomputedError::ComponentGeometry {
                    component: component_index,
                    expected_width,
                    expected_height,
                    actual_width,
                    actual_height,
                });
            }

            components.push(PrecomputedHtj2k53Component {
                x_rsiz,
                y_rsiz,
                dwt: component_to_j2k_dwt(component, component_index)?,
            });
        }

        Ok(PrecomputedHtj2k53Image {
            width: reference_width,
            height: reference_height,
            bit_depth,
            signed,
            components,
        })
    }
}

impl WaveletImage97<f32> {
    /// Convert validated floating-point 9/7 bands into the precomputed HTJ2K
    /// encoder representation.
    pub fn to_precomputed_htj2k_97(
        &self,
        reference_width: u32,
        reference_height: u32,
    ) -> Result<PrecomputedHtj2k97Image, WaveletToPrecomputedError> {
        if reference_width == 0 || reference_height == 0 {
            return Err(WaveletToPrecomputedError::InvalidReferenceDimensions {
                width: reference_width,
                height: reference_height,
            });
        }
        self.validate()
            .map_err(WaveletToPrecomputedError::Validation)?;

        let first = self
            .components
            .first()
            .ok_or(WaveletToPrecomputedError::Validation(
                WaveletValidationError::NoComponents,
            ))?;
        let bit_depth = first.bit_depth;
        let signed = first.is_signed;

        let mut components = Vec::with_capacity(self.components.len());
        for (component_index, component) in self.components.iter().enumerate() {
            if component.bit_depth != bit_depth {
                return Err(WaveletToPrecomputedError::MixedBitDepth {
                    component: component_index,
                    expected: bit_depth,
                    actual: component.bit_depth,
                });
            }
            if component.is_signed != signed {
                return Err(WaveletToPrecomputedError::MixedSignedness {
                    component: component_index,
                    expected: signed,
                    actual: component.is_signed,
                });
            }
            let x_rsiz = u8::try_from(component.sampling.x_rsiz).map_err(|_| {
                WaveletToPrecomputedError::SamplingTooLarge {
                    component: component_index,
                    x_rsiz: component.sampling.x_rsiz,
                    y_rsiz: component.sampling.y_rsiz,
                }
            })?;
            let y_rsiz = u8::try_from(component.sampling.y_rsiz).map_err(|_| {
                WaveletToPrecomputedError::SamplingTooLarge {
                    component: component_index,
                    x_rsiz: component.sampling.x_rsiz,
                    y_rsiz: component.sampling.y_rsiz,
                }
            })?;

            let expected_width = reference_width.div_ceil(u32::from(x_rsiz));
            let expected_height = reference_height.div_ceil(u32::from(y_rsiz));
            let actual_width = usize_to_u32(component.width, component_index, "component width")?;
            let actual_height =
                usize_to_u32(component.height, component_index, "component height")?;
            if actual_width != expected_width || actual_height != expected_height {
                return Err(WaveletToPrecomputedError::ComponentGeometry {
                    component: component_index,
                    expected_width,
                    expected_height,
                    actual_width,
                    actual_height,
                });
            }

            components.push(PrecomputedHtj2k97Component {
                x_rsiz,
                y_rsiz,
                dwt: component_to_j2k_dwt97(component, component_index)?,
            });
        }

        Ok(PrecomputedHtj2k97Image {
            width: reference_width,
            height: reference_height,
            bit_depth,
            signed,
            components,
        })
    }
}

/// Validation failure for an HTJ2K-ready wavelet image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaveletValidationError {
    /// Image has no components.
    NoComponents,
    /// Component has no decomposition levels.
    NoLevels {
        /// Component index.
        component: usize,
    },
    /// Component bit depth is outside JPEG 2000's representable precision.
    InvalidBitDepth {
        /// Component index.
        component: usize,
        /// Reported component bit depth.
        bit_depth: u8,
    },
    /// Component sampling factor is zero.
    InvalidSampling {
        /// Component index.
        component: usize,
        /// Horizontal sampling factor.
        x_rsiz: u16,
        /// Vertical sampling factor.
        y_rsiz: u16,
    },
    /// A band length does not equal `width * height`.
    BandLength {
        /// Component index.
        component: usize,
        /// Band name.
        band: &'static str,
        /// Expected coefficient count.
        expected: usize,
        /// Actual coefficient count.
        actual: usize,
    },
    /// A band's declared geometry does not match recursive 5/3 expectations.
    BandGeometry {
        /// Component index.
        component: usize,
        /// Band name.
        band: &'static str,
        /// Expected band width.
        expected_width: usize,
        /// Expected band height.
        expected_height: usize,
        /// Actual band width.
        actual_width: usize,
        /// Actual band height.
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

/// Failure while adapting a validated wavelet image to the native precomputed
/// HTJ2K encoder representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaveletToPrecomputedError {
    /// Wavelet descriptor validation failed.
    Validation(WaveletValidationError),
    /// Reference-grid dimensions cannot be zero.
    InvalidReferenceDimensions {
        /// Reference-grid width.
        width: u32,
        /// Reference-grid height.
        height: u32,
    },
    /// Component sampling factors exceed the native encoder's current SIZ
    /// representation.
    SamplingTooLarge {
        /// Component index.
        component: usize,
        /// Horizontal sampling factor.
        x_rsiz: u16,
        /// Vertical sampling factor.
        y_rsiz: u16,
    },
    /// Component dimensions do not match the provided reference grid and SIZ
    /// sampling factors.
    ComponentGeometry {
        /// Component index.
        component: usize,
        /// Expected component width.
        expected_width: u32,
        /// Expected component height.
        expected_height: u32,
        /// Actual component width.
        actual_width: u32,
        /// Actual component height.
        actual_height: u32,
    },
    /// Components use different bit depths, but the native precomputed encoder
    /// currently stores one precision for the image.
    MixedBitDepth {
        /// Component index.
        component: usize,
        /// Expected shared bit depth.
        expected: u8,
        /// Actual component bit depth.
        actual: u8,
    },
    /// Components use mixed signedness, but the native precomputed encoder
    /// currently stores one signedness flag for the image.
    MixedSignedness {
        /// Component index.
        component: usize,
        /// Expected shared signedness.
        expected: bool,
        /// Actual component signedness.
        actual: bool,
    },
    /// A wavelet dimension exceeds the native encoder's current u32 geometry.
    DimensionTooLarge {
        /// Component index.
        component: usize,
        /// Field whose value was too large.
        field: &'static str,
        /// Oversized value.
        value: usize,
    },
}

impl fmt::Display for WaveletToPrecomputedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Validation(err) => write!(f, "{err}"),
            Self::InvalidReferenceDimensions { width, height } => {
                write!(f, "invalid reference-grid dimensions {width}x{height}")
            }
            Self::SamplingTooLarge {
                component,
                x_rsiz,
                y_rsiz,
            } => write!(
                f,
                "component {component} sampling XRsiz={x_rsiz}, YRsiz={y_rsiz} exceeds encoder range"
            ),
            Self::ComponentGeometry {
                component,
                expected_width,
                expected_height,
                actual_width,
                actual_height,
            } => write!(
                f,
                "component {component} is {actual_width}x{actual_height}, expected {expected_width}x{expected_height} from reference grid"
            ),
            Self::MixedBitDepth {
                component,
                expected,
                actual,
            } => write!(
                f,
                "component {component} has bit depth {actual}, expected {expected}"
            ),
            Self::MixedSignedness {
                component,
                expected,
                actual,
            } => write!(
                f,
                "component {component} signedness is {actual}, expected {expected}"
            ),
            Self::DimensionTooLarge {
                component,
                field,
                value,
            } => write!(
                f,
                "component {component} {field} value {value} exceeds encoder range"
            ),
        }
    }
}

impl std::error::Error for WaveletToPrecomputedError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Validation(err) => Some(err),
            Self::InvalidReferenceDimensions { .. }
            | Self::SamplingTooLarge { .. }
            | Self::ComponentGeometry { .. }
            | Self::MixedBitDepth { .. }
            | Self::MixedSignedness { .. }
            | Self::DimensionTooLarge { .. } => None,
        }
    }
}

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

fn component_to_j2k_dwt(
    component: &WaveletComponent53<i32>,
    component_index: usize,
) -> Result<J2kForwardDwt53Output, WaveletToPrecomputedError> {
    let mut current_width = component.width;
    let mut current_height = component.height;
    let mut levels = Vec::with_capacity(component.levels.len());

    for level in &component.levels {
        let low_width = current_width.div_ceil(2);
        let low_height = current_height.div_ceil(2);
        let high_width = current_width / 2;
        let high_height = current_height / 2;

        levels.push(J2kForwardDwt53Level {
            hl: i32_to_f32(&level.hl.coefficients),
            lh: i32_to_f32(&level.lh.coefficients),
            hh: i32_to_f32(&level.hh.coefficients),
            width: usize_to_u32(current_width, component_index, "level width")?,
            height: usize_to_u32(current_height, component_index, "level height")?,
            low_width: usize_to_u32(low_width, component_index, "low-pass width")?,
            low_height: usize_to_u32(low_height, component_index, "low-pass height")?,
            high_width: usize_to_u32(high_width, component_index, "high-pass width")?,
            high_height: usize_to_u32(high_height, component_index, "high-pass height")?,
        });

        current_width = low_width;
        current_height = low_height;
    }
    levels.reverse();

    Ok(J2kForwardDwt53Output {
        ll: i32_to_f32(&component.final_ll.coefficients),
        ll_width: usize_to_u32(component.final_ll.width, component_index, "LL width")?,
        ll_height: usize_to_u32(component.final_ll.height, component_index, "LL height")?,
        levels,
    })
}

fn component_to_j2k_dwt97(
    component: &WaveletComponent97<f32>,
    component_index: usize,
) -> Result<J2kForwardDwt97Output, WaveletToPrecomputedError> {
    let mut current_width = component.width;
    let mut current_height = component.height;
    let mut levels = Vec::with_capacity(component.levels.len());

    for level in &component.levels {
        let low_width = current_width.div_ceil(2);
        let low_height = current_height.div_ceil(2);
        let high_width = current_width / 2;
        let high_height = current_height / 2;

        levels.push(J2kForwardDwt97Level {
            hl: level.hl.coefficients.clone(),
            lh: level.lh.coefficients.clone(),
            hh: level.hh.coefficients.clone(),
            width: usize_to_u32(current_width, component_index, "level width")?,
            height: usize_to_u32(current_height, component_index, "level height")?,
            low_width: usize_to_u32(low_width, component_index, "low-pass width")?,
            low_height: usize_to_u32(low_height, component_index, "low-pass height")?,
            high_width: usize_to_u32(high_width, component_index, "high-pass width")?,
            high_height: usize_to_u32(high_height, component_index, "high-pass height")?,
        });

        current_width = low_width;
        current_height = low_height;
    }
    levels.reverse();

    Ok(J2kForwardDwt97Output {
        ll: component.final_ll.coefficients.clone(),
        ll_width: usize_to_u32(component.final_ll.width, component_index, "LL width")?,
        ll_height: usize_to_u32(component.final_ll.height, component_index, "LL height")?,
        levels,
    })
}

fn i32_to_f32(coefficients: &[i32]) -> Vec<f32> {
    coefficients
        .iter()
        .map(|&coefficient| coefficient as f32)
        .collect()
}

fn usize_to_u32(
    value: usize,
    component: usize,
    field: &'static str,
) -> Result<u32, WaveletToPrecomputedError> {
    u32::try_from(value).map_err(|_| WaveletToPrecomputedError::DimensionTooLarge {
        component,
        field,
        value,
    })
}
