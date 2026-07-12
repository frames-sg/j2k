// SPDX-License-Identifier: MIT OR Apache-2.0

//! Allocation-free JP2/JPH color-specification resolution.

use crate::{parse::ParsedImageInfo, J2kColorSpec, J2kError, J2kFileColorSpec};
use j2k_core::{BufferError, Colorspace, Unsupported};

#[derive(Clone, Copy)]
pub(super) enum ColorSelection<'a> {
    Options {
        legacy: J2kFileColorSpec<'a>,
        explicit: &'a [J2kFileColorSpec<'a>],
    },
    Inspected(&'a [J2kColorSpec]),
}

#[derive(Clone, Copy)]
pub(super) enum PlannedColorSpec<'a> {
    Enumerated(u32),
    IccProfile(&'a [u8]),
}

impl ColorSelection<'_> {
    pub(super) fn for_each_resolved<'a, F>(
        &'a self,
        parsed: &ParsedImageInfo,
        mut visit: F,
    ) -> Result<(), J2kError>
    where
        F: FnMut(PlannedColorSpec<'a>) -> Result<(), J2kError>,
    {
        match *self {
            Self::Options { legacy, explicit } => {
                if explicit.is_empty() {
                    return visit(resolve_color_spec(parsed, legacy)?);
                }
                for &color in explicit {
                    visit(resolve_color_spec(parsed, color)?)?;
                }
            }
            Self::Inspected(color_specs) => {
                let has_representable = color_specs
                    .iter()
                    .any(|color| J2kFileColorSpec::from_inspected(color).is_some());
                if !has_representable {
                    return visit(resolve_color_spec(parsed, J2kFileColorSpec::Infer)?);
                }
                for color in color_specs {
                    if let Some(color) = J2kFileColorSpec::from_inspected(color) {
                        visit(resolve_color_spec(parsed, color)?)?;
                    }
                }
            }
        }
        Ok(())
    }

    pub(super) fn writes_rgba_cdef(self, parsed: &ParsedImageInfo) -> Result<bool, J2kError> {
        if parsed.info.components != 4 {
            return Ok(false);
        }
        Ok(matches!(
            self.first_resolved(parsed)?,
            PlannedColorSpec::Enumerated(16)
        ))
    }

    fn first_resolved<'a>(
        &'a self,
        parsed: &ParsedImageInfo,
    ) -> Result<PlannedColorSpec<'a>, J2kError> {
        let mut first = None;
        self.for_each_resolved(parsed, |color| {
            if first.is_none() {
                first = Some(color);
            }
            Ok(())
        })?;
        first.ok_or(J2kError::InternalInvariant {
            what: "JP2/JPH color selection produced no COLR box",
        })
    }
}

impl PlannedColorSpec<'_> {
    pub(super) fn payload_len(self) -> Result<usize, J2kError> {
        match self {
            Self::Enumerated(_) => Ok(7),
            Self::IccProfile(profile) => {
                profile
                    .len()
                    .checked_add(3)
                    .ok_or(J2kError::Buffer(BufferError::SizeOverflow {
                        what: "JP2/JPH ICC COLR payload",
                    }))
            }
        }
    }
}

fn resolve_color_spec<'a>(
    parsed: &ParsedImageInfo,
    color: J2kFileColorSpec<'a>,
) -> Result<PlannedColorSpec<'a>, J2kError> {
    match color {
        J2kFileColorSpec::Infer => Ok(PlannedColorSpec::Enumerated(
            inferred_enumerated_colorspace(parsed.info.components, parsed.info.colorspace)?,
        )),
        J2kFileColorSpec::Enumerated(colorspace) => Ok(PlannedColorSpec::Enumerated(
            enumerated_colorspace_code(colorspace)?,
        )),
        J2kFileColorSpec::IccProfile(profile) => Ok(PlannedColorSpec::IccProfile(profile)),
    }
}

fn inferred_enumerated_colorspace(
    components: u16,
    colorspace: Colorspace,
) -> Result<u32, J2kError> {
    match colorspace {
        Colorspace::Grayscale | Colorspace::SGray => Ok(17),
        Colorspace::YCbCr => Ok(18),
        Colorspace::Rgb | Colorspace::SRgb | Colorspace::Rct | Colorspace::Ict => Ok(16),
        Colorspace::IccTagged if components == 1 => Ok(17),
        Colorspace::IccTagged if components == 3 || components == 4 => Ok(16),
        _ => Err(J2kError::Unsupported(Unsupported {
            what: "JP2/JPH wrapping for this colorspace requires an ICC profile",
        })),
    }
}

fn enumerated_colorspace_code(colorspace: Colorspace) -> Result<u32, J2kError> {
    match colorspace {
        Colorspace::Grayscale | Colorspace::SGray => Ok(17),
        Colorspace::YCbCr => Ok(18),
        Colorspace::Rgb | Colorspace::SRgb | Colorspace::Rct | Colorspace::Ict => Ok(16),
        _ => Err(J2kError::Unsupported(Unsupported {
            what: "JP2/JPH enumerated colorspace must be sRGB, sGray, or YCbCr",
        })),
    }
}
