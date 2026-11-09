// SPDX-License-Identifier: MIT OR Apache-2.0

//! Pixel-integrity policy shared by CPU decode and coefficient consumers.

use alloc::vec::Vec;

use super::{DecodeOutcome, Decoder, Rect};
use crate::parse::header::ParsedHeader;
use crate::{ColorTransform, DecodeOptions, JpegError, Warning};

#[derive(Clone, Copy, Debug)]
pub(super) struct DecodeIntegrity {
    options: DecodeOptions,
    eof_complete: bool,
    /// Whether the frame converts colour, so an Adobe transform byte selects
    /// how its samples are interpreted. Only three- and four-component frames
    /// do; the byte has no effect on grayscale or two-component frames.
    converts_colour: bool,
}

const fn converts_colour(component_count: usize) -> bool {
    matches!(component_count, 3 | 4)
}

/// `options` with only the colour hints that apply to the frame.
///
/// `ColorTransform` overrides apply only to three-component frames. An ignored
/// RGB or YCbCr hint cannot establish CMYK versus YCCK interpretation, so a
/// four-component frame keeps automatic detection.
fn applicable_hints(options: DecodeOptions, component_count: usize) -> DecodeOptions {
    if component_count == 3 {
        options
    } else {
        options.with_color_transform(ColorTransform::Auto)
    }
}

impl DecodeIntegrity {
    #[expect(
        clippy::inline_always,
        reason = "per-image decode setup on the small-image hot path"
    )]
    #[inline(always)]
    pub(super) fn from_header(
        header: &ParsedHeader,
        options: DecodeOptions,
    ) -> Result<Self, JpegError> {
        let converts_colour = converts_colour(header.component_ids.len());
        if header.warnings.is_empty() {
            return Ok(Self {
                options,
                eof_complete: true,
                converts_colour,
            });
        }
        let options = applicable_hints(options, header.component_ids.len());
        let eof_complete = !header.warnings.contains(&Warning::MissingEoi)
            || !matches!(
                header.sof_kind,
                crate::SofKind::Progressive8 | crate::SofKind::Progressive12
            )
            || progressive_bands_complete(header);
        let policy = Self {
            options,
            eof_complete,
            converts_colour,
        };
        // Reject an unresolved interpretation before output allocation. Recovery
        // and EOF policy is checked after entropy validation so existing typed
        // entropy errors are preserved rather than masked by a recovery notice.
        policy.validate_header_warnings(&header.warnings)?;
        Ok(policy)
    }

    pub(super) fn sequential(options: DecodeOptions, component_count: usize) -> Self {
        Self {
            options: applicable_hints(options, component_count),
            eof_complete: true,
            converts_colour: converts_colour(component_count),
        }
    }

    /// Reject an unresolved interpretation among header warnings before any
    /// output is allocated, for a parsed header or a cached header template.
    pub(super) fn validate_header_warnings(self, warnings: &[Warning]) -> Result<(), JpegError> {
        warnings
            .iter()
            .try_for_each(|warning| self.validate_interpretation(warning))
    }

    /// Whether a reserved Adobe transform leaves the colour interpretation to
    /// the libjpeg-turbo default: a colour frame without an applicable hint.
    fn defaults_interpretation(self) -> bool {
        self.converts_colour && self.options.color_transform() == ColorTransform::Auto
    }

    fn validate_interpretation(self, warning: &Warning) -> Result<(), JpegError> {
        match warning {
            // libjpeg-turbo reads a reserved transform as YCbCr for three
            // components and YCCK for four, the colour space the header
            // reports. Strict decode refuses that guess; lenient decode makes
            // it and `validate` reports the image as recovered.
            Warning::AdobeApp14Ambiguous { raw_transform }
                if self.defaults_interpretation() && self.options.is_strict() =>
            {
                Err(JpegError::AmbiguousColorTransform {
                    raw_transform: *raw_transform,
                })
            }
            Warning::PrecisionClamped { from_bits, .. } => {
                Err(JpegError::UnsupportedBitDepth { depth: *from_bits })
            }
            // There is no producer carrying proof of an authoritative cache
            // fallback today. Never publish output from an unverified cache.
            Warning::TableCacheMismatch { .. } => Err(JpegError::RecoveryRequired {
                warning: warning.clone(),
            }),
            _ => Ok(()),
        }
    }

    pub(super) fn validate(self, warnings: &[Warning]) -> Result<bool, JpegError> {
        let mut recovered = false;
        for warning in warnings {
            self.validate_interpretation(warning)?;
            let needs_recovery = match warning {
                Warning::MissingEoi => !self.eof_complete,
                // Only lenient decode gets here with the interpretation left
                // to the libjpeg-turbo default; strict was rejected above.
                Warning::AdobeApp14Ambiguous { .. } => self.defaults_interpretation(),
                // Current producers do not emit these notices. A future
                // supported salvage must initialize all output before success.
                Warning::SofDimensionsPatched { .. } | Warning::RestartRecovered { .. } => true,
                Warning::NonstandardTables
                | Warning::IccProfileIgnored { .. }
                | Warning::UnknownAppMarker { .. }
                | Warning::UnknownColorProfile
                // validate_interpretation already rejects these under either policy.
                | Warning::PrecisionClamped { .. }
                | Warning::TableCacheMismatch { .. } => false,
            };
            if needs_recovery && self.options.is_strict() {
                return Err(JpegError::RecoveryRequired {
                    warning: warning.clone(),
                });
            }
            recovered |= needs_recovery;
        }
        Ok(recovered)
    }
}

fn progressive_bands_complete(header: &ParsedHeader) -> bool {
    let mut complete = [0u64; 4];
    for parsed in &header.progressive_scans {
        for component in &parsed.scan.components {
            let Some(index) = header
                .component_ids
                .iter()
                .position(|&id| id == component.id)
            else {
                return false;
            };
            let band = (u64::MAX << parsed.scan.ss) & (u64::MAX >> (63 - parsed.scan.se));
            if parsed.scan.al == 0 {
                complete[index] |= band;
            } else {
                complete[index] &= !band;
            }
        }
    }
    complete[..header.component_ids.len()]
        .iter()
        .all(|&component| component == u64::MAX)
}

impl Decoder<'_> {
    #[expect(
        clippy::inline_always,
        reason = "successful decode handoff on the small-image hot path"
    )]
    #[inline(always)]
    pub(super) fn finish_outcome(
        &self,
        decoded: Rect,
        warnings: Vec<Warning>,
    ) -> Result<DecodeOutcome, JpegError> {
        let recovered = if warnings.is_empty() {
            false
        } else {
            self.integrity.validate(&warnings)?
        };
        Ok(DecodeOutcome {
            recovered,
            decoded,
            warnings,
        })
    }

    /// Validate a completed coefficient decode before it is published.
    pub(crate) fn validate_completed_warnings(
        &self,
        warnings: &[Warning],
    ) -> Result<bool, JpegError> {
        let header_recovered = self.integrity.validate(&self.warnings)?;
        Ok(header_recovered | self.integrity.validate(warnings)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::TableKind;

    #[test]
    fn strictness_classifies_all_warning_meanings_without_treating_cosmetics_as_recovery() {
        assert!(DecodeOptions::default().is_strict());
        for strict in [true, false] {
            let options = DecodeOptions::default().with_strict(strict);
            let policy = DecodeIntegrity::sequential(options, 3);
            for warning in [
                Warning::MissingEoi,
                Warning::NonstandardTables,
                Warning::IccProfileIgnored { size: 64 },
                Warning::UnknownAppMarker {
                    marker: 0xed,
                    size: 4,
                },
                Warning::UnknownColorProfile,
            ] {
                assert!(!policy.validate(&[warning]).unwrap());
            }
            for warning in [
                Warning::SofDimensionsPatched {
                    from: (0, 0),
                    to: (8, 8),
                },
                Warning::RestartRecovered { offset: 42 },
            ] {
                let result = policy.validate(core::slice::from_ref(&warning));
                if strict {
                    assert!(
                        matches!(result, Err(JpegError::RecoveryRequired { warning: actual }) if actual == warning)
                    );
                } else {
                    assert!(result.unwrap());
                }
            }
            assert!(matches!(
                policy.validate(&[Warning::PrecisionClamped {
                    from_bits: 12,
                    to_bits: 8
                }]),
                Err(JpegError::UnsupportedBitDepth { depth: 12 })
            ));
            assert!(matches!(
                policy.validate(&[Warning::TableCacheMismatch {
                    which: TableKind::HuffmanDc,
                    id: 0
                }]),
                Err(JpegError::RecoveryRequired { .. })
            ));
            let ambiguous = Warning::AdobeApp14Ambiguous { raw_transform: 7 };
            for hint in [
                ColorTransform::Auto,
                ColorTransform::ForceRgb,
                ColorTransform::ForceYCbCr,
            ] {
                let hinted = options.with_color_transform(hint);
                // Strict refuses libjpeg-turbo's guess; lenient makes it and
                // reports recovery. Three-component hints resolve the
                // interpretation outright; four-component frames ignore them.
                for components in [3, 4] {
                    let result = DecodeIntegrity::sequential(hinted, components)
                        .validate(core::slice::from_ref(&ambiguous));
                    if components == 3 && hint != ColorTransform::Auto {
                        assert!(!result.unwrap(), "{hint:?} resolves three components");
                    } else if strict {
                        assert!(matches!(
                            result,
                            Err(JpegError::AmbiguousColorTransform { raw_transform: 7 })
                        ));
                    } else {
                        assert!(result.unwrap(), "{hint:?} with {components} components");
                    }
                    let header = DecodeIntegrity::sequential(hinted, components)
                        .validate_header_warnings(core::slice::from_ref(&ambiguous));
                    assert_eq!(
                        header.is_err(),
                        strict && (components == 4 || hint == ColorTransform::Auto)
                    );
                }
            }
            for components in [1, 2] {
                let uncoloured = DecodeIntegrity::sequential(options, components);
                assert!(!uncoloured
                    .validate(core::slice::from_ref(&ambiguous))
                    .unwrap());
            }
            let unverified = DecodeIntegrity {
                options,
                eof_complete: false,
                converts_colour: true,
            };
            let result = unverified.validate(&[Warning::MissingEoi]);
            if strict {
                assert!(matches!(result, Err(JpegError::RecoveryRequired { .. })));
            } else {
                assert!(result.unwrap());
            }
        }
    }
}
