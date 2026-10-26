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
        // ColorTransform overrides apply only to three-component frames. An
        // ignored RGB hint cannot establish CMYK versus YCCK interpretation.
        let options = if header.component_ids.as_slice().len() == 3 {
            options
        } else {
            options.with_color_transform(ColorTransform::Auto)
        };
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
        for warning in &header.warnings {
            policy.validate_interpretation(warning)?;
        }
        Ok(policy)
    }

    pub(super) const fn sequential(options: DecodeOptions, component_count: usize) -> Self {
        Self {
            options,
            eof_complete: true,
            converts_colour: converts_colour(component_count),
        }
    }

    fn validate_interpretation(self, warning: &Warning) -> Result<(), JpegError> {
        match warning {
            Warning::AdobeApp14Ambiguous { raw_transform }
                if self.converts_colour
                    && self.options.color_transform() == ColorTransform::Auto =>
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
                // Current producers do not emit these notices. A future
                // supported salvage must initialize all output before success.
                Warning::SofDimensionsPatched { .. } | Warning::RestartRecovered { .. } => true,
                Warning::NonstandardTables
                | Warning::AdobeApp14Ambiguous { .. }
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
            assert!(matches!(
                policy.validate(core::slice::from_ref(&ambiguous)),
                Err(JpegError::AmbiguousColorTransform { raw_transform: 7 })
            ));
            for hint in [ColorTransform::ForceRgb, ColorTransform::ForceYCbCr] {
                let hinted = DecodeIntegrity::sequential(options.with_color_transform(hint), 3);
                assert!(!hinted.validate(core::slice::from_ref(&ambiguous)).unwrap());
            }
            assert!(matches!(
                DecodeIntegrity::sequential(options, 4).validate(core::slice::from_ref(&ambiguous)),
                Err(JpegError::AmbiguousColorTransform { raw_transform: 7 })
            ));
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
