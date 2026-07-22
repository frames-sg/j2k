// SPDX-License-Identifier: MIT OR Apache-2.0

//! Facade-owned decode validation policy.

/// Validation policy used by prepared and one-shot J2K batch decoding.
///
/// Decode geometry belongs to [`crate::DecodeRequest`], so this value does not
/// carry a second target-resolution setting that could conflict with the
/// request. Palette and component mappings are validated by batch preparation;
/// the native decoder always resolves them on non-batch facade paths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeSettings {
    strict: bool,
}

impl DecodeSettings {
    /// Compatibility policy that permits recoverable optional-metadata errors.
    #[must_use]
    pub const fn lenient() -> Self {
        Self { strict: false }
    }

    /// Fail-closed validation for malformed codec and container metadata.
    #[must_use]
    pub const fn strict() -> Self {
        Self { strict: true }
    }

    /// Whether strict validation is enabled.
    #[must_use]
    pub const fn is_strict(self) -> bool {
        self.strict
    }

    /// Whether recoverable optional-metadata errors may be tolerated.
    #[must_use]
    pub const fn lenient_tolerance_enabled(self) -> bool {
        !self.strict
    }

    pub(crate) const fn to_native(
        self,
        target_resolution: Option<(u32, u32)>,
    ) -> j2k_native::DecodeSettings {
        j2k_native::DecodeSettings {
            resolve_palette_indices: true,
            strict: self.strict,
            target_resolution,
        }
    }
}

impl Default for DecodeSettings {
    fn default() -> Self {
        Self::lenient()
    }
}

#[cfg(test)]
mod tests {
    use super::DecodeSettings;

    #[test]
    fn facade_policy_preserves_strictness_and_internal_geometry() {
        assert!(DecodeSettings::default().lenient_tolerance_enabled());
        assert!(!DecodeSettings::lenient().is_strict());
        assert!(DecodeSettings::strict().is_strict());

        let native = DecodeSettings::strict().to_native(Some((17, 11)));
        assert!(native.resolve_palette_indices);
        assert!(native.strict);
        assert_eq!(native.target_resolution, Some((17, 11)));
    }
}
