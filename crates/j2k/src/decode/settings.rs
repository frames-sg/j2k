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
    partial_codestream: bool,
}

impl DecodeSettings {
    /// Compatibility policy for explicitly recoverable JP2/JPH metadata.
    ///
    /// Lenient mode may ignore malformed trailing boxes after required
    /// metadata, ignore malformed optional `cdef` or `pclr` boxes, and infer
    /// one undeclared alpha component. Raw codestream and entropy validation,
    /// bounds and overflow checks, allocation limits, and all resource-safety
    /// checks remain strict.
    #[must_use]
    pub const fn lenient() -> Self {
        Self {
            strict: false,
            partial_codestream: false,
        }
    }

    /// Reject malformed codec and container metadata.
    #[must_use]
    pub const fn strict() -> Self {
        Self {
            strict: true,
            partial_codestream: false,
        }
    }

    /// Accept input that may be only the beginning of a codestream, such as
    /// the bytes a progressive viewer has fetched so far.
    ///
    /// The main header must be complete. Decoding then succeeds for any
    /// request whose packets are all present, with the same samples the
    /// complete codestream gives; a request that needs a missing packet
    /// fails with [`crate::J2kError::IncompleteCodestream`].
    /// [`crate::J2kDecoder::codestream_completeness`] reports which
    /// reductions can be decoded. Raw codestreams and JP2/JPH files are
    /// accepted; the JP2/JPH boxes before the codestream must be complete.
    /// Only the CPU decoder reads partial input.
    #[must_use]
    pub const fn with_partial_codestream(mut self, partial: bool) -> Self {
        self.partial_codestream = partial;
        self
    }

    /// Whether the input may be the beginning of a codestream; see
    /// [`Self::with_partial_codestream`].
    #[must_use]
    pub const fn is_partial_codestream(self) -> bool {
        self.partial_codestream
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
            partial_codestream: self.partial_codestream,
        }
    }
}

impl Default for DecodeSettings {
    fn default() -> Self {
        Self::strict()
    }
}

#[cfg(test)]
mod tests {
    use super::DecodeSettings;

    #[test]
    fn facade_policy_preserves_strictness_and_internal_geometry() {
        assert!(DecodeSettings::default().is_strict());
        assert!(!DecodeSettings::default().lenient_tolerance_enabled());
        assert!(!DecodeSettings::lenient().is_strict());
        assert!(DecodeSettings::strict().is_strict());

        assert!(!DecodeSettings::default().is_partial_codestream());
        let partial = DecodeSettings::strict().with_partial_codestream(true);
        assert!(partial.is_partial_codestream() && partial.is_strict());
        assert!(partial.to_native(None).partial_codestream);

        let native = DecodeSettings::strict().to_native(Some((17, 11)));
        assert!(!native.partial_codestream);
        assert!(native.resolve_palette_indices);
        assert!(native.strict);
        assert_eq!(native.target_resolution, Some((17, 11)));
    }
}
