// SPDX-License-Identifier: MIT OR Apache-2.0

//! Aggregate-budgeted exact comparison of two decoded images.

use alloc::vec::Vec;

use super::{allocation::combine_retained_bytes, DecoderContext, Image};
use crate::{DecodedNativeComponents, RawBitmap, Result, ValidationError};

impl Image<'_> {
    /// Decode and compare two images while keeping every simultaneous owner
    /// inside one native decode allocation boundary.
    ///
    /// This adapter is used by recode validation, where both parsed images and
    /// the first decoded result remain live during the second decode.
    ///
    /// # Errors
    ///
    /// Returns an error if either decode fails or their aggregate retained
    /// metadata and output capacities exceed the native decode cap.
    #[doc(hidden)]
    pub fn decoded_samples_equal(&self, other: &Image<'_>) -> Result<bool> {
        self.decoded_samples_equal_with_external_bytes(other, 0)
    }

    /// Compare decoded samples while an owned encoded byte vector remains
    /// live, accounting its allocator-returned capacity in both decode phases.
    ///
    /// # Errors
    ///
    /// Returns an error if either decode fails or the combined retained owners
    /// exceed the native decode cap.
    #[doc(hidden)]
    pub fn decoded_samples_equal_with_retained_bytes(
        &self,
        other: &Image<'_>,
        retained_encoded_bytes: &Vec<u8>,
    ) -> Result<bool> {
        self.decoded_samples_equal_with_external_bytes(other, retained_encoded_bytes.capacity())
    }

    fn decoded_samples_equal_with_external_bytes(
        &self,
        other: &Image<'_>,
        external_bytes: usize,
    ) -> Result<bool> {
        if self.has_uniform_component_precision() && other.has_uniform_component_precision() {
            self.packed_samples_equal(other, external_bytes)
        } else {
            self.component_samples_equal(other, external_bytes)
        }
    }

    fn packed_samples_equal(&self, other: &Image<'_>, external_bytes: usize) -> Result<bool> {
        let source_metadata =
            combine_retained_bytes(external_bytes, self.retained_metadata_bytes()?)?;
        let paired_metadata =
            combine_retained_bytes(source_metadata, other.retained_metadata_bytes()?)?;
        let source = {
            let mut context = DecoderContext::default();
            self.decode_native_with_context_and_retained_baseline(&mut context, paired_metadata)?
        };
        let source_bytes = source
            .allocated_bytes()
            .ok_or(ValidationError::ImageTooLarge)?;
        let encoded_baseline = combine_retained_bytes(paired_metadata, source_bytes)?;
        let encoded = {
            let mut context = DecoderContext::default();
            other
                .decode_native_with_context_and_retained_baseline(&mut context, encoded_baseline)?
        };
        Ok(raw_bitmaps_equal(&source, &encoded))
    }

    fn component_samples_equal(&self, other: &Image<'_>, external_bytes: usize) -> Result<bool> {
        let source_metadata =
            combine_retained_bytes(external_bytes, self.retained_metadata_bytes()?)?;
        let paired_metadata =
            combine_retained_bytes(source_metadata, other.retained_metadata_bytes()?)?;
        let source = {
            let mut context = DecoderContext::default();
            self.decode_native_components_with_context_and_retained_baseline(
                &mut context,
                paired_metadata,
            )?
        };
        let source_bytes = source
            .allocated_bytes()
            .ok_or(ValidationError::ImageTooLarge)?;
        let encoded_baseline = combine_retained_bytes(paired_metadata, source_bytes)?;
        let encoded = {
            let mut context = DecoderContext::default();
            other.decode_native_components_with_context_and_retained_baseline(
                &mut context,
                encoded_baseline,
            )?
        };
        Ok(native_components_equal(&source, &encoded))
    }

    fn has_uniform_component_precision(&self) -> bool {
        if self.settings.resolve_palette_indices && self.boxes.palette.is_some() {
            // Palette columns carry their own bit depth and signedness. The
            // codestream header describes only the index component, so it
            // cannot prove that the resolved output can share one packed
            // native representation.
            return false;
        }
        let Some(first) = self.header.component_infos.first() else {
            return false;
        };
        self.header
            .component_infos
            .iter()
            .all(|component| component.size_info.precision == first.size_info.precision)
    }
}

fn raw_bitmaps_equal(source: &RawBitmap, encoded: &RawBitmap) -> bool {
    source.width == encoded.width
        && source.height == encoded.height
        && source.bit_depth == encoded.bit_depth
        && source.signed == encoded.signed
        && source.component_signed == encoded.component_signed
        && source.num_components == encoded.num_components
        && source.bytes_per_sample == encoded.bytes_per_sample
        && source.data == encoded.data
}

fn native_components_equal(
    source: &DecodedNativeComponents,
    encoded: &DecodedNativeComponents,
) -> bool {
    source.dimensions() == encoded.dimensions()
        && source.planes().len() == encoded.planes().len()
        && source
            .planes()
            .iter()
            .zip(encoded.planes())
            .all(|(source, encoded)| {
                source.dimensions() == encoded.dimensions()
                    && source.sampling() == encoded.sampling()
                    && source.bit_depth() == encoded.bit_depth()
                    && source.signed() == encoded.signed()
                    && source.data() == encoded.data()
            })
}

#[cfg(test)]
mod tests;
