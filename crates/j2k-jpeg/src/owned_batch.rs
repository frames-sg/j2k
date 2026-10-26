// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tensor-shaped batches: same-size JPEGs decoded into one dense RGB8 buffer.

use alloc::vec::Vec;

use j2k_core::{
    checked_batch_count_product, try_batch_reserve_for_push, BatchAllocationBudget,
    BatchInfrastructureError, BatchLayout, PixelFormat, TileBatchError, TileBatchOptions,
};

use crate::{
    DecodeOptions, JpegBatchSession, JpegError, PreparedJpeg, PreparedJpegTileJob, Warning,
};

mod admission;

/// One complete JPEG submitted to [`JpegBatchDecoder::decode`].
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct JpegBatchInput<'a> {
    /// Complete JPEG interchange bytes.
    bytes: &'a [u8],
    /// Decode options, such as a container-derived color-transform override.
    options: DecodeOptions,
}

impl<'a> JpegBatchInput<'a> {
    /// Decode `bytes` with default options.
    #[must_use]
    pub fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            options: DecodeOptions::default(),
        }
    }
    /// Replace the per-image decode options.
    #[must_use]
    pub const fn with_options(mut self, options: DecodeOptions) -> Self {
        self.options = options;
        self
    }

    /// Complete JPEG interchange bytes.
    #[must_use]
    pub const fn bytes(&self) -> &'a [u8] {
        self.bytes
    }

    /// Per-image decode options.
    #[must_use]
    pub const fn options(&self) -> DecodeOptions {
        self.options
    }
}

/// Same-size JPEGs decoded into one dense RGB8 allocation.
///
/// Grayscale inputs are expanded to RGB, so every group has three channels.
#[derive(Debug)]
pub struct JpegBatchGroup {
    dimensions: (u32, u32),
    layout: BatchLayout,
    source_indices: Vec<usize>,
    warnings: Vec<Vec<Warning>>,
    recovered: Vec<bool>,
    samples: Vec<u8>,
}

impl JpegBatchGroup {
    /// Decoded width and height shared by every image.
    #[must_use]
    pub const fn dimensions(&self) -> (u32, u32) {
        self.dimensions
    }

    /// Channel ordering of [`Self::samples`].
    #[must_use]
    pub const fn layout(&self) -> BatchLayout {
        self.layout
    }

    /// Original input indices in batch order.
    #[must_use]
    pub fn source_indices(&self) -> &[usize] {
        &self.source_indices
    }

    /// Non-fatal warnings for each image.
    #[must_use]
    pub fn warnings(&self) -> &[Vec<Warning>] {
        &self.warnings
    }

    /// Whether each row required lenient recovery or unverified completion.
    #[must_use]
    pub fn recovered(&self) -> &[bool] {
        &self.recovered
    }

    /// `[N, 3, H, W]` or `[N, H, W, 3]` RGB8 samples, following [`Self::layout`].
    #[must_use]
    pub fn samples(&self) -> &[u8] {
        &self.samples
    }

    /// Consume the group into its indices, warnings, recovery flags, and samples.
    #[must_use]
    pub fn into_parts(self) -> (Vec<usize>, Vec<Vec<Warning>>, Vec<bool>, Vec<u8>) {
        (
            self.source_indices,
            self.warnings,
            self.recovered,
            self.samples,
        )
    }
}

/// Decoded groups plus per-input failures in input order.
#[derive(Debug)]
pub struct JpegBatchDecodeResult {
    groups: Vec<JpegBatchGroup>,
    errors: Vec<TileBatchError<JpegError>>,
}

impl JpegBatchDecodeResult {
    /// Successfully decoded same-size groups in unspecified order.
    #[must_use]
    pub fn groups(&self) -> &[JpegBatchGroup] {
        &self.groups
    }

    /// Indexed errors for every input that failed to parse or decode.
    #[must_use]
    pub fn errors(&self) -> &[TileBatchError<JpegError>] {
        &self.errors
    }

    /// Consume the result into its groups and indexed errors.
    #[must_use]
    pub fn into_parts(self) -> (Vec<JpegBatchGroup>, Vec<TileBatchError<JpegError>>) {
        (self.groups, self.errors)
    }
}

/// Options for a persistent dense JPEG decoder.
#[derive(Debug, Clone, Copy, Default)]
#[non_exhaustive]
pub struct JpegBatchDecodeOptions {
    /// Dense tensor channel ordering.
    pub layout: BatchLayout,
    /// CPU worker limit, or the shared pool when unspecified.
    pub workers: Option<core::num::NonZeroUsize>,
}

/// Persistent CPU decoder for tensor-shaped JPEG batches.
///
/// Inputs are grouped by decoded size in unspecified order. An input that
/// fails is reported by index and removed from its group; the rest of the group
/// is kept.
#[derive(Debug)]
pub struct JpegBatchDecoder {
    session: JpegBatchSession,
    layout: BatchLayout,
}

impl JpegBatchDecoder {
    /// Create a decoder with the given output layout and CPU worker options.
    #[must_use]
    pub fn new(options: JpegBatchDecodeOptions) -> Self {
        Self {
            session: JpegBatchSession::new(TileBatchOptions::new(options.workers)),
            layout: options.layout,
        }
    }

    /// Decode `inputs` into dense same-size groups.
    ///
    /// # Errors
    ///
    /// Returns [`BatchInfrastructureError`] for allocation or scheduling
    /// failures. Per-input codec failures are in
    /// [`JpegBatchDecodeResult::errors`].
    pub fn decode(
        &mut self,
        inputs: &[JpegBatchInput<'_>],
    ) -> Result<JpegBatchDecodeResult, BatchInfrastructureError> {
        self.decode_with_output_limit(inputs, crate::decoder::DEFAULT_MAX_DECODE_BYTES)
    }

    fn decode_with_output_limit(
        &mut self,
        inputs: &[JpegBatchInput<'_>],
        output_limit: usize,
    ) -> Result<JpegBatchDecodeResult, BatchInfrastructureError> {
        let mut budget = BatchAllocationBudget::new("JPEG batch metadata");
        let mut errors = budget.try_vec(inputs.len(), "JPEG batch errors")?;
        let mut plans: Vec<(admission::Admission, Vec<usize>)> =
            budget.try_vec(inputs.len(), "JPEG batch size groups")?;
        let mut ctx = crate::DecoderContext::new();
        for (index, &input) in inputs.iter().enumerate() {
            let plan = match admission::admit(input, &mut ctx) {
                Ok(plan) => plan,
                Err(source) => {
                    errors.push(TileBatchError { index, source });
                    continue;
                }
            };
            let scratch = if self.layout == BatchLayout::Nchw {
                plan.image_len
            } else {
                0
            };
            let requested = plan
                .image_len
                .saturating_add(scratch)
                .saturating_add(plan.worker_bytes);
            let cap = crate::decoder::DEFAULT_MAX_DECODE_BYTES;
            if requested > cap || plan.image_len.saturating_add(scratch) > output_limit {
                errors.push(TileBatchError {
                    index,
                    source: JpegError::MemoryCapExceeded { requested, cap },
                });
                continue;
            }
            let group = plans
                .iter()
                .position(|(seen, _)| seen.dimensions == plan.dimensions)
                .unwrap_or_else(|| {
                    plans.push((admission::Admission { ..plan }, Vec::new()));
                    plans.len() - 1
                });
            plans[group].0.worker_bytes = plans[group].0.worker_bytes.max(plan.worker_bytes);
            try_batch_reserve_for_push(&mut plans[group].1, "JPEG batch group members")?;
            plans[group].1.push(index);
        }
        drop(ctx);
        let mut groups = budget.try_vec(inputs.len(), "JPEG batch groups")?;
        for (plan, members) in plans {
            let scratch = if self.layout == BatchLayout::Nchw {
                plan.image_len
            } else {
                0
            };
            let output_budget =
                output_limit.min(crate::decoder::DEFAULT_MAX_DECODE_BYTES - plan.worker_bytes);
            let count = (output_budget - scratch) / plan.image_len;
            let count = count.max(1);
            for chunk in members.chunks(count) {
                if let Some(group) =
                    self.decode_group(inputs, plan.dimensions, chunk, &mut errors)?
                {
                    groups.push(group);
                }
            }
        }
        errors.sort_unstable_by_key(|error| error.index);
        Ok(JpegBatchDecodeResult { groups, errors })
    }

    fn decode_group(
        &mut self,
        inputs: &[JpegBatchInput<'_>],
        dimensions: (u32, u32),
        members: &[usize],
        errors: &mut Vec<TileBatchError<JpegError>>,
    ) -> Result<Option<JpegBatchGroup>, BatchInfrastructureError> {
        let channels = PixelFormat::Rgb8.bytes_per_pixel();
        let pixels = checked_batch_count_product(
            dimensions.0 as usize,
            dimensions.1 as usize,
            "JPEG batch image pixels",
        )?;
        let image_len = checked_batch_count_product(pixels, channels, "JPEG batch image")?;
        let total_len =
            checked_batch_count_product(image_len, members.len(), "JPEG batch samples")?;
        let scratch_len = if self.layout == BatchLayout::Nchw {
            image_len
        } else {
            0
        };
        self.session.prepare_owned_output(total_len + scratch_len)?;
        let mut budget = BatchAllocationBudget::new("JPEG batch group");
        let mut samples = budget.try_filled(total_len, 0_u8, "JPEG batch samples")?;

        let mut image = budget.try_vec(scratch_len, "JPEG batch planar scratch")?;
        let output_bytes = samples.capacity() + image.capacity();
        let mut jobs = budget.try_vec(members.len(), "JPEG batch jobs")?;
        jobs.extend(members.iter().zip(samples.chunks_exact_mut(image_len)).map(
            |(&index, out)| PreparedJpegTileJob {
                input: PreparedJpeg::Borrowed(inputs[index].bytes),
                out,
                stride: dimensions.0 as usize * channels,
                options: inputs[index].options,
            },
        ));
        let results = self
            .session
            .decode_prepared_jpeg_tiles_rgb8_with_external_live(&mut jobs, output_bytes)?;
        drop(jobs);

        let mut source_indices = budget.try_vec(members.len(), "JPEG batch source indices")?;
        let mut warnings = budget.try_vec(members.len(), "JPEG batch warnings")?;
        let mut recovered = budget.try_vec(members.len(), "JPEG batch recovery flags")?;
        for (slot, (&index, result)) in members.iter().zip(results).enumerate() {
            match result {
                Ok(tile) => {
                    let kept = source_indices.len();
                    if kept != slot {
                        samples.copy_within(
                            slot * image_len..(slot + 1) * image_len,
                            kept * image_len,
                        );
                    }
                    source_indices.push(index);
                    recovered.push(tile.recovered);
                    warnings.push(tile.warnings);
                }
                Err(source) => errors.push(TileBatchError { index, source }),
            }
        }
        if source_indices.is_empty() {
            return Ok(None);
        }
        samples.truncate(source_indices.len() * image_len);
        if self.layout == BatchLayout::Nchw {
            for slot in samples.chunks_exact_mut(image_len) {
                image.clear();
                image.extend_from_slice(slot);
                to_planar(&image, slot, pixels);
            }
        }
        Ok(Some(JpegBatchGroup {
            dimensions,
            layout: self.layout,
            source_indices,
            warnings,
            recovered,
            samples,
        }))
    }
}

/// Write one interleaved RGB image as three planes.
fn to_planar(interleaved: &[u8], planar: &mut [u8], pixels: usize) {
    let (red, rest) = planar.split_at_mut(pixels);
    let (green, blue) = rest.split_at_mut(pixels);
    for (&[r, g, b], ((red, green), blue)) in interleaved
        .as_chunks::<3>()
        .0
        .iter()
        .zip(red.iter_mut().zip(green.iter_mut()).zip(blue.iter_mut()))
    {
        (*red, *green, *blue) = (r, g, b);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use j2k_test_support::JPEG_BASELINE_420_16X16;

    #[test]
    fn output_budget_splits_groups_without_losing_rows_or_pixels() {
        let inputs = [JpegBatchInput::new(JPEG_BASELINE_420_16X16); 7];
        for layout in [BatchLayout::Nchw, BatchLayout::Nhwc] {
            let options = JpegBatchDecodeOptions {
                layout,
                ..JpegBatchDecodeOptions::default()
            };
            let expected = JpegBatchDecoder::new(options).decode(&inputs).unwrap();
            let split = JpegBatchDecoder::new(options)
                .decode_with_output_limit(&inputs, 3 * 16 * 16 * 3)
                .unwrap();
            assert!(split.errors().is_empty());
            assert!(split.groups().len() > 1);
            assert_eq!(
                split
                    .groups()
                    .iter()
                    .flat_map(|g| g.source_indices().iter().copied())
                    .collect::<Vec<_>>(),
                (0..7).collect::<Vec<_>>()
            );
            assert_eq!(
                split
                    .groups()
                    .iter()
                    .flat_map(|g| g.samples().iter().copied())
                    .collect::<Vec<_>>(),
                expected.groups()[0].samples()
            );
            assert!(split
                .groups()
                .iter()
                .all(|g| g.recovered().iter().all(|&r| !r)));
        }
    }
}
