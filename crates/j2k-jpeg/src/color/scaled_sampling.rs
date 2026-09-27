// SPDX-License-Identifier: MIT OR Apache-2.0

//! libjpeg-turbo's per-component DCT scaling and upsampler choice.
//!
//! When decoding at a reduced scale, libjpeg-turbo (`jdmaster.c`) gives each
//! component its own reduced IDCT: a subsampled component is decoded with a
//! larger IDCT whenever that replaces upsampling, so 4:2:0 chroma at 1/2 is
//! decoded with a full 8x8 IDCT and needs no upsampling at all. `jdsample.c`
//! then picks each component's upsampler: the smoothing ("fancy") filters
//! apply only above 1/8 scale, and the 2:1 horizontal filters only when the
//! component is more than two samples wide; everything else replicates.
//!
//! Every CPU decode path takes its geometry from [`ScaledSampling`] so all of
//! them reproduce libjpeg-turbo (and therefore `OpenSlide`) bit for bit.

use crate::info::{DownscaleFactor, SamplingFactors};

/// How one component's decoded samples reach the output grid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Upsample {
    /// The component is already at output resolution.
    None,
    /// `h2v1_fancy_upsample`: 2:1 horizontal triangle filter.
    FancyH2V1,
    /// `h1v2_fancy_upsample`: 1:2 vertical triangle filter.
    FancyH1V2,
    /// `h2v2_fancy_upsample`: 2:1 horizontal and vertical triangle filter.
    FancyH2V2,
    /// `int_upsample` and friends: each sample is repeated `h_ratio` times
    /// across and each row `v_ratio` times down.
    Replicate,
}

/// Decode geometry of one component at a DCT scale.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ScaledComponent {
    /// Reduced IDCT output size per block side: 8, 4, 2 or 1.
    pub(crate) idct_size: u32,
    /// Output pixels per decoded sample, horizontally.
    pub(crate) h_ratio: u32,
    /// Output pixels per decoded sample, vertically.
    pub(crate) v_ratio: u32,
    /// Decoded samples across the whole image (`downsampled_width`).
    pub(crate) width: u32,
    /// Decoded sample rows in the whole image (`downsampled_height`).
    pub(crate) height: u32,
    /// Upsampler libjpeg-turbo selects for this component.
    pub(crate) upsample: Upsample,
}

/// Per-component geometry for one image at one DCT scale.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ScaledSampling {
    components: [ScaledComponent; 4],
    /// Sampling factors counted in luma-sized output blocks: each component's
    /// factors times its IDCT enlargement. Component planes, MCU strides and
    /// upsampling ratios follow these; the entropy decode keeps the coded
    /// factors.
    pub(crate) effective: SamplingFactors,
}

impl ScaledSampling {
    /// Geometry of an image with `sampling` and full-size `dimensions`
    /// decoded at `downscale`.
    pub(crate) fn new(
        sampling: SamplingFactors,
        downscale: DownscaleFactor,
        dimensions: (u32, u32),
    ) -> Self {
        let block_size = downscale.output_block_size();
        let max_h = u32::from(sampling.max_h);
        let max_v = u32::from(sampling.max_v);
        // jdsample.c: `do_fancy = do_fancy_upsampling && min_DCT_scaled_size > 1`.
        let smoothing = block_size > 1;
        let mut components = [ScaledComponent {
            idct_size: block_size,
            h_ratio: 1,
            v_ratio: 1,
            width: 0,
            height: 0,
            upsample: Upsample::None,
        }; 4];
        let mut effective = [(1u8, 1u8); 4];
        for (index, (h, v)) in sampling.iter().enumerate() {
            let (h, v) = (u32::from(h), u32::from(v));
            // jdmaster.c `jpeg_calc_output_dimensions`: grow the IDCT while
            // both upsampling ratios stay even.
            let mut idct_size = block_size;
            while idct_size < 8
                && (max_h * block_size).is_multiple_of(h * idct_size * 2)
                && (max_v * block_size).is_multiple_of(v * idct_size * 2)
            {
                idct_size *= 2;
            }
            let h_ratio = max_h * block_size / (h * idct_size);
            let v_ratio = max_v * block_size / (v * idct_size);
            let width = downsampled_extent(dimensions.0, h * idct_size, max_h);
            let height = downsampled_extent(dimensions.1, v * idct_size, max_v);
            let upsample = match (h_ratio, v_ratio) {
                (1, 1) => Upsample::None,
                (2, 1) if smoothing && width > 2 => Upsample::FancyH2V1,
                (1, 2) if smoothing => Upsample::FancyH1V2,
                (2, 2) if smoothing && width > 2 => Upsample::FancyH2V2,
                _ => Upsample::Replicate,
            };
            components[index] = ScaledComponent {
                idct_size,
                h_ratio,
                v_ratio,
                width,
                height,
                upsample,
            };
            effective[index] = (
                effective_factor(h, idct_size, block_size),
                effective_factor(v, idct_size, block_size),
            );
        }
        Self {
            components,
            effective: SamplingFactors::from_validated_components(&effective[..sampling.len()]),
        }
    }

    /// Geometry of the component at `index` (declaration and plane order).
    pub(crate) fn component(&self, index: usize) -> ScaledComponent {
        debug_assert!(index < self.effective.len());
        self.components[index]
    }

    /// Whether every component decodes straight to output resolution.
    pub(crate) fn is_unsampled(&self) -> bool {
        self.components[..self.effective.len()]
            .iter()
            .all(|component| component.upsample == Upsample::None)
    }

    /// Whether this is three-component data whose chroma both use the 4:2:0
    /// smoothing filter against a 2x2 first component, the shape the
    /// dedicated 4:2:0 emitters handle.
    pub(crate) fn is_fancy_420(&self) -> bool {
        self.effective.components() == [(2, 2), (1, 1), (1, 1)]
            && self.components[1].upsample == Upsample::FancyH2V2
            && self.components[2].upsample == Upsample::FancyH2V2
    }
}

/// `jdiv_round_up(image_extent * factor * idct_size, max_factor * DCTSIZE)`.
fn downsampled_extent(image_extent: u32, factor_times_idct: u32, max_factor: u32) -> u32 {
    let numerator = u64::from(image_extent) * u64::from(factor_times_idct);
    let denominator = u64::from(max_factor) * 8;
    u32::try_from(numerator.div_ceil(denominator)).unwrap_or(u32::MAX)
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "IDCT enlargement never lifts a factor above the 1..=4 maximum"
)]
fn effective_factor(factor: u32, idct_size: u32, block_size: u32) -> u8 {
    (factor * idct_size / block_size) as u8
}

#[cfg(test)]
mod tests {
    use super::{ScaledSampling, Upsample};
    use crate::info::{DownscaleFactor, SamplingFactors};

    fn chroma(luma: (u8, u8), downscale: DownscaleFactor, width: u32) -> (u32, u32, u32, Upsample) {
        let sampling = SamplingFactors::from_components(&[luma, (1, 1), (1, 1)]).unwrap();
        let chroma = ScaledSampling::new(sampling, downscale, (width, 64)).component(1);
        (
            chroma.idct_size,
            chroma.h_ratio,
            chroma.v_ratio,
            chroma.upsample,
        )
    }

    #[test]
    fn subsampled_chroma_uses_libjpeg_turbo_idct_sizes() {
        use DownscaleFactor::{Eighth, Full, Half, Quarter};
        use Upsample::{FancyH1V2, FancyH2V1, FancyH2V2, None, Replicate};
        // 4:2:0 chroma is decoded at twice the luma IDCT size below full.
        assert_eq!(chroma((2, 2), Full, 64), (8, 2, 2, FancyH2V2));
        assert_eq!(chroma((2, 2), Half, 64), (8, 1, 1, None));
        assert_eq!(chroma((2, 2), Quarter, 64), (4, 1, 1, None));
        assert_eq!(chroma((2, 2), Eighth, 64), (2, 1, 1, None));
        // 4:2:2 and 4:4:0 keep the luma size; 1/8 replicates.
        assert_eq!(chroma((2, 1), Half, 64), (4, 2, 1, FancyH2V1));
        assert_eq!(chroma((2, 1), Eighth, 64), (1, 2, 1, Replicate));
        assert_eq!(chroma((1, 2), Quarter, 64), (2, 1, 2, FancyH1V2));
        assert_eq!(chroma((1, 2), Eighth, 64), (1, 1, 2, Replicate));
        // 4:1:0 chroma grows once, leaving a smoothed 2:1 horizontal step.
        assert_eq!(chroma((4, 2), Half, 64), (8, 2, 1, FancyH2V1));
        assert_eq!(chroma((4, 2), Eighth, 64), (2, 2, 1, Replicate));
        // 4:1:1 and 1x4 always replicate.
        assert_eq!(chroma((4, 1), Half, 64), (4, 4, 1, Replicate));
        assert_eq!(chroma((1, 4), Full, 64), (8, 1, 4, Replicate));
    }

    #[test]
    fn narrow_chroma_replicates_instead_of_smoothing() {
        use DownscaleFactor::{Full, Quarter};
        // Four pixels of 4:2:0 or 4:2:2 leave two chroma samples.
        assert_eq!(chroma((2, 2), Full, 4).3, Upsample::Replicate);
        assert_eq!(chroma((2, 1), Full, 4).3, Upsample::Replicate);
        assert_eq!(chroma((2, 2), Full, 5).3, Upsample::FancyH2V2);
        // 9 pixels at 1/4 leave ceil(9 * 2 / 16) = 2 chroma samples.
        assert_eq!(chroma((2, 1), Quarter, 9).3, Upsample::Replicate);
        assert_eq!(chroma((2, 1), Quarter, 17).3, Upsample::FancyH2V1);
        // The vertical filter has no width limit.
        assert_eq!(chroma((1, 2), Full, 2).3, Upsample::FancyH1V2);
    }

    #[test]
    fn effective_sampling_counts_luma_sized_blocks() {
        let sampling = SamplingFactors::from_components(&[(2, 2), (1, 1), (1, 1)]).unwrap();
        let half = ScaledSampling::new(sampling, DownscaleFactor::Half, (64, 64));
        assert_eq!(half.effective.components(), [(2, 2), (2, 2), (2, 2)]);
        assert!(half.is_unsampled());
        assert!(!half.is_fancy_420());
        let full = ScaledSampling::new(sampling, DownscaleFactor::Full, (64, 64));
        assert_eq!(full.effective.components(), [(2, 2), (1, 1), (1, 1)]);
        assert!(full.is_fancy_420());
    }
}
