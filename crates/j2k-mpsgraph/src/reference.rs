// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{allocation::try_vec, Error};

/// Fixed RGB weights used by the reference graph and its CPU oracle.
pub const RGB8_REFERENCE_CHANNEL_WEIGHTS: [f32; 3] = [0.2126, 0.7152, 0.0722];

/// CPU oracle for the RGB8/NHWC reference graph.
#[expect(
    clippy::cast_precision_loss,
    reason = "the validated image pixel count is intentionally converted to the F32 graph arithmetic domain"
)]
pub fn rgb8_nhwc_reference_cpu(
    pixels: &[u8],
    batch: usize,
    height: usize,
    width: usize,
) -> Result<Vec<f32>, Error> {
    let image_samples = height
        .checked_mul(width)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or(Error::TensorShapeOverflow)?;
    let expected = batch
        .checked_mul(image_samples)
        .ok_or(Error::TensorShapeOverflow)?;
    if pixels.len() != expected {
        return Err(Error::InvalidTensorContract {
            reason: "RGB8/NHWC CPU oracle input length does not match its shape",
        });
    }
    let spatial_pixels = height
        .checked_mul(width)
        .ok_or(Error::TensorShapeOverflow)?;
    if spatial_pixels == 0 || batch == 0 {
        return Err(Error::InvalidTensorContract {
            reason: "RGB8/NHWC CPU oracle dimensions must be nonzero",
        });
    }
    let mut scores = try_vec(batch, "RGB8/NHWC CPU oracle scores")?;
    for image in pixels.chunks_exact(image_samples) {
        let weighted_sum = image.chunks_exact(3).fold(0.0_f32, |sum, rgb| {
            sum + f32::from(rgb[0]) * RGB8_REFERENCE_CHANNEL_WEIGHTS[0]
                + f32::from(rgb[1]) * RGB8_REFERENCE_CHANNEL_WEIGHTS[1]
                + f32::from(rgb[2]) * RGB8_REFERENCE_CHANNEL_WEIGHTS[2]
        });
        scores.push(weighted_sum / (255.0 * spatial_pixels as f32));
    }
    Ok(scores)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oracle_reduces_each_image_independently() {
        let pixels = [255, 0, 0, 0, 255, 0];
        let scores = rgb8_nhwc_reference_cpu(&pixels, 2, 1, 1).expect("valid oracle input");
        assert_eq!(scores, [0.2126, 0.7152]);
    }

    #[test]
    fn oracle_rejects_shape_length_mismatch() {
        assert!(matches!(
            rgb8_nhwc_reference_cpu(&[0; 5], 1, 1, 2),
            Err(Error::InvalidTensorContract { .. })
        ));
    }

    #[test]
    fn oracle_rejects_zero_batch_and_spatial_dimensions() {
        for (batch, height, width) in [(0, 1, 1), (1, 0, 1), (1, 1, 0)] {
            assert!(matches!(
                rgb8_nhwc_reference_cpu(&[], batch, height, width),
                Err(Error::InvalidTensorContract { .. })
            ));
        }
    }
}
