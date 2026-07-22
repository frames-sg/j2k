// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k::BatchLayout;

use super::Error;

pub(super) fn validate_stacked_grayscale_destination_indices(
    dimensions: (u32, u32),
    count: usize,
) -> Result<(), Error> {
    let plane_elements = usize::try_from(dimensions.0)
        .ok()
        .and_then(|width| width.checked_mul(dimensions.1 as usize))
        .ok_or_else(index_span_overflow)?;
    validate_stacked_destination_index_span(plane_elements, count)
}

pub(super) fn validate_stacked_color_destination_indices(
    dimensions: (u32, u32),
    channels: usize,
    layout: BatchLayout,
    count: usize,
    broadcast_planes: bool,
) -> Result<(), Error> {
    if !matches!(layout, BatchLayout::Nchw | BatchLayout::Nhwc) {
        return Err(Error::UnsupportedMetalRequest {
            reason: "J2K Metal exact color destination received an unknown batch layout",
        });
    }
    let plane_elements = usize::try_from(dimensions.0)
        .ok()
        .and_then(|width| width.checked_mul(dimensions.1 as usize))
        .ok_or_else(index_span_overflow)?;
    let output_elements = plane_elements
        .checked_mul(channels)
        .ok_or_else(index_span_overflow)?;
    let input_count = if broadcast_planes { 1 } else { count };
    validate_stacked_destination_index_span(plane_elements, input_count)?;
    validate_stacked_destination_index_span(output_elements, count)
}

fn validate_stacked_destination_index_span(
    elements_per_image: usize,
    image_count: usize,
) -> Result<(), Error> {
    let aggregate_elements = elements_per_image
        .checked_mul(image_count)
        .ok_or_else(index_span_overflow)?;
    u32::try_from(aggregate_elements).map_err(|_| index_span_overflow())?;
    Ok(())
}

fn index_span_overflow() -> Error {
    Error::MetalKernel {
        message: "J2K Metal stacked destination element span exceeds u32 shader indexing"
            .to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aggregate_span_accepts_u32_max_and_rejects_one_over() {
        assert!(validate_stacked_destination_index_span(u32::MAX as usize, 1).is_ok());
        assert!(validate_stacked_destination_index_span(65_536, 65_536).is_err());
    }

    #[test]
    fn broadcast_planes_count_one_input_plane_only() {
        let plane_elements = 65_536;
        let batch_count = 65_536;
        assert!(validate_stacked_destination_index_span(plane_elements, 1).is_ok());
        assert!(validate_stacked_destination_index_span(plane_elements, batch_count).is_err());
    }

    #[test]
    fn grayscale_plane_len_times_count_must_fit_shader_indexing() {
        assert!(validate_stacked_grayscale_destination_indices((u32::MAX, 1), 1).is_ok());
        assert!(validate_stacked_grayscale_destination_indices((65_536, 1), 65_536).is_err());
    }

    #[test]
    fn color_channel_spans_are_checked_for_both_tensor_layouts() {
        let dimensions = (1_u32 << 29, 1);
        for layout in [BatchLayout::Nchw, BatchLayout::Nhwc] {
            assert!(
                validate_stacked_color_destination_indices(dimensions, 3, layout, 2, false,)
                    .is_ok()
            );
            assert!(
                validate_stacked_color_destination_indices(dimensions, 3, layout, 3, false,)
                    .is_err()
            );
            assert!(
                validate_stacked_color_destination_indices(dimensions, 4, layout, 1, false,)
                    .is_ok()
            );
            assert!(
                validate_stacked_color_destination_indices(dimensions, 4, layout, 2, false,)
                    .is_err()
            );
        }
    }
}
