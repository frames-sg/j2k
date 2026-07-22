// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    dispatch_2d_pipeline, size_of, Buffer, ComputeCommandEncoderRef, Error, J2kGrayStoreParams,
    MetalRuntime, PixelFormat,
};
use j2k_metal_support::MetalImageDestination;

#[derive(Clone, Copy)]
pub(in crate::compute) struct GrayStoreDestinationRequest<'a> {
    pub(in crate::compute) runtime: &'a MetalRuntime,
    pub(in crate::compute) encoder: &'a ComputeCommandEncoderRef,
    pub(in crate::compute) input: &'a Buffer,
    pub(in crate::compute) input_offset_bytes: usize,
    pub(in crate::compute) params: J2kGrayStoreParams,
    pub(in crate::compute) dims: (u32, u32),
    pub(in crate::compute) fmt: PixelFormat,
    pub(in crate::compute) destination: &'a MetalImageDestination,
    pub(in crate::compute) destination_item_index: usize,
}

pub(in crate::compute) fn encode_gray_store_to_destination_in_encoder(
    request: GrayStoreDestinationRequest<'_>,
) -> Result<(), Error> {
    let GrayStoreDestinationRequest {
        runtime,
        encoder,
        input,
        input_offset_bytes,
        mut params,
        dims,
        fmt,
        destination,
        destination_item_index,
    } = request;
    destination
        .validate_device(&runtime.device)
        .and_then(|()| destination.validate_image(dims, fmt))
        .map_err(|source| {
            crate::error::metal_kernel_support_error(
                "J2K Metal grayscale final-store destination validation failed",
                source,
            )
        })?;
    let layout = destination.layout();
    let bytes_per_sample = fmt.bytes_per_sample();
    if !layout.pitch_bytes().is_multiple_of(bytes_per_sample) {
        return Err(Error::MetalKernel {
            message: "J2K Metal grayscale destination pitch is not sample aligned".to_string(),
        });
    }
    params.output_stride =
        u32::try_from(layout.pitch_bytes() / bytes_per_sample).map_err(|_| Error::MetalKernel {
            message: "J2K Metal grayscale destination stride exceeds u32".to_string(),
        })?;
    let item_offset_bytes = layout
        .image_offset_bytes(destination_item_index)
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal grayscale destination item index exceeds group".to_string(),
        })?;
    if !item_offset_bytes.is_multiple_of(bytes_per_sample) {
        return Err(Error::MetalKernel {
            message: "J2K Metal grayscale destination item stride is not sample aligned"
                .to_string(),
        });
    }
    params.output_item_offset =
        u32::try_from(item_offset_bytes / bytes_per_sample).map_err(|_| Error::MetalKernel {
            message: "J2K Metal grayscale destination item offset exceeds u32".to_string(),
        })?;
    let pipeline = match fmt {
        PixelFormat::Gray8 => &runtime.store_component_gray_u8,
        PixelFormat::Gray16 => &runtime.store_component_gray_u16,
        PixelFormat::GrayI16 => &runtime.store_component_gray_i16,
        _ => {
            return Err(Error::UnsupportedMetalRequest {
                reason: "J2K Metal external final-store currently supports Gray8/Gray16/GrayI16",
            });
        }
    };

    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(input), input_offset_bytes as u64);
    // SAFETY: `MetalImageDestination` validated this exact allocation range,
    // and the decode-into submission retains exclusive access until command
    // completion or a same-device consumer dependency is registered. The raw
    // handle does not escape this encoder binding.
    encoder.set_buffer(
        1,
        Some(unsafe { destination.raw_buffer() }),
        layout.byte_offset() as u64,
    );
    encoder.set_bytes(
        2,
        size_of::<J2kGrayStoreParams>() as u64,
        (&raw const params).cast(),
    );
    dispatch_2d_pipeline(encoder, pipeline, (params.copy_width, params.copy_height));
    Ok(())
}
