// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{ComputeCommandEncoderRef, Error, J2kGrayStoreParams, MetalRuntime, PixelFormat};
use j2k_metal_support::MetalImageDestination;

#[derive(Clone, Copy)]
pub(in crate::engine) struct GrayStoreDestinationRequest<'a> {
    pub(in crate::engine) runtime: &'a MetalRuntime,
    pub(in crate::engine) encoder: &'a ComputeCommandEncoderRef,
    pub(in crate::engine) source: super::GrayStoreSource<'a>,
    pub(in crate::engine) params: J2kGrayStoreParams,
    pub(in crate::engine) dims: (u32, u32),
    pub(in crate::engine) fmt: PixelFormat,
    pub(in crate::engine) destination: &'a MetalImageDestination,
    pub(in crate::engine) destination_item_index: usize,
    /// How the source maps to output samples; see `gray_store_pipeline`.
    pub(in crate::engine) signing: super::GraySourceSigning,
}

pub(in crate::engine) fn encode_gray_store_to_destination_in_encoder(
    request: &GrayStoreDestinationRequest<'_>,
) -> Result<(), Error> {
    let GrayStoreDestinationRequest {
        runtime,
        encoder,
        source,
        mut params,
        dims,
        fmt,
        signing,
        destination,
        destination_item_index,
    } = *request;
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
    if !matches!(
        fmt,
        PixelFormat::Gray8 | PixelFormat::Gray16 | PixelFormat::GrayI16
    ) {
        return Err(Error::capability_rejected(
            j2k_core::CapabilityRejection::unsupported_format(
                "J2K Metal external final-store currently supports Gray8/Gray16/GrayI16",
            ),
        ));
    }
    // SAFETY: `MetalImageDestination` validated this exact allocation range,
    // and the decode-into submission retains exclusive access until command
    // completion or a same-device consumer dependency is registered. The raw
    // handle does not escape this encoder binding.
    let output = unsafe { destination.raw_buffer() };
    super::encode_gray_store(
        runtime,
        encoder,
        source,
        super::GrayStoreOutput {
            buffer: output,
            offset_bytes: layout.byte_offset(),
        },
        &params,
        fmt,
        signing,
    )?;
    Ok(())
}
