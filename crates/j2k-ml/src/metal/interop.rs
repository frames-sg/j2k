// SPDX-License-Identifier: MIT OR Apache-2.0

use burn_core::tensor::{DType, Int, Shape, Tensor};
use burn_cubecl::cubecl::{wgpu::WgpuRuntime, Runtime};
use burn_cubecl::{tensor::CubeTensor, CubeBackend};
use burn_fusion::{get_client, stream::OperationStreams, NoOp};
use burn_ir::{BackendIr, InitOperationIr, OperationIr};
use burn_wgpu::{graphics, RuntimeOptions, Wgpu, WgpuDevice, WgpuSetup};
use j2k::{BatchGroupInfo, PixelFormat};
use j2k_metal::{MetalBackendSession, MetalImageDestination};
use metal::foreign_types::ForeignType;

use crate::BurnDecodeError;

type InnerWgpu = CubeBackend<WgpuRuntime, f32, i32, u32>;

pub(super) fn paired_metal_runtime(
) -> Result<(MetalBackendSession, WgpuDevice, metal::CommandQueue), BurnDecodeError> {
    use std::sync::OnceLock;

    // CubeCL permits one `init_device` call for each externally created setup.
    // Cache the setup and its initialized logical device together so every
    // decoder shares the same client, allocator, Metal device, and queue.
    static DEFAULT_RUNTIME: OnceLock<(WgpuSetup, WgpuDevice)> = OnceLock::new();
    let (setup, burn) = DEFAULT_RUNTIME.get_or_init(|| {
        let setup = burn_wgpu::init_setup::<graphics::Metal>(
            &WgpuDevice::DefaultDevice,
            RuntimeOptions::default(),
        );
        let burn = burn_wgpu::init_device(setup.clone(), RuntimeOptions::default());
        (setup, burn)
    });
    paired_from_setup(setup, burn.clone())
}

fn paired_from_setup(
    setup: &WgpuSetup,
    burn: WgpuDevice,
) -> Result<(MetalBackendSession, WgpuDevice, metal::CommandQueue), BurnDecodeError> {
    // SAFETY: setup was explicitly created with the Metal graphics API. The
    // returned destruction guard is dropped while the cached setup stays live.
    let hal_device = unsafe { setup.device.as_hal::<wgpu_hal::api::Metal>() }.ok_or_else(|| {
        BurnDecodeError::AcceleratorInterop {
            backend: "Metal",
            message: "Burn wgpu setup did not expose a Metal device".to_string(),
        }
    })?;
    let retained = hal_device.retained_raw_handle();
    // SAFETY: the patched HAL accessor transfers one +1 Objective-C retain;
    // `metal::Device` adopts exactly that retain and releases it on drop.
    let metal_device = unsafe { metal::Device::from_ptr(retained.as_ptr().cast()) };
    drop(hal_device);
    // SAFETY: setup was created with the Metal graphics API. The retained
    // queue is the exact queue CubeCL/wgpu will use after setup is installed.
    let hal_queue = unsafe { setup.queue.as_hal::<wgpu_hal::api::Metal>() }.ok_or_else(|| {
        BurnDecodeError::AcceleratorInterop {
            backend: "Metal",
            message: "Burn wgpu setup did not expose a Metal command queue".to_string(),
        }
    })?;
    let retained_queue = hal_queue.retained_raw_handle();
    // SAFETY: the patched HAL accessor transfers one +1 Objective-C retain;
    // `metal::CommandQueue` adopts exactly that retain and releases it on drop.
    let metal_queue = unsafe { metal::CommandQueue::from_ptr(retained_queue.as_ptr().cast()) };
    drop(hal_queue);
    let codec = MetalBackendSession::with_command_queue(metal_device, metal_queue.clone())?;
    Ok((codec, burn, metal_queue))
}

/// Fresh Burn allocation kept private behind a pending codec status owner.
/// The payload is declared first so Drop retires Metal work before the `CubeCL`
/// allocation can be released.
pub(super) struct SubmittedBatchIntTensor<R, const D: usize> {
    payload: R,
    cube: burn_cubecl::tensor::CubeTensor<WgpuRuntime>,
    shape: Shape,
    dtype: DType,
    device: WgpuDevice,
}

impl<R, const D: usize> SubmittedBatchIntTensor<R, D> {
    pub(super) fn into_parts(
        self,
    ) -> (
        burn_cubecl::tensor::CubeTensor<WgpuRuntime>,
        Shape,
        DType,
        WgpuDevice,
        R,
    ) {
        let Self {
            payload,
            cube,
            shape,
            dtype,
            device,
        } = self;
        (cube, shape, dtype, device, payload)
    }
}

pub(super) fn fill_batch_int_tensor<R, const D: usize>(
    shape: [usize; D],
    dtype: DType,
    info: &BatchGroupInfo,
    device: &WgpuDevice,
    submit: impl FnOnce(MetalImageDestination) -> Result<R, BurnDecodeError>,
) -> Result<SubmittedBatchIntTensor<R, D>, BurnDecodeError> {
    let logical_len = shape
        .iter()
        .try_fold(dtype.size(), |size, dim| size.checked_mul(*dim))
        .ok_or(BurnDecodeError::SizeOverflow)?;
    let tracked_len = logical_len
        .checked_next_multiple_of(4)
        .ok_or(BurnDecodeError::SizeOverflow)?;
    let burn_shape = Shape::from(shape.to_vec());
    let client = WgpuRuntime::client(device);
    // wgpu buffer bindings and lazy initialization tracking operate on
    // four-byte-sized ranges. Reserve that rounded range as part of this
    // tensor's unique CubeCL allocation, while retaining the exact logical
    // shape in the tensor metadata. This makes the final 1-3 padding bytes
    // ours rather than borrowing bytes from a neighboring pooled allocation.
    let handle = client.empty(tracked_len);
    let cube =
        CubeTensor::new_contiguous(client, device.clone(), burn_shape.clone(), handle, dtype);
    let handle_len =
        usize::try_from(cube.handle.size_in_used()).map_err(|_| BurnDecodeError::SizeOverflow)?;
    if handle_len != tracked_len {
        return Err(interop(format!(
            "CubeCL tensor handle exposes {handle_len} bytes; expected {tracked_len} including tracker padding"
        )));
    }
    let resource = cube
        .client
        .get_resource(cube.handle.clone())
        .map_err(|error| interop(format!("CubeCL resource access failed: {error}")))?;
    let raw = resource.resource();
    let available = usize::try_from(raw.size).map_err(|_| BurnDecodeError::SizeOverflow)?;
    if logical_len > available {
        return Err(interop(format!(
            "CubeCL resource exposes {available} bytes for a {logical_len}-byte tensor"
        )));
    }
    let base = usize::try_from(raw.offset).map_err(|_| BurnDecodeError::SizeOverflow)?;
    let initialized_range = tracked_external_write_range(
        base,
        logical_len,
        available,
        usize::try_from(raw.buffer.size()).map_err(|_| BurnDecodeError::SizeOverflow)?,
    )?;
    // SAFETY: the resource is a live wgpu Metal buffer. The guard prevents
    // destruction while the ownership-bearing retained handle is adopted.
    let hal_buffer = unsafe { raw.buffer.as_hal::<wgpu_hal::api::Metal>() }
        .ok_or_else(|| interop("Burn tensor allocation is not Metal-backed"))?;
    let retained = hal_buffer.retained_raw_handle();
    // SAFETY: the patched HAL accessor transfers one +1 Objective-C retain;
    // `metal::Buffer` adopts it exactly once.
    let metal_buffer = unsafe { metal::Buffer::from_ptr(retained.as_ptr().cast()) };
    drop(hal_buffer);

    let pixel_format = pixel_format(info)?;
    let row_bytes = usize::try_from(info.dimensions.0)
        .ok()
        .and_then(|width| width.checked_mul(pixel_format.bytes_per_pixel()))
        .ok_or(BurnDecodeError::SizeOverflow)?;
    let image_bytes = row_bytes
        .checked_mul(info.dimensions.1 as usize)
        .ok_or(BurnDecodeError::SizeOverflow)?;
    let image_count = shape[0];
    let layout = j2k_metal::MetalImageLayout::new_batch(
        base,
        info.dimensions,
        row_bytes,
        pixel_format,
        image_count,
        image_bytes,
    )
    .map_err(|error| interop(error.to_string()))?;
    // SAFETY: `cube` has not been registered with Burn, no tensor alias has
    // escaped, and the managed resource remains live through the completed
    // codec submission. The destination validates the exact suballocation.
    let destination = unsafe { MetalImageDestination::from_exclusive_buffer(metal_buffer, layout) }
        .map_err(|error| interop(error.to_string()))?;
    let output = submit(destination)?;
    // SAFETY: `cube` is still private and uniquely owned. The submission
    // callback has registered the codec producer dependency on Burn's exact
    // consumer queue before returning, and the codec final store initializes
    // every logical byte in this dense tensor subrange. CubeCL binds the final
    // 1-3 bytes as inaccessible alignment padding with shader bounds checks,
    // so the tracker range mirrors its four-byte-rounded binding. Without this
    // handoff wgpu's first use would zero the decoded allocation.
    unsafe {
        raw.buffer
            .mark_external_write_initialized(initialized_range)
    }
    .map_err(|error| interop(error.to_string()))?;
    drop(resource);
    Ok(SubmittedBatchIntTensor {
        payload: output,
        cube,
        shape: burn_shape,
        dtype,
        device: device.clone(),
    })
}

fn tracked_external_write_range(
    base: usize,
    logical_len: usize,
    allocation_len: usize,
    buffer_len: usize,
) -> Result<std::ops::Range<u64>, BurnDecodeError> {
    if !base.is_multiple_of(4) {
        return Err(interop(format!(
            "CubeCL tensor suballocation offset {base} is not four-byte aligned"
        )));
    }
    let tracked_len = logical_len
        .checked_next_multiple_of(4)
        .ok_or(BurnDecodeError::SizeOverflow)?;
    let end = base
        .checked_add(tracked_len)
        .ok_or(BurnDecodeError::SizeOverflow)?;
    let allocation_end = base
        .checked_add(allocation_len)
        .ok_or(BurnDecodeError::SizeOverflow)?;
    if end > allocation_end {
        return Err(interop(format!(
            "CubeCL tensor tracker range {base}..{end} exceeds its exact {base}..{allocation_end} suballocation"
        )));
    }
    if end > buffer_len {
        return Err(interop(format!(
            "CubeCL tensor tracker range {base}..{end} exceeds its {buffer_len}-byte wgpu buffer"
        )));
    }
    Ok(
        u64::try_from(base).map_err(|_| BurnDecodeError::SizeOverflow)?
            ..u64::try_from(end).map_err(|_| BurnDecodeError::SizeOverflow)?,
    )
}

fn pixel_format(info: &BatchGroupInfo) -> Result<PixelFormat, BurnDecodeError> {
    info.native_pixel_format()
        .ok_or(BurnDecodeError::UnsupportedCodecContract)
}

pub(super) fn register_int_tensor<const D: usize>(
    cube: burn_cubecl::tensor::CubeTensor<WgpuRuntime>,
    shape: Shape,
    dtype: DType,
    device: &WgpuDevice,
) -> Tensor<Wgpu, D, Int> {
    let fusion = get_client::<InnerWgpu>(device);
    let handle = <InnerWgpu as BackendIr>::int_tensor_handle(cube);
    let desc = InitOperationIr::create(shape, dtype, || fusion.register_tensor_handle(handle));
    let primitive = fusion
        .register(
            OperationStreams::default(),
            OperationIr::Init(desc),
            NoOp::<InnerWgpu>::new(),
        )
        .remove(0);
    Tensor::<Wgpu, D, Int>::from_primitive(primitive)
}

fn interop(message: impl Into<String>) -> BurnDecodeError {
    BurnDecodeError::AcceleratorInterop {
        backend: "Metal",
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::tracked_external_write_range;

    #[cfg(target_os = "macos")]
    #[test]
    fn paired_runtime_uses_burns_exact_command_queue() {
        let (codec, _device, burn_queue) =
            super::paired_metal_runtime().expect("paired Metal runtime");
        assert!(
            codec
                .uses_command_queue(&burn_queue)
                .expect("initialized codec queue identity"),
            "the codec must submit on Burn's exact queue so prior and future Burn work is ordered"
        );
    }

    #[test]
    fn external_write_tracker_range_covers_odd_tensor_tails_without_crossing_buffer() {
        assert_eq!(tracked_external_write_range(4, 5, 8, 12).unwrap(), 4..12);
        assert_eq!(
            tracked_external_write_range(32, 30, 32, 64).unwrap(),
            32..64
        );
        assert_eq!(
            tracked_external_write_range(256, 33, 36, 292).unwrap(),
            256..292
        );
    }

    #[test]
    fn external_write_tracker_range_rejects_invalid_suballocations_before_submission() {
        assert!(tracked_external_write_range(2, 5, 8, 16).is_err());
        assert!(tracked_external_write_range(8, 5, 8, 15).is_err());
        assert!(tracked_external_write_range(usize::MAX - 3, 5, 8, usize::MAX).is_err());
        assert!(tracked_external_write_range(8, 5, 5, 64).is_err());
    }
}
