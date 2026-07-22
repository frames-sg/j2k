// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    native_decode_error, profile, CudaHtj2kColorDecodePlans, CudaHtj2kDecodePlan,
    CudaHtj2kDecodeProfileDetail, CudaHtj2kProfileReport, CudaHtj2kTransform, DecodeSettings,
    DeviceDecodePlan, DeviceDecodeRequest, Downscale, Error, J2kDecoder, NativeDecoderContext,
    NativeImage, PixelFormat, Rect,
};
#[cfg(feature = "cuda-runtime")]
use crate::allocation::HostPhaseBudget;
#[cfg(feature = "cuda-runtime")]
use j2k_native::{J2kReferencedClassicPlan, J2kReferencedHtj2kPlan};
#[cfg(feature = "cuda-runtime")]
mod color_owners;
#[cfg(feature = "cuda-runtime")]
use self::color_owners::{
    flatten_cuda_color_components, flatten_referenced_classic_cuda_color_tile_components,
    flatten_referenced_cuda_color_tile_components,
};

#[cfg(feature = "cuda-runtime")]
mod color;
#[cfg(feature = "cuda-runtime")]
mod color_decoder;
#[cfg(feature = "cuda-runtime")]
mod color_referenced;
mod grayscale;

#[cfg(feature = "cuda-runtime")]
pub(super) use self::color::{
    build_cuda_color_plan_from_bytes_for_device_plan_with_profile,
    build_cuda_htj2k_color_plans_from_bytes_with_profile,
};
#[cfg(feature = "cuda-runtime")]
pub(super) use self::color_referenced::{
    build_cuda_classic_color_plans_from_referenced_with_profile,
    build_cuda_htj2k_color_plans_from_referenced_with_profile,
};
pub(super) use self::grayscale::{
    build_cuda_classic_grayscale_plans_from_referenced_with_profile,
    build_cuda_htj2k_grayscale_plan_from_bytes_for_device_plan_with_profile,
    build_cuda_htj2k_grayscale_plan_from_bytes_with_profile,
    build_cuda_htj2k_grayscale_plans_from_referenced_with_profile,
};

#[cfg(feature = "cuda-runtime")]
const fn rgba_bit_depths_from_rgb(bit_depths: [u8; 3]) -> [u8; 4] {
    [bit_depths[0], bit_depths[1], bit_depths[2], 0]
}
