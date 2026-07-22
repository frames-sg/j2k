// SPDX-License-Identifier: MIT OR Apache-2.0

mod batch;
mod cache;
mod execution;
mod plan_resolution;
mod planning;
mod profile;

#[cfg(test)]
pub(crate) use self::batch::{
    decode_region_scaled_color_batch_direct_to_device,
    decode_repeated_region_scaled_color_batch_direct_to_device,
};
pub(crate) use self::batch::{
    decode_region_scaled_color_batch_direct_to_device_routed,
    decode_region_scaled_grayscale_batch_direct_to_device_routed,
    decode_repeated_region_scaled_color_batch_direct_to_device_routed,
};
pub(crate) use self::cache::REGION_SCALED_COLOR_PLAN_CACHE_CAP;
#[cfg(test)]
pub(crate) use self::cache::{
    region_scaled_color_plan_builds_for_test, region_scaled_color_plan_test_lock_for_test,
    reset_region_scaled_color_plan_builds_for_test, reset_region_scaled_color_plan_cache_for_test,
};
pub(crate) use self::execution::{
    decode_region_scaled_direct_to_surface, decode_region_scaled_direct_to_surface_with_session,
};
pub(crate) use self::planning::benchmark_region_scaled_direct_plan_prepare;

#[cfg(test)]
mod tests;
