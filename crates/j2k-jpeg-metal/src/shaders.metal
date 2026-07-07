// SPDX-License-Identifier: MIT OR Apache-2.0

// The JPEG Metal kernels are split across:
// - shaders_shared.metal
// - shaders_encode.metal
// - shaders_decode_helpers.metal
// - shaders_pack_444.metal
// - shaders_decode_fast420.metal
// - shaders_decode_fast422_regions.metal
// - shaders_decode_fast444.metal
// - shaders_pack_subsampled.metal
// Keep compute.rs concatenation order byte-compatible with the original monolith.
