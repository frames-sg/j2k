// SPDX-License-Identifier: MIT OR Apache-2.0

// Classic tier-1 Metal encode code is split across:
// - encode_bitstream_classic_core.metal
// - encode_bitstream_classic_tokens.metal
// - encode_bitstream_classic_symbol_plan.metal
// - encode_bitstream_classic_kernels.metal
// Keep compute/shader_source.rs concatenation order byte-compatible with the original chunk.
