// SPDX-License-Identifier: MIT OR Apache-2.0

// The Metal encode bitstream kernels are split across:
// - encode_bitstream_shared.metal
// - encode_bitstream_classic_core.metal
// - encode_bitstream_classic_tokens.metal
// - encode_bitstream_classic_symbol_plan.metal
// - encode_bitstream_classic_kernels.metal
// - encode_bitstream_ht.metal
// - encode_bitstream_packetize.metal
// Keep compute/shader_source.rs concatenation order byte-compatible with the original monolith.
