// SPDX-License-Identifier: MIT OR Apache-2.0

// Token packing for externally generated classic Tier-1 symbols.

use alloc::vec::Vec;

use super::super::arithmetic_encoder::{ArithmeticEncoder, ArithmeticEncoderContext};
use crate::writer::BitWriter;

use super::segments::{push_segment, reset_contexts};
use super::EncodedCodeBlockWithSegments;

#[derive(Debug, Clone, Copy)]
pub(crate) struct ClassicTier1TokenSegment {
    pub(crate) token_bit_offset: u32,
    pub(crate) token_bit_count: u32,
    pub(crate) start_coding_pass: u8,
    pub(crate) end_coding_pass: u8,
    pub(crate) use_arithmetic: bool,
}

pub(crate) fn pack_classic_selective_bypass_tier1_tokens(
    token_bytes: &[u8],
    token_segments: &[ClassicTier1TokenSegment],
    number_of_coding_passes: u8,
    missing_bit_planes: u8,
) -> Result<EncodedCodeBlockWithSegments, &'static str> {
    let mut reader = ClassicTier1TokenReader::new(token_bytes);
    let mut contexts = [ArithmeticEncoderContext::default(); 19];
    reset_contexts(&mut contexts);
    let mut data = Vec::new();
    let mut segments = Vec::with_capacity(token_segments.len());

    for segment in token_segments {
        if segment.start_coding_pass > segment.end_coding_pass {
            return Err("classic Tier-1 token segment pass range is invalid");
        }
        if segment.end_coding_pass > number_of_coding_passes {
            return Err("classic Tier-1 token segment exceeds coding passes");
        }
        let token_bit_offset = usize::try_from(segment.token_bit_offset)
            .map_err(|_| "classic Tier-1 token bit offset exceeds usize")?;
        let token_bit_count = usize::try_from(segment.token_bit_count)
            .map_err(|_| "classic Tier-1 token bit count exceeds usize")?;
        reader.seek(token_bit_offset)?;
        if segment.use_arithmetic {
            if token_bit_count % 6 != 0 {
                return Err("classic Tier-1 MQ token segment is not aligned to 6-bit symbols");
            }
            let symbol_count = token_bit_count / 6;
            let mut encoder =
                ArithmeticEncoder::with_capacity(symbol_count.saturating_div(16) + 32);
            for _ in 0..symbol_count {
                let token = reader.read_bits(6)?;
                let ctx = (token & 0x1F) as usize;
                if ctx >= contexts.len() {
                    return Err("classic Tier-1 MQ token context is out of range");
                }
                let bit = (token >> 5) & 1;
                encoder.encode(bit, &mut contexts[ctx]);
            }
            push_segment(
                &mut data,
                &mut segments,
                segment.start_coding_pass,
                segment.end_coding_pass,
                encoder.finish(),
                f64::EPSILON,
                true,
            );
        } else {
            let mut writer = BitWriter::new();
            for _ in 0..token_bit_count {
                writer.write_bit(reader.read_bits(1)?);
            }
            push_segment(
                &mut data,
                &mut segments,
                segment.start_coding_pass,
                segment.end_coding_pass,
                writer.finish(),
                f64::EPSILON,
                false,
            );
        }
    }

    Ok(EncodedCodeBlockWithSegments {
        data,
        segments,
        num_coding_passes: number_of_coding_passes,
        num_zero_bitplanes: missing_bit_planes,
    })
}

struct ClassicTier1TokenReader<'a> {
    bytes: &'a [u8],
    bit_pos: usize,
}

impl<'a> ClassicTier1TokenReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, bit_pos: 0 }
    }

    fn seek(&mut self, bit_pos: usize) -> Result<(), &'static str> {
        if bit_pos > self.bytes.len().saturating_mul(8) {
            return Err("classic Tier-1 token offset exceeds token buffer");
        }
        self.bit_pos = bit_pos;
        Ok(())
    }

    fn read_bits(&mut self, count: u8) -> Result<u32, &'static str> {
        let end = self
            .bit_pos
            .checked_add(usize::from(count))
            .ok_or("classic Tier-1 token bit range overflows")?;
        if end > self.bytes.len().saturating_mul(8) {
            return Err("classic Tier-1 token read exceeds token buffer");
        }
        let mut value = 0u32;
        for _ in 0..count {
            let byte = self.bytes[self.bit_pos / 8];
            let shift = 7 - (self.bit_pos % 8);
            value = (value << 1) | u32::from((byte >> shift) & 1);
            self.bit_pos += 1;
        }
        Ok(value)
    }
}
