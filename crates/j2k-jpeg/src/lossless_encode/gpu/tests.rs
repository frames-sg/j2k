// SPDX-License-Identifier: MIT OR Apache-2.0

//! A host emulation of the device stages, checked against CPU encode.
//! Metal and CUDA kernels implement these stages one for one.

use super::*;
use crate::adapter::JpegBaselineHuffmanTable;
use crate::lossless_encode::{
    assemble_jpeg_lossless_frame, encode_jpeg_lossless, jpeg_lossless_huffman_table,
    JpegLosslessSamples,
};
use j2k_codec_math::jpeg::lossless::{
    additional_bits, difference_category, predict_at, DIFFERENCE_SYMBOLS,
};

fn sample(params: &JpegLosslessGpuEncodeParams, input: &[u8], row: u32, index: u32) -> u16 {
    let offset = (params.input_offset_bytes
        + row * params.pitch_bytes
        + index * params.sample_bytes) as usize;
    if params.sample_bytes == 2 {
        u16::from_ne_bytes([input[offset], input[offset + 1]])
    } else {
        u16::from(input[offset])
    }
}

/// Stage 1: tokens and the category histogram.
fn classify(params: &JpegLosslessGpuEncodeParams, input: &[u8]) -> (Vec<u32>, [u64; 17]) {
    let row_samples = params.width * params.components;
    let mut histogram = [0_u64; DIFFERENCE_SYMBOLS];
    let tokens = (0..params.sample_count)
        .map(|global| {
            let (row, index) = (global / row_samples, global % row_samples);
            let x = index / params.components;
            let shift = params.point_transform;
            let at = |row, index| sample(params, input, row, index) >> shift;
            let ra = if x > 0 {
                at(row, index - params.components)
            } else {
                0
            };
            let rb = if row > 0 { at(row - 1, index) } else { 0 };
            let rc = if x > 0 && row > 0 {
                at(row - 1, index - params.components)
            } else {
                0
            };
            let prediction = predict_at(
                u8::try_from(params.predictor).expect("predictor"),
                u16::try_from(params.initial_predictor).expect("seed"),
                x as usize,
                (row % params.interval_rows) as usize,
                [ra, rb, rc],
            );
            let (category, bits) = difference_category(at(row, index).wrapping_sub(prediction));
            histogram[usize::from(category)] += 1;
            u32::from(category) | u32::from(bits) << 16
        })
        .collect();
    (tokens, histogram)
}

fn token_bits(table: &JpegBaselineHuffmanTable, token: u32) -> u32 {
    let category = (token & 0xFF) as u8;
    u32::from(table.lens[usize::from(category)] + additional_bits(category))
}

fn chunk_tokens(params: &JpegLosslessGpuEncodeParams, chunk: u32) -> core::ops::Range<usize> {
    let (row, k) = (chunk / params.chunks_per_row, chunk % params.chunks_per_row);
    let x0 = k * params.chunk_pixels;
    let x1 = (x0 + params.chunk_pixels).min(params.width);
    let base = row * params.width * params.components;
    (base + x0 * params.components) as usize..(base + x1 * params.components) as usize
}

/// Stages 3-5: offsets and the packed word buffer, in memory byte order.
#[expect(
    clippy::cast_possible_truncation,
    reason = "stream words are the low 32 bits of the bit accumulator, as on the device"
)]
fn pack(
    params: &JpegLosslessGpuEncodeParams,
    tokens: &[u32],
    table: &JpegBaselineHuffmanTable,
) -> (Vec<u8>, Vec<u32>) {
    let chunks = params.chunks();
    let chunk_bits: Vec<u32> = (0..chunks)
        .map(|chunk| {
            chunk_tokens(params, chunk)
                .map(|i| token_bits(table, tokens[i]))
                .sum()
        })
        .collect();
    let mut chunk_offsets = vec![0_u32; chunks as usize];
    let mut row_bits = vec![0_u32; params.height as usize];
    for row in 0..params.height {
        let mut running = 0;
        for k in 0..params.chunks_per_row {
            let chunk = (row * params.chunks_per_row + k) as usize;
            chunk_offsets[chunk] = running;
            running += chunk_bits[chunk];
        }
        row_bits[row as usize] = running;
    }
    let mut row_offsets = vec![0_u32; params.height as usize];
    let mut interval_bits = vec![0_u32; params.intervals as usize];
    for interval in 0..params.intervals {
        let first = interval * params.interval_rows;
        let last = (first + params.interval_rows).min(params.height);
        let mut running = 0;
        for row in first..last {
            row_offsets[row as usize] = running;
            running += row_bits[row as usize];
        }
        interval_bits[interval as usize] = running;
    }
    let interval_bytes: Vec<u32> = interval_bits.iter().map(|bits| bits.div_ceil(8)).collect();
    let mut bases = vec![0_u32; params.intervals as usize];
    let mut running = 0;
    for (base, &bytes) in bases.iter_mut().zip(&interval_bytes) {
        *base = running;
        running += bytes;
    }
    let mut words = vec![0_u32; params.packed_words as usize];
    let mut or_word = |index: u32, stream: u32| words[index as usize] |= stream.swap_bytes();
    for chunk in 0..chunks {
        let row = chunk / params.chunks_per_row;
        let interval = row / params.interval_rows;
        let start = bases[interval as usize] * 8
            + row_offsets[row as usize]
            + chunk_offsets[chunk as usize];
        let (mut word, lead) = (start / 32, start % 32);
        let (mut acc, mut acc_bits) = (0_u64, lead);
        for i in chunk_tokens(params, chunk) {
            let category = (tokens[i] & 0xFF) as u8;
            let extra = additional_bits(category);
            let code = u64::from(table.codes[usize::from(category)]);
            let len = u32::from(table.lens[usize::from(category)]) + u32::from(extra);
            let bits = u64::from(tokens[i] >> 16) & ((1 << extra) - 1);
            acc = (acc << len) | (code << extra) | bits;
            acc_bits += len;
            while acc_bits >= 32 {
                acc_bits -= 32;
                or_word(word, (acc >> acc_bits) as u32);
                acc &= (1 << acc_bits) - 1;
                word += 1;
            }
        }
        if acc_bits > 0 {
            or_word(word, (acc << (32 - acc_bits)) as u32);
        }
        let last_row = ((interval + 1) * params.interval_rows).min(params.height) - 1;
        if row == last_row && chunk % params.chunks_per_row == params.chunks_per_row - 1 {
            let end = bases[interval as usize] * 8 + interval_bits[interval as usize];
            let pad = (8 - end % 8) % 8;
            if pad > 0 {
                let shift = 32 - end % 32 - pad;
                or_word(end / 32, ((1_u32 << pad) - 1) << shift);
            }
        }
    }
    let bytes = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    (bytes, interval_bytes)
}

fn check(format: PixelFormat, width: u32, height: u32, options: JpegLosslessEncodeOptions) {
    let components = if matches!(format, PixelFormat::Gray8 | PixelFormat::Gray16) {
        1
    } else {
        3
    };
    let wide = matches!(format, PixelFormat::Gray16 | PixelFormat::Rgb16);
    let max = (1_u32 << options.precision) - 1;
    let count = (width * height * components) as usize;
    // Noise plus ramps and jumps, so every category and both code-word
    // boundaries occur.
    let values: Vec<u16> = (0..u32::try_from(count).expect("count"))
        .map(|i| {
            let value = (i.wrapping_mul(2_654_435_761) >> 7) % 37
                + i * 13 % (max / 3 + 1)
                + (i % 7) * (max / 9);
            u16::try_from(value.min(max)).expect("sample")
        })
        .collect();
    // A padded device layout: two spare bytes at the front, four per row.
    let sample_bytes = if wide { 2 } else { 1 };
    let pitch = (width * components) as usize * sample_bytes + 4;
    let mut input = vec![0xAB_u8; 2 + pitch * height as usize];
    for (i, &value) in values.iter().enumerate() {
        let (row, index) = (
            i / (width * components) as usize,
            i % (width * components) as usize,
        );
        let at = 2 + row * pitch + index * sample_bytes;
        if wide {
            input[at..at + 2].copy_from_slice(&value.to_ne_bytes());
        } else {
            input[at] = u8::try_from(value).expect("8-bit");
        }
    }
    let tile = JpegLosslessGpuEncodeTile {
        byte_offset: 2,
        width,
        height,
        pitch_bytes: pitch,
        format,
        buffer_len: input.len(),
    };
    let planned = plan_jpeg_lossless_gpu_encode(tile, options, JpegBackend::Metal).expect("plan");
    let (tokens, histogram) = classify(&planned.params, &input);
    let table = jpeg_lossless_huffman_table(&histogram).expect("table");
    let (packed, lengths) = pack(&planned.params, &tokens, &table.codes());
    let entropy = finish_jpeg_lossless_gpu_entropy(&packed, &lengths).expect("finish");
    let device = assemble_jpeg_lossless_frame(&planned.plan, &table, &entropy, JpegBackend::Cpu)
        .expect("frame");
    let narrow: Vec<u8> = values
        .iter()
        .map(|&v| u8::try_from(v).unwrap_or(u8::MAX))
        .collect();
    let samples = match format {
        PixelFormat::Gray8 => JpegLosslessSamples::Gray8 {
            data: &narrow,
            width,
            height,
        },
        PixelFormat::Rgb8 => JpegLosslessSamples::Rgb8 {
            data: &narrow,
            width,
            height,
        },
        PixelFormat::Gray16 => JpegLosslessSamples::Gray16 {
            data: &values,
            width,
            height,
        },
        _ => JpegLosslessSamples::Rgb16 {
            data: &values,
            width,
            height,
        },
    };
    let cpu = encode_jpeg_lossless(samples, options).expect("cpu");
    assert!(
        device.data == cpu.data,
        "{format:?} {width}x{height} {options:?}: device stages differ from CPU encode"
    );
}

#[test]
fn device_stages_reproduce_cpu_encode() {
    let mut options = JpegLosslessEncodeOptions::new(16);
    check(PixelFormat::Gray16, 61, 37, options);
    check(PixelFormat::Gray16, 32, 5, options);
    options.predictor = 7;
    options.restart_rows = Some(3);
    check(PixelFormat::Gray16, 65, 11, options);
    options.restart_rows = Some(1);
    check(PixelFormat::Rgb16, 33, 7, options);
    options.predictor = 4;
    options.point_transform = 3;
    options.restart_rows = None;
    check(PixelFormat::Rgb16, 1, 9, options);
    let mut narrow = JpegLosslessEncodeOptions::new(8);
    for predictor in 1..=7 {
        narrow.predictor = predictor;
        check(PixelFormat::Gray8, 97, 13, narrow);
        check(PixelFormat::Rgb8, 31, 6, narrow);
    }
    narrow.precision = 5;
    narrow.restart_rows = Some(2);
    check(PixelFormat::Gray8, 64, 9, narrow);
}

#[test]
fn finishing_stuffs_bytes_and_numbers_restart_markers() {
    let packed = [0x12, 0xFF, 0x34, 0xFF, 0xFF, 0x00, 0x00];
    let entropy = finish_jpeg_lossless_gpu_entropy(&packed, &[2, 1, 2, 0]).expect("finish");
    assert_eq!(
        entropy,
        [0x12, 0xFF, 0x00, 0xFF, 0xD0, 0x34, 0xFF, 0xD1, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0xD2]
    );
    assert!(finish_jpeg_lossless_gpu_entropy(&packed, &[5, 3]).is_err());
}

#[test]
fn device_tiles_are_validated() {
    let tile = JpegLosslessGpuEncodeTile {
        byte_offset: 1,
        width: 4,
        height: 2,
        pitch_bytes: 8,
        format: PixelFormat::Gray16,
        buffer_len: 64,
    };
    let options = JpegLosslessEncodeOptions::new(12);
    assert!(matches!(
        plan_jpeg_lossless_gpu_encode(tile, options, JpegBackend::Metal),
        Err(JpegLosslessGpuEncodeError::UnalignedSamples { alignment: 2 })
    ));
    let short = JpegLosslessGpuEncodeTile {
        byte_offset: 0,
        pitch_bytes: 6,
        ..tile
    };
    assert!(matches!(
        plan_jpeg_lossless_gpu_encode(short, options, JpegBackend::Metal),
        Err(JpegLosslessGpuEncodeError::PitchTooShort {
            row_bytes: 8,
            pitch_bytes: 6
        })
    ));
    let outside = JpegLosslessGpuEncodeTile {
        byte_offset: 50,
        ..tile
    };
    assert!(matches!(
        plan_jpeg_lossless_gpu_encode(outside, options, JpegBackend::Metal),
        Err(JpegLosslessGpuEncodeError::InputRangeExceedsBuffer { .. })
    ));
    let mut cuda = options;
    cuda.backend = JpegBackend::Cuda;
    assert!(matches!(
        plan_jpeg_lossless_gpu_encode(
            JpegLosslessGpuEncodeTile {
                byte_offset: 0,
                ..tile
            },
            cuda,
            JpegBackend::Metal
        ),
        Err(JpegLosslessGpuEncodeError::UnsupportedBackend { .. })
    ));
}
