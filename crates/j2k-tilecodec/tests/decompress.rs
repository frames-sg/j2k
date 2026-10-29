// SPDX-License-Identifier: MIT OR Apache-2.0

use flate2::{
    write::{DeflateEncoder, ZlibEncoder},
    Compression,
};
use j2k_core::{CodecError, ScratchPool, TileDecompress};
use j2k_tilecodec::{
    DeflateCodec, DeflatePool, LzwCodec, LzwPool, NoPool, TileCodecError, UncompressedCodec,
    ZstdCodec, ZstdPool,
};
use std::io::Write;
use weezl::{encode::Encoder, BitOrder};

fn sample_bytes() -> Vec<u8> {
    (0..=255).cycle().take(8192).collect()
}

#[test]
#[cfg_attr(miri, ignore = "flate2 zlib backend calls foreign functions")]
fn deflate_codec_decodes_zlib_wrapped_payload() {
    let source = sample_bytes();
    let mut compressor = ZlibEncoder::new(Vec::new(), Compression::default());
    compressor.write_all(&source).expect("write zlib");
    let encoded_bytes = compressor.finish().expect("finish zlib");

    let mut pool = DeflatePool::new();
    let mut out = vec![0_u8; source.len()];
    let written =
        DeflateCodec::decompress_into(&mut pool, &encoded_bytes, &mut out).expect("deflate");

    assert_eq!(written, source.len());
    assert_eq!(&out[..written], source.as_slice());
    assert!(pool.bytes_allocated() >= source.len());
}

#[test]
#[cfg_attr(miri, ignore = "flate2 zlib backend calls foreign functions")]
fn deflate_codec_decodes_raw_deflate_payload() {
    let source = sample_bytes();
    let mut compressor = DeflateEncoder::new(Vec::new(), Compression::default());
    compressor.write_all(&source).expect("write deflate");
    let encoded_bytes = compressor.finish().expect("finish deflate");

    let mut pool = DeflatePool::new();
    let mut out = vec![0_u8; source.len()];
    let written =
        DeflateCodec::decompress_into(&mut pool, &encoded_bytes, &mut out).expect("deflate");

    assert_eq!(written, source.len());
    assert_eq!(&out[..written], source.as_slice());
}

#[test]
#[cfg_attr(miri, ignore = "zstd backend calls foreign functions")]
fn zstd_codec_roundtrips_payload() {
    let source = sample_bytes();
    let encoded = zstd::stream::encode_all(std::io::Cursor::new(&source), 1).expect("zstd encode");

    let mut pool = ZstdPool::new();
    let mut out = vec![0_u8; source.len()];
    let written = ZstdCodec::decompress_into(&mut pool, &encoded, &mut out).expect("zstd");

    assert_eq!(written, source.len());
    assert_eq!(&out[..written], source.as_slice());
}

#[test]
fn lzw_codec_decodes_tiff_early_change_payloads() {
    // Varied bytes push the dictionary past 510 codes, where TIFF LZW widens
    // its codes one entry earlier than standard LZW.
    let source = (0_u32..64 * 1024)
        .map(|index| {
            u8::try_from((index.wrapping_mul(2_654_435_761) >> 24) & 0xff).expect("masked")
        })
        .collect::<Vec<_>>();
    let encoded_bytes = Encoder::with_tiff_size_switch(BitOrder::Msb, 8)
        .encode(&source)
        .expect("TIFF lzw encode");

    let mut pool = LzwPool::new();
    let mut out = std::iter::repeat_n(0_u8, source.len()).collect::<Vec<_>>();
    let written =
        LzwCodec::decompress_into(&mut pool, &encoded_bytes, &mut out).expect("TIFF lzw decode");

    assert_eq!(written, source.len());
    assert_eq!(out, source);
}

/// Whether weezl consumes all of `encoded` in one call without reaching the
/// end-of-information code: the final code then still sits in its bit buffer
/// and only a further call, with no input left, completes the stream.
fn lzw_end_code_pending_after_last_byte(encoded: &[u8], decoded_len: usize) -> bool {
    let mut decoder = weezl::decode::Decoder::with_tiff_size_switch(BitOrder::Msb, 8);
    let mut out = vec![0_u8; decoded_len + 1];
    let result = decoder.decode_bytes(encoded, &mut out);
    result.consumed_in == encoded.len() && matches!(result.status, Ok(weezl::LzwStatus::Ok))
}

/// Deterministic short TIFF LZW streams and their sources.
fn short_lzw_streams() -> impl Iterator<Item = (u32, u32, Vec<u8>, Vec<u8>)> {
    (1_u32..=96).flat_map(|length| {
        (0_u32..8).map(move |seed| {
            let source = (0..length)
                .map(|index| {
                    let mixed = index.wrapping_add(seed * 977).wrapping_mul(2_654_435_761);
                    u8::try_from((mixed >> (24 + seed % 4)) & 0xff).expect("masked")
                })
                .collect::<Vec<_>>();
            let encoded = Encoder::with_tiff_size_switch(BitOrder::Msb, 8)
                .encode(&source)
                .expect("TIFF lzw encode");
            (length, seed, source, encoded)
        })
    })
}

#[test]
fn lzw_codec_finishes_streams_whose_last_codes_are_buffered() {
    // weezl reads input ahead of decoding, so for many short streams the call
    // that consumes the last byte still holds codes up to the end code. A
    // decoder that stops once the input is consumed misreports them as
    // truncated.
    let mut pending = 0;
    for (length, seed, source, encoded) in short_lzw_streams() {
        {
            pending += usize::from(lzw_end_code_pending_after_last_byte(&encoded, source.len()));

            let mut pool = LzwPool::new();
            let mut out = vec![0_u8; source.len()];
            let written = LzwCodec::decompress_into(&mut pool, &encoded, &mut out)
                .unwrap_or_else(|error| panic!("length {length} seed {seed}: {error}"));
            assert_eq!(written, source.len(), "length {length} seed {seed}");
            assert_eq!(out, source, "length {length} seed {seed}");
        }
    }
    assert!(pending > 0, "no stream left its final codes buffered");
}

#[test]
fn lzw_codec_rejects_every_prefix_that_loses_data_as_truncated() {
    // Finishing buffered codes after the input ends must not accept a stream
    // cut short. A cut that only loses end-code bits still fills the output,
    // which libtiff reads, so it must decode exactly.
    let mut complete_without_end_code = 0;
    for (length, seed, source, encoded) in short_lzw_streams() {
        for cut in 0..encoded.len() {
            let mut pool = LzwPool::new();
            let mut out = vec![0_u8; source.len()];
            let context = format!("length {length} seed {seed} cut {cut}/{}", encoded.len());
            match LzwCodec::decompress_into(&mut pool, &encoded[..cut], &mut out) {
                Ok(written) => {
                    assert_eq!(written, source.len(), "{context}");
                    assert_eq!(out, source, "{context}");
                    complete_without_end_code += 1;
                }
                Err(err) => assert!(CodecError::is_truncated(&err), "{context}: {err:?}"),
            }
        }
    }
    assert!(complete_without_end_code > 0, "no cut kept every data code");
}

/// MSB-first TIFF LZW codes that spell `source` one literal at a time,
/// widening codes as a TIFF decoder grows its table, optionally ending with
/// the end-of-information code.
fn literal_lzw_stream(source: &[u8], end_code: bool) -> Vec<u8> {
    const CLEAR: u32 = 256;
    const END: u32 = 257;
    let (mut bits, mut bit_count, mut out) = (0_u64, 0_u32, Vec::new());
    let mut push = |code: u32, width: u32| {
        bits = (bits << width) | u64::from(code);
        bit_count += width;
        while bit_count >= 8 {
            bit_count -= 8;
            out.push(u8::try_from((bits >> bit_count) & 0xff).expect("masked"));
        }
        bits &= (1 << bit_count) - 1;
    };
    let (mut width, mut next_code) = (9_u32, 258_u32);
    push(CLEAR, width);
    for (index, &byte) in source.iter().enumerate() {
        push(u32::from(byte), width);
        // Every code after the first after a clear adds one table entry; TIFF
        // widens one entry early.
        if index > 0 {
            next_code += 1;
            if next_code > (1 << width) - 2 && width < 12 {
                width += 1;
            }
        }
    }
    if end_code {
        push(END, width);
    }
    if bit_count > 0 {
        out.push(u8::try_from((bits << (8 - bit_count)) & 0xff).expect("masked"));
    }
    out
}

#[test]
fn lzw_codec_reads_streams_without_an_end_code_that_fill_the_output() {
    // libtiff reads a strip whose codes stop without an end code as long as
    // they produce the whole strip, and reports one that stops short.
    for length in [1_usize, 2, 7, 253, 254, 255, 600, 1500] {
        let source = (0..length)
            .map(|index| u8::try_from(index.wrapping_mul(37) % 251).expect("below 251"))
            .collect::<Vec<_>>();
        let with_end_code = literal_lzw_stream(&source, true);
        let mut reference = vec![0_u8; length];
        assert_eq!(
            LzwCodec::decompress_into(&mut LzwPool::new(), &with_end_code, &mut reference)
                .expect("literal stream with an end code"),
            length
        );
        assert_eq!(
            reference, source,
            "length {length}: literal stream encoding"
        );

        let without_end_code = literal_lzw_stream(&source, false);
        let mut out = vec![0_u8; length];
        let written = LzwCodec::decompress_into(&mut LzwPool::new(), &without_end_code, &mut out)
            .unwrap_or_else(|error| panic!("length {length}: {error}"));
        assert_eq!(written, length, "length {length}");
        assert_eq!(out, source, "length {length}");

        let mut larger = vec![0_u8; length + 1];
        let err = LzwCodec::decompress_into(&mut LzwPool::new(), &without_end_code, &mut larger)
            .expect_err("a stream that stops short of the output must not decode");
        assert!(CodecError::is_truncated(&err), "length {length}: {err:?}");
    }
}

#[test]
fn lzw_codec_roundtrips_payload() {
    let source = sample_bytes();
    let mut compressor = Encoder::with_tiff_size_switch(BitOrder::Msb, 8);
    let encoded_bytes = compressor.encode(&source).expect("lzw encode");

    let mut pool = LzwPool::new();
    let mut out = vec![0_u8; source.len()];
    let written = LzwCodec::decompress_into(&mut pool, &encoded_bytes, &mut out).expect("lzw");

    assert_eq!(written, source.len());
    assert_eq!(&out[..written], source.as_slice());
}

#[test]
#[cfg_attr(miri, ignore = "deflate and zstd backends call foreign functions")]
fn malformed_payloads_are_not_truncated_for_codec_error_classifier() {
    let source = sample_bytes();

    // Stored Deflate block with an invalid LEN/NLEN pair; this cannot be
    // accepted by either the zlib-wrapped or raw-Deflate fallback path.
    let deflate = [0x01_u8, 0x01, 0x00, 0x00, 0x00];

    let mut deflate_pool = DeflatePool::new();
    let mut zstd_pool = ZstdPool::new();
    let mut out = vec![0_u8; source.len()];

    for err in [
        DeflateCodec::decompress_into(&mut deflate_pool, &deflate, &mut out)
            .expect_err("malformed deflate must fail"),
        ZstdCodec::decompress_into(&mut zstd_pool, &[0, 1, 2, 3], &mut out)
            .expect_err("malformed zstd must fail"),
    ] {
        assert!(matches!(&err, TileCodecError::Malformed { .. }), "{err:?}");
        assert!(!CodecError::is_truncated(&err), "{err:?}");
        let source = std::error::Error::source(&err)
            .and_then(|source| source.downcast_ref::<std::io::Error>())
            .expect("malformed backend error must retain its I/O source");
        assert_ne!(source.kind(), std::io::ErrorKind::UnexpectedEof);
    }
}

#[test]
fn truncated_payloads_are_input_errors_for_codec_error_classifier() {
    let source = sample_bytes();
    let mut lzw_encoder = Encoder::with_tiff_size_switch(BitOrder::Msb, 8);
    let mut lzw = lzw_encoder.encode(&source).expect("lzw encode");
    // Losing only the end code still fills the output, so cut into the data.
    lzw.truncate(lzw.len() / 2);

    let mut lzw_pool = LzwPool::new();
    let mut out = vec![0_u8; source.len()];
    let err = LzwCodec::decompress_into(&mut lzw_pool, &lzw, &mut out)
        .expect_err("truncated lzw must fail");

    assert!(matches!(err, TileCodecError::Input(_)), "{err:?}");
    assert!(CodecError::is_truncated(&err), "{err:?}");
}

#[test]
fn uncompressed_codec_copies_input_verbatim() {
    let source = sample_bytes();
    let mut pool = NoPool;
    let mut out = vec![0_u8; source.len()];
    let written = UncompressedCodec::decompress_into(&mut pool, &source, &mut out).expect("copy");

    assert_eq!(written, source.len());
    assert_eq!(out, source);
    assert_eq!(
        UncompressedCodec::expected_size(&source).expect("size"),
        Some(source.len())
    );
}

#[test]
#[cfg_attr(miri, ignore = "deflate and zstd setup calls foreign functions")]
fn codecs_reject_undersized_output() {
    let source = sample_bytes();

    let mut zlib = ZlibEncoder::new(Vec::new(), Compression::default());
    zlib.write_all(&source).expect("write zlib");
    let deflate = zlib.finish().expect("finish zlib");

    let zstd = zstd::stream::encode_all(std::io::Cursor::new(&source), 1).expect("zstd encode");
    let mut lzw_encoder = Encoder::with_tiff_size_switch(BitOrder::Msb, 8);
    let lzw = lzw_encoder.encode(&source).expect("lzw encode");

    let mut deflate_pool = DeflatePool::new();
    let mut zstd_pool = ZstdPool::new();
    let mut lzw_pool = LzwPool::new();
    let mut no_pool = NoPool;
    let mut tiny = vec![0_u8; source.len() / 2];

    assert!(matches!(
        DeflateCodec::decompress_into(&mut deflate_pool, &deflate, &mut tiny),
        Err(TileCodecError::Buffer(_))
    ));
    assert!(matches!(
        ZstdCodec::decompress_into(&mut zstd_pool, &zstd, &mut tiny),
        Err(TileCodecError::Buffer(_))
    ));
    assert!(matches!(
        LzwCodec::decompress_into(&mut lzw_pool, &lzw, &mut tiny),
        Err(TileCodecError::Buffer(_))
    ));
    assert!(matches!(
        UncompressedCodec::decompress_into(&mut no_pool, &source, &mut tiny),
        Err(TileCodecError::Buffer(_))
    ));
}

#[test]
#[cfg_attr(miri, ignore = "flate2 zlib backend calls foreign functions")]
fn deflate_codec_rejects_oversized_zlib_without_full_scratch_allocation() {
    let source = vec![0xA5; 1 << 20];
    let mut compressor = ZlibEncoder::new(Vec::new(), Compression::best());
    compressor.write_all(&source).expect("write zlib");
    let encoded_bytes = compressor.finish().expect("finish zlib");

    let mut pool = DeflatePool::new();
    let mut tiny = vec![0_u8; 128];
    let err = DeflateCodec::decompress_into(&mut pool, &encoded_bytes, &mut tiny).unwrap_err();

    assert!(matches!(
        err,
        TileCodecError::Buffer(j2k_core::BufferError::OutputTooSmall {
            required,
            have: 128
        }) if required == 129
    ));
    assert!(
        pool.bytes_allocated() < source.len() / 16,
        "deflate scratch grew to {} bytes for {} decoded bytes",
        pool.bytes_allocated(),
        source.len()
    );
}

#[test]
#[cfg_attr(miri, ignore = "zstd backend calls foreign functions")]
fn zstd_codec_rejects_oversized_payload_without_full_scratch_allocation() {
    let source = vec![0x5A; 1 << 20];
    let encoded = zstd::stream::encode_all(std::io::Cursor::new(&source), 19).expect("zstd encode");

    let mut pool = ZstdPool::new();
    let mut tiny = vec![0_u8; 128];
    let err = ZstdCodec::decompress_into(&mut pool, &encoded, &mut tiny).unwrap_err();

    assert!(matches!(
        err,
        TileCodecError::Buffer(j2k_core::BufferError::OutputTooSmall {
            required,
            have: 128
        }) if required == 129
    ));
    assert!(
        pool.bytes_allocated() < source.len() / 16,
        "zstd scratch grew to {} bytes for {} decoded bytes",
        pool.bytes_allocated(),
        source.len()
    );
}

#[test]
#[cfg_attr(miri, ignore = "large LZW stress case is too slow under Miri")]
fn lzw_codec_rejects_oversized_payload_without_full_scratch_allocation() {
    let source = vec![0x33; 1 << 20];
    let mut compressor = Encoder::with_tiff_size_switch(BitOrder::Msb, 8);
    let encoded_bytes = compressor.encode(&source).expect("lzw encode");

    let mut pool = LzwPool::new();
    let mut tiny = vec![0_u8; 128];
    let err = LzwCodec::decompress_into(&mut pool, &encoded_bytes, &mut tiny).unwrap_err();

    assert!(matches!(
        err,
        TileCodecError::Buffer(j2k_core::BufferError::OutputTooSmall {
            required,
            have: 128
        }) if required == 129
    ));
    assert!(
        pool.bytes_allocated() < source.len() / 16,
        "lzw scratch grew to {} bytes for {} decoded bytes",
        pool.bytes_allocated(),
        source.len()
    );
}

#[test]
#[cfg_attr(miri, ignore = "zstd backend calls foreign functions")]
fn pools_can_be_reused_across_calls() {
    let source = sample_bytes();
    let encoded = zstd::stream::encode_all(std::io::Cursor::new(&source), 1).expect("zstd encode");
    let mut pool = ZstdPool::new();
    let mut out = vec![0_u8; source.len()];

    let first = ZstdCodec::decompress_into(&mut pool, &encoded, &mut out).expect("first decode");
    let second = ZstdCodec::decompress_into(&mut pool, &encoded, &mut out).expect("second decode");

    assert_eq!(first, source.len());
    assert_eq!(second, source.len());
    assert_eq!(&out[..second], source.as_slice());
    let allocated_before_reset = pool.bytes_allocated();
    assert!(allocated_before_reset >= source.len());

    pool.reset();
    assert_eq!(pool.bytes_allocated(), allocated_before_reset);
}

#[test]
fn lzw_codec_rejects_short_codes_without_an_end_code_as_truncated() {
    // Zero bytes decode as three literal codes, far short of the output, and
    // never reach the end code.
    let mut pool = LzwPool::new();
    let mut out = vec![0_u8; 64];
    let err = LzwCodec::decompress_into(&mut pool, &[0; 4], &mut out)
        .expect_err("a stream without an end code must not decode");
    assert!(CodecError::is_truncated(&err), "{err:?}");
}
