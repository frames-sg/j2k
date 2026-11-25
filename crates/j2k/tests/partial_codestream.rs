// SPDX-License-Identifier: MIT OR Apache-2.0

//! Decoding the start of a codestream, as a progressive viewer does while
//! the rest is still arriving.

use j2k::{
    encode_j2k_lossless, wrap_j2k_codestream, DecodeSettings, J2kBlockCodingMode, J2kDecoder,
    J2kError, J2kFileWrapOptions, J2kLosslessEncodeOptions, J2kLosslessSamples, J2kMarkerSegment,
    J2kProgressionOrder, J2kScratchPool,
};
use j2k_core::{CodecError, PixelFormat, Rect};

const WIDTH: u32 = 96;
const HEIGHT: u32 = 72;
const LEVELS: u8 = 4;

struct Source {
    bytes: Vec<u8>,
    components: u16,
    bit_depth: u8,
    format: PixelFormat,
}

/// A 12-bit gray image with structure at every scale, like a CT slice.
fn gray12() -> Source {
    let mut bytes = Vec::with_capacity((WIDTH * HEIGHT * 2) as usize);
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let value = ((x * 37 + y * 11) ^ (x * y)) % 4096;
            bytes.extend_from_slice(&(value as u16).to_le_bytes());
        }
    }
    Source {
        bytes,
        components: 1,
        bit_depth: 12,
        format: PixelFormat::Gray16,
    }
}

fn rgb8() -> Source {
    let bytes = (0..HEIGHT)
        .flat_map(|y| {
            (0..WIDTH).flat_map(move |x| {
                [
                    ((x * 3 + y) & 0xff) as u8,
                    (((x ^ y) * 5) & 0xff) as u8,
                    ((x * y) % 251) as u8,
                ]
            })
        })
        .collect();
    Source {
        bytes,
        components: 3,
        bit_depth: 8,
        format: PixelFormat::Rgb8,
    }
}

fn encode(source: &Source, options: &J2kLosslessEncodeOptions) -> Vec<u8> {
    let samples = J2kLosslessSamples::new(
        &source.bytes,
        WIDTH,
        HEIGHT,
        source.components,
        source.bit_depth,
        false,
    )
    .expect("samples");
    encode_j2k_lossless(samples, options)
        .expect("encode")
        .codestream
}

fn options(mode: J2kBlockCodingMode, progression: J2kProgressionOrder) -> J2kLosslessEncodeOptions {
    J2kLosslessEncodeOptions::default()
        .with_block_coding_mode(mode)
        .with_progression(progression)
        .with_max_decomposition_levels(Some(LEVELS))
}

fn partial() -> DecodeSettings {
    DecodeSettings::strict().with_partial_codestream(true)
}

fn whole_image() -> Rect {
    Rect {
        x: 0,
        y: 0,
        w: WIDTH,
        h: HEIGHT,
    }
}

fn decode_at(
    decoder: &mut J2kDecoder<'_>,
    format: PixelFormat,
    area: Rect,
    reduction: u8,
) -> Result<Vec<u8>, J2kError> {
    let plan = decoder.region_decode_plan(format, area, reduction)?;
    let mut out = vec![0_u8; plan.buffer_bytes];
    decoder.decode_region_scaled_pow2_into(
        &mut J2kScratchPool::new(),
        &mut out,
        plan.row_bytes,
        format,
        area,
        reduction,
    )?;
    Ok(out)
}

/// What the complete codestream decodes to at each reduction.
fn references(codestream: &[u8], format: PixelFormat) -> Vec<Vec<u8>> {
    let mut decoder = J2kDecoder::new(codestream).expect("complete codestream");
    (0..=LEVELS)
        .map(|reduction| decode_at(&mut decoder, format, whole_image(), reduction).expect("decode"))
        .collect()
}

/// Prefix lengths to try: a spread over the whole input, plus the last bytes.
fn prefix_lengths(len: usize) -> Vec<usize> {
    let step = (len / 240).max(1);
    let mut lengths: Vec<usize> = (1..len).step_by(step).collect();
    lengths.extend([len.saturating_sub(2), len.saturating_sub(1), len]);
    lengths.sort_unstable();
    lengths.dedup();
    lengths
}

/// Decode every prefix of `file` and check that each reports a reduction
/// that decodes to the complete codestream's samples, that one level less
/// is refused, and that the usable reduction only improves as bytes arrive.
/// Returns the distinct reductions seen before the input was complete.
fn check_prefixes(file: &[u8], source: &Source, references: &[Vec<u8>]) -> Vec<u8> {
    let mut previous: Option<u8> = None;
    let mut opened = false;
    let mut seen = Vec::new();
    for len in prefix_lengths(file.len()) {
        let prefix = &file[..len];
        let mut decoder = match J2kDecoder::new_with_settings(prefix, partial()) {
            Ok(decoder) => decoder,
            Err(error) => {
                assert!(
                    !opened,
                    "prefix {len} failed after a shorter one opened: {error}"
                );
                assert!(error.is_truncated(), "prefix {len}: {error}");
                continue;
            }
        };
        opened = true;
        let completeness = decoder
            .codestream_completeness()
            .unwrap_or_else(|error| panic!("prefix {len}: completeness: {error}"));
        let reduction = completeness.min_exact_reduction();
        if let (Some(before), current) = (previous, reduction) {
            assert!(
                current.is_some_and(|current| current <= before),
                "prefix {len}: reduction went from {before} to {current:?}"
            );
        }
        if len < file.len() && !seen.contains(&reduction.unwrap_or(u8::MAX)) {
            seen.push(reduction.unwrap_or(u8::MAX));
        }
        if let Some(reduction) = reduction {
            let samples = decode_at(&mut decoder, source.format, whole_image(), reduction)
                .unwrap_or_else(|error| panic!("prefix {len} at {reduction}: {error}"));
            assert_eq!(
                samples,
                references[usize::from(reduction)],
                "prefix {len} at reduction {reduction}"
            );
            if reduction > 0 {
                let refused = decode_at(&mut decoder, source.format, whole_image(), reduction - 1);
                assert!(
                    matches!(refused, Err(J2kError::IncompleteCodestream)),
                    "prefix {len} at {}: {refused:?}",
                    reduction - 1
                );
            }
            previous = Some(reduction);
        } else {
            let refused = decode_at(&mut decoder, source.format, whole_image(), LEVELS);
            let error = refused.expect_err("nothing complete yet");
            assert!(error.is_truncated(), "prefix {len}: {error}");
        }
        if len == file.len() {
            assert!(completeness.is_complete());
            assert_eq!(reduction, Some(0));
        }
    }
    assert!(opened, "no prefix opened");
    seen
}

#[test]
fn htj2k_rpcl_gray_prefixes_decode_progressively() {
    let source = gray12();
    let codestream = encode(
        &source,
        &options(
            J2kBlockCodingMode::HighThroughput,
            J2kProgressionOrder::Rpcl,
        ),
    );
    let references = references(&codestream, source.format);
    let seen = check_prefixes(&codestream, &source, &references);
    // Resolution-major order makes each level usable as soon as it arrives.
    for reduction in 1..=LEVELS {
        assert!(
            seen.contains(&reduction),
            "reduction {reduction} never usable: {seen:?}"
        );
    }
}

#[test]
fn htj2k_rpcl_with_packet_markers_and_lengths() {
    let source = gray12();
    let codestream = encode(
        &source,
        &options(
            J2kBlockCodingMode::HighThroughput,
            J2kProgressionOrder::Rpcl,
        )
        .with_marker_segments(&[
            J2kMarkerSegment::Sop,
            J2kMarkerSegment::Eph,
            J2kMarkerSegment::Plt,
            J2kMarkerSegment::Tlm,
        ]),
    );
    let references = references(&codestream, source.format);
    let seen = check_prefixes(&codestream, &source, &references);
    assert!(seen.contains(&LEVELS), "{seen:?}");
}

#[test]
fn classic_rpcl_rgb_with_layers_and_packed_headers() {
    let source = rgb8();
    for markers in [[J2kMarkerSegment::Ppt], [J2kMarkerSegment::Ppm]] {
        let codestream = encode(
            &source,
            &options(J2kBlockCodingMode::Classic, J2kProgressionOrder::Rpcl)
                .with_quality_layers(2)
                .with_marker_segments(&markers),
        );
        let references = references(&codestream, source.format);
        let seen = check_prefixes(&codestream, &source, &references);
        assert!(seen.len() > 1, "{markers:?}: {seen:?}");
    }
}

#[test]
fn layer_progressive_codestream_reports_what_it_holds() {
    let source = gray12();
    let codestream = encode(
        &source,
        &options(J2kBlockCodingMode::Classic, J2kProgressionOrder::Lrcp).with_quality_layers(3),
    );
    let references = references(&codestream, source.format);
    check_prefixes(&codestream, &source, &references);
}

#[test]
fn jph_file_cut_inside_its_codestream_box() {
    let source = gray12();
    let codestream = encode(
        &source,
        &options(
            J2kBlockCodingMode::HighThroughput,
            J2kProgressionOrder::Rpcl,
        ),
    );
    let file = wrap_j2k_codestream(&codestream, J2kFileWrapOptions::jph()).expect("wrap");
    let references = references(&codestream, source.format);
    let seen = check_prefixes(&file, &source, &references);
    assert!(seen.contains(&LEVELS), "{seen:?}");
}

#[test]
fn tiles_that_arrived_decode_while_later_tiles_are_missing() {
    let source = gray12();
    let (tile_w, tile_h) = (48, 36);
    let codestream = encode(
        &source,
        &options(
            J2kBlockCodingMode::HighThroughput,
            J2kProgressionOrder::Rpcl,
        )
        .with_max_decomposition_levels(Some(3))
        .with_tile_size(Some((tile_w, tile_h)))
        .with_tile_part_packet_limit(Some(4))
        .with_marker_segments(&[J2kMarkerSegment::Tlm]),
    );
    let first_tile = Rect {
        x: 0,
        y: 0,
        w: tile_w,
        h: tile_h,
    };
    let last_tile = Rect {
        x: tile_w,
        y: tile_h,
        w: WIDTH - tile_w,
        h: HEIGHT - tile_h,
    };
    let mut full = J2kDecoder::new(&codestream).expect("complete");
    let first_reference = decode_at(&mut full, source.format, first_tile, 0).expect("tile");

    let mut checked = false;
    for len in prefix_lengths(codestream.len()) {
        let Ok(mut decoder) = J2kDecoder::new_with_settings(&codestream[..len], partial()) else {
            continue;
        };
        let completeness = decoder.codestream_completeness().expect("completeness");
        let tiles = completeness.tile_min_exact_reduction();
        assert_eq!(tiles.len(), 4);
        if tiles[0] == Some(0) && tiles[3] != Some(0) {
            assert_ne!(completeness.min_exact_reduction(), Some(0));
            let samples = decode_at(&mut decoder, source.format, first_tile, 0)
                .unwrap_or_else(|error| panic!("prefix {len}: first tile: {error}"));
            assert_eq!(samples, first_reference, "prefix {len}");
            let refused = decode_at(&mut decoder, source.format, last_tile, 0);
            assert!(
                matches!(refused, Err(J2kError::IncompleteCodestream)),
                "prefix {len}: last tile: {refused:?}"
            );
            checked = true;
        }
    }
    assert!(checked, "no prefix held the first tile without the last");
}

#[test]
fn strict_decoding_still_rejects_a_cut_off_codestream() {
    let source = gray12();
    let codestream = encode(
        &source,
        &options(
            J2kBlockCodingMode::HighThroughput,
            J2kProgressionOrder::Rpcl,
        ),
    );
    let prefix = &codestream[..codestream.len() * 2 / 3];
    if let Ok(mut decoder) = J2kDecoder::new(prefix) {
        assert!(decode_at(&mut decoder, source.format, whole_image(), LEVELS).is_err());
        assert!(decoder.codestream_completeness().is_err());
    }
}

#[test]
fn partial_input_is_never_a_passthrough_candidate() {
    let source = gray12();
    let codestream = encode(
        &source,
        &options(
            J2kBlockCodingMode::HighThroughput,
            J2kProgressionOrder::Rpcl,
        ),
    );
    let view = j2k::J2kView::parse_with_settings(&codestream, partial()).expect("view");
    assert!(view.passthrough_candidate().is_none());
    let view = j2k::J2kView::parse(&codestream).expect("view");
    assert!(view.passthrough_candidate().is_some());
}

#[derive(Default)]
struct CollectRows {
    rows: Vec<u16>,
}

impl j2k_core::RowSink<u16> for CollectRows {
    type Error = J2kError;

    fn write_row(&mut self, _y: u32, row: &[u16]) -> Result<(), Self::Error> {
        self.rows.extend_from_slice(row);
        Ok(())
    }
}

/// Every whole-image decode refuses a prefix that lacks full resolution
/// rather than returning samples decoded from missing packets.
#[test]
fn every_full_resolution_decode_refuses_missing_packets() {
    let source = gray12();
    let codestream = encode(
        &source,
        &options(
            J2kBlockCodingMode::HighThroughput,
            J2kProgressionOrder::Rpcl,
        ),
    );
    let references = references(&codestream, source.format);
    let (len, reduction) = prefix_lengths(codestream.len())
        .into_iter()
        .find_map(|len| {
            let mut decoder = J2kDecoder::new_with_settings(&codestream[..len], partial()).ok()?;
            let reduction = decoder
                .codestream_completeness()
                .ok()?
                .min_exact_reduction()?;
            (reduction == 2).then_some((len, reduction))
        })
        .expect("a prefix that decodes at reduction 2");
    let prefix = &codestream[..len];
    let open = || J2kDecoder::new_with_settings(prefix, partial()).expect("decoder");
    let incomplete = |result: Result<(), J2kError>, what: &str| {
        assert!(
            matches!(result, Err(J2kError::IncompleteCodestream)),
            "{what}: {result:?}"
        );
    };

    let mut out = vec![0_u8; (WIDTH * HEIGHT * 2) as usize];
    incomplete(
        open()
            .decode_into(&mut out, WIDTH as usize * 2, PixelFormat::Gray16)
            .map(drop),
        "decode_into",
    );
    incomplete(open().decode_components().map(drop), "decode_components");
    incomplete(
        open().decode_native_components().map(drop),
        "decode_native_components",
    );
    incomplete(open().decode_srgb8().map(drop), "decode_srgb8");
    let rows = open().decode_rows_u16_bounded(
        &mut CollectRows::default(),
        j2k::J2kRowDecodeOptions::new(8),
    );
    assert!(
        matches!(
            rows,
            Err(j2k_core::DecodeRowsError::Decode(
                J2kError::IncompleteCodestream
            ))
        ),
        "decode_rows_u16_bounded: {rows:?}"
    );
    incomplete(
        open()
            .decode_native_components_at_reduction(reduction - 1)
            .map(drop),
        "decode_native_components_at_reduction",
    );

    // The reported reduction decodes through the other entry points too.
    let native = open()
        .decode_native_components_at_reduction(reduction)
        .expect("native components at the reported reduction");
    assert_eq!(native.planes()[0].bytes_per_sample(), 2);
    assert_eq!(
        native.planes()[0].data(),
        references[usize::from(reduction)]
    );
}
