// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k::{
    encode_j2k_lossless_components, EncodeBackendPreference, J2kBlockCodingMode, J2kCodec,
    J2kComponentPlane, J2kContext, J2kDecoder, J2kError, J2kLosslessComponentPlane,
    J2kLosslessComponentSamples, J2kLosslessEncodeOptions, J2kRowDecodeOptions,
    ReversibleTransform,
};
use j2k_core::{
    BufferError, DecoderContext, Downscale, ImageDecodeRows, PixelFormat, Rect, RowSink,
    TileBatchDecode,
};
use j2k_native::{
    encode, encode_htj2k, encode_precomputed_htj2k_53, DecodeSettings, EncodeOptions, Image,
    J2kForwardDwt53Level, J2kForwardDwt53Output, PrecomputedHtj2k53Component,
    PrecomputedHtj2k53Image,
};
use j2k_test_support::{
    crop_interleaved_bytes, crop_interleaved_u8, openhtj2k_refinement_fixture,
    openhtj2k_refinement_odd_fixture, openhtj2k_refinement_odd_pixels, openhtj2k_refinement_pixels,
    wrap_jp2_codestream, wrap_jp2_rgba_codestream, PixelRect,
};

type OpenHtFixture = (&'static str, fn() -> &'static [u8], fn() -> &'static [u8]);

const OPENHT_REFINEMENT_FIXTURES: &[OpenHtFixture] = &[
    (
        "ds0_ht_12_b11",
        openhtj2k_refinement_fixture,
        openhtj2k_refinement_pixels,
    ),
    (
        "ds0_ht_09_b11",
        openhtj2k_refinement_odd_fixture,
        openhtj2k_refinement_odd_pixels,
    ),
];

fn encode_codestream(
    pixels: &[u8],
    width: u32,
    height: u32,
    components: u8,
    bit_depth: u8,
    reversible: bool,
) -> Vec<u8> {
    let options = EncodeOptions {
        reversible,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    encode(
        pixels,
        width,
        height,
        components.into(),
        bit_depth,
        false,
        &options,
    )
    .expect("encode")
}

fn rewrite_component_descriptor(bytes: &mut [u8], component: usize, ssiz: u8) {
    let siz_offset = bytes
        .windows(2)
        .position(|marker| marker == [0xff, 0x51])
        .expect("SIZ marker");
    bytes[siz_offset + 40 + component * 3] = ssiz;
}

fn unsigned_29_bytes(sample: u32) -> [u8; 4] {
    [
        (sample & 0xff) as u8,
        ((sample >> 8) & 0xff) as u8,
        ((sample >> 16) & 0xff) as u8,
        ((sample >> 24) & 0x1f) as u8,
    ]
}

fn encode_signed_codestream(pixels: &[u8], width: u32, height: u32, bit_depth: u8) -> Vec<u8> {
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        use_mct: false,
        ..EncodeOptions::default()
    };
    encode(pixels, width, height, 1, bit_depth, true, &options).expect("encode signed")
}

fn encode_codestream_with_levels(
    pixels: &[u8],
    width: u32,
    height: u32,
    components: u8,
    bit_depth: u8,
    reversible: bool,
    levels: u8,
) -> Vec<u8> {
    let options = EncodeOptions {
        reversible,
        num_decomposition_levels: levels,
        ..EncodeOptions::default()
    };
    encode(
        pixels,
        width,
        height,
        components.into(),
        bit_depth,
        false,
        &options,
    )
    .expect("encode")
}

fn encode_ht_codestream(
    pixels: &[u8],
    width: u32,
    height: u32,
    components: u8,
    bit_depth: u8,
) -> Vec<u8> {
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    encode_htj2k(
        pixels,
        width,
        height,
        components.into(),
        bit_depth,
        false,
        &options,
    )
    .expect("encode ht")
}

fn backend_decode_u8(bytes: &[u8]) -> Vec<u8> {
    Image::new(bytes, &DecodeSettings::default())
        .expect("backend image")
        .decode()
        .expect("backend decode")
}

fn backend_decode_u8_scaled(bytes: &[u8], target_resolution: (u32, u32)) -> Vec<u8> {
    let settings = DecodeSettings {
        target_resolution: Some(target_resolution),
        ..DecodeSettings::default()
    };
    Image::new(bytes, &settings)
        .expect("backend image")
        .decode()
        .expect("backend decode")
}

fn backend_decode_u8_region(bytes: &[u8], roi: Rect) -> Vec<u8> {
    let mut context = j2k_native::DecoderContext::default();
    Image::new(bytes, &DecodeSettings::default())
        .expect("backend image")
        .decode_region_with_context((roi.x, roi.y, roi.w, roi.h), &mut context)
        .expect("backend region decode")
        .data
}

fn locally_inspectable_codestream_without_decode_headers() -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&[0xFF, 0x4F]);

    bytes.extend_from_slice(&[0xFF, 0x51]);
    bytes.extend_from_slice(&41_u16.to_be_bytes());
    bytes.extend_from_slice(&0_u16.to_be_bytes());
    bytes.extend_from_slice(&2_u32.to_be_bytes());
    bytes.extend_from_slice(&2_u32.to_be_bytes());
    bytes.extend_from_slice(&0_u32.to_be_bytes());
    bytes.extend_from_slice(&0_u32.to_be_bytes());
    bytes.extend_from_slice(&2_u32.to_be_bytes());
    bytes.extend_from_slice(&2_u32.to_be_bytes());
    bytes.extend_from_slice(&0_u32.to_be_bytes());
    bytes.extend_from_slice(&0_u32.to_be_bytes());
    bytes.extend_from_slice(&1_u16.to_be_bytes());
    bytes.extend_from_slice(&[7, 1, 1]);

    bytes.extend_from_slice(&[0xFF, 0x52]);
    bytes.extend_from_slice(&12_u16.to_be_bytes());
    bytes.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]);

    bytes.extend_from_slice(&[0xFF, 0x90]);
    bytes
}

fn crop_u8(full: &[u8], full_width: usize, channels: usize, roi: Rect) -> Vec<u8> {
    crop_interleaved_u8(full, full_width, channels, pixel_rect(roi))
}

fn crop_bytes(full: &[u8], full_width: usize, bytes_per_pixel: usize, roi: Rect) -> Vec<u8> {
    crop_interleaved_bytes(full, full_width, bytes_per_pixel, pixel_rect(roi))
}

fn pixel_rect(roi: Rect) -> PixelRect {
    PixelRect::new(roi.x, roi.y, roi.w, roi.h)
}

#[derive(Default)]
struct CollectRowsU8 {
    rows: Vec<u8>,
}

impl RowSink<u8> for CollectRowsU8 {
    type Error = J2kError;

    fn write_row(&mut self, _y: u32, row: &[u8]) -> Result<(), Self::Error> {
        self.rows.extend_from_slice(row);
        Ok(())
    }
}

#[derive(Default)]
struct CollectRowsU16 {
    rows: Vec<u16>,
}

impl RowSink<u16> for CollectRowsU16 {
    type Error = J2kError;

    fn write_row(&mut self, _y: u32, row: &[u16]) -> Result<(), Self::Error> {
        self.rows.extend_from_slice(row);
        Ok(())
    }
}

#[test]
fn decode_rgb8_codestream_roundtrips_reversible_pixels() {
    let pixels = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];
    let codestream = encode_codestream(&pixels, 2, 2, 3, 8, true);
    let expected = backend_decode_u8(&codestream);
    let mut decoder = J2kDecoder::new(&codestream).expect("decoder");
    let mut out = [0_u8; 12];
    let outcome = decoder
        .decode_into(&mut out, 2 * 3, PixelFormat::Rgb8)
        .expect("decode");
    assert_eq!(outcome.decoded, j2k_core::Rect::full((2, 2)));
    assert_eq!(out, expected.as_slice());
}

#[test]
fn decoder_new_rejects_codestream_that_only_header_inspection_accepts() {
    let malformed = locally_inspectable_codestream_without_decode_headers();

    J2kDecoder::inspect(&malformed).expect("header inspection still succeeds");
    let Err(err) = J2kDecoder::new(&malformed) else {
        panic!("decoder construction must validate backend");
    };

    assert!(
        matches!(err, J2kError::Backend(_)),
        "expected backend construction error, got {err:?}"
    );
}

#[test]
fn decoder_reuses_native_context_across_multiple_decode_calls() {
    let pixels = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];
    let codestream = encode_codestream(&pixels, 2, 2, 3, 8, true);
    let expected = backend_decode_u8(&codestream);
    let mut decoder = J2kDecoder::new(&codestream).expect("decoder");

    let mut full = [0_u8; 12];
    decoder
        .decode_into(&mut full, 2 * 3, PixelFormat::Rgb8)
        .expect("first decode");
    assert_eq!(full, expected.as_slice());

    let mut scaled = [0_u8; 3];
    decoder
        .decode_scaled_into(
            &mut j2k::J2kScratchPool::new(),
            &mut scaled,
            3,
            PixelFormat::Rgb8,
            Downscale::Half,
        )
        .expect("scaled decode");

    let mut second = [0_u8; 12];
    decoder
        .decode_into(&mut second, 2 * 3, PixelFormat::Rgb8)
        .expect("second decode");
    assert_eq!(second, expected.as_slice());
}

#[test]
fn decode_rgba8_fills_opaque_alpha_for_rgb_source() {
    let pixels = [1, 2, 3, 4, 5, 6];
    let codestream = encode_codestream(&pixels, 2, 1, 3, 8, true);
    let mut decoder = J2kDecoder::new(&codestream).expect("decoder");
    let mut out = [0_u8; 8];
    decoder
        .decode_into(&mut out, 2 * 4, PixelFormat::Rgba8)
        .expect("decode");
    assert_eq!(out, [1, 2, 3, 255, 4, 5, 6, 255]);
}

#[test]
fn decode_gray8_jp2_roundtrips_reversible_pixels() {
    let pixels = [3, 9, 27, 81];
    let codestream = encode_codestream(&pixels, 2, 2, 1, 8, true);
    let jp2 = wrap_jp2_codestream(&codestream, 2, 2, 1, 8, 17);
    let mut decoder = J2kDecoder::new(&jp2).expect("decoder");
    let mut out = [0_u8; 4];
    decoder
        .decode_into(&mut out, 2, PixelFormat::Gray8)
        .expect("decode");
    assert_eq!(out, pixels);
}

#[test]
fn decode_components_exposes_signed_gray8_public_samples() {
    let pixels = [(-10_i8) as u8, (-1_i8) as u8, 0_i8 as u8, 12_i8 as u8];
    let codestream = encode_signed_codestream(&pixels, 2, 2, 8);

    let support = J2kDecoder::inspect_support(&codestream).expect("inspect support");
    assert!(support.has_signed_components());
    let mut decoder = J2kDecoder::new(&codestream).expect("decoder");
    let components = decoder.decode_components().expect("decode components");

    assert_eq!(components.dimensions(), (2, 2));
    assert_eq!(components.planes().len(), 1);
    assert!(components.planes()[0].signed());
    assert_eq!(components.planes()[0].bit_depth(), 8);
    assert_eq!(components.planes()[0].dimensions(), (2, 2));
    let samples = components.planes()[0]
        .samples()
        .iter()
        .map(|sample| sample.round() as i8)
        .collect::<Vec<_>>();
    assert_eq!(samples, [-10, -1, 0, 12]);
}

#[test]
fn decode_gray16_roundtrips_native_samples() {
    let samples = [0_u16, 1024, 2048, 4095];
    let pixels: Vec<u8> = samples.into_iter().flat_map(u16::to_le_bytes).collect();
    let codestream = encode_codestream(&pixels, 2, 2, 1, 12, true);
    let mut decoder = J2kDecoder::new(&codestream).expect("decoder");
    let mut out = [0_u8; 8];
    decoder
        .decode_into(&mut out, 2 * 2, PixelFormat::Gray16)
        .expect("decode");
    assert_eq!(out, pixels.as_slice());
}

#[test]
fn decode_components_exposes_signed_gray16_public_samples() {
    let samples = [-300_i16, -1, 0, 300];
    let pixels = samples
        .iter()
        .flat_map(|sample| sample.to_le_bytes())
        .collect::<Vec<_>>();
    let codestream = encode_signed_codestream(&pixels, 2, 2, 16);

    let support = J2kDecoder::inspect_support(&codestream).expect("inspect support");
    assert!(support.has_signed_components());
    let mut decoder = J2kDecoder::new(&codestream).expect("decoder");
    let components = decoder.decode_components().expect("decode components");

    assert_eq!(components.dimensions(), (2, 2));
    assert_eq!(components.planes().len(), 1);
    assert!(components.planes()[0].signed());
    assert_eq!(components.planes()[0].bit_depth(), 16);
    assert_eq!(components.planes()[0].dimensions(), (2, 2));
    let decoded_samples = components.planes()[0]
        .samples()
        .iter()
        .map(|sample| sample.round() as i16)
        .collect::<Vec<_>>();
    assert_eq!(decoded_samples, samples);
}

#[test]
fn decode_region_components_exposes_plane_dimensions() {
    let pixels = [1_u8, 2, 3, 4, 5, 6, 7, 8, 9];
    let codestream = encode_codestream(&pixels, 3, 3, 1, 8, true);
    let mut decoder = J2kDecoder::new(&codestream).expect("decoder");

    let components = decoder
        .decode_region_components(Rect {
            x: 1,
            y: 1,
            w: 2,
            h: 1,
        })
        .expect("decode component region");

    assert_eq!(components.dimensions(), (2, 1));
    assert_eq!(components.planes().len(), 1);
    assert_eq!(components.planes()[0].dimensions(), (2, 1));
    assert_eq!(components.planes()[0].samples(), &[5.0, 6.0]);
}

#[test]
fn decode_native_components_exposes_mixed_public_planes() {
    let pixels = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];
    let mut codestream = encode_codestream(&pixels, 2, 2, 3, 8, true);
    rewrite_component_descriptor(&mut codestream, 1, 11);
    rewrite_component_descriptor(&mut codestream, 2, 0x87);
    let mut decoder = J2kDecoder::new(&codestream).expect("decoder");

    let components = decoder
        .decode_native_components()
        .expect("native component decode");

    assert_eq!(components.dimensions(), (2, 2));
    assert_eq!(components.planes().len(), 3);
    assert_eq!(components.planes()[0].bit_depth(), 8);
    assert_eq!(components.planes()[0].bytes_per_sample(), 1);
    assert_eq!(components.planes()[0].data().len(), 4);
    assert_eq!(components.planes()[1].bit_depth(), 12);
    assert_eq!(components.planes()[1].bytes_per_sample(), 2);
    assert_eq!(components.planes()[1].data().len(), 8);
    assert_eq!(components.planes()[2].bit_depth(), 8);
    assert!(components.planes()[2].signed());
    assert_eq!(components.planes()[2].data().len(), 4);

    let region = decoder
        .decode_native_region_components(Rect {
            x: 1,
            y: 0,
            w: 1,
            h: 2,
        })
        .expect("native component region decode");
    assert_eq!(region.dimensions(), (1, 2));
    assert!(region
        .planes()
        .iter()
        .all(|plane| plane.dimensions() == (1, 2)));
    assert_eq!(region.planes()[1].data().len(), 4);
}

#[test]
fn decode_native_components_exposes_high_bit_public_plane() {
    let samples = [0_u32, 1, (1_u32 << 28) + 7, (1_u32 << 29) - 1];
    let pixels = samples
        .iter()
        .flat_map(|sample| unsigned_29_bytes(*sample))
        .collect::<Vec<_>>();
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        use_mct: false,
        ..EncodeOptions::default()
    };
    let codestream = encode(&pixels, 2, 2, 1, 29, false, &options).expect("encode gray29");
    let mut decoder = J2kDecoder::new(&codestream).expect("decoder");

    let components = decoder
        .decode_native_components()
        .expect("native high-bit component decode");

    assert_eq!(components.dimensions(), (2, 2));
    assert_eq!(components.planes().len(), 1);
    assert_eq!(components.planes()[0].bit_depth(), 29);
    assert_eq!(components.planes()[0].bytes_per_sample(), 4);
    assert!(!components.planes()[0].signed());
    assert_eq!(components.planes()[0].data(), pixels);
}

#[test]
fn decode_native_region_components_exposes_high_bit_public_plane() {
    let samples = [
        0_u32,
        1,
        2,
        3,
        (1_u32 << 28) + 4,
        (1_u32 << 28) + 5,
        (1_u32 << 29) - 3,
        (1_u32 << 29) - 2,
        (1_u32 << 29) - 1,
    ];
    let pixels = samples
        .iter()
        .flat_map(|sample| unsigned_29_bytes(*sample))
        .collect::<Vec<_>>();
    let expected = [samples[4], samples[5]]
        .iter()
        .flat_map(|sample| unsigned_29_bytes(*sample))
        .collect::<Vec<_>>();
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 2,
        use_mct: false,
        ..EncodeOptions::default()
    };
    let codestream = encode(&pixels, 3, 3, 1, 29, false, &options).expect("encode gray29");
    let mut decoder = J2kDecoder::new(&codestream).expect("decoder");

    let components = decoder
        .decode_native_region_components(Rect {
            x: 1,
            y: 1,
            w: 2,
            h: 1,
        })
        .expect("native high-bit component region decode");

    assert_eq!(components.dimensions(), (2, 1));
    assert_eq!(components.planes().len(), 1);
    assert_eq!(components.planes()[0].dimensions(), (2, 1));
    assert_eq!(components.planes()[0].bit_depth(), 29);
    assert_eq!(components.planes()[0].bytes_per_sample(), 4);
    assert_eq!(components.planes()[0].data(), expected);
}

#[test]
fn decode_native_region_components_covers_sampled_high_bit_public_planes() {
    let pack_u29 = |values: &[u32]| {
        values
            .iter()
            .flat_map(|sample| unsigned_29_bytes(*sample))
            .collect::<Vec<_>>()
    };
    let full = (0_u32..25).map(|idx| 1_000 + idx).collect::<Vec<_>>();
    let quarter_grid_values = (0_u32..9)
        .map(|idx| (1_u32 << 28) + 100 + idx)
        .collect::<Vec<_>>();
    let offset_grid_values = (0_u32..9)
        .map(|idx| (1_u32 << 28) + 1_000 + idx)
        .collect::<Vec<_>>();
    let full_bytes = pack_u29(&full);
    let quarter_plane_bytes = pack_u29(&quarter_grid_values);
    let offset_plane_bytes = pack_u29(&offset_grid_values);
    let planes = [
        J2kLosslessComponentPlane {
            data: &full_bytes,
            x_rsiz: 1,
            y_rsiz: 1,
        },
        J2kLosslessComponentPlane {
            data: &quarter_plane_bytes,
            x_rsiz: 2,
            y_rsiz: 2,
        },
        J2kLosslessComponentPlane {
            data: &offset_plane_bytes,
            x_rsiz: 2,
            y_rsiz: 2,
        },
    ];
    let samples = J2kLosslessComponentSamples::new(&planes, 5, 5, 29, false)
        .expect("sampled high-bit component samples");
    let encoded = encode_j2k_lossless_components(
        samples,
        &J2kLosslessEncodeOptions::default()
            .with_backend(EncodeBackendPreference::CpuOnly)
            .with_block_coding_mode(J2kBlockCodingMode::Classic)
            .with_reversible_transform(ReversibleTransform::None53)
            .with_max_decomposition_levels(Some(1)),
    )
    .expect("sampled high-bit encode");
    let mut decoder = J2kDecoder::new(&encoded.codestream).expect("decoder");

    let components = decoder
        .decode_native_region_components(Rect {
            x: 1,
            y: 1,
            w: 3,
            h: 3,
        })
        .expect("sampled high-bit native component region decode");

    assert_eq!(components.dimensions(), (3, 3));
    assert_eq!(components.planes().len(), 3);
    assert_eq!(components.planes()[0].dimensions(), (3, 3));
    assert_eq!(components.planes()[0].sampling(), (1, 1));
    assert_eq!(
        components.planes()[0].data(),
        pack_u29(&[
            full[6], full[7], full[8], full[11], full[12], full[13], full[16], full[17], full[18]
        ])
    );
    assert_eq!(components.planes()[1].dimensions(), (3, 3));
    assert_eq!(components.planes()[1].sampling(), (2, 2));
    assert_eq!(
        components.planes()[1].data(),
        pack_u29(&[
            quarter_grid_values[0],
            quarter_grid_values[1],
            quarter_grid_values[1],
            quarter_grid_values[3],
            quarter_grid_values[4],
            quarter_grid_values[4],
            quarter_grid_values[3],
            quarter_grid_values[4],
            quarter_grid_values[4],
        ])
    );
    assert_eq!(components.planes()[2].dimensions(), (3, 3));
    assert_eq!(components.planes()[2].sampling(), (2, 2));
    assert_eq!(
        components.planes()[2].data(),
        pack_u29(&[
            offset_grid_values[0],
            offset_grid_values[1],
            offset_grid_values[1],
            offset_grid_values[3],
            offset_grid_values[4],
            offset_grid_values[4],
            offset_grid_values[3],
            offset_grid_values[4],
            offset_grid_values[4],
        ])
    );
}

#[test]
fn decode_components_exposes_public_sampling_metadata() {
    let image = PrecomputedHtj2k53Image {
        width: 16,
        height: 16,
        bit_depth: 8,
        signed: false,
        components: vec![
            PrecomputedHtj2k53Component {
                x_rsiz: 1,
                y_rsiz: 1,
                dwt: zero_dwt53(16, 16),
            },
            PrecomputedHtj2k53Component {
                x_rsiz: 2,
                y_rsiz: 2,
                dwt: zero_dwt53(8, 8),
            },
            PrecomputedHtj2k53Component {
                x_rsiz: 2,
                y_rsiz: 2,
                dwt: zero_dwt53(8, 8),
            },
        ],
    };
    let options = EncodeOptions {
        num_decomposition_levels: 1,
        reversible: true,
        use_ht_block_coding: true,
        use_mct: false,
        validate_high_throughput_codestream: false,
        ..EncodeOptions::default()
    };
    let codestream =
        encode_precomputed_htj2k_53(&image, &options).expect("encode subsampled HTJ2K");
    let mut decoder = J2kDecoder::new(&codestream).expect("decoder");

    let components = decoder.decode_components().expect("decode components");
    let sampling = components
        .planes()
        .iter()
        .map(J2kComponentPlane::sampling)
        .collect::<Vec<_>>();
    let dimensions = components
        .planes()
        .iter()
        .map(J2kComponentPlane::dimensions)
        .collect::<Vec<_>>();

    assert_eq!(sampling, [(1, 1), (2, 2), (2, 2)]);
    assert_eq!(dimensions, [(16, 16), (16, 16), (16, 16)]);
}

#[test]
fn public_decode_matches_openhtj2k_refinement_fixtures() {
    for (name, codestream, expected) in OPENHT_REFINEMENT_FIXTURES {
        let codestream = codestream();
        let expected = expected();
        let mut decoder = J2kDecoder::new(codestream).unwrap_or_else(|err| {
            panic!("{name}: public decoder did not accept OpenHTJ2K fixture: {err}")
        });
        let info = decoder.info();
        let width = usize::try_from(info.dimensions.0).expect("width");
        let height = usize::try_from(info.dimensions.1).expect("height");
        let mut output = vec![0_u8; width * height];

        decoder
            .decode_into(&mut output, width, PixelFormat::Gray8)
            .unwrap_or_else(|err| panic!("{name}: public decode failed: {err}"));

        assert_eq!(output.as_slice(), expected, "{name}: decoded pixels");
    }
}

#[test]
fn decode_gray16_widens_8bit_samples_to_full_u16_range() {
    let pixels = [0_u8, 64, 128, 255];
    let codestream = encode_codestream(&pixels, 2, 2, 1, 8, true);
    let mut decoder = J2kDecoder::new(&codestream).expect("decoder");
    let mut out = [0_u8; 8];
    decoder
        .decode_into(&mut out, 2 * 2, PixelFormat::Gray16)
        .expect("decode");
    let expected: Vec<u8> = [0_u16, 16448, 32896, 65535]
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect();
    assert_eq!(out, expected.as_slice());
}

#[test]
fn decode_rgb16_roundtrips_native_samples() {
    let samples = [0_u16, 1, 2, 1024, 2048, 3072];
    let pixels: Vec<u8> = samples.into_iter().flat_map(u16::to_le_bytes).collect();
    let codestream = encode_codestream(&pixels, 2, 1, 3, 12, true);
    let mut decoder = J2kDecoder::new(&codestream).expect("decoder");
    let mut out = [0_u8; 12];
    decoder
        .decode_into(&mut out, 2 * 3 * 2, PixelFormat::Rgb16)
        .expect("decode");
    assert_eq!(out, pixels.as_slice());
}

#[test]
fn decode_rgba16_fills_opaque_alpha_for_rgb_source() {
    let samples = [0_u16, 1, 2, 1024, 2048, 3072];
    let pixels: Vec<u8> = samples.into_iter().flat_map(u16::to_le_bytes).collect();
    let codestream = encode_codestream(&pixels, 2, 1, 3, 12, true);
    let mut decoder = J2kDecoder::new(&codestream).expect("decoder");
    let mut out = [0_u8; 16];
    decoder
        .decode_into(&mut out, 2 * 4 * 2, PixelFormat::Rgba16)
        .expect("decode");
    let expected: Vec<u8> = [0_u16, 1, 2, 4095, 1024, 2048, 3072, 4095]
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect();
    assert_eq!(out, expected.as_slice());
}

#[test]
fn decode_rgba16_preserves_jp2_alpha_channel() {
    let samples = [0_u16, 1, 2, 3, 1024, 2048, 3072, 4095];
    let pixels: Vec<u8> = samples.into_iter().flat_map(u16::to_le_bytes).collect();
    let codestream = encode_codestream(&pixels, 2, 1, 4, 12, true);
    let jp2 = wrap_jp2_rgba_codestream(&codestream, 2, 1, 12);
    let mut decoder = J2kDecoder::new(&jp2).expect("decoder");
    let mut out = [0_u8; 16];
    decoder
        .decode_into(&mut out, 2 * 4 * 2, PixelFormat::Rgba16)
        .expect("decode");
    assert_eq!(out, pixels.as_slice());
}

#[test]
fn decode_rgba16_roi_scaled_and_region_scaled_preserve_alpha() {
    let samples: Vec<u16> = (0..4 * 4 * 4).map(|sample| sample * 3).collect();
    let pixels: Vec<u8> = samples.into_iter().flat_map(u16::to_le_bytes).collect();
    let codestream = encode_codestream(&pixels, 4, 4, 4, 12, true);
    let jp2 = wrap_jp2_rgba_codestream(&codestream, 4, 4, 12);
    let fmt = PixelFormat::Rgba16;
    let bytes_per_pixel = fmt.bytes_per_pixel();

    let mut full_decoder = J2kDecoder::new(&jp2).expect("full decoder");
    let full_stride = 4 * bytes_per_pixel;
    let mut full = vec![0_u8; full_stride * 4];
    full_decoder
        .decode_into(&mut full, full_stride, fmt)
        .expect("full decode");

    let roi = Rect {
        x: 1,
        y: 1,
        w: 2,
        h: 2,
    };
    let mut region_decoder = J2kDecoder::new(&jp2).expect("region decoder");
    let region_stride = roi.w as usize * bytes_per_pixel;
    let mut region = vec![0_u8; region_stride * roi.h as usize];
    let region_outcome = region_decoder
        .decode_region_into(
            &mut j2k::J2kScratchPool::new(),
            &mut region,
            region_stride,
            fmt,
            roi,
        )
        .expect("region decode");
    assert_eq!(region_outcome.decoded, roi);
    assert_eq!(region, crop_bytes(&full, 4, bytes_per_pixel, roi));

    let scale = Downscale::Half;
    let scaled_dims = (2, 2);
    let scaled_stride = scaled_dims.0 as usize * bytes_per_pixel;
    let mut scaled_decoder = J2kDecoder::new(&jp2).expect("scaled decoder");
    let mut scaled = vec![0_u8; scaled_stride * scaled_dims.1 as usize];
    let scaled_outcome = scaled_decoder
        .decode_scaled_into(
            &mut j2k::J2kScratchPool::new(),
            &mut scaled,
            scaled_stride,
            fmt,
            scale,
        )
        .expect("scaled decode");
    assert_eq!(scaled_outcome.decoded, Rect::full(scaled_dims));

    let scaled_roi = roi.scaled_covering(scale);
    let mut region_scaled_decoder = J2kDecoder::new(&jp2).expect("region scaled decoder");
    let region_scaled_stride = scaled_roi.w as usize * bytes_per_pixel;
    let mut region_scaled = vec![0_u8; region_scaled_stride * scaled_roi.h as usize];
    let region_scaled_outcome = region_scaled_decoder
        .decode_region_scaled_into(
            &mut j2k::J2kScratchPool::new(),
            &mut region_scaled,
            region_scaled_stride,
            fmt,
            roi,
            scale,
        )
        .expect("region scaled decode");
    assert_eq!(region_scaled_outcome.decoded, scaled_roi);
    assert_eq!(
        region_scaled,
        crop_bytes(&scaled, scaled_dims.0 as usize, bytes_per_pixel, scaled_roi)
    );
}

#[test]
fn decode_rejects_small_output_buffer() {
    let pixels = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];
    let codestream = encode_codestream(&pixels, 2, 2, 3, 8, true);
    let mut decoder = J2kDecoder::new(&codestream).expect("decoder");
    let mut out = [0_u8; 11];
    let err = decoder
        .decode_into(&mut out, 6, PixelFormat::Rgb8)
        .unwrap_err();
    assert!(matches!(
        err,
        J2kError::Buffer(BufferError::OutputTooSmall { .. })
    ));
}

#[test]
fn decode_rejects_too_small_stride() {
    let pixels = [10, 20, 30, 40, 50, 60];
    let codestream = encode_codestream(&pixels, 2, 1, 3, 8, true);
    let mut decoder = J2kDecoder::new(&codestream).expect("decoder");
    let mut out = [0_u8; 6];
    let err = decoder
        .decode_into(&mut out, 5, PixelFormat::Rgb8)
        .unwrap_err();
    assert!(matches!(
        err,
        J2kError::Buffer(BufferError::StrideTooSmall { .. })
    ));
}

#[test]
fn decode_scaled_into_matches_backend_target_resolution_decode() {
    let pixels: Vec<u8> = (0_u8..48).collect();
    let codestream = encode_codestream(&pixels, 4, 4, 3, 8, true);
    let expected = backend_decode_u8_scaled(&codestream, (2, 2));
    let mut decoder = J2kDecoder::new(&codestream).expect("decoder");
    let mut pool = j2k::J2kScratchPool::new();
    let mut out = [0_u8; 12];
    let outcome = decoder
        .decode_scaled_into(
            &mut pool,
            &mut out,
            2 * 3,
            PixelFormat::Rgb8,
            Downscale::Half,
        )
        .expect("scaled decode");
    assert_eq!(outcome.decoded, Rect::full((2, 2)));
    assert_eq!(out, expected.as_slice());
}

#[test]
fn reused_decoder_scaled_decodes_match_fresh_decodes_across_scales() {
    let pixels: Vec<u8> = (0..16 * 16 * 3)
        .map(|index| ((index * 13 + 17) & 0xFF) as u8)
        .collect();
    let codestream = encode_codestream_with_levels(&pixels, 16, 16, 3, 8, true, 2);
    let mut reused = J2kDecoder::new(&codestream).expect("reused decoder");
    let mut reused_pool = j2k::J2kScratchPool::new();

    for scale in [Downscale::Half, Downscale::Quarter, Downscale::Half] {
        let dims = (
            16_u32.div_ceil(scale.denominator()),
            16_u32.div_ceil(scale.denominator()),
        );
        let stride = dims.0 as usize * PixelFormat::Rgb8.bytes_per_pixel();
        let mut expected = vec![0_u8; stride * dims.1 as usize];
        let mut fresh = J2kDecoder::new(&codestream).expect("fresh decoder");
        fresh
            .decode_scaled_into(
                &mut j2k::J2kScratchPool::new(),
                &mut expected,
                stride,
                PixelFormat::Rgb8,
                scale,
            )
            .expect("fresh scaled decode");

        let mut actual = vec![0_u8; stride * dims.1 as usize];
        let outcome = reused
            .decode_scaled_into(
                &mut reused_pool,
                &mut actual,
                stride,
                PixelFormat::Rgb8,
                scale,
            )
            .expect("reused scaled decode");

        assert_eq!(outcome.decoded, Rect::full(dims));
        assert_eq!(actual, expected, "scale {scale:?}");
    }
}

#[test]
fn reused_decoder_region_scaled_decodes_match_fresh_decodes_across_scales() {
    let pixels: Vec<u8> = (0..16 * 16 * 3)
        .map(|index| ((index * 11 + 29) & 0xFF) as u8)
        .collect();
    let codestream = encode_codestream_with_levels(&pixels, 16, 16, 3, 8, true, 2);
    let roi = Rect {
        x: 3,
        y: 2,
        w: 9,
        h: 10,
    };
    let mut reused = J2kDecoder::new(&codestream).expect("reused decoder");
    let mut reused_pool = j2k::J2kScratchPool::new();

    for scale in [Downscale::Quarter, Downscale::Half, Downscale::Quarter] {
        let scaled_roi = roi.scaled_covering(scale);
        let stride = scaled_roi.w as usize * PixelFormat::Rgb8.bytes_per_pixel();
        let mut expected = vec![0_u8; stride * scaled_roi.h as usize];
        let mut fresh = J2kDecoder::new(&codestream).expect("fresh decoder");
        fresh
            .decode_region_scaled_into(
                &mut j2k::J2kScratchPool::new(),
                &mut expected,
                stride,
                PixelFormat::Rgb8,
                roi,
                scale,
            )
            .expect("fresh region scaled decode");

        let mut actual = vec![0_u8; stride * scaled_roi.h as usize];
        let outcome = reused
            .decode_region_scaled_into(
                &mut reused_pool,
                &mut actual,
                stride,
                PixelFormat::Rgb8,
                roi,
                scale,
            )
            .expect("reused region scaled decode");

        assert_eq!(outcome.decoded, scaled_roi);
        assert_eq!(actual, expected, "scale {scale:?}");
    }
}

#[test]
fn decode_region_into_matches_cropping_full_decode() {
    let pixels = [0_u8, 1, 2, 3, 4, 5, 6, 7, 8];
    let codestream = encode_codestream(&pixels, 3, 3, 1, 8, true);
    let mut decoder = J2kDecoder::new(&codestream).expect("decoder");
    let mut full = [0_u8; 9];
    decoder
        .decode_into(&mut full, 3, PixelFormat::Gray8)
        .expect("full decode");

    let roi = Rect {
        x: 1,
        y: 1,
        w: 2,
        h: 2,
    };
    let expected = crop_u8(&full, 3, 1, roi);
    let mut pool = j2k::J2kScratchPool::new();
    let mut out = [0_u8; 4];
    let outcome = decoder
        .decode_region_into(&mut pool, &mut out, 2, PixelFormat::Gray8, roi)
        .expect("region decode");
    assert_eq!(outcome.decoded, roi);
    assert_eq!(out, expected.as_slice());
}

#[test]
fn decode_region_scaled_into_matches_cropping_scaled_decode_for_supported_formats() {
    let roi = Rect {
        x: 1,
        y: 0,
        w: 2,
        h: 3,
    };
    let scale = Downscale::Half;
    let scaled_roi = roi.scaled_covering(scale);

    let rgb8_pixels: Vec<u8> = (0_u8..48).collect();
    let rgb8_codestream = encode_codestream(&rgb8_pixels, 4, 4, 3, 8, true);
    for fmt in [PixelFormat::Rgb8, PixelFormat::Rgba8] {
        let mut scaled_decoder = J2kDecoder::new(&rgb8_codestream).expect("scaled decoder");
        let scaled_stride = 2 * fmt.bytes_per_pixel();
        let mut scaled = vec![0_u8; scaled_stride * 2];
        scaled_decoder
            .decode_scaled_into(
                &mut j2k::J2kScratchPool::new(),
                &mut scaled,
                scaled_stride,
                fmt,
                scale,
            )
            .expect("scaled decode");
        let expected = crop_bytes(&scaled, 2, fmt.bytes_per_pixel(), scaled_roi);

        let mut decoder = J2kDecoder::new(&rgb8_codestream).expect("decoder");
        let stride = scaled_roi.w as usize * fmt.bytes_per_pixel();
        let mut out = vec![0_u8; stride * scaled_roi.h as usize];
        let outcome = decoder
            .decode_region_scaled_into(
                &mut j2k::J2kScratchPool::new(),
                &mut out,
                stride,
                fmt,
                roi,
                scale,
            )
            .expect("region scaled decode");
        assert_eq!(outcome.decoded, scaled_roi);
        assert_eq!(out, expected, "format {fmt:?}");
    }

    let gray8_pixels: Vec<u8> = (0_u8..16).collect();
    let gray8_codestream = encode_codestream(&gray8_pixels, 4, 4, 1, 8, true);
    let mut gray8_scaled_decoder =
        J2kDecoder::new(&gray8_codestream).expect("gray8 scaled decoder");
    let mut gray8_scaled = vec![0_u8; 2 * 2];
    gray8_scaled_decoder
        .decode_scaled_into(
            &mut j2k::J2kScratchPool::new(),
            &mut gray8_scaled,
            2,
            PixelFormat::Gray8,
            scale,
        )
        .expect("gray8 scaled decode");
    let expected_gray8 = crop_bytes(
        &gray8_scaled,
        2,
        PixelFormat::Gray8.bytes_per_pixel(),
        scaled_roi,
    );
    let mut gray8_decoder = J2kDecoder::new(&gray8_codestream).expect("gray8 decoder");
    let mut gray8_out = vec![0_u8; scaled_roi.w as usize * scaled_roi.h as usize];
    let gray8_stride = scaled_roi.w as usize * PixelFormat::Gray8.bytes_per_pixel();
    let outcome = gray8_decoder
        .decode_region_scaled_into(
            &mut j2k::J2kScratchPool::new(),
            &mut gray8_out,
            gray8_stride,
            PixelFormat::Gray8,
            roi,
            scale,
        )
        .expect("gray8 region scaled decode");
    assert_eq!(outcome.decoded, scaled_roi);
    assert_eq!(gray8_out, expected_gray8);

    let gray16_samples = [
        0_u16, 64, 128, 192, 256, 512, 768, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 3072, 4095,
    ];
    let gray16_pixels: Vec<u8> = gray16_samples
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect();
    let gray16_codestream = encode_codestream(&gray16_pixels, 4, 4, 1, 12, true);
    let mut gray16_scaled_decoder =
        J2kDecoder::new(&gray16_codestream).expect("gray16 scaled decoder");
    let mut gray16_scaled = vec![0_u8; 2 * 2 * 2];
    gray16_scaled_decoder
        .decode_scaled_into(
            &mut j2k::J2kScratchPool::new(),
            &mut gray16_scaled,
            2 * 2,
            PixelFormat::Gray16,
            scale,
        )
        .expect("gray16 scaled decode");
    let expected_gray16 = crop_bytes(
        &gray16_scaled,
        2,
        PixelFormat::Gray16.bytes_per_pixel(),
        scaled_roi,
    );
    let mut gray16_decoder = J2kDecoder::new(&gray16_codestream).expect("gray16 decoder");
    let mut gray16_out = vec![0_u8; scaled_roi.w as usize * scaled_roi.h as usize * 2];
    let gray16_stride = scaled_roi.w as usize * PixelFormat::Gray16.bytes_per_pixel();
    let outcome = gray16_decoder
        .decode_region_scaled_into(
            &mut j2k::J2kScratchPool::new(),
            &mut gray16_out,
            gray16_stride,
            PixelFormat::Gray16,
            roi,
            scale,
        )
        .expect("gray16 region scaled decode");
    assert_eq!(outcome.decoded, scaled_roi);
    assert_eq!(gray16_out, expected_gray16);

    let rgb16_samples = [
        0_u16, 1, 2, 64, 65, 66, 128, 129, 130, 192, 193, 194, 256, 257, 258, 512, 513, 514, 768,
        769, 770, 1024, 1025, 1026, 1280, 1281, 1282, 1536, 1537, 1538, 1792, 1793, 1794, 2048,
        2049, 2050, 2304, 2305, 2306, 2560, 2561, 2562, 3072, 3073, 3074, 4093, 4094, 4095,
    ];
    let rgb16_pixels: Vec<u8> = rgb16_samples
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect();
    let rgb16_codestream = encode_codestream(&rgb16_pixels, 4, 4, 3, 12, true);
    let mut rgb16_scaled_decoder =
        J2kDecoder::new(&rgb16_codestream).expect("rgb16 scaled decoder");
    let mut rgb16_scaled = vec![0_u8; 2 * 2 * 3 * 2];
    rgb16_scaled_decoder
        .decode_scaled_into(
            &mut j2k::J2kScratchPool::new(),
            &mut rgb16_scaled,
            2 * 3 * 2,
            PixelFormat::Rgb16,
            scale,
        )
        .expect("rgb16 scaled decode");
    let expected_rgb16 = crop_bytes(
        &rgb16_scaled,
        2,
        PixelFormat::Rgb16.bytes_per_pixel(),
        scaled_roi,
    );
    let mut rgb16_decoder = J2kDecoder::new(&rgb16_codestream).expect("rgb16 decoder");
    let mut rgb16_out = vec![0_u8; scaled_roi.w as usize * scaled_roi.h as usize * 3 * 2];
    let rgb16_stride = scaled_roi.w as usize * PixelFormat::Rgb16.bytes_per_pixel();
    let outcome = rgb16_decoder
        .decode_region_scaled_into(
            &mut j2k::J2kScratchPool::new(),
            &mut rgb16_out,
            rgb16_stride,
            PixelFormat::Rgb16,
            roi,
            scale,
        )
        .expect("rgb16 region scaled decode");
    assert_eq!(outcome.decoded, scaled_roi);
    assert_eq!(rgb16_out, expected_rgb16);
}

#[test]
fn decode_region_scaled_htj2k_gray8_matches_cropping_scaled_decode() {
    let pixels: Vec<u8> = (0_u8..16).collect();
    let codestream = encode_ht_codestream(&pixels, 4, 4, 1, 8);
    let roi = Rect {
        x: 1,
        y: 0,
        w: 2,
        h: 3,
    };
    let scale = Downscale::Half;
    let scaled_roi = roi.scaled_covering(scale);

    let mut scaled_decoder = J2kDecoder::new(&codestream).expect("scaled decoder");
    let mut scaled = vec![0_u8; 2 * 2];
    scaled_decoder
        .decode_scaled_into(
            &mut j2k::J2kScratchPool::new(),
            &mut scaled,
            2,
            PixelFormat::Gray8,
            scale,
        )
        .expect("scaled decode");
    let expected = crop_bytes(&scaled, 2, PixelFormat::Gray8.bytes_per_pixel(), scaled_roi);

    let mut decoder = J2kDecoder::new(&codestream).expect("decoder");
    let stride = scaled_roi.w as usize * PixelFormat::Gray8.bytes_per_pixel();
    let mut out = vec![0_u8; stride * scaled_roi.h as usize];
    let outcome = decoder
        .decode_region_scaled_into(
            &mut j2k::J2kScratchPool::new(),
            &mut out,
            stride,
            PixelFormat::Gray8,
            roi,
            scale,
        )
        .expect("region scaled decode");

    assert_eq!(outcome.decoded, scaled_roi);
    assert_eq!(out, expected);
}

#[test]
fn decode_region_scaled_none_matches_region_decode() {
    let pixels = [0_u8, 1, 2, 3, 4, 5, 6, 7, 8];
    let codestream = encode_codestream(&pixels, 3, 3, 1, 8, true);
    let roi = Rect {
        x: 1,
        y: 1,
        w: 2,
        h: 2,
    };

    let mut expected_decoder = J2kDecoder::new(&codestream).expect("expected decoder");
    let mut expected = [0_u8; 4];
    expected_decoder
        .decode_region_into(
            &mut j2k::J2kScratchPool::new(),
            &mut expected,
            2,
            PixelFormat::Gray8,
            roi,
        )
        .expect("region decode");

    let mut decoder = J2kDecoder::new(&codestream).expect("decoder");
    let mut out = [0_u8; 4];
    let outcome = decoder
        .decode_region_scaled_into(
            &mut j2k::J2kScratchPool::new(),
            &mut out,
            2,
            PixelFormat::Gray8,
            roi,
            Downscale::None,
        )
        .expect("region scaled none decode");

    assert_eq!(outcome.decoded, roi);
    assert_eq!(out, expected);
}

#[test]
fn native_backend_region_decode_matches_cropping_full_decode() {
    let pixels = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];
    let codestream = encode_codestream(&pixels, 2, 2, 3, 8, true);
    let roi = Rect {
        x: 1,
        y: 0,
        w: 1,
        h: 2,
    };
    let expected = crop_u8(&backend_decode_u8(&codestream), 2, 3, roi);
    let actual = backend_decode_u8_region(&codestream, roi);
    assert_eq!(actual, expected);
}

#[test]
fn decode_rows_u8_matches_full_rgb8_decode() {
    let pixels = [10_u8, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];
    let codestream = encode_codestream(&pixels, 2, 2, 3, 8, true);
    let mut decoder = J2kDecoder::new(&codestream).expect("decoder");
    let mut full = [0_u8; 12];
    decoder
        .decode_into(&mut full, 2 * 3, PixelFormat::Rgb8)
        .expect("full decode");

    let mut sink = CollectRowsU8::default();
    <J2kDecoder<'_> as ImageDecodeRows<'_, u8>>::decode_rows(&mut decoder, &mut sink)
        .expect("row decode");
    assert_eq!(sink.rows, full);
}

#[test]
fn decode_rows_u16_matches_full_gray16_decode() {
    let samples = [0_u16, 1024, 2048, 4095];
    let pixels: Vec<u8> = samples.into_iter().flat_map(u16::to_le_bytes).collect();
    let codestream = encode_codestream(&pixels, 2, 2, 1, 12, true);
    let mut decoder = J2kDecoder::new(&codestream).expect("decoder");
    let mut full = [0_u8; 8];
    decoder
        .decode_into(&mut full, 2 * 2, PixelFormat::Gray16)
        .expect("full decode");

    let mut sink = CollectRowsU16::default();
    <J2kDecoder<'_> as ImageDecodeRows<'_, u16>>::decode_rows(&mut decoder, &mut sink)
        .expect("row decode");
    let collected: Vec<u8> = sink.rows.into_iter().flat_map(u16::to_le_bytes).collect();
    assert_eq!(collected, full);
}

#[test]
fn decode_rows_u8_bounded_matches_full_rgba8_decode_one_row_at_a_time() {
    let pixels: Vec<u8> = (0_u8..32).collect();
    let codestream = encode_codestream(&pixels, 4, 2, 4, 8, true);
    let jp2 = wrap_jp2_rgba_codestream(&codestream, 4, 2, 8);
    let mut decoder = J2kDecoder::new(&jp2).expect("decoder");
    let mut full = [0_u8; 32];
    decoder
        .decode_into(&mut full, 4 * 4, PixelFormat::Rgba8)
        .expect("full decode");

    let mut sink = CollectRowsU8::default();
    decoder
        .decode_rows_u8_bounded(&mut sink, J2kRowDecodeOptions::new(1))
        .expect("bounded row decode");
    assert_eq!(sink.rows, full);
}

#[test]
fn decode_rows_u16_bounded_matches_full_rgba16_decode_one_row_at_a_time() {
    let samples: Vec<u16> = (0..4 * 2 * 4).map(|sample| sample * 11).collect();
    let pixels: Vec<u8> = samples.into_iter().flat_map(u16::to_le_bytes).collect();
    let codestream = encode_codestream(&pixels, 4, 2, 4, 12, true);
    let jp2 = wrap_jp2_rgba_codestream(&codestream, 4, 2, 12);
    let mut decoder = J2kDecoder::new(&jp2).expect("decoder");
    let mut full = vec![0_u8; 4 * 2 * 4 * 2];
    decoder
        .decode_into(&mut full, 4 * 4 * 2, PixelFormat::Rgba16)
        .expect("full decode");

    let mut sink = CollectRowsU16::default();
    decoder
        .decode_rows_u16_bounded(&mut sink, J2kRowDecodeOptions::new(1))
        .expect("bounded row decode");
    let collected: Vec<u8> = sink.rows.into_iter().flat_map(u16::to_le_bytes).collect();
    assert_eq!(collected, full);
}

#[test]
fn tile_batch_decode_matches_borrowed_decoder_decode() {
    let pixels = [10_u8, 20, 30, 40, 50, 60];
    let codestream = encode_codestream(&pixels, 2, 1, 3, 8, true);
    let mut decoder = J2kDecoder::new(&codestream).expect("decoder");
    let mut expected = [0_u8; 6];
    decoder
        .decode_into(&mut expected, 2 * 3, PixelFormat::Rgb8)
        .expect("decoder decode");

    let mut ctx = DecoderContext::<J2kContext>::new();
    let mut pool = j2k::J2kScratchPool::new();
    let mut out = [0_u8; 6];
    let outcome = <J2kCodec as TileBatchDecode>::decode_tile(
        &mut ctx,
        &mut pool,
        &codestream,
        &mut out,
        2 * 3,
        PixelFormat::Rgb8,
    )
    .expect("tile decode");
    assert_eq!(outcome.decoded, Rect::full((2, 1)));
    assert_eq!(out, expected);
    assert_eq!(ctx.cache_stats(), j2k_core::CacheStats::default());
}

#[test]
fn tile_batch_region_decode_matches_decoder_region_decode() {
    let pixels = [0_u8, 1, 2, 3, 4, 5, 6, 7, 8];
    let codestream = encode_codestream(&pixels, 3, 3, 1, 8, true);
    let roi = Rect {
        x: 1,
        y: 1,
        w: 2,
        h: 2,
    };
    let mut decoder = J2kDecoder::new(&codestream).expect("decoder");
    let mut pool = j2k::J2kScratchPool::new();
    let mut expected = [0_u8; 4];
    decoder
        .decode_region_into(&mut pool, &mut expected, 2, PixelFormat::Gray8, roi)
        .expect("decoder region");

    let mut ctx = DecoderContext::<J2kContext>::new();
    let mut out = [0_u8; 4];
    <J2kCodec as TileBatchDecode>::decode_tile_region(
        &mut ctx,
        &mut pool,
        &codestream,
        &mut out,
        2,
        PixelFormat::Gray8,
        roi,
    )
    .expect("tile region");
    assert_eq!(out, expected);
}

#[test]
fn tile_batch_scaled_decode_matches_decoder_scaled_decode() {
    let pixels: Vec<u8> = (0_u8..48).collect();
    let codestream = encode_codestream(&pixels, 4, 4, 3, 8, true);
    let mut decoder = J2kDecoder::new(&codestream).expect("decoder");
    let mut pool = j2k::J2kScratchPool::new();
    let mut expected = [0_u8; 12];
    decoder
        .decode_scaled_into(
            &mut pool,
            &mut expected,
            2 * 3,
            PixelFormat::Rgb8,
            Downscale::Half,
        )
        .expect("decoder scaled");

    let mut ctx = DecoderContext::<J2kContext>::new();
    let mut out = [0_u8; 12];
    <J2kCodec as TileBatchDecode>::decode_tile_scaled(
        &mut ctx,
        &mut pool,
        &codestream,
        &mut out,
        2 * 3,
        PixelFormat::Rgb8,
        Downscale::Half,
    )
    .expect("tile scaled");
    assert_eq!(out, expected);
}

#[test]
fn tile_batch_region_scaled_decode_matches_decoder_region_scaled_decode() {
    let pixels: Vec<u8> = (0_u8..48).collect();
    let codestream = encode_codestream(&pixels, 4, 4, 3, 8, true);
    let roi = Rect {
        x: 1,
        y: 0,
        w: 2,
        h: 3,
    };
    let scale = Downscale::Half;
    let scaled_roi = roi.scaled_covering(scale);
    let stride = scaled_roi.w as usize * PixelFormat::Rgb8.bytes_per_pixel();

    let mut decoder = J2kDecoder::new(&codestream).expect("decoder");
    let mut pool = j2k::J2kScratchPool::new();
    let mut expected = vec![0_u8; stride * scaled_roi.h as usize];
    decoder
        .decode_region_scaled_into(
            &mut pool,
            &mut expected,
            stride,
            PixelFormat::Rgb8,
            roi,
            scale,
        )
        .expect("decoder region scaled");

    let mut ctx = DecoderContext::<J2kContext>::new();
    let mut out = vec![0_u8; stride * scaled_roi.h as usize];
    let outcome = <J2kCodec as TileBatchDecode>::decode_tile_region_scaled(
        &mut ctx,
        &mut pool,
        &codestream,
        &mut out,
        stride,
        PixelFormat::Rgb8,
        roi,
        scale,
    )
    .expect("tile region scaled");
    assert_eq!(outcome.decoded, scaled_roi);
    assert_eq!(out, expected);
}

#[test]
fn decode_region_into_rejects_out_of_bounds_roi() {
    let pixels = [0_u8, 1, 2, 3];
    let codestream = encode_codestream(&pixels, 2, 2, 1, 8, true);
    let mut decoder = J2kDecoder::new(&codestream).expect("decoder");
    let mut pool = j2k::J2kScratchPool::new();
    let mut out = [0_u8; 4];
    let err = decoder
        .decode_region_into(
            &mut pool,
            &mut out,
            2,
            PixelFormat::Gray8,
            Rect {
                x: 1,
                y: 1,
                w: 2,
                h: 2,
            },
        )
        .unwrap_err();
    assert!(matches!(err, J2kError::InvalidRegion { .. }));
}

#[test]
fn decode_htj2k_gray8_roundtrips_reversible_pixels() {
    let pixels = [3_u8, 9, 27, 81];
    let codestream = encode_ht_codestream(&pixels, 2, 2, 1, 8);
    let mut decoder = J2kDecoder::new(&codestream).expect("decoder");
    let mut out = [0_u8; 4];
    decoder
        .decode_into(&mut out, 2, PixelFormat::Gray8)
        .expect("ht decode");
    assert_eq!(out, pixels);
}

#[test]
fn decode_htj2k_scaled_into_matches_native_target_resolution_decode() {
    let pixels: Vec<u8> = (0_u8..16).collect();
    let codestream = encode_ht_codestream(&pixels, 4, 4, 1, 8);
    let expected = backend_decode_u8_scaled(&codestream, (2, 2));
    let mut decoder = J2kDecoder::new(&codestream).expect("decoder");
    let mut pool = j2k::J2kScratchPool::new();
    let mut out = [0_u8; 4];
    decoder
        .decode_scaled_into(&mut pool, &mut out, 2, PixelFormat::Gray8, Downscale::Half)
        .expect("scaled decode");
    assert_eq!(out, expected.as_slice());
}

#[test]
fn decode_rows_u8_matches_full_gray8_decode_for_htj2k() {
    let pixels = [2_u8, 4, 6, 8];
    let codestream = encode_ht_codestream(&pixels, 2, 2, 1, 8);
    let mut decoder = J2kDecoder::new(&codestream).expect("decoder");
    let mut full = [0_u8; 4];
    decoder
        .decode_into(&mut full, 2, PixelFormat::Gray8)
        .expect("full decode");

    let mut sink = CollectRowsU8::default();
    <J2kDecoder<'_> as ImageDecodeRows<'_, u8>>::decode_rows(&mut decoder, &mut sink)
        .expect("row decode");
    assert_eq!(sink.rows, full);
}

#[test]
fn tile_batch_decode_matches_borrowed_decoder_for_htj2k() {
    let pixels = [7_u8, 11, 13, 17];
    let codestream = encode_ht_codestream(&pixels, 2, 2, 1, 8);
    let mut decoder = J2kDecoder::new(&codestream).expect("decoder");
    let mut expected = [0_u8; 4];
    decoder
        .decode_into(&mut expected, 2, PixelFormat::Gray8)
        .expect("decoder decode");

    let mut ctx = DecoderContext::<J2kContext>::new();
    let mut pool = j2k::J2kScratchPool::new();
    let mut out = [0_u8; 4];
    <J2kCodec as TileBatchDecode>::decode_tile(
        &mut ctx,
        &mut pool,
        &codestream,
        &mut out,
        2,
        PixelFormat::Gray8,
    )
    .expect("tile decode");
    assert_eq!(out, expected);
}

fn zero_dwt53(width: u32, height: u32) -> J2kForwardDwt53Output {
    let low_width = width.div_ceil(2);
    let low_height = height.div_ceil(2);
    let high_width = width / 2;
    let high_height = height / 2;

    J2kForwardDwt53Output {
        ll: vec![0.0; (low_width * low_height) as usize],
        ll_width: low_width,
        ll_height: low_height,
        levels: vec![J2kForwardDwt53Level {
            hl: vec![0.0; (high_width * low_height) as usize],
            lh: vec![0.0; (low_width * high_height) as usize],
            hh: vec![0.0; (high_width * high_height) as usize],
            width,
            height,
            low_width,
            low_height,
            high_width,
            high_height,
        }],
    }
}
