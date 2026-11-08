// SPDX-License-Identifier: MIT OR Apache-2.0

//! `decode_srgb8` of a Gray, RGB or RGBA image without an ICC profile takes
//! the full-image route `decode_into` takes: when the image's component
//! planes would pass the decoder's planes budget, its rows stream into the
//! returned buffer instead of being held as whole-image planes first.
//!
//! Heap peaks come from the process-wide counting allocator of
//! `j2k_alloc_probe`, so this binary holds one test and decodes serially.

use j2k::{
    wrap_j2k_codestream, CpuDecodeParallelism, J2kDecoder, J2kFileColorSpec, J2kFileWrapOptions,
    J2kSrgb8Layout, Rect,
};
use j2k_core::{Colorspace, PixelFormat};
use j2k_native::{encode, EncodeOptions};

const WIDTH: u32 = 512;
const HEIGHT: u32 = 384;
/// Small allocations no decode plans: warning vectors, boxed jobs, rounding.
const SLACK_BYTES: usize = 64 * 1024;

fn noisy_pixels(components: u32) -> Vec<u8> {
    let mut state = 0x2545_f491_u32;
    (0..WIDTH * HEIGHT * components)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state >> 24) as u8
        })
        .collect()
}

fn fixture(components: u16) -> Vec<u8> {
    let options = EncodeOptions {
        num_decomposition_levels: 4,
        reversible: true,
        use_ht_block_coding: true,
        ..EncodeOptions::default()
    };
    let pixels = noisy_pixels(u32::from(components));
    let codestream =
        encode(&pixels, WIDTH, HEIGHT, components, 8, false, &options).expect("encode fixture");
    if components == 4 {
        wrap_j2k_codestream(
            &codestream,
            J2kFileWrapOptions::jph().with_color(J2kFileColorSpec::Enumerated(Colorspace::SRgb)),
        )
        .expect("wrap RGBA JPH")
    } else {
        codestream
    }
}

fn decoder(bytes: &[u8], planes_budget: usize) -> J2kDecoder<'_> {
    let mut decoder = J2kDecoder::new(bytes).expect("decoder");
    decoder.set_cpu_decode_parallelism(CpuDecodeParallelism::Serial);
    decoder.set_full_decode_planes_budget(planes_budget);
    decoder
}

/// The result of `operation` and the most heap bytes live at once while it
/// ran, above those live before it.
fn peak_during<R>(operation: impl FnOnce() -> R) -> (R, usize) {
    let before = j2k_alloc_probe::reset_peak();
    let result = operation();
    (result, j2k_alloc_probe::peak_bytes() - before)
}

/// `decode_into` of the whole image at `planes_budget`, with its output
/// allocated before the measurement, and the peak it held.
fn decode_into_peak(bytes: &[u8], fmt: PixelFormat, planes_budget: usize) -> (Vec<u8>, usize) {
    let stride = WIDTH as usize * fmt.bytes_per_pixel();
    let mut out = vec![0_u8; stride * HEIGHT as usize];
    let mut decoder = decoder(bytes, planes_budget);
    let (outcome, peak) = peak_during(|| decoder.decode_into(&mut out, stride, fmt));
    outcome.expect("decode_into");
    (out, peak)
}

#[test]
fn srgb8_streams_rows_where_decode_into_streams_them() {
    // The RGBA image's alpha needs a channel-definition box, which keeps its
    // region decode from streaming rows: both its routes hold the planes.
    for (components, fmt, layout, streams) in [
        (1, PixelFormat::Gray8, J2kSrgb8Layout::Gray, true),
        (3, PixelFormat::Rgb8, J2kSrgb8Layout::Rgb, true),
        (4, PixelFormat::Rgba8, J2kSrgb8Layout::Rgba, false),
    ] {
        let bytes = fixture(components);
        let (planes, planes_peak) = decode_into_peak(&bytes, fmt, usize::MAX);
        let (streamed, streamed_peak) = decode_into_peak(&bytes, fmt, 0);
        assert_eq!(streamed, planes, "{fmt:?}: routes disagree");
        let output_bytes = planes.len();
        // The fixture separates the routes: the planes hold more than the
        // streamed decode plus a second copy of the output.
        assert_eq!(
            streamed_peak + output_bytes + SLACK_BYTES < planes_peak,
            streams,
            "{fmt:?}: streamed {streamed_peak}, output {output_bytes}, planes {planes_peak}"
        );

        for (budget, route_peak) in [(usize::MAX, planes_peak), (0, streamed_peak)] {
            let mut decoder = decoder(&bytes, budget);
            let (image, srgb8_peak) = peak_during(|| decoder.decode_srgb8());
            let image = image.unwrap_or_else(|error| panic!("{fmt:?} at {budget}: {error:?}"));
            assert_eq!(image.layout(), layout, "{fmt:?} at {budget}");
            assert_eq!(image.dimensions(), (WIDTH, HEIGHT), "{fmt:?} at {budget}");
            assert!(
                image.data() == planes,
                "{fmt:?} at {budget}: samples differ"
            );
            // decode_srgb8 allocates the output decode_into was handed.
            assert!(
                srgb8_peak <= route_peak + output_bytes + SLACK_BYTES,
                "{fmt:?} at {budget}: decode_srgb8 held {srgb8_peak} bytes, \
                 decode_into {route_peak} beside its {output_bytes}-byte output"
            );
        }
    }

    // The explicit ICC route also reconstructs only the region. Measuring a
    // fresh context includes native scratch as well as both conversion buffers;
    // a crop of a whole-source decode cannot meet this separation.
    let raw = fixture(3);
    let profile = moxcms::ColorProfile::new_pro_photo_rgb()
        .encode()
        .expect("ICC profile");
    let bytes = wrap_j2k_codestream(
        &raw,
        J2kFileWrapOptions::jph().with_color(J2kFileColorSpec::IccProfile(&profile)),
    )
    .expect("ICC image");
    let mut full_decoder = decoder(&bytes, usize::MAX);
    let (full, full_peak) = peak_during(|| full_decoder.decode_srgb8());
    let full = full.expect("whole-source normalized reference");
    let roi = Rect {
        x: 7,
        y: 9,
        w: 63,
        h: 47,
    };
    let mut region_decoder = decoder(&bytes, usize::MAX);
    let (region, region_peak) = peak_during(|| region_decoder.decode_region_srgb8(roi));
    let region = region.expect("bounded normalized region");
    for (y, row) in region.data().chunks_exact(roi.w as usize * 3).enumerate() {
        let start = ((roi.y as usize + y) * WIDTH as usize + roi.x as usize) * 3;
        assert_eq!(row, &full.data()[start..start + row.len()]);
    }
    assert!(
        region_peak + full.data().len() + SLACK_BYTES < full_peak,
        "small-region peak {region_peak}, whole-source peak {full_peak}"
    );
}
