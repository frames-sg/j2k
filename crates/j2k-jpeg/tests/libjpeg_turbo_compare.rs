// SPDX-License-Identifier: MIT OR Apache-2.0

#[path = "../benches/common/libjpeg_turbo.rs"]
mod libjpeg_turbo;
#[path = "../benches/common/libjpeg_turbo_extended.rs"]
mod libjpeg_turbo_extended;
#[cfg(all(has_libjpeg_turbo, has_libjpeg_turbo_v3))]
#[path = "../benches/common/libjpeg_turbo_v2.rs"]
mod libjpeg_turbo_v2;

use j2k_jpeg::{DecodeRequest, Decoder, Downscale, PixelFormat, Rect};
use j2k_test_support::{
    JPEG_BASELINE_420_16X16, JPEG_BASELINE_422_16X8, JPEG_BASELINE_422_16X8_RGB, JPEG_GRAYSCALE_8X8,
};

/// Whether libjpeg-turbo comparisons can run. They skip when it is missing,
/// unless `J2K_REQUIRE_LIBJPEG_TURBO` makes that a failure.
fn turbo_available() -> bool {
    let available = libjpeg_turbo::is_available();
    assert!(
        available || std::env::var_os("J2K_REQUIRE_LIBJPEG_TURBO").is_none(),
        "J2K_REQUIRE_LIBJPEG_TURBO is set but libjpeg-turbo is unavailable"
    );
    available
}

#[test]
fn turbojpeg_rgb_and_region_match_j2k_fixtures() {
    if !turbo_available() {
        return;
    }

    let bytes = JPEG_BASELINE_420_16X16;
    let dec = Decoder::new(bytes).expect("j2k decoder");
    let mut turbo = libjpeg_turbo::TurboJpegDecoder::new().expect("turbojpeg decoder");

    let info = turbo.inspect(bytes).expect("turbojpeg inspect");
    assert_eq!((info.width, info.height), (16, 16));
    assert_eq!(info.subsamp, 2, "fixture should report 4:2:0 sampling");

    let (rgb, _) = dec
        .decode_request(DecodeRequest::full(PixelFormat::Rgb8))
        .expect("j2k rgb");
    let turbo_rgb = turbo.decode_rgb(bytes).expect("turbojpeg rgb");
    assert_eq!(turbo_rgb, rgb);

    let prepared = turbo.prepare_rgb(bytes).expect("turbojpeg prepare RGB");
    assert_eq!((prepared.width, prepared.height), (16, 16));
    assert_eq!(prepared.subsamp, 2);
    let mut prepared_rgb = vec![0_u8; 16 * 16 * 3];
    turbo
        .decode_prepared_rgb_into(bytes, &mut prepared_rgb, 16 * 3, 16, 16)
        .expect("turbojpeg prepared RGB");
    assert_eq!(prepared_rgb, turbo_rgb);

    let gray_decoder = Decoder::new(JPEG_GRAYSCALE_8X8).expect("j2k grayscale decoder");
    let (gray, _) = gray_decoder
        .decode_request(DecodeRequest::full(PixelFormat::Gray8))
        .expect("j2k grayscale");
    let turbo_gray = turbo
        .decode_gray(JPEG_GRAYSCALE_8X8)
        .expect("turbojpeg grayscale");
    assert_eq!(turbo_gray, gray);

    let (scaled, _) = dec
        .decode_request(DecodeRequest::scaled(PixelFormat::Rgb8, Downscale::Quarter))
        .expect("j2k scaled");
    let turbo_scaled = turbo
        .decode_scaled_rgb(bytes, Downscale::Quarter)
        .expect("turbojpeg scaled");
    assert_eq!(turbo_scaled.len(), scaled.len());
    assert!(!turbo_scaled.is_empty());

    let roi = Rect {
        x: 4,
        y: 4,
        w: 8,
        h: 8,
    };
    let turbo_region_a = turbo
        .decode_region_rgb(bytes, roi)
        .expect("turbojpeg region");
    let turbo_region_b = turbo
        .decode_region_rgb(bytes, roi)
        .expect("turbojpeg region");
    assert_eq!(turbo_region_a, turbo_region_b);
    assert_eq!(turbo_region_a.len(), crop_rgb(&turbo_rgb, 16, roi).len());

    let turbo_region_scaled_a = turbo
        .decode_region_scaled_rgb(bytes, roi, Downscale::Quarter)
        .expect("turbojpeg scaled region");
    let turbo_region_scaled_b = turbo
        .decode_region_scaled_rgb(bytes, roi, Downscale::Quarter)
        .expect("turbojpeg scaled region");
    assert_eq!(turbo_region_scaled_a, turbo_region_scaled_b);
    assert_eq!(turbo_region_scaled_a.len(), 2 * 2 * 3);

    let bytes_422 = JPEG_BASELINE_422_16X8;
    let decoder_422 = Decoder::new(bytes_422).expect("j2k 4:2:2 decoder");
    let info_422 = turbo.inspect(bytes_422).expect("turbojpeg 4:2:2 inspect");
    assert_eq!((info_422.width, info_422.height), (16, 8));
    assert_eq!(info_422.subsamp, 1, "fixture should report 4:2:2 sampling");
    let (rgb_422, _) = decoder_422
        .decode_request(DecodeRequest::full(PixelFormat::Rgb8))
        .expect("j2k 4:2:2 RGB");
    let turbo_rgb_422 = turbo.decode_rgb(bytes_422).expect("turbojpeg 4:2:2 RGB");
    assert_eq!(turbo_rgb_422, JPEG_BASELINE_422_16X8_RGB);
    assert_eq!(rgb_422, turbo_rgb_422);
}

#[cfg(all(has_libjpeg_turbo, has_libjpeg_turbo_v3))]
#[test]
fn legacy_turbojpeg_abi_matches_the_v3_adapter() {
    let bytes = JPEG_BASELINE_420_16X16;
    let mut current = libjpeg_turbo::TurboJpegDecoder::new().expect("v3 turbojpeg decoder");
    let mut legacy = libjpeg_turbo_v2::TurboJpegDecoder::new().expect("legacy turbojpeg decoder");

    assert_eq!(
        legacy.decode_rgb(bytes).expect("legacy full RGB"),
        current.decode_rgb(bytes).expect("v3 full RGB")
    );
    assert_eq!(
        legacy
            .decode(JPEG_GRAYSCALE_8X8, 6, None, Downscale::None)
            .expect("legacy grayscale"),
        current
            .decode(JPEG_GRAYSCALE_8X8, 6, None, Downscale::None)
            .expect("v3 grayscale")
    );
    assert_eq!(
        legacy
            .decode(bytes, 0, None, Downscale::Quarter)
            .expect("legacy scaled RGB"),
        current
            .decode(bytes, 0, None, Downscale::Quarter)
            .expect("v3 scaled RGB")
    );

    let roi = Rect {
        x: 4,
        y: 4,
        w: 8,
        h: 8,
    };
    assert_eq!(
        legacy
            .decode(bytes, 0, Some(roi), Downscale::None)
            .expect("legacy region RGB"),
        current
            .decode(bytes, 0, Some(roi), Downscale::None)
            .expect("v3 region RGB")
    );
    assert_eq!(
        legacy
            .decode(bytes, 0, Some(roi), Downscale::Quarter)
            .expect("legacy scaled region RGB"),
        current
            .decode(bytes, 0, Some(roi), Downscale::Quarter)
            .expect("v3 scaled region RGB")
    );

    let (width, height, _) = legacy
        .prepare_full_frame(bytes)
        .expect("legacy prepare RGB");
    let pitch = width as usize * 3;
    let mut prepared = vec![0_u8; pitch * height as usize];
    legacy
        .decompress(bytes, &mut prepared, pitch, 0)
        .expect("legacy prepared RGB");
    assert_eq!(
        prepared,
        current.decode_rgb(bytes).expect("v3 prepared reference")
    );
}

fn crop_rgb(full: &[u8], width: usize, roi: Rect) -> Vec<u8> {
    let stride = width * 3;
    let mut out = vec![0u8; roi.w as usize * roi.h as usize * 3];
    for row in 0..roi.h as usize {
        let src_start = (roi.y as usize + row) * stride + roi.x as usize * 3;
        let src_end = src_start + roi.w as usize * 3;
        let dst_start = row * roi.w as usize * 3;
        out[dst_start..dst_start + roi.w as usize * 3].copy_from_slice(&full[src_start..src_end]);
    }
    out
}

#[test]
fn rgba420_and_non_row_aligned_restarts_match_turbo_with_padded_output() {
    use j2k_jpeg::{encode_jpeg_baseline, JpegEncodeOptions, JpegSamples, JpegSubsampling};
    if !turbo_available() {
        return;
    }
    let (width, height) = (35, 33);
    let pixels = j2k_test_support::patterned_rgb8(width, height);
    let mut turbo = libjpeg_turbo::TurboJpegDecoder::new().unwrap();
    for restart_interval in [None, Some(1), Some(7)] {
        let encoded = encode_jpeg_baseline(
            JpegSamples::Rgb8 {
                data: &pixels,
                width,
                height,
            },
            JpegEncodeOptions {
                subsampling: JpegSubsampling::Ybr420,
                restart_interval,
                ..Default::default()
            },
        )
        .unwrap();
        let reference = turbo.decode_rgb(&encoded.data).unwrap();
        let decoder = Decoder::new(&encoded.data).unwrap();
        let stride = width as usize * 4 + 5;
        let mut output = vec![0xa5; stride * height as usize];
        decoder
            .decode_into(&mut output, stride, PixelFormat::Rgba8)
            .unwrap();
        for (row, rgb) in output
            .chunks_exact(stride)
            .zip(reference.chunks_exact(width as usize * 3))
        {
            for (rgba, expected) in row[..width as usize * 4]
                .chunks_exact(4)
                .zip(rgb.chunks_exact(3))
            {
                assert_eq!(&rgba[..3], expected, "restart {restart_interval:?}");
                assert_eq!(rgba[3], 255);
            }
            assert_eq!(&row[width as usize * 4..], &[0xa5; 5]);
        }
        let roi = Rect {
            x: 17,
            y: 15,
            w: 13,
            h: 17,
        };
        let (region, _) = decoder
            .decode_request(DecodeRequest::region(PixelFormat::Rgba8, roi))
            .unwrap();
        let expected = crop_rgb(&reference, width as usize, roi);
        for (rgba, rgb) in region.chunks_exact(4).zip(expected.chunks_exact(3)) {
            assert_eq!(&rgba[..3], rgb);
            assert_eq!(rgba[3], 255);
        }
    }
}

/// 4:2:0 frames whose height ends inside an MCU row. The encoded rows below
/// the frame height hold contrasting colours, so blending decoded MCU padding
/// into the last output row, instead of replicating the last real chroma row
/// as libjpeg-turbo does, shows up as a large difference.
#[test]
fn ybr420_bottom_rows_match_turbo_when_height_is_not_mcu_aligned() {
    use j2k_jpeg::{encode_jpeg_baseline, JpegEncodeOptions, JpegSamples, JpegSubsampling};
    if !turbo_available() {
        return;
    }
    let mut turbo = libjpeg_turbo::TurboJpegDecoder::new().unwrap();
    let width = 130u32;
    // Heights = 2 (mod 4) and 4, 8, 12 (mod 16), with aligned and odd controls.
    for height in [34u32, 64, 65, 66, 68, 72, 76] {
        let encoded_height = height.div_ceil(16) * 16;
        let pixels = rgb8_with_contrasting_rows_below(width, encoded_height, height);
        for restart_interval in [None, Some(1), Some(7)] {
            let mut jpeg = encode_jpeg_baseline(
                JpegSamples::Rgb8 {
                    data: &pixels,
                    width,
                    height: encoded_height,
                },
                JpegEncodeOptions {
                    subsampling: JpegSubsampling::Ybr420,
                    restart_interval,
                    ..Default::default()
                },
            )
            .unwrap()
            .data;
            set_baseline_frame_height(&mut jpeg, height);
            let context = format!("height {height}, restart {restart_interval:?}");
            let reference = turbo.decode_rgb(&jpeg).unwrap();
            let decoder = Decoder::new(&jpeg).unwrap();

            let (rgb, _) = decoder
                .decode_request(DecodeRequest::full(PixelFormat::Rgb8))
                .unwrap();
            assert_rgb_rows_eq(&rgb, 3, &reference, width, &format!("{context}, RGB8"));
            let (rgba, _) = decoder
                .decode_request(DecodeRequest::full(PixelFormat::Rgba8))
                .unwrap();
            assert_rgb_rows_eq(&rgba, 4, &reference, width, &format!("{context}, RGBA8"));

            let roi = Rect {
                x: 17,
                y: height - 9,
                w: 90,
                h: 9,
            };
            let expected = crop_rgb(&reference, width as usize, roi);
            let (region, _) = decoder
                .decode_request(DecodeRequest::region(PixelFormat::Rgb8, roi))
                .unwrap();
            assert_rgb_rows_eq(&region, 3, &expected, roi.w, &format!("{context}, region"));
        }
    }
}

fn rgb8_with_contrasting_rows_below(width: u32, height: u32, real_height: u32) -> Vec<u8> {
    let mut pixels = j2k_test_support::patterned_rgb8(width, height);
    let row_len = width as usize * 3;
    for (x, pixel) in pixels[real_height as usize * row_len..]
        .chunks_exact_mut(3)
        .enumerate()
    {
        let magenta_or_blue = if (x % width as usize / 3).is_multiple_of(2) {
            255
        } else {
            0
        };
        pixel.copy_from_slice(&[magenta_or_blue, 0, 255]);
    }
    pixels
}

/// Rewrites the SOF0 frame height, walking marker segments from SOI.
fn set_baseline_frame_height(jpeg: &mut [u8], height: u32) {
    let mut offset = 2;
    loop {
        assert_eq!(jpeg[offset], 0xFF, "marker expected at byte {offset}");
        let marker = jpeg[offset + 1];
        if marker == 0xC0 {
            let height = u16::try_from(height).unwrap().to_be_bytes();
            jpeg[offset + 5..offset + 7].copy_from_slice(&height);
            return;
        }
        assert_ne!(marker, 0xDA, "no SOF0 before the first scan");
        let len = usize::from(u16::from_be_bytes([jpeg[offset + 2], jpeg[offset + 3]]));
        offset += 2 + len;
    }
}

fn assert_rgb_rows_eq(actual: &[u8], channels: usize, expected_rgb: &[u8], width: u32, what: &str) {
    let width = width as usize;
    assert_eq!(
        actual.len() / channels,
        expected_rgb.len() / 3,
        "{what}: size"
    );
    for (row, (actual_row, expected_row)) in actual
        .chunks_exact(width * channels)
        .zip(expected_rgb.chunks_exact(width * 3))
        .enumerate()
    {
        let matches = actual_row
            .chunks_exact(channels)
            .zip(expected_row.chunks_exact(3))
            .all(|(actual, expected)| &actual[..3] == expected);
        assert!(matches, "{what}: row {row} differs from libjpeg-turbo");
    }
}

/// Compression and 12-bit decompression through `TurboJPEG` 3: fixtures and
/// oracles the shared benchmark adapter does not provide.
#[cfg(all(has_libjpeg_turbo, has_libjpeg_turbo_v3))]
mod turbo_codec {
    use std::ffi::{c_char, c_int, c_void, CStr};

    const TJINIT_COMPRESS: c_int = 0;
    const TJINIT_DECOMPRESS: c_int = 1;
    const TJPARAM_QUALITY: c_int = 3;
    const TJPARAM_SUBSAMP: c_int = 4;
    const TJPARAM_PROGRESSIVE: c_int = 12;
    pub(crate) const TJPF_RGB: c_int = 0;
    pub(crate) const TJPF_GRAY: c_int = 6;
    pub(crate) const TJSAMP_444: c_int = 0;
    pub(crate) const TJSAMP_422: c_int = 1;
    pub(crate) const TJSAMP_420: c_int = 2;
    pub(crate) const TJSAMP_GRAY: c_int = 3;
    pub(crate) const TJSAMP_440: c_int = 4;

    unsafe extern "C" {
        fn tj3Init(init_type: c_int) -> *mut c_void;
        fn tj3Destroy(handle: *mut c_void);
        fn tj3GetErrorStr(handle: *mut c_void) -> *mut c_char;
        fn tj3Set(handle: *mut c_void, param: c_int, value: c_int) -> c_int;
        fn tj3Free(buffer: *mut c_void);
        fn tj3Compress8(
            handle: *mut c_void,
            src: *const u8,
            width: c_int,
            pitch: c_int,
            height: c_int,
            pixel_format: c_int,
            jpeg: *mut *mut u8,
            jpeg_size: *mut usize,
        ) -> c_int;
        fn tj3Compress12(
            handle: *mut c_void,
            src: *const i16,
            width: c_int,
            pitch: c_int,
            height: c_int,
            pixel_format: c_int,
            jpeg: *mut *mut u8,
            jpeg_size: *mut usize,
        ) -> c_int;
        fn tj3Decompress12(
            handle: *mut c_void,
            jpeg: *const u8,
            jpeg_size: usize,
            dst: *mut i16,
            pitch: c_int,
            pixel_format: c_int,
        ) -> c_int;
    }

    /// Encoding parameters for a generated fixture.
    #[derive(Clone, Copy, Debug)]
    pub(crate) struct Encode {
        pub(crate) width: u32,
        pub(crate) height: u32,
        pub(crate) pixel_format: c_int,
        pub(crate) subsampling: c_int,
        pub(crate) quality: c_int,
        pub(crate) progressive: bool,
    }

    struct Handle(*mut c_void);

    impl Handle {
        fn new(init_type: c_int) -> Self {
            // SAFETY: tj3Init has no preconditions; a null result is checked.
            let handle = unsafe { tj3Init(init_type) };
            assert!(!handle.is_null(), "tj3Init returned null");
            Self(handle)
        }

        fn check(&self, rc: c_int, what: &str) {
            if rc == 0 {
                return;
            }
            // SAFETY: the handle is live; TurboJPEG returns a static C string.
            let message = unsafe { CStr::from_ptr(tj3GetErrorStr(self.0)) };
            panic!("{what}: {}", message.to_string_lossy());
        }

        fn configure(&self, spec: Encode) {
            for (param, value) in [
                (TJPARAM_QUALITY, spec.quality),
                (TJPARAM_SUBSAMP, spec.subsampling),
                (TJPARAM_PROGRESSIVE, c_int::from(spec.progressive)),
            ] {
                // SAFETY: the handle is live and the parameter ids are TurboJPEG 3's.
                self.check(unsafe { tj3Set(self.0, param, value) }, "tj3Set");
            }
        }
    }

    impl Drop for Handle {
        fn drop(&mut self) {
            // SAFETY: the handle came from tj3Init and is destroyed once.
            unsafe { tj3Destroy(self.0) };
        }
    }

    fn channels(pixel_format: c_int) -> usize {
        if pixel_format == TJPF_GRAY {
            1
        } else {
            3
        }
    }

    fn c(value: usize) -> c_int {
        c_int::try_from(value).expect("dimension fits c_int")
    }

    /// Copies a TurboJPEG-allocated JPEG buffer out and frees it.
    fn take_jpeg(jpeg: *mut u8, size: usize) -> Vec<u8> {
        // SAFETY: TurboJPEG allocated `size` bytes at `jpeg` for this call.
        let bytes = unsafe { std::slice::from_raw_parts(jpeg, size) }.to_vec();
        // SAFETY: the buffer came from TurboJPEG and is freed once.
        unsafe { tj3Free(jpeg.cast()) };
        bytes
    }

    pub(crate) fn compress8(pixels: &[u8], spec: Encode) -> Vec<u8> {
        let handle = Handle::new(TJINIT_COMPRESS);
        handle.configure(spec);
        let pitch = spec.width as usize * channels(spec.pixel_format);
        assert_eq!(pixels.len(), pitch * spec.height as usize);
        let (mut jpeg, mut size) = (std::ptr::null_mut(), 0usize);
        // SAFETY: `pixels` holds `height` rows of `pitch` samples; TurboJPEG
        // allocates the output buffer, which take_jpeg frees.
        let rc = unsafe {
            tj3Compress8(
                handle.0,
                pixels.as_ptr(),
                c(spec.width as usize),
                c(pitch),
                c(spec.height as usize),
                spec.pixel_format,
                &raw mut jpeg,
                &raw mut size,
            )
        };
        handle.check(rc, "tj3Compress8");
        take_jpeg(jpeg, size)
    }

    pub(crate) fn compress12(samples: &[i16], spec: Encode) -> Vec<u8> {
        let handle = Handle::new(TJINIT_COMPRESS);
        handle.configure(spec);
        let pitch = spec.width as usize * channels(spec.pixel_format);
        assert_eq!(samples.len(), pitch * spec.height as usize);
        let (mut jpeg, mut size) = (std::ptr::null_mut(), 0usize);
        // SAFETY: as for compress8, with 12-bit samples in `i16`.
        let rc = unsafe {
            tj3Compress12(
                handle.0,
                samples.as_ptr(),
                c(spec.width as usize),
                c(pitch),
                c(spec.height as usize),
                spec.pixel_format,
                &raw mut jpeg,
                &raw mut size,
            )
        };
        handle.check(rc, "tj3Compress12");
        take_jpeg(jpeg, size)
    }

    /// Full-frame 12-bit decode with libjpeg-turbo's defaults (ISLOW IDCT,
    /// fancy upsampling).
    pub(crate) fn decompress12(
        jpeg: &[u8],
        width: u32,
        height: u32,
        pixel_format: c_int,
    ) -> Vec<i16> {
        let handle = Handle::new(TJINIT_DECOMPRESS);
        let pitch = width as usize * channels(pixel_format);
        let mut out = vec![0i16; pitch * height as usize];
        // SAFETY: `out` holds `height` rows of `pitch` samples.
        let rc = unsafe {
            tj3Decompress12(
                handle.0,
                jpeg.as_ptr(),
                jpeg.len(),
                out.as_mut_ptr(),
                c(pitch),
                pixel_format,
            )
        };
        handle.check(rc, "tj3Decompress12");
        out
    }
}

/// Smooth gradients plus deterministic noise, so blocks carry many AC terms.
#[cfg(all(has_libjpeg_turbo, has_libjpeg_turbo_v3))]
fn textured_samples(width: u32, height: u32, channels: u32, max: u32) -> Vec<u32> {
    let mut state = 0x9e37_79b9_u32;
    let span = width + height;
    let mut samples = Vec::with_capacity((width * height * channels) as usize);
    for y in 0..height {
        for x in 0..width {
            for channel in 0..channels {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                let base = ((x + y) * max / span + channel * max / 3 + (x * y) % 97) % (max + 1);
                let noise = state % (max / 8 + 1);
                samples.push((base + noise).min(max));
            }
        }
    }
    samples
}

#[cfg(all(has_libjpeg_turbo, has_libjpeg_turbo_v3))]
#[test]
fn ycbcr440_matches_turbo_h1v2_fancy_upsampling() {
    use turbo_codec::{Encode, TJPF_RGB, TJSAMP_440};

    if !turbo_available() {
        return;
    }
    let mut turbo = libjpeg_turbo::TurboJpegDecoder::new().expect("turbojpeg decoder");
    // Odd, even, and MCU-aligned heights; baseline and progressive.
    for (width, height) in [(64, 64), (67, 45), (40, 34), (33, 17)] {
        for (quality, progressive) in [(75, false), (95, false), (90, true)] {
            let rgb = textured_samples(width, height, 3, 255)
                .into_iter()
                .map(|sample| u8::try_from(sample).expect("8-bit sample"))
                .collect::<Vec<_>>();
            let spec = Encode {
                width,
                height,
                pixel_format: TJPF_RGB,
                subsampling: TJSAMP_440,
                quality,
                progressive,
            };
            let jpeg = turbo_codec::compress8(&rgb, spec);
            let expected = turbo.decode_rgb(&jpeg).expect("turbojpeg 4:4:0 decode");
            let (actual, _) = Decoder::new(&jpeg)
                .expect("4:4:0 decoder")
                .decode_request(DecodeRequest::full(PixelFormat::Rgb8))
                .expect("4:4:0 decode");
            assert_rgb_rows_eq(&actual, 3, &expected, width, &format!("{spec:?}"));
        }
    }
}

#[cfg(all(has_libjpeg_turbo, has_libjpeg_turbo_v3))]
#[test]
fn extended12_decodes_match_turbo() {
    use turbo_codec::{
        Encode, TJPF_GRAY, TJPF_RGB, TJSAMP_420, TJSAMP_422, TJSAMP_444, TJSAMP_GRAY,
    };

    if !turbo_available() {
        return;
    }
    for (pixel_format, subsampling) in [
        (TJPF_GRAY, TJSAMP_GRAY),
        (TJPF_RGB, TJSAMP_444),
        (TJPF_RGB, TJSAMP_422),
        (TJPF_RGB, TJSAMP_420),
    ] {
        let (channels, format) = if pixel_format == TJPF_GRAY {
            (1, PixelFormat::Gray16)
        } else {
            (3, PixelFormat::Rgb16)
        };
        for (width, height) in [(64, 48), (67, 45)] {
            for quality in [75, 95, 100] {
                let samples = textured_samples(width, height, channels, 4095)
                    .into_iter()
                    .map(|sample| i16::try_from(sample).expect("12-bit sample"))
                    .collect::<Vec<_>>();
                let spec = Encode {
                    width,
                    height,
                    pixel_format,
                    subsampling,
                    quality,
                    progressive: false,
                };
                let jpeg = turbo_codec::compress12(&samples, spec);
                let expected = turbo_codec::decompress12(&jpeg, width, height, pixel_format);
                let (actual, _) = Decoder::new(&jpeg)
                    .expect("12-bit decoder")
                    .decode_request(DecodeRequest::full(format))
                    .expect("12-bit decode");
                let actual = actual
                    .chunks_exact(2)
                    .map(|pair| i16::from_ne_bytes([pair[0], pair[1]]))
                    .collect::<Vec<_>>();
                assert_eq!(actual.len(), expected.len(), "{spec:?}: size");
                if let Some(index) = actual.iter().zip(&expected).position(|(a, e)| a != e) {
                    let max_delta = actual
                        .iter()
                        .zip(&expected)
                        .map(|(a, e)| (i32::from(*a) - i32::from(*e)).abs())
                        .max()
                        .unwrap_or(0);
                    panic!(
                        "{spec:?}: sample {index} is {} (turbo {}); max delta {max_delta}",
                        actual[index], expected[index]
                    );
                }
            }
        }
    }
}
