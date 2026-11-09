// SPDX-License-Identifier: MIT OR Apache-2.0

//! The cross-backend baseline JPEG encode matrix.
//!
//! The CPU, Metal and CUDA baseline encoders produce identical bytes for the
//! same samples and options: they share the colour conversion, sampling,
//! tables and entropy coding, and the integer forward DCT of
//! `j2k_codec_math::jpeg::fdct`. Every backend encodes these cases and checks
//! the framed digest of its frames against the one golden,
//! [`JPEG_ENCODE_MATRIX_SHA256`].

use sha2::{Digest, Sha256};

/// Framed SHA-256 of every case of [`jpeg_encode_matrix_cases`], in order,
/// encoded by any baseline encoder backend.
pub const JPEG_ENCODE_MATRIX_SHA256: &str =
    "8efe6ea0d1850bdf3c7221db3f06f8814f0ebb0997d42751cf51f7908c9cb661";

const DIGEST_DOMAIN: &[u8] = b"J2K-JPEG-BASELINE-ENCODE-MATRIX\0";

/// Frame sizes: a single pixel, sizes that end inside an MCU in both
/// directions, and one that spans several MCUs of every sampling.
const SIZES: [(u32, u32); 4] = [(1, 1), (17, 9), (33, 31), (70, 45)];

/// Low and high quality; quality 100 quantizes by one, where rounding ties
/// are most frequent.
const QUALITIES: [u8; 2] = [20, 100];

const RESTART_INTERVALS: [Option<u16>; 2] = [None, Some(2)];

/// Component sampling of one matrix case.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JpegEncodeMatrixSampling {
    /// One grayscale component.
    Gray,
    /// RGB as YCbCr 4:4:4.
    Ybr444,
    /// RGB as YCbCr 4:2:2.
    Ybr422,
    /// RGB as YCbCr 4:2:0.
    Ybr420,
}

/// One baseline encode of the cross-backend matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JpegEncodeMatrixCase {
    /// Component sampling.
    pub sampling: JpegEncodeMatrixSampling,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Encoder quality, 1 to 100.
    pub quality: u8,
    /// Restart interval in MCUs, if any.
    pub restart_interval: Option<u16>,
}

impl JpegEncodeMatrixCase {
    /// Interleaved samples per pixel: 1 for grayscale, 3 for RGB.
    pub fn channels(self) -> usize {
        if self.sampling == JpegEncodeMatrixSampling::Gray {
            1
        } else {
            3
        }
    }

    /// The case's interleaved Gray8 or Rgb8 samples, row-major and tightly
    /// packed.
    ///
    /// Each 8x8 cell holds one of four textures: a gradient, uniform noise,
    /// two levels (which put exact rounding ties on many frequencies) or a
    /// flat value, so the encoders' constant-block, repeated-edge and general
    /// paths all run.
    ///
    /// # Panics
    ///
    /// Never: a channel index is at most 2.
    pub fn samples(self) -> Vec<u8> {
        let channels = self.channels();
        let mut samples = Vec::new();
        let mut noise = 0x5eed_f00d_u32 ^ u32::from(self.quality) ^ (self.width << 8);
        for y in 0..self.height {
            for x in 0..self.width {
                for channel in 0..channels {
                    let channel = u32::try_from(channel).expect("channel index fits u32");
                    noise = noise.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    samples.push(texture_sample(x, y, channel, noise));
                }
            }
        }
        samples
    }

    /// A short description for assertion messages.
    pub fn label(self) -> String {
        format!(
            "{:?} {}x{} quality {} restart {:?}",
            self.sampling, self.width, self.height, self.quality, self.restart_interval
        )
    }
}

fn texture_sample(x: u32, y: u32, channel: u32, noise: u32) -> u8 {
    let cell = (x / 8 + y / 8 + channel) % 4;
    let value = match cell {
        0 => x * 9 + y * 5 + channel * 40,
        1 => noise >> 24,
        2 => {
            if (noise >> 20) & 1 == 1 {
                200 - channel * 30
            } else {
                40 + channel * 25
            }
        }
        _ => 60 + (x / 8) * 37 + (y / 8) * 11 + channel * 50,
    };
    u8::try_from(value & 0xff).expect("masked sample fits u8")
}

/// Framed SHA-256 of [`jpeg_encode_transcode_fixtures`], in order, encoded
/// by any baseline encoder backend.
pub const JPEG_ENCODE_TRANSCODE_FIXTURES_SHA256: &str =
    "f095869ddad3f3937dd986676051ab4d1e240d4be9865298b5ab9d8631277ea3";

/// One named input of [`jpeg_encode_transcode_fixtures`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JpegEncodeFixture {
    /// The fixture's name in the JPEG-to-HTJ2K transcode matrix.
    pub name: &'static str,
    /// Sampling, size, quality and restart interval.
    pub case: JpegEncodeMatrixCase,
    /// Interleaved Gray8 or Rgb8 samples, row-major and tightly packed.
    pub samples: Vec<u8>,
}

/// The synthetic inputs of the JPEG-to-HTJ2K transcode test matrix.
///
/// The transcode tests encode them with the baseline encoder; before the
/// encoders shared `j2k_codec_math::jpeg::fdct`, six of the RGB ones encoded
/// differently on aarch64 and x86-64. The samples are smooth ramps with
/// wrap-around edges under a checkerboard, at quality 90.
pub fn jpeg_encode_transcode_fixtures() -> Vec<JpegEncodeFixture> {
    use JpegEncodeMatrixSampling::{Gray, Ybr420, Ybr422, Ybr444};
    let specs = [
        ("synthetic_444_224_restart", Ybr444, (224, 224), Some(28)),
        ("synthetic_420_224_restart", Ybr420, (224, 224), Some(28)),
        ("synthetic_420_512_restart", Ybr420, (512, 512), Some(64)),
        ("synthetic_gray_100x37_restart", Gray, (100, 37), Some(3)),
        ("synthetic_422_131x77", Ybr422, (131, 77), None),
        ("synthetic_420_97x61_restart", Ybr420, (97, 61), Some(5)),
        ("synthetic_444_45x67_restart", Ybr444, (45, 67), Some(1)),
        ("synthetic_gray_1031x9", Gray, (1031, 9), None),
        (
            "synthetic_420_1100x1030_restart",
            Ybr420,
            (1100, 1030),
            Some(69),
        ),
    ];
    specs
        .into_iter()
        .map(|(name, sampling, (width, height), restart_interval)| {
            let channels = if sampling == Gray { 1 } else { 3 };
            let mut samples = Vec::new();
            for y in 0..height {
                for x in 0..width {
                    samples.extend((0..channels).map(|channel| pattern_sample(x, y, channel)));
                }
            }
            JpegEncodeFixture {
                name,
                case: JpegEncodeMatrixCase {
                    sampling,
                    width,
                    height,
                    quality: 90,
                    restart_interval,
                },
                samples,
            }
        })
        .collect()
}

/// The transcode matrix's sample: ramps that wrap around, plus a
/// checkerboard of 3x5 cells, so every subband carries energy.
fn pattern_sample(x: u32, y: u32, channel: u32) -> u8 {
    let ramp = match channel {
        0 => x * 5 + y * 3 + 17,
        1 => (x ^ y) * 2 + 41,
        _ => x * 11 + y * 13 + 73,
    };
    let checker = if (x / 3 + y / 5).is_multiple_of(2) {
        0
    } else {
        37
    };
    u8::try_from((ramp + checker) & 0xff).expect("masked sample fits u8")
}

/// Every case of the matrix: each sampling, size, quality and restart
/// setting.
pub fn jpeg_encode_matrix_cases() -> Vec<JpegEncodeMatrixCase> {
    let mut cases = Vec::new();
    for sampling in [
        JpegEncodeMatrixSampling::Gray,
        JpegEncodeMatrixSampling::Ybr444,
        JpegEncodeMatrixSampling::Ybr422,
        JpegEncodeMatrixSampling::Ybr420,
    ] {
        for (width, height) in SIZES {
            for quality in QUALITIES {
                for restart_interval in RESTART_INTERVALS {
                    cases.push(JpegEncodeMatrixCase {
                        sampling,
                        width,
                        height,
                        quality,
                        restart_interval,
                    });
                }
            }
        }
    }
    cases
}

/// Inputs smaller than the frames a GPU encoder codes them in.
///
/// Each case comes with its frame size. They cover every sampling, a frame
/// that ends inside the input's last MCU and one several MCUs past it, odd
/// heights, and restart intervals.
pub fn jpeg_encode_padded_cases() -> Vec<(JpegEncodeMatrixCase, (u32, u32))> {
    use JpegEncodeMatrixSampling::{Gray, Ybr420, Ybr422, Ybr444};
    let geometries = [
        (Ybr444, (17, 9), (24, 13)),
        (Ybr420, (17, 9), (24, 13)),
        (Ybr422, (17, 9), (24, 13)),
        (Gray, (17, 9), (24, 13)),
        (Ybr420, (5, 3), (40, 21)),
        (Ybr444, (33, 31), (35, 33)),
    ];
    let mut cases = Vec::new();
    for (sampling, (width, height), frame) in geometries {
        for (quality, restart_interval) in [(20, None), (100, Some(1))] {
            let case = JpegEncodeMatrixCase {
                sampling,
                width,
                height,
                quality,
                restart_interval,
            };
            cases.push((case, frame));
        }
    }
    cases
}

/// `samples`, tightly packed at `size` with `channels` interleaved samples per
/// pixel, extended to `frame` by repeating the last column and row.
///
/// The CPU encoder codes every sample beyond its image as the nearest image
/// sample, so this is the image a GPU encoder codes when its frame extends
/// beyond its input.
///
/// # Panics
///
/// Panics if `samples` does not hold `size`, or `frame` is smaller than
/// `size` or empty.
pub fn replicate_edges(
    samples: &[u8],
    size: (u32, u32),
    channels: usize,
    frame: (u32, u32),
) -> Vec<u8> {
    let (width, height) = (size.0 as usize, size.1 as usize);
    assert_eq!(
        samples.len(),
        width * height * channels,
        "samples hold size"
    );
    assert!(width > 0 && height > 0 && frame.0 >= size.0 && frame.1 >= size.1);
    let mut extended = Vec::new();
    for y in 0..frame.1 as usize {
        let row = &samples[y.min(height - 1) * width * channels..][..width * channels];
        extended.extend_from_slice(row);
        let last = &row[(width - 1) * channels..];
        for _ in width..frame.0 as usize {
            extended.extend_from_slice(last);
        }
    }
    extended
}

/// The framed SHA-256 of encoded frames: each frame's length, then its bytes.
///
/// # Panics
///
/// Panics if a frame's length does not fit in a `u64`.
pub fn jpeg_encode_matrix_digest<'a>(frames: impl IntoIterator<Item = &'a [u8]>) -> String {
    let mut digest = Sha256::new();
    digest.update(DIGEST_DOMAIN);
    for frame in frames {
        digest.update(
            u64::try_from(frame.len())
                .expect("frame length fits u64")
                .to_le_bytes(),
        );
        digest.update(frame);
    }
    format!("{:x}", digest.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replicate_edges_repeats_the_last_column_and_row() {
        let samples = [1u8, 2, 3, 4, 5, 6];
        assert_eq!(
            replicate_edges(&samples, (3, 2), 1, (4, 3)),
            [1, 2, 3, 3, 4, 5, 6, 6, 4, 5, 6, 6]
        );
        assert_eq!(
            replicate_edges(&[7, 8, 9], (1, 1), 3, (2, 2)),
            [7, 8, 9, 7, 8, 9, 7, 8, 9, 7, 8, 9]
        );
    }

    #[test]
    fn matrix_covers_every_sampling_size_quality_and_restart() {
        let cases = jpeg_encode_matrix_cases();
        assert_eq!(
            cases.len(),
            4 * SIZES.len() * QUALITIES.len() * RESTART_INTERVALS.len()
        );
        for case in &cases {
            let samples = case.samples();
            assert_eq!(
                samples.len(),
                case.width as usize * case.height as usize * case.channels()
            );
        }
    }
}
