// SPDX-License-Identifier: MIT OR Apache-2.0

//! Opt-in (`J2K_ENCODE_BENCH_V3`) comparison of the lossless and
//! optimized-Huffman encoders against `TurboJPEG` 3.
//!
//! Inputs are the `J2K_ENCODE_BENCH_INPUTS` PGM/PPM files, else patterned
//! images. Each input also yields a 12-bit grayscale case from its luma, with
//! deterministic low bits, encoded at precision 12 and 16 as DICOM stores it.

use std::hint::black_box;
use std::sync::Arc;

use criterion::Criterion;
use j2k_jpeg::{
    encode_jpeg_baseline_optimized, encode_jpeg_lossless, JpegBackend, JpegEncodeOptions,
    JpegLosslessEncodeOptions, JpegLosslessSamples, JpegSamples, JpegSubsampling,
};
use j2k_test_support::{patterned_gray8, patterned_rgb8};

use super::turbo_v3::{TurboV3Encoder, TurboV3Mode};

struct Image8 {
    name: String,
    width: u32,
    height: u32,
    gray: bool,
    data: Arc<[u8]>,
}

enum Encode {
    Lossless8,
    Lossless16 {
        precision: u8,
        data: Arc<[u16]>,
    },
    Optimized {
        quality: u8,
        subsampling: JpegSubsampling,
    },
}

struct V3Case {
    name: String,
    image: Arc<Image8>,
    encode: Encode,
}

impl V3Case {
    fn j2k(&self) -> Vec<u8> {
        let Image8 {
            width,
            height,
            gray,
            ref data,
            ..
        } = *self.image;
        let encoded = match &self.encode {
            Encode::Lossless8 => {
                let samples = if gray {
                    JpegLosslessSamples::Gray8 {
                        data,
                        width,
                        height,
                    }
                } else {
                    JpegLosslessSamples::Rgb8 {
                        data,
                        width,
                        height,
                    }
                };
                encode_jpeg_lossless(samples, lossless_options(8))
            }
            Encode::Lossless16 { precision, data } => encode_jpeg_lossless(
                JpegLosslessSamples::Gray16 {
                    data,
                    width,
                    height,
                },
                lossless_options(*precision),
            ),
            Encode::Optimized {
                quality,
                subsampling,
            } => {
                let samples = if gray {
                    JpegSamples::Gray8 {
                        data,
                        width,
                        height,
                    }
                } else {
                    JpegSamples::Rgb8 {
                        data,
                        width,
                        height,
                    }
                };
                encode_jpeg_baseline_optimized(
                    samples,
                    JpegEncodeOptions {
                        quality: *quality,
                        subsampling: *subsampling,
                        restart_interval: None,
                        backend: JpegBackend::Cpu,
                    },
                )
            }
        };
        encoded.expect("j2k encode").data
    }

    fn turbo_encoder(&self) -> TurboV3Encoder {
        let mode = match self.encode {
            Encode::Lossless8 => TurboV3Mode::LosslessSv1 { precision: 8 },
            Encode::Lossless16 { precision, .. } => TurboV3Mode::LosslessSv1 { precision },
            Encode::Optimized {
                quality,
                subsampling,
            } => TurboV3Mode::Optimized {
                quality,
                subsampling,
            },
        };
        let gray = self.image.gray || matches!(self.encode, Encode::Lossless16 { .. });
        TurboV3Encoder::new(self.image.width, self.image.height, gray, mode)
    }

    fn turbo(&self, encoder: &mut TurboV3Encoder, wide12: &[i16]) -> Vec<u8> {
        match &self.encode {
            Encode::Lossless16 { precision, .. } if *precision <= 12 => encoder.encode12(wide12),
            Encode::Lossless16 { data, .. } => encoder.encode16(data),
            Encode::Lossless8 | Encode::Optimized { .. } => encoder.encode8(&self.image.data),
        }
    }

    /// The 12-bit API takes signed storage; empty for other cases.
    fn wide12(&self) -> Vec<i16> {
        match &self.encode {
            Encode::Lossless16 { precision, data } if *precision <= 12 => data
                .iter()
                .map(|&sample| i16::try_from(sample).expect("12-bit sample"))
                .collect(),
            _ => Vec::new(),
        }
    }
}

fn lossless_options(precision: u8) -> JpegLosslessEncodeOptions {
    let mut options = JpegLosslessEncodeOptions::new(precision);
    options.backend = JpegBackend::Cpu;
    options
}

pub(super) fn bench(c: &mut Criterion) {
    let cases = cases();
    let mut group = c.benchmark_group("jpeg_cpu_encode_vs_turbojpeg3");
    for case in &cases {
        let wide12 = case.wide12();
        let mut turbo = case.turbo_encoder();
        let ours = case.j2k();
        let theirs = case.turbo(&mut turbo, &wide12);
        println!(
            "j2k_turbo3_compare case={} j2k_bytes={} turbojpeg_bytes={} identical={}",
            case.name,
            ours.len(),
            theirs.len(),
            ours == theirs
        );
        group.bench_function(format!("j2k/{}", case.name), |b| {
            b.iter(|| black_box(case.j2k()));
        });
        group.bench_function(format!("libjpeg-turbo/{}", case.name), |b| {
            b.iter(|| black_box(case.turbo(&mut turbo, &wide12)));
        });
    }
    group.finish();
}

fn cases() -> Vec<V3Case> {
    let filter = std::env::var("J2K_ENCODE_BENCH_V3_FILTER").ok();
    let mut cases = Vec::new();
    for image in images().into_iter().map(Arc::new) {
        cases.push(V3Case {
            name: format!("{}_lossless_sv1", image.name),
            image: Arc::clone(&image),
            encode: Encode::Lossless8,
        });
        let gray12: Arc<[u16]> = gray12_from(&image).into();
        for precision in [12, 16] {
            cases.push(V3Case {
                name: format!("{}_gray12_lossless_sv1_p{precision}", image.name),
                image: Arc::clone(&image),
                encode: Encode::Lossless16 {
                    precision,
                    data: Arc::clone(&gray12),
                },
            });
        }
        let samplings: &[(JpegSubsampling, &str)] = if image.gray {
            &[(JpegSubsampling::Gray, "gray")]
        } else {
            &[
                (JpegSubsampling::Ybr420, "420"),
                (JpegSubsampling::Ybr444, "444"),
            ]
        };
        for &(subsampling, suffix) in samplings {
            cases.push(V3Case {
                name: format!("{}_optimized_q90_{suffix}", image.name),
                image: Arc::clone(&image),
                encode: Encode::Optimized {
                    quality: 90,
                    subsampling,
                },
            });
        }
    }
    cases.retain(|case| {
        filter
            .as_deref()
            .is_none_or(|filter| case.name.contains(filter))
    });
    assert!(
        !cases.is_empty(),
        "no TurboJPEG 3 comparison cases selected"
    );
    cases
}

fn images() -> Vec<Image8> {
    let roots = j2k_test_support::paths_from_env("J2K_ENCODE_BENCH_INPUTS");
    if roots.is_empty() {
        return vec![
            Image8 {
                name: "patterned_rgb8_512".into(),
                width: 512,
                height: 512,
                gray: false,
                data: patterned_rgb8(512, 512).into(),
            },
            Image8 {
                name: "patterned_gray8_512".into(),
                width: 512,
                height: 512,
                gray: true,
                data: patterned_gray8(512, 512).into(),
            },
        ];
    }
    let mut paths = Vec::new();
    for root in roots {
        super::inputs::collect_pnm(root, &mut paths);
    }
    paths.sort();
    paths.dedup();
    paths
        .into_iter()
        // The shared reader accepts only 8-bit PNM; deeper inputs are derived.
        .filter_map(|path| {
            let image = j2k_test_support::read_pnm_image(&path).ok()?;
            Some(Image8 {
                name: path.file_stem()?.to_string_lossy().into_owned(),
                width: image.width,
                height: image.height,
                gray: image.channels == 1,
                data: image.pixels.into(),
            })
        })
        .collect()
}

/// Luma scaled to 12 bits, with low bits from a fixed LCG standing in for the
/// sensor detail an 8-bit source lacks.
fn gray12_from(image: &Image8) -> Vec<u16> {
    let mut state = 0x2545_f491_u32;
    let channels = if image.gray { 1 } else { 3 };
    image
        .data
        .chunks_exact(channels)
        .map(|pixel| {
            let luma = if image.gray {
                u32::from(pixel[0])
            } else {
                (77 * u32::from(pixel[0]) + 150 * u32::from(pixel[1]) + 29 * u32::from(pixel[2]))
                    >> 8
            };
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            u16::try_from((luma << 4) | (state >> 28)).expect("12-bit sample")
        })
        .collect()
}
