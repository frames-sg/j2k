// SPDX-License-Identifier: MIT OR Apache-2.0

//! GPU time by step kind for the single-image grayscale route. Each step kind
//! is encoded once and then `EXTRA + 1` times; the difference in median decode
//! time over `EXTRA` is that kind's cost. One decoder is reused per image, so
//! its plan is prepared once and the timings cover submission and GPU work.
//! `J2K_STEP_PROFILE_FILES` (a path list) profiles those `Gray16` codestreams
//! instead of the generated images, and `J2K_STEP_PROFILE_INTO` decodes into
//! one reused destination instead of a new surface each time.
//! `J2K_STEP_PROFILE_LOSSY` encodes the generated images with the 9/7
//! transform, and `J2K_STEP_PROFILE_COMPARE_IDWT` also times the inverse
//! transform with the single-pass tile kernels turned off.
//!
//! ```sh
//! J2K_REQUIRE_METAL_RUNTIME=1 cargo test --profile gpu-quick -p j2k-metal --lib -- \
//!     single_image_step_profile --include-ignored --nocapture --test-threads=1
//! ```

use std::fmt::Write as _;
use std::time::Instant;

use j2k::{BackendRequest, PixelFormat};
use j2k_native::{encode_htj2k, EncodeOptions};

use crate::engine::test_counters::{set_single_step_repeats_for_test, SingleStepKind};
use crate::{J2kDecoder, MetalDecodeRequest};

const EXTRA: u8 = 4;
const SAMPLES: usize = 41;

/// Restores single encoding even if a probe panics.
struct SingleStepsOnDrop;

impl Drop for SingleStepsOnDrop {
    fn drop(&mut self) {
        for kind in [
            SingleStepKind::Tier1,
            SingleStepKind::Idwt,
            SingleStepKind::Store,
            SingleStepKind::HtZero,
            SingleStepKind::HtVlc,
        ] {
            set_single_step_repeats_for_test(kind, 1);
        }
        crate::engine::test_counters::set_fused_2d_idwt_for_test(true);
    }
}

/// A 16-bit lossless HTJ2K grayscale codestream with smooth content and
/// noise, roughly like a medical image.
fn gray16(width: u32, height: u32, reversible: bool) -> Vec<u8> {
    let mut state = 0x9E37_79B9_u32;
    let mut pixels = Vec::with_capacity(width as usize * height as usize * 2);
    for y in 0..height {
        for x in 0..width {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let smooth = (x * 7 + y * 5) % 3000 + 1000;
            let value = u16::try_from(smooth + (state & 63)).expect("fits u16");
            pixels.extend_from_slice(&value.to_le_bytes());
        }
    }
    encode_htj2k(
        &pixels,
        width,
        height,
        1,
        16,
        false,
        &EncodeOptions {
            reversible,
            ..EncodeOptions::default()
        },
    )
    .expect("encode profile fixture")
}

fn median_decode_ms(
    decoder: &mut J2kDecoder<'_>,
    destination: Option<&j2k_metal_support::MetalImageDestination>,
) -> f64 {
    let mut samples = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let start = Instant::now();
        if let Some(destination) = destination {
            decoder
                .decode_into_metal(destination)
                .expect("Metal decode into destination");
        } else {
            let surface = decoder
                .decode_request_to_device(MetalDecodeRequest::full(
                    PixelFormat::Gray16,
                    BackendRequest::Metal,
                ))
                .expect("Metal decode");
            std::hint::black_box(surface);
        }
        samples.push(start.elapsed().as_secs_f64() * 1e3);
    }
    samples.sort_by(f64::total_cmp);
    samples[SAMPLES / 2]
}

#[test]
#[ignore = "GPU step attribution; run explicitly with --include-ignored --nocapture"]
fn single_image_step_profile() {
    if !super::runtime::should_run_metal_runtime() {
        return;
    }
    let _restore = SingleStepsOnDrop;
    let images: Vec<(String, Vec<u8>)> = match std::env::var_os("J2K_STEP_PROFILE_FILES") {
        Some(paths) => std::env::split_paths(&paths)
            .map(|path| {
                let bytes = std::fs::read(&path).expect("read profile codestream");
                (path.display().to_string(), bytes)
            })
            .collect(),
        None => [(512, 512), (1024, 1024), (3072, 4096)]
            .into_iter()
            .map(|(width, height)| {
                let reversible = std::env::var_os("J2K_STEP_PROFILE_LOSSY").is_none();
                let label = format!("{width}x{height}{}", if reversible { "" } else { " 9/7" });
                (label, gray16(width, height, reversible))
            })
            .collect(),
    };
    for (label, bytes) in &images {
        let mut decoder = J2kDecoder::new(bytes).expect("parse profile codestream");
        let destination = std::env::var_os("J2K_STEP_PROFILE_INTO").map(|_| {
            let dimensions = decoder.inner().info().dimensions;
            let pitch = dimensions.0 as usize * PixelFormat::Gray16.bytes_per_pixel();
            let device = j2k_metal_support::system_default_device().expect("Metal device");
            let buffer =
                j2k_metal_support::checked_shared_buffer(&device, pitch * dimensions.1 as usize)
                    .expect("destination buffer");
            let layout =
                j2k_metal_support::MetalImageLayout::new(0, dimensions, pitch, PixelFormat::Gray16)
                    .expect("destination layout");
            // SAFETY: every profile decode completes before the next starts,
            // and nothing else reads or writes this buffer.
            unsafe {
                j2k_metal_support::MetalImageDestination::from_exclusive_buffer(buffer, layout)
            }
            .expect("destination")
        });
        let destination = destination.as_ref();
        median_decode_ms(&mut decoder, destination);
        let base = median_decode_ms(&mut decoder, destination);
        let mut line = format!("{label}: total {base:.3} ms");
        for (name, kind) in [
            ("tier1", SingleStepKind::Tier1),
            ("idwt", SingleStepKind::Idwt),
            ("store", SingleStepKind::Store),
            ("ht zero", SingleStepKind::HtZero),
            ("ht vlc", SingleStepKind::HtVlc),
        ] {
            set_single_step_repeats_for_test(kind, EXTRA + 1);
            let repeated = median_decode_ms(&mut decoder, destination);
            set_single_step_repeats_for_test(kind, 1);
            write!(
                line,
                ", {name} {:.3} ms",
                (repeated - base) / f64::from(EXTRA)
            )
            .expect("format into a String");
        }
        if std::env::var_os("J2K_STEP_PROFILE_COMPARE_IDWT").is_some() {
            crate::engine::test_counters::set_fused_2d_idwt_for_test(false);
            let two_pass_base = median_decode_ms(&mut decoder, destination);
            set_single_step_repeats_for_test(SingleStepKind::Idwt, EXTRA + 1);
            let two_pass_repeated = median_decode_ms(&mut decoder, destination);
            set_single_step_repeats_for_test(SingleStepKind::Idwt, 1);
            crate::engine::test_counters::set_fused_2d_idwt_for_test(true);
            write!(
                line,
                ", two-pass total {two_pass_base:.3} ms, two-pass idwt {:.3} ms",
                (two_pass_repeated - two_pass_base) / f64::from(EXTRA)
            )
            .expect("format into a String");
        }
        std::println!("{line}");
    }
}
