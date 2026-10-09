// SPDX-License-Identifier: MIT OR Apache-2.0

//! GPU time by step kind for the single-image grayscale route. Each step kind
//! is encoded once and then `EXTRA + 1` times; the difference in median decode
//! time over `EXTRA` is that kind's cost. The decodes are GPU-bound, so wall
//! time tracks GPU time.
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
        ] {
            set_single_step_repeats_for_test(kind, 1);
        }
    }
}

/// A 16-bit lossless HTJ2K grayscale codestream with smooth content and
/// noise, roughly like a medical image.
fn gray16(width: u32, height: u32) -> Vec<u8> {
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
        &EncodeOptions::default(),
    )
    .expect("encode profile fixture")
}

fn median_decode_ms(bytes: &[u8]) -> f64 {
    let mut samples = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let start = Instant::now();
        let mut decoder = J2kDecoder::new(bytes).expect("parse profile fixture");
        let surface = decoder
            .decode_request_to_device(MetalDecodeRequest::full(
                PixelFormat::Gray16,
                BackendRequest::Metal,
            ))
            .expect("Metal decode");
        std::hint::black_box(surface);
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
    for (width, height) in [(512, 512), (1024, 1024), (3072, 4096)] {
        let bytes = gray16(width, height);
        for _ in 0..3 {
            median_decode_ms(&bytes);
        }
        let base = median_decode_ms(&bytes);
        let mut line = format!("{width}x{height}: total {base:.3} ms");
        for (name, kind) in [
            ("tier1", SingleStepKind::Tier1),
            ("idwt", SingleStepKind::Idwt),
            ("store", SingleStepKind::Store),
        ] {
            set_single_step_repeats_for_test(kind, EXTRA + 1);
            let repeated = median_decode_ms(&bytes);
            set_single_step_repeats_for_test(kind, 1);
            write!(
                line,
                ", {name} {:.3} ms",
                (repeated - base) / f64::from(EXTRA)
            )
            .expect("format into a String");
        }
        std::println!("{line}");
    }
}
