// SPDX-License-Identifier: MIT OR Apache-2.0

//! GPU time by stage for the resident color batch route, by truncation: each
//! variant encodes the dispatches up to one `DecodeStageLimit`, and successive
//! differences of the median GPU times attribute the cost. Variants are sampled
//! round-robin after a warm-up so clock ramping affects each alike.
//!
//! ```sh
//! J2K_REQUIRE_METAL_RUNTIME=1 cargo test --profile gpu-quick -p j2k-metal --lib -- \
//!     decode_stage_profile --include-ignored --nocapture --test-threads=1
//! ```

use std::sync::Arc;
use std::time::{Duration, Instant};

use j2k::{BatchDecodeOptions, BatchLayout, EncodedImage, PreparedBatch};
use j2k_native::{encode_htj2k, EncodeOptions};

use crate::engine::test_counters::{
    begin_direct_destination_capture_for_test, end_direct_destination_capture_for_test,
    force_per_image_idwt_for_test, set_decode_stage_limit_for_test, DecodeStageLimit,
};
use crate::metal_types::prelude::*;

const LIMITS: [DecodeStageLimit; 4] = [
    DecodeStageLimit::HtVlc,
    DecodeStageLimit::Tier1,
    DecodeStageLimit::Idwt,
    DecodeStageLimit::Full,
];
const SAMPLES: usize = 21;
const WARM_UP: Duration = Duration::from_secs(2);

/// Restores full decoding even if a probe panics.
struct FullStagesOnDrop;

impl Drop for FullStagesOnDrop {
    fn drop(&mut self) {
        set_decode_stage_limit_for_test(DecodeStageLimit::Full);
        force_per_image_idwt_for_test(false);
    }
}

/// Distinct 9/7 HT RGB codestreams, as `metal_idwt97_geometry_distinct` builds them.
fn distinct_batch(width: u32, height: u32, count: u8) -> Vec<EncodedImage> {
    let options = EncodeOptions {
        reversible: false,
        num_decomposition_levels: 3,
        guard_bits: 2,
        ..EncodeOptions::default()
    };
    (0..count)
        .map(|index| {
            let mut pixels = j2k_test_support::patterned_rgb8(width, height);
            pixels[0] = pixels[0].wrapping_add(index);
            let bytes = encode_htj2k(&pixels, width, height, 3, 8, false, &options)
                .expect("encode stage-profile fixture");
            EncodedImage::full(Arc::from(bytes))
        })
        .collect()
}

/// GPU seconds of one resident decode truncated at `limit`.
fn decode_gpu_seconds(
    decoder: &mut crate::MetalBatchDecoder,
    prepared: &PreparedBatch,
    limit: DecodeStageLimit,
) -> f64 {
    set_decode_stage_limit_for_test(limit);
    begin_direct_destination_capture_for_test();
    let result = decoder
        .decode_prepared(prepared)
        .expect("stage-profile decode");
    let command_buffers = end_direct_destination_capture_for_test();
    assert!(
        result.errors().is_empty(),
        "stage-profile decode at {limit:?}: {:?}",
        result.errors()
    );
    assert!(
        !command_buffers.is_empty(),
        "the batch did not take the direct color destination route"
    );
    command_buffers
        .iter()
        .map(|command_buffer| {
            command_buffer.waitUntilCompleted();
            command_buffer.GPUEndTime() - command_buffer.GPUStartTime()
        })
        .sum()
}

fn median_ms(samples: &mut [f64]) -> f64 {
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2] * 1e3
}

#[test]
#[ignore = "GPU stage attribution; run explicitly with --include-ignored --nocapture"]
fn decode_stage_profile() {
    if !super::runtime::should_run_metal_runtime() {
        return;
    }
    let _restore = FullStagesOnDrop;
    for (width, height, count) in [(1024, 1024, 16), (640, 480, 16)] {
        let mut decoder =
            crate::MetalBatchDecoder::system_default_with_options(BatchDecodeOptions {
                layout: BatchLayout::Nhwc,
                ..BatchDecodeOptions::default()
            })
            .expect("Metal batch decoder");
        let prepared = decoder
            .prepare(distinct_batch(width, height, count))
            .expect("prepare stage-profile batch");

        let warm_until = Instant::now() + WARM_UP;
        while Instant::now() < warm_until {
            for limit in LIMITS {
                decode_gpu_seconds(&mut decoder, &prepared, limit);
            }
        }
        let mut samples = [(); 4].map(|()| Vec::with_capacity(SAMPLES));
        for _ in 0..SAMPLES {
            for (slot, limit) in samples.iter_mut().zip(LIMITS) {
                slot.push(decode_gpu_seconds(&mut decoder, &prepared, limit));
            }
        }
        let [vlc, tier1, idwt, full] = samples.map(|mut slot| median_ms(&mut slot));
        println!(
            "{width}x{height} x{count} GPU ms (median of {SAMPLES}): full {full:.3} | \
             zero-fill+VLC {vlc:.3}, MagSgn {:.3}, IDWT {:.3}, ICT+store {:.3}",
            tier1 - vlc,
            idwt - tier1,
            full - idwt,
        );
    }
}

/// IDWT and full GPU time with the per-image 9/7 IDWT route forced (the
/// pre-P36 behaviour above 20 MiB) and with the batched route, round-robin.
#[test]
#[ignore = "GPU A/B probe; run explicitly with --include-ignored --nocapture"]
fn batched_idwt_route_profile() {
    if !super::runtime::should_run_metal_runtime() {
        return;
    }
    let _restore = FullStagesOnDrop;
    let routes = [true, false];
    for (width, height, count) in [(1024, 1024, 16), (640, 480, 16)] {
        let mut decoder =
            crate::MetalBatchDecoder::system_default_with_options(BatchDecodeOptions {
                layout: BatchLayout::Nhwc,
                ..BatchDecodeOptions::default()
            })
            .expect("Metal batch decoder");
        let prepared = decoder
            .prepare(distinct_batch(width, height, count))
            .expect("prepare IDWT probe batch");
        let mut sample = |per_image: bool, stage: DecodeStageLimit| {
            force_per_image_idwt_for_test(per_image);
            decode_gpu_seconds(&mut decoder, &prepared, stage)
        };
        let warm_until = Instant::now() + WARM_UP;
        while Instant::now() < warm_until {
            for per_image in routes {
                sample(per_image, DecodeStageLimit::Full);
            }
        }
        let mut samples = [(); 6].map(|()| Vec::with_capacity(SAMPLES));
        for _ in 0..SAMPLES {
            for (index, per_image) in routes.into_iter().enumerate() {
                samples[index * 3].push(sample(per_image, DecodeStageLimit::Tier1));
                samples[index * 3 + 1].push(sample(per_image, DecodeStageLimit::Idwt));
                samples[index * 3 + 2].push(sample(per_image, DecodeStageLimit::Full));
            }
        }
        let [t1_a, idwt_a, full_a, t1_b, idwt_b, full_b] =
            samples.map(|mut slot| median_ms(&mut slot));
        println!(
            "{width}x{height} x{count} GPU ms: per-image IDWT {:.3} full {full_a:.3} | \
             batched IDWT {:.3} full {full_b:.3}",
            idwt_a - t1_a,
            idwt_b - t1_b,
        );
    }
}
