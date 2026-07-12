// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    encode_precomputed_htj2k_53_with_accelerator_and_max_host_bytes,
    encode_precomputed_htj2k_97_with_accelerator_and_max_host_bytes, map_encode_error,
    record_encode_dispatch_delta, Instant, J2kEncodeStageAccelerator, JpegToHtj2kError,
    JpegToHtj2kOptions, PrecomputedComponentBatch, PrecomputedHtj2k53Image,
    PrecomputedHtj2k97Image, TranscodeTimingReport,
};

pub(super) fn encode_component_batch<E: J2kEncodeStageAccelerator>(
    width: u32,
    height: u32,
    precomputed_components: PrecomputedComponentBatch,
    options: &JpegToHtj2kOptions,
    encode_accelerator: &mut E,
    timings: &mut TranscodeTimingReport,
    max_host_bytes: usize,
) -> Result<(Vec<u8>, u128), JpegToHtj2kError> {
    let encode_start = Instant::now();
    let encode_dispatch_before = encode_accelerator.dispatch_report();
    let native_encode_options = options.encode_options.to_native()?;
    let codestream = match precomputed_components {
        PrecomputedComponentBatch::Dwt53(components) => {
            let precomputed = PrecomputedHtj2k53Image {
                width,
                height,
                bit_depth: 8,
                signed: false,
                components,
            };
            encode_precomputed_htj2k_53_with_accelerator_and_max_host_bytes(
                &precomputed,
                &native_encode_options,
                encode_accelerator,
                max_host_bytes,
            )
            .map_err(map_encode_error)?
        }
        PrecomputedComponentBatch::Dwt97(components) => {
            let precomputed = PrecomputedHtj2k97Image {
                width,
                height,
                bit_depth: 8,
                signed: false,
                components,
            };
            encode_precomputed_htj2k_97_with_accelerator_and_max_host_bytes(
                &precomputed,
                &native_encode_options,
                encode_accelerator,
                max_host_bytes,
            )
            .map_err(map_encode_error)?
        }
    };
    record_encode_dispatch_delta(
        timings,
        encode_dispatch_before,
        encode_accelerator.dispatch_report(),
    );
    let encode_us = encode_start.elapsed().as_micros();
    timings.htj2k_encode_us = encode_us;
    Ok((codestream, encode_us))
}
