// SPDX-License-Identifier: MIT OR Apache-2.0

pub(crate) mod classification;
mod compare_only;
mod full_frame_env;
mod full_frame_policy;
mod input_loader;
mod inputs;
mod libjpeg_turbo;
mod libjpeg_turbo_extended;
mod null_sink;
mod row_stream_env;
mod row_stream_policy;
mod shared_drivers;
mod turbo_basic;
mod turbo_compare;

pub(crate) use self::classification::DecodeMode;
pub(crate) use self::compare_only::{
    j2k_decode_reused, j2k_decode_with_scratch, output_geometry,
    J2KTileBatchRegionScaledRgbSession, J2KTileBatchRgbOutputBuffers, J2KTileBatchRgbScratch,
    J2KTileBatchRgbSession, J2KTileBatchScaledRgbSession,
};
pub(crate) use self::full_frame_env::should_compare_full_frame;
pub(crate) use self::inputs::load_bench_inputs;
pub(crate) use self::libjpeg_turbo::TurboJpegDecoder;
pub(crate) use self::row_stream_env::should_bench_decode_rows_rgb;
pub(crate) use self::shared_drivers::{
    centered_roi, j2k_decode, j2k_decode_region, j2k_decode_region_scaled, j2k_decode_rows,
    j2k_decode_scaled, j2k_decode_tile_batch_region_scaled, j2k_decode_tile_batch_scaled,
    j2k_inspect, jpeg_decoder_decode, jpeg_decoder_decode_batch_region_scaled,
    jpeg_decoder_decode_batch_scaled, jpeg_decoder_decode_region,
    jpeg_decoder_decode_region_scaled, jpeg_decoder_decode_scaled, jpeg_decoder_inspect,
    zune_decode, zune_decode_batch_region_scaled, zune_decode_batch_scaled, zune_decode_region,
    zune_decode_region_scaled, zune_decode_scaled, zune_inspect,
};
pub(crate) use self::turbo_basic::{libjpeg_turbo_available, libjpeg_turbo_decode_batch};
pub(crate) use self::turbo_compare::{
    libjpeg_turbo_decode, libjpeg_turbo_decode_batch_region_scaled,
    libjpeg_turbo_decode_batch_scaled, libjpeg_turbo_decode_region,
    libjpeg_turbo_decode_region_scaled, libjpeg_turbo_decode_scaled, libjpeg_turbo_inspect,
    TurboJpegBatchRgbOutputBuffers,
};
