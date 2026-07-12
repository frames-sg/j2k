// SPDX-License-Identifier: MIT OR Apache-2.0

pub(super) fn generated_rgb_jpeg(
    subsampling: j2k_jpeg::JpegSubsampling,
    width: u32,
    height: u32,
    restart_interval: Option<u16>,
) -> Vec<u8> {
    let rgb = j2k_test_support::gpu_bench_rgb8(width, height);
    j2k_jpeg::encode_jpeg_baseline(
        j2k_jpeg::JpegSamples::Rgb8 {
            data: &rgb,
            width,
            height,
        },
        j2k_jpeg::JpegEncodeOptions {
            quality: 90,
            subsampling,
            restart_interval,
            backend: j2k_jpeg::JpegBackend::Cpu,
        },
    )
    .expect("generated JPEG")
    .data
}
