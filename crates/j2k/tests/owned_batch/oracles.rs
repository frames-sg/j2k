// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k::{BatchLayout, CpuBatchSamples, Downscale, Rect};

pub(super) fn native_request_oracle(
    image: &j2k::PreparedImage,
    layout: BatchLayout,
) -> CpuBatchSamples {
    let plan = image.plan();
    let target_resolution = (plan.scale() != Downscale::None).then_some((
        plan.source_dims().0.div_ceil(plan.scale().denominator()),
        plan.source_dims().1.div_ceil(plan.scale().denominator()),
    ));
    let decoded = j2k_native::Image::new(
        image.bytes(),
        &j2k_native::DecodeSettings {
            target_resolution,
            ..j2k_native::DecodeSettings::strict()
        },
    )
    .expect("parse broad native RGBA oracle");
    let output = plan.output_rect();
    let raw = if output == Rect::full((decoded.width(), decoded.height())) {
        decoded.decode_native().expect("decode broad native oracle")
    } else {
        decoded
            .decode_native_region((output.x, output.y, output.w, output.h))
            .expect("decode broad native region oracle")
    };
    let pixel_count = output.w as usize * output.h as usize;
    let channels = raw.num_components as usize;
    if raw.signed {
        CpuBatchSamples::I16(apply_batch_layout(
            raw.data
                .chunks_exact(2)
                .map(|sample| i16::from_le_bytes([sample[0], sample[1]]))
                .collect(),
            pixel_count,
            channels,
            layout,
        ))
    } else if raw.bit_depth <= 8 {
        CpuBatchSamples::U8(apply_batch_layout(raw.data, pixel_count, channels, layout))
    } else {
        CpuBatchSamples::U16(apply_batch_layout(
            raw.data
                .chunks_exact(2)
                .map(|sample| u16::from_le_bytes([sample[0], sample[1]]))
                .collect(),
            pixel_count,
            channels,
            layout,
        ))
    }
}

pub(super) fn apply_batch_layout<T: Copy>(
    samples: Vec<T>,
    pixel_count: usize,
    channels: usize,
    layout: BatchLayout,
) -> Vec<T> {
    match layout {
        BatchLayout::Nhwc => samples,
        BatchLayout::Nchw => (0..channels)
            .flat_map(|channel| {
                let samples = &samples;
                (0..pixel_count).map(move |pixel| samples[pixel * channels + channel])
            })
            .collect(),
        _ => panic!("unsupported test batch layout"),
    }
}

pub(super) fn decoded_samples_for_source(
    result: &j2k::CpuBatchDecodeResult,
    source_index: usize,
) -> CpuBatchSamples {
    let group = result
        .groups()
        .iter()
        .find(|group| group.source_indices().contains(&source_index))
        .expect("decoded source group");
    let image_position = group
        .source_indices()
        .iter()
        .position(|index| *index == source_index)
        .expect("source position inside decoded group");
    let image_count = group.source_indices().len();
    match group.samples() {
        CpuBatchSamples::U8(samples) => {
            let samples_per_image = samples.len() / image_count;
            let start = image_position * samples_per_image;
            CpuBatchSamples::U8(samples[start..start + samples_per_image].to_vec())
        }
        CpuBatchSamples::U16(samples) => {
            let samples_per_image = samples.len() / image_count;
            let start = image_position * samples_per_image;
            CpuBatchSamples::U16(samples[start..start + samples_per_image].to_vec())
        }
        CpuBatchSamples::I16(samples) => {
            let samples_per_image = samples.len() / image_count;
            let start = image_position * samples_per_image;
            CpuBatchSamples::I16(samples[start..start + samples_per_image].to_vec())
        }
        _ => panic!("unsupported test batch sample type"),
    }
}
