#![no_main]

use j2k::{Downscale, J2kDecoder, J2kScratchPool, PixelFormat, Rect};
use libfuzzer_sys::fuzz_target;

const MAX_OUTPUT_BYTES: usize = 1 << 20;

fuzz_target!(|data: &[u8]| {
    let Ok(mut decoder) = J2kDecoder::new(data) else {
        return;
    };

    let info = decoder.info();
    let dims = info.dimensions;
    let components = info.components;
    if dims.0 == 0 || dims.1 == 0 {
        return;
    }

    let scale = downscale_from(data.first().copied().unwrap_or_default());
    let fmt = match components {
        1 => PixelFormat::Gray8,
        4 => PixelFormat::Rgba8,
        _ => PixelFormat::Rgb8,
    };

    let full_frame = data.get(1).copied().unwrap_or_default() & 1 == 0;
    let roi = bounded_rect(dims, data.get(2..).unwrap_or_default());
    let out_dims = if full_frame {
        scaled_dims(dims, scale)
    } else {
        let scaled = roi.scaled_covering(scale);
        (scaled.w, scaled.h)
    };

    let Some((stride, len)) = output_geometry(out_dims, fmt) else {
        return;
    };
    if len == 0 || len > MAX_OUTPUT_BYTES {
        return;
    }

    let mut out = vec![0_u8; len];
    let mut pool = J2kScratchPool::new();
    if full_frame {
        let _ = decoder.decode_scaled_into(&mut pool, &mut out, stride, fmt, scale);
    } else {
        let _ = decoder.decode_region_scaled_into(&mut pool, &mut out, stride, fmt, roi, scale);
    }
});

fn downscale_from(value: u8) -> Downscale {
    match value & 3 {
        0 => Downscale::None,
        1 => Downscale::Half,
        2 => Downscale::Quarter,
        _ => Downscale::Eighth,
    }
}

fn bounded_rect(dims: (u32, u32), data: &[u8]) -> Rect {
    let x = bounded_u32(data.first().copied().unwrap_or_default(), dims.0);
    let y = bounded_u32(data.get(1).copied().unwrap_or_default(), dims.1);
    let max_w = dims.0.saturating_sub(x).max(1);
    let max_h = dims.1.saturating_sub(y).max(1);
    let w = 1 + bounded_u32(data.get(2).copied().unwrap_or_default(), max_w);
    let h = 1 + bounded_u32(data.get(3).copied().unwrap_or_default(), max_h);
    Rect { x, y, w, h }
}

fn bounded_u32(value: u8, max_exclusive: u32) -> u32 {
    if max_exclusive == 0 {
        0
    } else {
        u32::from(value) % max_exclusive
    }
}

fn scaled_dims(dims: (u32, u32), scale: Downscale) -> (u32, u32) {
    let denom = scale.denominator();
    (dims.0.div_ceil(denom), dims.1.div_ceil(denom))
}

fn output_geometry(dims: (u32, u32), fmt: PixelFormat) -> Option<(usize, usize)> {
    let stride = dims.0 as usize * fmt.bytes_per_pixel();
    let len = stride.checked_mul(dims.1 as usize)?;
    Some((stride, len))
}
