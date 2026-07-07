#![no_main]

use j2k::{J2kDecoder, J2kScratchPool, PixelFormat};
use j2k_transcode::{jpeg_to_htj2k, JpegToHtj2kOptions};
use libfuzzer_sys::fuzz_target;

const MAX_INPUT_BYTES: usize = 256 * 1024;
const MAX_OUTPUT_BYTES: usize = 4 * 1024 * 1024;
const MAX_DECODE_BYTES: usize = 1 << 20;

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_INPUT_BYTES {
        return;
    }

    let Ok(encoded) = jpeg_to_htj2k(data, &JpegToHtj2kOptions::lossless_53()) else {
        return;
    };

    if encoded.codestream.len() > MAX_OUTPUT_BYTES {
        return;
    }

    let mut decoder = match J2kDecoder::new(&encoded.codestream) {
        Ok(decoder) => decoder,
        Err(err) => panic!("transcoded HTJ2K codestream must parse: {err}"),
    };

    let info = decoder.info();
    let dims = info.dimensions;
    let fmt = if info.components == 1 {
        PixelFormat::Gray8
    } else {
        PixelFormat::Rgb8
    };
    let Some((stride, len)) = output_geometry(dims, fmt) else {
        return;
    };
    if len == 0 || len > MAX_DECODE_BYTES {
        return;
    }

    let mut out = vec![0_u8; len];
    let mut pool = J2kScratchPool::new();
    let _ = decoder.decode_into_with_scratch(&mut pool, &mut out, stride, fmt);
});

fn output_geometry(dims: (u32, u32), fmt: PixelFormat) -> Option<(usize, usize)> {
    let stride = dims.0 as usize * fmt.bytes_per_pixel();
    let len = stride.checked_mul(dims.1 as usize)?;
    Some((stride, len))
}
