#![no_main]

//! Decode input as a codestream that may be cut off, and check that the
//! reported reduction agrees with what decoding accepts.

use j2k::{DecodeSettings, J2kDecoder, J2kError, J2kScratchPool, PixelFormat, Rect};
use libfuzzer_sys::fuzz_target;

const MAX_OUTPUT_BYTES: usize = 1 << 20;

fn decode_at(
    decoder: &mut J2kDecoder<'_>,
    format: PixelFormat,
    area: Rect,
    reduction: u8,
) -> Option<Result<(), J2kError>> {
    let plan = match decoder.region_decode_plan(format, area, reduction) {
        Ok(plan) => plan,
        Err(error) => return Some(Err(error)),
    };
    if plan.buffer_bytes == 0 || plan.buffer_bytes > MAX_OUTPUT_BYTES {
        return None;
    }
    let mut out = vec![0_u8; plan.buffer_bytes];
    Some(
        decoder
            .decode_region_scaled_pow2_into(
                &mut J2kScratchPool::new(),
                &mut out,
                plan.row_bytes,
                format,
                area,
                reduction,
            )
            .map(drop),
    )
}

fuzz_target!(|data: &[u8]| {
    let settings = DecodeSettings::strict().with_partial_codestream(true);
    let Ok(mut decoder) = J2kDecoder::new_with_settings(data, settings) else {
        return;
    };
    let (width, height) = decoder.info().dimensions;
    if width == 0 || height == 0 {
        return;
    }
    let Ok(report) = decoder.codestream_completeness() else {
        return;
    };
    let Some(reduction) = report.min_exact_reduction() else {
        return;
    };
    let format = if decoder.info().components == 1 {
        PixelFormat::Gray8
    } else {
        PixelFormat::Rgb8
    };
    let area = Rect {
        x: 0,
        y: 0,
        w: width,
        h: height,
    };
    // Other failures (unsupported coding, malformed entropy data, limits)
    // are allowed; a refusal for missing data at the reported reduction is
    // not, and neither is success one level shallower.
    if let Some(result) = decode_at(&mut decoder, format, area, reduction) {
        assert!(
            !matches!(result, Err(J2kError::IncompleteCodestream)),
            "reported reduction {reduction} was refused for missing data"
        );
    }
    if reduction > 0 {
        if let Some(result) = decode_at(&mut decoder, format, area, reduction - 1) {
            assert!(
                result.is_err(),
                "reduction {} decoded below the reported minimum",
                reduction - 1
            );
        }
    }
});
