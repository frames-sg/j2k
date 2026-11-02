// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared driver for full-image route parity integration tests.

use j2k::{CpuDecodeParallelism, J2kDecoder, J2kScratchPool};
use j2k_core::{Downscale, PixelFormat};
use j2k_native::DecodeErrorClass;

pub(super) const PADDING: u8 = 0xab;

/// Why a decode failed: the class of the native decode error behind it, or
/// the facade's own error where there is none.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Failure {
    Native(DecodeErrorClass),
    Facade(String),
}

impl Failure {
    fn of(error: &j2k::J2kError) -> Self {
        let mut next: Option<&(dyn std::error::Error + 'static)> = Some(error);
        while let Some(current) = next {
            if let Some(native) = current.downcast_ref::<j2k_native::DecodeError>() {
                return Self::Native(native.classify());
            }
            next = current.source();
        }
        Self::Facade(error.to_string())
    }
}

pub(super) fn decoder(
    bytes: &[u8],
    planes_budget: Option<usize>,
    parallelism: CpuDecodeParallelism,
) -> J2kDecoder<'_> {
    let mut decoder = J2kDecoder::new(bytes).expect("decoder");
    decoder.set_cpu_decode_parallelism(parallelism);
    if let Some(bytes) = planes_budget {
        decoder.set_full_decode_planes_budget(bytes);
    }
    decoder
}

/// The image at `scale` in a buffer padded past each row, or the native
/// error class of the failure, from a decoder whose planes budget is
/// `planes_budget` when one is given.
pub(super) fn decode(
    bytes: &[u8],
    planes_budget: Option<usize>,
    parallelism: CpuDecodeParallelism,
    fmt: PixelFormat,
    scale: Downscale,
    padding: usize,
) -> Result<Vec<u8>, Failure> {
    let mut decoder = decoder(bytes, planes_budget, parallelism);
    let (width, height) = decoder.info().dimensions;
    let divisor = scale.denominator();
    let (width, height) = (width.div_ceil(divisor), height.div_ceil(divisor));
    let row = width as usize * fmt.bytes_per_pixel();
    let stride = row + padding;
    let mut out = vec![PADDING; stride * height as usize];
    let result = match scale {
        Downscale::None => decoder.decode_into(&mut out, stride, fmt),
        _ => decoder.decode_scaled_into(&mut J2kScratchPool::new(), &mut out, stride, fmt, scale),
    };
    result.map_err(|error| Failure::of(&error))?;
    for line in out.chunks(stride) {
        assert!(
            line[row..].iter().all(|&byte| byte == PADDING),
            "stride padding untouched"
        );
    }
    Ok(out)
}
