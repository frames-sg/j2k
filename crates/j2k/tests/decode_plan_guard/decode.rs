// SPDX-License-Identifier: MIT OR Apache-2.0

//! One guarded decode: the request, output, parallelism and planes budget of
//! a case, the peak heap it holds, and the hash of what it wrote.

use std::convert::Infallible;

use j2k::{
    CpuDecodeParallelism, J2kDecodedNativeComponents, J2kDecoder, J2kError, J2kRowDecodeOptions,
    J2kScratchPool,
};
use j2k_core::{DecodeRowsError, Downscale, PixelFormat, Rect, RowSink};

/// Rows a bounded rows decode holds at once: several stripes per image.
const STRIPE_ROWS: u32 = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum Request {
    Full,
    Region,
    Scaled(Downscale),
    RegionScaled(Downscale),
    /// The bounded rows API, in the image's own 8- or 16-bit row layout.
    Rows,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum Output {
    Packed(PixelFormat),
    Native,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Budget {
    Default,
    /// The decoder's test hook holding the full-image planes to these bytes.
    Planes(usize),
}

/// What a decode wrote, or the class of the error it returned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Outcome {
    Bytes(u64),
    Error(String),
}

impl core::fmt::Display for Outcome {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Bytes(hash) => write!(f, "ok:{hash:016x}"),
            Self::Error(class) => write!(f, "err:{class}"),
        }
    }
}

pub(super) struct Measured {
    pub(super) outcome: Outcome,
    /// Most heap bytes live at once during the decode, above those live
    /// just before it (the decoder and the caller's output buffer).
    pub(super) peak: usize,
    /// Bytes of the output: the caller's buffer, or the native planes the
    /// decode returns (which `peak` includes).
    pub(super) output: usize,
}

/// FNV-1a, 64 bits.
#[derive(Clone, Copy)]
pub(super) struct Hash(u64);

impl Hash {
    pub(super) const fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }

    pub(super) fn update(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 = (self.0 ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3);
        }
    }

    pub(super) const fn finish(self) -> u64 {
        self.0
    }
}

pub(super) fn hash(bytes: &[u8]) -> u64 {
    let mut hash = Hash::new();
    hash.update(bytes);
    hash.finish()
}

/// The hash of native planes: each plane's geometry, precision and bytes.
pub(super) fn native_hash(planes: &[NativePlane]) -> u64 {
    let mut hash = Hash::new();
    for plane in planes {
        hash.update(&plane.dimensions.0.to_le_bytes());
        hash.update(&plane.dimensions.1.to_le_bytes());
        hash.update(&[
            plane.bit_depth,
            u8::from(plane.signed),
            plane.bytes_per_sample,
        ]);
        hash.update(&plane.data);
    }
    hash.finish()
}

/// One native component plane, owned, so a reference can be cropped.
pub(super) struct NativePlane {
    pub(super) dimensions: (u32, u32),
    pub(super) bit_depth: u8,
    pub(super) signed: bool,
    pub(super) bytes_per_sample: u8,
    pub(super) data: Vec<u8>,
}

pub(super) fn native_planes(decoded: &J2kDecodedNativeComponents) -> Vec<NativePlane> {
    decoded
        .planes()
        .iter()
        .map(|plane| NativePlane {
            dimensions: plane.dimensions(),
            bit_depth: plane.bit_depth(),
            signed: plane.signed(),
            bytes_per_sample: plane.bytes_per_sample(),
            data: plane.data().to_vec(),
        })
        .collect()
}

/// The variant of `error`, without the sizes and messages it carries.
pub(super) fn error_class(error: &J2kError) -> String {
    let debug = format!("{error:?}");
    debug
        .split(|c: char| !c.is_ascii_alphanumeric())
        .next()
        .unwrap_or_default()
        .to_string()
}

/// Bytes per row and rows of the output of `request` on an image of
/// `size`, at `scale`'s resolution for scaled requests.
pub(super) fn output_rect(request: Request, size: (u32, u32), region: Rect) -> Rect {
    let scaled = |scale: Downscale| {
        let divisor = scale.denominator();
        Rect::full((size.0.div_ceil(divisor), size.1.div_ceil(divisor)))
    };
    match request {
        Request::Full | Request::Rows => Rect::full(size),
        Request::Region => region,
        Request::Scaled(scale) => scaled(scale),
        Request::RegionScaled(scale) => region.scaled_covering(scale),
    }
}

/// The 8- or 16-bit row layout of the bounded rows API for `components`.
pub(super) const fn row_format(components: u16, sixteen: bool) -> Option<PixelFormat> {
    Some(match (components, sixteen) {
        (1, false) => PixelFormat::Gray8,
        (3, false) => PixelFormat::Rgb8,
        (4, false) => PixelFormat::Rgba8,
        (1, true) => PixelFormat::Gray16,
        (3, true) => PixelFormat::Rgb16,
        (4, true) => PixelFormat::Rgba16,
        _ => return None,
    })
}

/// A row sink that hashes the rows in order, as little-endian bytes.
struct HashRows {
    hash: Hash,
    next: u32,
}

impl RowSink<u8> for HashRows {
    type Error = Infallible;

    fn write_row(&mut self, y: u32, row: &[u8]) -> Result<(), Infallible> {
        assert_eq!(y, self.next, "rows arrive in order");
        self.next += 1;
        self.hash.update(row);
        Ok(())
    }
}

impl RowSink<u16> for HashRows {
    type Error = Infallible;

    fn write_row(&mut self, y: u32, row: &[u16]) -> Result<(), Infallible> {
        assert_eq!(y, self.next, "rows arrive in order");
        self.next += 1;
        for sample in row {
            self.hash.update(&sample.to_le_bytes());
        }
        Ok(())
    }
}

fn rows_outcome<T>(
    result: Result<T, DecodeRowsError<J2kError, Infallible>>,
    sink: &HashRows,
) -> Outcome {
    match result {
        Ok(_) => Outcome::Bytes(sink.hash.finish()),
        Err(DecodeRowsError::Decode(error)) => Outcome::Error(error_class(&error)),
        Err(DecodeRowsError::Sink(never)) => match never {},
    }
}

fn packed_outcome<T>(result: Result<T, J2kError>, out: &[u8]) -> Outcome {
    match result {
        Ok(_) => Outcome::Bytes(hash(out)),
        Err(error) => Outcome::Error(error_class(&error)),
    }
}

/// One decode of `bytes`, on the current thread's pool, measured from the
/// moment the decoder exists and the output buffer is allocated.
pub(super) struct Decode<'a> {
    pub(super) bytes: &'a [u8],
    pub(super) size: (u32, u32),
    pub(super) region: Rect,
    pub(super) request: Request,
    pub(super) output: Output,
    pub(super) parallelism: CpuDecodeParallelism,
    pub(super) budget: Budget,
}

impl Decode<'_> {
    pub(super) fn run(&self) -> Measured {
        let mut decoder = match J2kDecoder::new(self.bytes) {
            Ok(decoder) => decoder,
            Err(error) => {
                return Measured {
                    outcome: Outcome::Error(error_class(&error)),
                    peak: 0,
                    output: 0,
                }
            }
        };
        decoder.set_cpu_decode_parallelism(self.parallelism);
        if let Budget::Planes(bytes) = self.budget {
            decoder.set_full_decode_planes_budget(bytes);
        }
        match self.output {
            Output::Native => self.native(&mut decoder),
            Output::Packed(fmt) if self.request == Request::Rows => Self::rows(&mut decoder, fmt),
            Output::Packed(fmt) => self.packed(&mut decoder, fmt),
        }
    }

    fn packed(&self, decoder: &mut J2kDecoder<'_>, fmt: PixelFormat) -> Measured {
        let rect = output_rect(self.request, self.size, self.region);
        let stride = rect.w as usize * fmt.bytes_per_pixel();
        let mut out = vec![0_u8; stride * rect.h as usize];
        let mut pool = J2kScratchPool::new();
        let base = j2k_alloc_probe::reset_peak();
        let result = match self.request {
            Request::Full => decoder.decode_into(&mut out, stride, fmt),
            Request::Region => {
                decoder.decode_region_into(&mut pool, &mut out, stride, fmt, self.region)
            }
            Request::Scaled(scale) => {
                decoder.decode_scaled_into(&mut pool, &mut out, stride, fmt, scale)
            }
            Request::RegionScaled(scale) => decoder.decode_region_scaled_into(
                &mut pool,
                &mut out,
                stride,
                fmt,
                self.region,
                scale,
            ),
            Request::Rows => unreachable!("rows decode through a sink"),
        };
        let peak = j2k_alloc_probe::peak_bytes().saturating_sub(base);
        Measured {
            outcome: packed_outcome(result, &out),
            peak,
            output: out.len(),
        }
    }

    fn rows(decoder: &mut J2kDecoder<'_>, fmt: PixelFormat) -> Measured {
        let mut sink = HashRows {
            hash: Hash::new(),
            next: 0,
        };
        let options = J2kRowDecodeOptions::new(STRIPE_ROWS);
        let base = j2k_alloc_probe::reset_peak();
        let outcome = if fmt.bytes_per_pixel() == fmt.channels() {
            let result = decoder.decode_rows_u8_bounded(&mut sink, options);
            rows_outcome(result, &sink)
        } else {
            let result = decoder.decode_rows_u16_bounded(&mut sink, options);
            rows_outcome(result, &sink)
        };
        let peak = j2k_alloc_probe::peak_bytes().saturating_sub(base);
        Measured {
            outcome,
            peak,
            output: 0,
        }
    }

    fn native(&self, decoder: &mut J2kDecoder<'_>) -> Measured {
        let base = j2k_alloc_probe::reset_peak();
        let result = match self.request {
            Request::Full => decoder.decode_native_components(),
            Request::Region => decoder.decode_native_region_components(self.region),
            _ => unreachable!("native components decode the full image or a region"),
        };
        let peak = j2k_alloc_probe::peak_bytes().saturating_sub(base);
        match result {
            Ok(components) => {
                let planes = native_planes(&components);
                Measured {
                    outcome: Outcome::Bytes(native_hash(&planes)),
                    peak,
                    output: planes.iter().map(|plane| plane.data.len()).sum(),
                }
            }
            Err(error) => Measured {
                outcome: Outcome::Error(error_class(&error)),
                peak,
                output: 0,
            },
        }
    }
}
