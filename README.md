# J2K: JPEG 2000 and HTJ2K for Rust

[![crates.io](https://img.shields.io/crates/v/j2k.svg)](https://crates.io/crates/j2k)
[![docs.rs](https://img.shields.io/docsrs/j2k)](https://docs.rs/j2k)
[![Full validation](https://github.com/frames-sg/j2k/actions/workflows/full-validation.yml/badge.svg?branch=main&event=push)](https://github.com/frames-sg/j2k/actions/workflows/full-validation.yml?query=branch%3Amain+event%3Apush)
[![downloads](https://img.shields.io/crates/d/j2k.svg)](https://crates.io/crates/j2k)
[![license](https://img.shields.io/crates/l/j2k.svg)](#license)

J2K reads and writes JPEG 2000 images. It is written in Rust, runs on the CPU
on Linux, macOS, and Windows, and can use an NVIDIA GPU (CUDA) or an Apple GPU
(Metal) for the workloads where that was measured to be faster.

JPEG 2000 is the image format used in medical imaging (DICOM), digital
pathology slide scans, satellite and aerial imagery, and archives. Its files
are often very large and split into tiles. HTJ2K (High-Throughput JPEG 2000)
is a newer version of the format with a much faster block coder. J2K handles
both.

```bash
cargo add j2k
```

## What it does

- **Decode** a whole image, one region, a lower-resolution version, or a
  single tile. Region and low-resolution decodes skip the work for pixels
  outside the request, so pulling a thumbnail or a viewport out of a
  multi-gigabyte slide doesn't decode the whole image.
- **Decode many images at once.** The batch API decodes in parallel and
  returns results grouped by shape, ready to turn into tensors.
- **Encode** lossless or lossy JPEG 2000 and HTJ2K. Lossy encode accepts a
  1–100 quality setting (HTJ2K), a target file size, a bits-per-pixel
  rate, or a target PSNR.
- **Convert JPEG 2000 to lossless HTJ2K** with identical pixels. Reversible
  5/3 grayscale and RGB images with unsigned 8- or 16-bit samples are re-coded
  without decoding to pixels; other inputs are decoded and re-encoded
  losslessly.
- **Convert baseline JPEG to HTJ2K** from the JPEG's DCT coefficients, without
  decoding to pixels (`j2k-transcode`).
- **Read and write JP2 and JPH files** as well as raw codestreams.

It does not support JPEG 2000 Part 2 (JPX) extensions.

## Examples

Decode a file to 8-bit RGB (or gray or RGBA, depending on the image):

```rust
use j2k::J2kDecoder;

let bytes = std::fs::read("image.jp2")?;
let mut decoder = J2kDecoder::new(&bytes)?;
let (width, height) = decoder.info().dimensions;

let image = decoder.decode_srgb8()?;
println!("{width}x{height}, {:?}, {} bytes", image.layout(), image.data().len());
```

Decode a 1024 x 1024 area at quarter resolution without decoding the rest of
the image:

```rust
use j2k::{J2kDecoder, J2kScratchPool, PixelFormat, Rect};

let mut decoder = J2kDecoder::new(&bytes)?;
let mut scratch = J2kScratchPool::new();

let area = Rect { x: 4096, y: 8192, w: 1024, h: 1024 }; // full-resolution pixels
let plan = decoder.region_decode_plan(PixelFormat::Rgb8, area, 2)?;
let mut pixels = vec![0_u8; plan.buffer_bytes];
decoder.decode_region_scaled_pow2_into(
    &mut scratch,
    &mut pixels,
    plan.row_bytes,
    PixelFormat::Rgb8,
    area,
    plan.reduction_levels,
)?;
```

Related calls:

- `plan.buffer_len(stride)` sizes a padded buffer, and
  `decoder.max_reduction_levels()` is the deepest reduction every tile and
  component supports. A reduction the image does not have, or a region with no
  samples at that reduction, is an error.
- `srgb8_region_decode_plan` and `decode_region_srgb8_at_reduction` return
  8-bit sRGB with straight alpha, applying palettes and ICC profiles.
- `PixelFormat::Gray16` and `Rgb16` give 16-bit output, and
  `decode_native_components()` gives each component at its original bit depth.
- The `*_with_control` methods take a `J2kDecodeControl`: a working-memory
  limit, an output-size limit, and a cancellation token for one operation.
- `PreparedImageOperation` with `CpuBatchDecoder::decode_operations` decodes
  many regions of one prepared image without parsing it again.

Supported combinations and limits are in
[docs/public-support.md](docs/public-support.md#region-planning).

Encode 8-bit RGB pixels as lossless HTJ2K:

```rust
use j2k::{encode_j2k_lossless, J2kBlockCodingMode, J2kLosslessEncodeOptions, J2kLosslessSamples};

let samples = J2kLosslessSamples::new(&rgb, width, height, 3, 8, false)?;
let options = J2kLosslessEncodeOptions::default()
    .with_block_coding_mode(J2kBlockCodingMode::HighThroughput);
let encoded = encode_j2k_lossless(samples, &options)?;
std::fs::write("image.j2c", &encoded.codestream)?;
```

`encoded.codestream` is a raw codestream. Use `wrap_j2k_codestream` to put it
in a `.jp2` or `.jph` file. By default the encoder decodes its own output and
checks it against the input before returning.

More runnable examples are in `crates/*/examples/`, for example
`cargo run -p j2k --example decode_generated`.

## Command-line tool

```bash
cargo install j2k-cli
j2k inspect image.jp2       # size, color space, bit depth, components, levels, tiles
j2k transcode input.jpg output.j2k --htj2k --lossless-53   # JPEG to HTJ2K
j2k decode -i image.jp2 -d 0,0,1024,1024 -r 2 -o tile.png  # opj_decompress options
```

`j2k decode` takes OpenJPEG `opj_decompress` options with the same meanings,
and the binary run as `opj_decompress` (for example through a symlink) behaves
as that tool, so image servers that call `opj_decompress` can run j2k. See
[`crates/j2k-cli/README.md`](crates/j2k-cli/README.md) for the options and the
differences. To encode images, use the library.

[`docs/iiif.md`](docs/iiif.md) shows how to serve IIIF Image API 3.0 requests
and `info.json` with the Rust, C, Python and command-line interfaces, and how
to run j2k behind Cantaloupe.

## C and C++

[`j2k-c`](crates/j2k-c/README.md) is a CPU C ABI (`j2k.h`), a
header-only C++17 wrapper (`j2k.hpp`), and a relocatable shared-library
package with CMake configuration; Cargo also emits a static archive. It offers
batch and explicit core whole-image decode routes, inspects images, keeps one
decoder across codestreams and reports their compressed transfer syntax, and
decodes full images, regions, and reduced resolutions into caller-owned buffers
as native U8, U16, or I16 samples. It streams full images row by row without a
whole-image buffer, and encodes losslessly or to a rate, size, or PSNR target.
Build a package with
`python3 crates/j2k-c/scripts/package.py --output dist/j2k-c`; consumers need
only a C/C++ compiler and either CMake or pkg-config.

## GPU support

On the GPU, the CPU still reads the file's structure, and the GPU does the
expensive parts: block decoding, the wavelet transform, color conversion, and
writing out pixels. Decoded pixels can stay in GPU memory, so they can go
straight into a GPU-side pipeline without a copy back to the CPU.

You choose the backend with `BackendRequest`:

- **`Auto`** (the default) uses the GPU only for image types and sizes where it
  was measured to be faster than the CPU and to give identical output.
  Everything else runs on the CPU. The routed cases are listed in
  [`xtask/routing-promotion-evidence.json`](xtask/routing-promotion-evidence.json):
  8-bit gray and RGB decodes of classic JPEG 2000 and HTJ2K above a minimum
  size per case on both backends, and some 8-bit encodes on Metal.
  [How a case gets routed](docs/benchmark-evidence.md#how-auto-routing-thresholds-are-chosen).
- **`Cuda`** or **`Metal`** forces the GPU. If the GPU path can't handle the
  image, you get an error.
- **`Cpu`** never touches the GPU.

A decode never retries on the CPU after a GPU error, whether the GPU was
requested or chosen by `Auto`. JPEG-to-HTJ2K transcode under `Auto` runs a
stage on the CPU when the GPU is unavailable or does not support the job.

GPU support lives in separate crates: `j2k-cuda` (Linux, needs an NVIDIA
driver; enable the `cuda-runtime` feature) and `j2k-metal` (macOS). See their
READMEs for which paths run on the GPU. Benchmark numbers are in
[docs/benchmark-evidence.md](docs/benchmark-evidence.md).

## Machine learning

- **PyTorch and NumPy.** The optional [`j2k` Python extension](crates/j2k-python/README.md),
  written in Rust, decodes to tensors of the image's own sample type: single
  images, batches that can mix 8-bit JPEG with JPEG 2000, regions, reduced
  resolutions, and selected wavelet bands. It also encodes. Built with CUDA
  it decodes into PyTorch-owned device memory; built with Metal it returns
  `mps` tensors on Apple Silicon.
- **Burn.** [`j2k-ml`](docs/j2k-ml.md) returns `U8`, `U16`, or `I16` tensors
  in NCHW or NHWC. `CpuBurnDecoder` (feature `cpu`) takes JPEG and JPEG 2000
  in one batch. `CudaDirectBurnDecoder` decodes into Burn-owned CUDA memory.
  `CudaUploadBurnDecoder` and `MetalUploadBurnDecoder` decode on the GPU and
  upload through host memory.
- **MPSGraph.** [`j2k-mpsgraph`](docs/j2k-mpsgraph.md) is an experimental
  Apple Silicon adapter that hands Metal-resident batches to MPSGraph without
  reading pixels back to the CPU.
- **Wavelet coefficients in Rust.** `J2kDecoder::decode_wavelets` returns
  exact 5/3 or dequantized 9/7 coefficients for the bands you select, and
  does not run the inverse transform.

## Which crate do I need?

Most code only needs `j2k`. Add a GPU crate if you want GPU decoding.

| You want to | Use |
| --- | --- |
| Read, write, or convert JPEG 2000 / HTJ2K | `j2k` |
| Decode with an NVIDIA GPU | `j2k` + `j2k-cuda` |
| Decode with an Apple GPU | `j2k` + `j2k-metal` |
| Read or write regular JPEG | `j2k-jpeg` |
| Decode batches of JPEGs into dense RGB8 tensors | `j2k_jpeg::JpegBatchDecoder` |
| Convert JPEG to HTJ2K | `j2k-transcode` (+ `j2k-transcode-cuda` or `j2k-transcode-metal`) |
| JPEG on the GPU | `j2k-jpeg-cuda` or `j2k-jpeg-metal` |
| Feed decoded images to Burn | `j2k-ml` |
| Feed decoded images to MPSGraph | `j2k-mpsgraph` |
| Feed pixels or wavelet bands to PyTorch/NumPy | Optional `j2k-python` extension (`j2k`) |
| Extract selected wavelet coefficients in Rust | `j2k::J2kDecoder::decode_wavelets` |
| Decompress Deflate, Zstd, or LZW tiles (as in TIFF) | `j2k-tilecodec` |
| A command-line tool | `j2k-cli` |
| Decode from C or C++ | `j2k-c` (C ABI, C++17 wrapper, CMake package) |

The remaining `j2k-*` crates are shared building blocks used by the crates
above. [docs/architecture.md](docs/architecture.md) shows how they fit
together, and [docs/stable-api-1.0.md](docs/stable-api-1.0.md) lists which
crates are stable.

## Correctness and safety

**Conformance.** JPEG 2000 has an official test suite (ISO/IEC 15444-4, also
published as ITU-T T.803). The CPU decoder passes all 160 test cases for the
profiles J2K supports (90 for classic JPEG 2000 and 70 for HTJ2K) on macOS,
Linux, and Windows. The CUDA and Metal builds pass the same 160; in those runs
81 cases use the GPU for some stages and 79 run on the CPU. These results
are for release 0.14.0; details and the exact profiles are in
[docs/t803-conformance.md](docs/t803-conformance.md).

**Safety.** The public API is safe Rust and is built to take untrusted files:
malformed input returns an error, memory use is capped, and the decoders are
fuzzed in CI. `unsafe` code is used only for FFI and GPU calls, SIMD, and a few
allocation and buffer helpers. Every host-side `unsafe` block in the published
crates explains why it is sound, and Clippy rejects one that doesn't.

**Copying instead of re-encoding.** If you're moving JPEG 2000 data between
containers (DICOM to TIFF, for example) and the destination expects the same
image format, copy the compressed bytes instead of decoding and re-encoding.
`J2kView::passthrough_candidate` describes the compressed data (format, size,
components, bit depth) so you can check it against what the destination
expects.

## How it compares

| | Language | GPU | License |
| --- | --- | --- | --- |
| **J2K** | Rust | CUDA and Metal | MIT / Apache-2.0 |
| [OpenJPEG](https://github.com/uclouvain/openjpeg) | C | none | BSD-2-Clause |
| [Grok](https://github.com/GrokImageCompression/grok) | C++ | optional closed-source CUDA/Metal plugin | AGPL-3.0 |
| [Kakadu](https://kakadusoftware.com) | C++ | none | commercial |
| [OpenJPH](https://github.com/aous72/OpenJPH) | C++ | none | BSD-2-Clause |
| NVIDIA nvJPEG2000 | C | CUDA only | proprietary |

J2K does not depend on any of them.

## Status

**Release status:** `0.14.0` is published and security-supported. See the
[changelog](CHANGELOG.md) for migration notes.

`j2k`, `j2k-core`, `j2k-jpeg`, and `j2k-tilecodec` are the stable crates.
The GPU, transcode, and ML crates are experimental: patch releases don't
break them, but a `0.x` minor release can (every break is listed with
migration notes). See [docs/stable-api-1.0.md](docs/stable-api-1.0.md).

The minimum supported Rust version on `main` is 1.97; the published 0.14.0
crates need 1.99.0. Pull-request CI checks the whole workspace with Rust 1.97
and tests with the pinned 1.99.0 toolchain.

## Documentation

- [API docs on docs.rs](https://docs.rs/j2k)
- [Project website](https://frames-sg.com/j2k/rust-jpeg2000-codec/)
- [Rust JPEG codec](https://frames-sg.com/rust-jpeg-codec/)
- [JPEG-to-HTJ2K transcoding](https://frames-sg.com/jpeg-to-htj2k-transcoding/)
- [GPU image decoding for AI pipelines](https://frames-sg.com/gpu-image-decoding-ai/)
- [docs/public-support.md](docs/public-support.md): every supported and
  unsupported feature of the format
- [docs/architecture.md](docs/architecture.md): how the crates fit together
- [docs/benchmark-evidence.md](docs/benchmark-evidence.md): GPU and CPU
  benchmark results
- [docs/env-vars.md](docs/env-vars.md): `J2K_*` environment variables
- [CHANGELOG.md](CHANGELOG.md): release notes
- [CONTRIBUTING.md](CONTRIBUTING.md): building, testing, benchmarking, and
  releasing

## Security

Report vulnerabilities as described in [SECURITY.md](SECURITY.md).

## License

Dual-licensed under either [MIT](LICENSE-MIT) or
[Apache-2.0](LICENSE-APACHE), at your option.
