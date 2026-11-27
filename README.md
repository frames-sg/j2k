<div align="center">
  <img src="docs/assets/favicon.svg" alt="The J2K logo" height="120"/>
  <br /><br />

[![Rust 1.97 or later](https://img.shields.io/badge/Rust-1.97%2B-ff4d00?style=flat-square&logo=rust&logoColor=white)](https://crates.io/crates/j2k)
[![License](https://img.shields.io/badge/License-MIT%2FApache--2.0-ff4d00?style=flat-square)](#license)
[![Current Crates.io Version](https://img.shields.io/crates/v/j2k.svg?style=flat-square&color=ff4d00)](https://crates.io/crates/j2k)
[![Full validation](https://img.shields.io/github/actions/workflow/status/frames-sg/j2k/full-validation.yml?branch=main&label=CI&style=flat-square)](https://github.com/frames-sg/j2k/actions/workflows/full-validation.yml)
[![Documentation](https://img.shields.io/badge/docs-latest-ff4d00?style=flat-square)](https://docs.rs/j2k)

  <p><b>JPEG 2000 and HTJ2K for Rust, on the CPU, CUDA, and Metal.</b></p>
</div>

---

JPEG 2000 is the image format behind medical imaging (DICOM), digital pathology
slide scans, satellite imagery, and archives. Its files are often huge and
split into tiles, and the format was designed so you can pull a region or a
lower resolution out of one without decoding the whole image. HTJ2K (High
Throughput JPEG 2000, Part 15) keeps the format and swaps in a much faster
block coder.

J2K reads and writes both. The codecs are pure, safe Rust with no C libraries
underneath. They run on the CPU on Linux, macOS, and Windows, and can hand the
heavy work to an NVIDIA GPU (CUDA) or an Apple GPU (Metal). The same workspace
also has a JPEG codec, a JPEG-to-HTJ2K transcoder, Python and C bindings, and a
command-line tool.

## Features

- **Decode** whole images, regions, reduced resolutions, or single tiles.
  Region and low-resolution decodes only do the work for the pixels you ask
  for, so a thumbnail or viewport from a multi-gigabyte slide is cheap.
- **Decode in batches** in parallel, with output grouped by shape and ready to
  turn into NCHW or NHWC tensors.
- **Encode** lossless or lossy JPEG 2000 and HTJ2K, to a quality, a file size,
  a bit rate, or a target PSNR.
- **Transcode** JPEG to HTJ2K from the DCT coefficients, and classic JPEG 2000
  to HTJ2K losslessly, without going through pixels when the input allows it.
- **Read and write** raw codestreams and JP2 and JPH files.
- **Untrusted input is safe to decode.** Malformed files return errors, memory
  use is capped, and the decoders are fuzzed in CI.

J2K implements JPEG 2000 Part 1 and HTJ2K (Part 15). Part 2 (JPX) extensions
are not supported. The [format support page](book/format-support.md) has the
details.

## Getting started

```bash
cargo add j2k
```

Decode a file to 8-bit RGB (or gray or RGBA, depending on the image):

```rust
use j2k::J2kDecoder;

let bytes = std::fs::read("image.jp2")?;
let mut decoder = J2kDecoder::new(&bytes)?;
let (width, height) = decoder.info().dimensions;

let image = decoder.decode_srgb8()?;
println!("{width}x{height}, {:?}, {} bytes", image.layout(), image.data().len());
```

<details>
<summary>
Decode a region at reduced resolution
</summary>
<br />

This decodes a 1024 x 1024 area at quarter resolution, without touching the
rest of the image:

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

The plan tells you how big the output is before you allocate it. See
[Decoding](book/decoding.md) for 16-bit and native-depth output, batches,
memory limits, and cancellation.

</details>

<details>
<summary>
Encode lossless HTJ2K
</summary>
<br />

```rust
use j2k::{encode_j2k_lossless, J2kBlockCodingMode, J2kLosslessEncodeOptions, J2kLosslessSamples};

let samples = J2kLosslessSamples::new(&rgb, width, height, 3, 8, false)?;
let options = J2kLosslessEncodeOptions::default()
    .with_block_coding_mode(J2kBlockCodingMode::HighThroughput);
let encoded = encode_j2k_lossless(samples, &options)?;
std::fs::write("image.j2c", &encoded.codestream)?;
```

`encoded.codestream` is a raw codestream; `wrap_j2k_codestream` puts it in a
JP2 or JPH file. By default the encoder decodes its own output and checks it
against the input before returning. See [Encoding](book/encoding.md) for lossy
encodes and rate targets.

</details>

The [J2K Book](book/README.md) is the user guide. API documentation is on
[docs.rs](https://docs.rs/j2k), and there are runnable examples in
`crates/*/examples/` (try `cargo run -p j2k --example decode_generated`).

## GPU backends

|                                 | CPU | CUDA | Metal |
| ------------------------------- | --- | ---- | ----- |
| JPEG 2000 and HTJ2K decode      | ☑️  | ☑️   | ☑️    |
| JPEG 2000 and HTJ2K encode      | ☑️  | ☑️   | ☑️    |
| JPEG decode and encode          | ☑️  | ☑️   | ☑️    |
| JPEG to HTJ2K transcode         | ☑️  | ☑️   | ☑️    |
| Platforms                       | Linux, macOS, Windows | Linux, NVIDIA driver | macOS |

The CPU still parses the file, and the GPU does the expensive parts: block
decoding, the wavelet transform, colour conversion, and writing pixels.
Decoded images can stay in GPU memory and go straight into a GPU pipeline.

You pick a backend with `BackendRequest`. `Cuda`, `Metal`, and `Cpu` do what
they say, and a GPU request that the GPU can't handle returns an error rather
than quietly running on the CPU. `Auto`, the default, uses the GPU only for
the image types and sizes where we measured it to be faster than the CPU with
identical output, and the CPU for everything else.

GPU support lives in separate crates, `j2k-cuda` and `j2k-metal`, so CPU-only
users don't build any of it. [GPU decoding and encoding](book/gpu.md) covers
setup and what runs where.

<details>
<summary>
Machine learning pipelines 🧠
</summary>
<br />

- **PyTorch and NumPy.** The [`j2k` Python package](crates/j2k-python/README.md)
  decodes straight to tensors in the image's own sample type: single images,
  batches that mix JPEG and JPEG 2000, regions, reduced resolutions, and
  selected wavelet bands. Built with CUDA it decodes into PyTorch-owned device
  memory; built with Metal it returns `mps` tensors.
- **Burn.** [`j2k-ml`](book/machine-learning.md) decodes batches into Burn
  tensors on the CPU, CUDA, or Metal.
- **MPSGraph.** [`j2k-mpsgraph`](book/machine-learning.md) hands
  Metal-resident batches to MPSGraph without a round trip through the CPU.
- **Wavelet coefficients.** `J2kDecoder::decode_wavelets` returns the 5/3 or
  9/7 coefficients for the bands you select, skipping the inverse transform.

</details>

## Crates

Most projects only need `j2k`. Add a GPU crate if you want GPU decoding.

| You want to                                    | Use                                                          |
| ---------------------------------------------- | ------------------------------------------------------------ |
| Read, write, or convert JPEG 2000 and HTJ2K    | `j2k`                                                        |
| Decode and encode on an NVIDIA GPU             | `j2k` + `j2k-cuda`                                           |
| Decode and encode on an Apple GPU              | `j2k` + `j2k-metal`                                          |
| Read or write regular JPEG                     | `j2k-jpeg`, plus `j2k-jpeg-cuda` or `j2k-jpeg-metal` for GPUs |
| Convert JPEG to HTJ2K                          | `j2k-transcode`, plus `j2k-transcode-cuda` or `j2k-transcode-metal` |
| Feed decoded images to Burn or MPSGraph        | `j2k-ml`, `j2k-mpsgraph`                                     |
| Decompress Deflate, Zstd, or LZW tiles (TIFF)  | `j2k-tilecodec`                                              |
| Use J2K from Python, C, or C++                 | [`j2k` on PyPI](crates/j2k-python/README.md), [`j2k-c`](crates/j2k-c/README.md) (optional Metal decode) |
| Use J2K from the shell                         | [`j2k-cli`](crates/j2k-cli/README.md)                        |

The other `j2k-*` crates are building blocks shared by the ones above. The
[architecture chapter](contributor-book/architecture.md) shows how they fit
together.

<details>
<summary>
Command line and image servers 🖼️
</summary>
<br />

```bash
cargo install j2k-cli
j2k inspect image.jp2                                      # size, colour, depth, tiles, levels
j2k transcode input.jpg output.j2k --htj2k --lossless-53   # JPEG to HTJ2K
j2k decode -i image.jp2 -d 0,0,1024,1024 -r 2 -o tile.png  # opj_decompress options
```

`j2k decode` accepts OpenJPEG's `opj_decompress` options, and the binary
behaves as `opj_decompress` when run under that name, so image servers such as
Cantaloupe can use J2K without changes. [Serving IIIF images](book/iiif.md)
walks through the setup.

</details>

## Correctness

J2K passes the official JPEG 2000 conformance suite (ISO/IEC 15444-4, also
published as ITU-T T.803) for every profile it supports: all 160 cases, 90 for
classic JPEG 2000 and 70 for HTJ2K, on Linux, macOS, and Windows, and on the
CUDA and Metal builds. [Conformance](book/conformance.md) has the details.

The public API is safe Rust. `unsafe` is limited to FFI and GPU calls, SIMD,
and a few buffer helpers, and every host-side `unsafe` block says why it is
sound.

## How it compares

|                                                      | Language | GPU                         | License          |
| ---------------------------------------------------- | -------- | --------------------------- | ---------------- |
| **J2K**                                              | Rust     | CUDA and Metal              | MIT / Apache-2.0 |
| [OpenJPEG](https://github.com/uclouvain/openjpeg)    | C        | none                        | BSD-2-Clause     |
| [Grok](https://github.com/GrokImageCompression/grok) | C++      | closed-source plugin        | AGPL-3.0         |
| [Kakadu](https://kakadusoftware.com)                 | C++      | none                        | commercial       |
| [OpenJPH](https://github.com/aous72/OpenJPH)         | C++      | none                        | BSD-2-Clause     |
| NVIDIA nvJPEG2000                                    | C        | CUDA only                   | proprietary      |

J2K doesn't depend on any of them. Speed comparisons are in
[Performance](book/performance.md).

## Status

J2K is pre-1.0. `j2k`, `j2k-core`, `j2k-jpeg`, and `j2k-tilecodec` are stable:
we don't break them in patch releases, and breaking changes in minor releases
come with migration notes. The GPU, transcode, and machine learning crates are
experimental and change more often. See [API stability](book/stability.md).

Release notes are on [GitHub Releases](https://github.com/frames-sg/j2k/releases),
and upcoming changes are in the [changelog](CHANGELOG.md). More about the
project is on [frames-sg.com](https://frames-sg.com/j2k/rust-jpeg2000-codec/).

## Contributing

Read the [contributing guidelines](CONTRIBUTING.md) and the
[code of conduct](CODE_OF_CONDUCT.md) first. The
[contributor book](contributor-book/README.md) covers the architecture,
testing, benchmarks, and releases. Report security issues as described in
[SECURITY.md](SECURITY.md).

## License

J2K is distributed under the terms of either the [MIT License](LICENSE-MIT) or
the [Apache License, Version 2.0](LICENSE-APACHE), at your option. Third-party
code and data that J2K includes or derives from are listed in
[NOTICES.md](NOTICES.md).
