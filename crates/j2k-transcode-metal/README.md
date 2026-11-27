# J2K Transcode Metal

> Metal acceleration for [J2K](https://github.com/frames-sg/j2k)'s JPEG-to-HTJ2K transcode

[![Current Crates.io Version](https://img.shields.io/crates/v/j2k-transcode-metal.svg)](https://crates.io/crates/j2k-transcode-metal)
[![Documentation](https://docs.rs/j2k-transcode-metal/badge.svg)](https://docs.rs/j2k-transcode-metal)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/frames-sg/j2k#license)

Run the transform stages of
[`j2k-transcode`](https://crates.io/crates/j2k-transcode)'s JPEG-to-HTJ2K
conversion on an Apple GPU. `jpeg_to_htj2k_with_metal_route` takes a
`BackendRequest` and reports where each stage ran, and
`MetalDctToWaveletStageAccelerator` plugs into the `j2k-transcode` API
directly. The output is byte-identical to the CPU and CUDA backends. `Auto`
keeps single tiles on the CPU and sends only large batches of same-shape tiles
to Metal; an explicit `Metal` request returns an error for anything Metal
can't run.

Metal needs macOS. On other platforms the crate builds, and explicit Metal
requests return `MetalTranscodeError::MetalUnavailable`. Device constructors
take [`objc2-metal`](https://crates.io/crates/objc2-metal) types, not
`metal-rs`. The API is experimental, so a minor release can change it (see
[API stability](https://github.com/frames-sg/j2k/blob/main/book/stability.md)).

## Usage

```rust
use j2k_core::BackendRequest;
use j2k_transcode::JpegToHtj2kOptions;
use j2k_transcode_metal::jpeg_to_htj2k_with_metal_route;

let jpeg = std::fs::read("tile.jpg")?;
let routed = jpeg_to_htj2k_with_metal_route(
    &jpeg,
    &JpegToHtj2kOptions::lossless_53(),
    BackendRequest::Auto,
)?;
println!("{:?}", routed.route.selected_transform_backend);
std::fs::write("tile.j2c", &routed.encoded.codestream)?;
```

The [GPU acceleration](https://github.com/frames-sg/j2k/blob/main/book/transcoding.md#gpu-acceleration)
section of the J2K Book covers batches, the `Auto` thresholds, and how to
change them.

## Feature flags

- `bench-internals`: internal, for this crate's benchmarks.

<!-- j2k-crate-footer -->

---

Part of [J2K](https://github.com/frames-sg/j2k), a JPEG 2000 and HTJ2K codec
for Rust. See the [J2K Book](https://github.com/frames-sg/j2k/tree/main/book)
and the [API documentation](https://docs.rs/j2k).
