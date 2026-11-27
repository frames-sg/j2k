# J2K Native

> The JPEG 2000 and HTJ2K engine behind [J2K](https://github.com/frames-sg/j2k)

[![Current Crates.io Version](https://img.shields.io/crates/v/j2k-native.svg)](https://crates.io/crates/j2k-native)
[![Documentation](https://docs.rs/j2k-native/badge.svg)](https://docs.rs/j2k-native)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/frames-sg/j2k#license)

This is the codec engine that [`j2k`](https://crates.io/crates/j2k) is built
on: codestream and JP2/JPH parsing, block coding, packets, wavelet transforms,
and the encoders. Most code should use `j2k`, the stable API on top of it. Use
`j2k-native` directly for its lower-level `Image` API, or to build without
`std`, SIMD, or threads. It reads the same files as `j2k`: J2K/J2C and HTJ2K
codestreams and JP2 and JPH files, without JPEG 2000 Part 2 (JPX) extensions.
Its API is experimental, so a minor release can change it (see
[API stability](https://github.com/frames-sg/j2k/blob/main/book/stability.md)).

Repeated region decodes of one `Image` are cheaper: with `std`, the second
region decode records the image's packet boundaries (at most 16 MiB per
image), and later region decodes reuse them instead of parsing the packets
again. This works for single-tile images in RPCL order with one tile-part,
inline packet headers, and no POC or PLT markers. Other layouts parse packets
on every request.

The engine started as an import of dicom-toolkit-jpeg2000 0.5.0, which is
derived from Hayro (see
[`NOTICES.md`](https://github.com/frames-sg/j2k/blob/main/NOTICES.md)). It
contains no unsafe code.

## Usage

```rust
use j2k_native::{DecodeSettings, Image};

let data = std::fs::read("image.jp2")?;
let image = Image::new(&data, &DecodeSettings::default())?;
println!("{}x{}, {:?}", image.width(), image.height(), image.color_space());
let pixels = image.decode()?;
```

## Feature flags

- `std` (default): standard library support and the region packet directory.
  Without it, the crate is `no_std` with `alloc`.
- `simd` (default): SIMD code paths, through `fearless_simd`.
- `parallel` (default): multithreaded decoding and encoding, through Rayon.
- `profile` (default): stage timing rows when `J2K_PROFILE_STAGES` is set.
  Without it, profiling compiles to nothing.
- `logging`: decoder diagnostics through the `log` crate.

<!-- j2k-crate-footer -->

---

Part of [J2K](https://github.com/frames-sg/j2k), a JPEG 2000 and HTJ2K codec
for Rust. See the [J2K Book](https://github.com/frames-sg/j2k/tree/main/book)
and the [API documentation](https://docs.rs/j2k).
