# The J2K Book

J2K is a JPEG 2000 and HTJ2K codec for Rust. It decodes and encodes classic
JPEG 2000 (Part 1) and High-Throughput JPEG 2000 (Part 15), reads and writes
JP2 and JPH files, and has a JPEG codec alongside. Everything runs on the
CPU; an NVIDIA GPU (CUDA) or an Apple GPU (Metal) can take over the heavy
stages where that was measured to be faster. Python, C, and C++ bindings and
a command-line tool are built on the same code.

This book is for people using J2K. It assumes you know Rust and roughly what
JPEG 2000 is, and it explains how the pieces fit together; the API
documentation has the details of each type and function.

## Chapters

- [Getting started](getting-started.md): install, a first decode and encode,
  and which crate you need.
- [Decoding](decoding.md): full images, regions, reduced resolutions, tiles,
  batches, limits, and wavelet coefficients.
- [Encoding](encoding.md): lossless and lossy encoding, quality targets,
  containers, and converting JPEG 2000 to HTJ2K.
- [Transcoding](transcoding.md): converting JPEG to HTJ2K without decoding
  to pixels.
- [JPEG](jpeg.md): decoding, encoding, and cropping regular JPEG.
- [GPU decoding and encoding](gpu.md): how CUDA and Metal support works and
  how to choose a backend.
- [Machine learning](machine-learning.md): decoding into Burn tensors.
- [Python, C, and the command line](bindings.md): the bindings and the `j2k`
  tool.
- [Serving IIIF images](iiif.md): image-server requests and Cantaloupe.

Reference:

- [Format support](format-support.md): which parts of the standard are
  supported, and their limits.
- [Conformance](conformance.md): results against the ISO/IEC 15444-4 test
  suite.
- [Performance](performance.md): benchmark results.
- [Environment variables](environment-variables.md): the `J2K_*` variables.
- [API stability](stability.md): which crates are stable.

## Elsewhere

- [API documentation](https://docs.rs/j2k) on docs.rs.
- The [contributor book](../contributor-book/README.md), for changing J2K
  itself: code layout, testing, benchmarks, and releases.
- [Release notes](https://github.com/frames-sg/j2k/releases) on GitHub.
- [Reporting a vulnerability](../SECURITY.md).
