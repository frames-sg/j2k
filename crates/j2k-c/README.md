# J2K C

> C and C++ API for [J2K](https://github.com/frames-sg/j2k)

[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/frames-sg/j2k#license)

`j2k-c` is a C ABI over the `j2k` CPU codec, a header-only C++17 wrapper,
and a script that packages both as a relocatable shared library with CMake
and pkg-config files, plus a static archive. Applications that consume the
package need no Rust toolchain.

It inspects headers, components and the reference grid; keeps one decoder
across codestreams and reports their compressed transfer syntax; decodes full
images, regions and reduced resolutions into buffers you own, in the image's
native `uint8`, `uint16` or `int16` samples, through the batch route or the
core whole-image route; streams full images row by row through a callback;
converts regions to 8-bit sRGB; and encodes interleaved samples losslessly or
lossy to a rate, size or PSNR target, as a raw codestream, JP2 or JPH. It also
decodes and encodes JPEG-LS codestreams and DICOM RLE Lossless frames. GPU
decoding and slide-container readers are not part of this API.

The crate isn't published to crates.io. It ships as the native packages that
`scripts/package.py` builds, attached to each GitHub release.

## Download a prebuilt package

Each [GitHub release](https://github.com/frames-sg/j2k/releases) has a
package for Linux x86-64 and arm64 (glibc 2.28 or newer), macOS 11 or newer
on Apple silicon, and Windows x86-64 (MSVC), each with a `.sha256` file. The
[quickstart](docs/quickstart.md#download-a-prebuilt-package) says which
archive to pick, how to check it, and how to use it from CMake or pkg-config,
including the runtime search path on Linux and macOS and where `j2k.dll` goes
on Windows.

## Build from source

With the workspace's pinned Rust toolchain, Python 3 and CMake 3.20 or newer,
from the repository root:

```sh
python3 crates/j2k-c/scripts/package.py --output dist/j2k-c
cmake -S dist/j2k-c/examples -B target/examples -DCMAKE_PREFIX_PATH="$PWD/dist/j2k-c"
cmake --build target/examples
target/examples/j2k_decode image.jp2
target/examples/j2k_decode_cpp image.jp2
```

The package holds the headers, the shared library, CMake files, the examples,
these docs, and license notices for every package the library links or builds
with. macOS and GNU/Linux packages also hold the static archive
`lib/libj2k.a` and pkg-config files. Packaging builds for the host (macOS,
GNU/Linux, or x86-64 Windows with Visual Studio's C++ tools), doesn't
cross-compile, and refuses to overwrite an existing output directory or
archive. `--archive tar.gz` or `--archive zip` also writes an archive of the
package and its `.sha256` file; `--name` sets the archive's name.

Applications call `find_package(J2K CONFIG REQUIRED)` and link `J2K::j2k`, or
`J2K::j2k_static` for the static archive. C code includes `j2k.h`; C++17 code
can use `j2k::decoder` from `j2k.hpp`.

- [Quickstart](docs/quickstart.md), which is also the package's `README.md`
- [API reference](docs/api.md)
- [Supported configurations](docs/supported.md)
- [Module layout](docs/architecture.md), for contributors
- [Release notes](https://github.com/frames-sg/j2k/releases)

## Development

Rust tests cover the adapter. The native consumer tests build C and C++
programs against an assembled package. Before changing the ABI or packaging,
run:

```sh
cargo clippy -p j2k-c --lib --locked -- -D warnings
cargo clippy -p j2k-c --tests --examples --locked -- -D warnings -A clippy::disallowed_methods -A clippy::disallowed_macros
cargo test -p j2k-c --locked
python3 crates/j2k-c/scripts/package.py --output dist/j2k-c
cargo run -p j2k-c --locked --example make_fixture -- target/fixture.j2c
cmake -S crates/j2k-c/native/tests -B target/native-tests -DCMAKE_PREFIX_PATH="$PWD/dist/j2k-c" -DJ2K_FIXTURE="$PWD/target/fixture.j2c"
cmake --build target/native-tests
ctest --test-dir target/native-tests --output-on-failure
```

The native tests also need pkg-config, except on Windows. To test an archive
the way a consumer uses it, unpacked into a new directory:

```sh
python3 crates/j2k-c/scripts/package.py --name j2k-c-test --archive tar.gz
python3 crates/j2k-c/scripts/test_package.py dist/j2k-c-test.tar.gz --fixture target/fixture.j2c --work target/consumer
```

Keep `include/j2k.h`, the Rust ABI definitions in `src/abi.rs`, the examples
and `docs/api.md` in step. A change to supported formats, limits or sample
handling updates `docs/supported.md` and gets an entry in the root
[`CHANGELOG.md`](../../CHANGELOG.md). ABI 1 layouts and ownership rules don't
change in place: an incompatible change needs a new ABI revision. Raw pointers
stay in `src/ffi.rs`, with a safety comment on each unsafe block.

<!-- j2k-crate-footer -->

---

Part of [J2K](https://github.com/frames-sg/j2k), a JPEG 2000 and HTJ2K codec
for Rust. See the [J2K Book](https://github.com/frames-sg/j2k/tree/main/book)
and the [API documentation](https://docs.rs/j2k).
