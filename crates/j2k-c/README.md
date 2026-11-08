# j2k-c: C and C++ API

A CPU C ABI, a C++17 ownership wrapper, and relocatable native binary packages
for the `j2k` JPEG 2000 / HTJ2K codec. It inspects headers, components, and the
reference grid; decodes full images, regions, and reduced resolutions into
caller-owned buffers; streams full images row by row through a callback without
a whole-image buffer; and encodes interleaved samples losslessly (classic or
HTJ2K, raw, JP2, or JPH). Output preserves native U8/U16/I16 sample values with
explicit precision and alpha metadata. Decoders are strict by default or
lenient on request, and run on the codec's shared CPU pool or on their own
threads. GPU decode, lossy encoding, and slide-container readers are not
exposed.

The additive exact-operation API plans regions at reductions 0–31 and offers
explicit eight-bit sRGB output. Native and raw sRGB exact operations support
per-operation working/output limits and cancellation, including the source
preparation an operation performs itself. Existing ABI-1
structures and the legacy 0–3 reduction API retain their layouts and behavior.

- [Consumer quickstart](docs/quickstart.md), also the package's `README.md`
- [C ABI contract](docs/api.md)
- [Supported configurations](docs/supported.md)
- [Module boundaries](docs/architecture.md)
- [Changelog](CHANGELOG.md)

## Build and use

Use the workspace's pinned Rust toolchain, Python 3, and CMake 3.20+. From the
repository root:

```sh
cargo test -p j2k-c --locked
python3 crates/j2k-c/scripts/package.py --output dist/j2k-c
cmake -S dist/j2k-c/examples -B target/examples -DCMAKE_PREFIX_PATH="$PWD/dist/j2k-c"
cmake --build target/examples
target/examples/j2k_decode image.jp2
target/examples/j2k_decode_cpp image.jp2
```

The package contains headers, the platform shared library, CMake configuration,
examples, documentation, and license notices for the packages the library links.
Consumers of the binary package need no Rust toolchain. Packaging uses the host
target and supports macOS and GNU/Linux; it does not cross-compile, and it
refuses to overwrite an existing output directory.

Applications use `find_package(J2K CONFIG REQUIRED)` and link
`J2K::j2k`. The C interface is `include/j2k.h`; C++17 callers can use
`j2k::decoder` from `include/j2k.hpp`.

## Checks

Rust tests cover adapter behavior; the native consumer tests cover the ABI and
the C++ wrapper against an installed package. Before changing the ABI or
packaging, run:

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

Keep the C header, Rust ABI definitions, examples, and contract documentation in
sync. Changes to supported formats, limits, or sample handling update
`docs/supported.md` and `CHANGELOG.md`. Preserve ABI-1 layouts and ownership
rules; an incompatible change needs a new ABI revision. Keep raw pointers in
`src/ffi.rs`, with a safety explanation on each unsafe block.

Licensed under either of the workspace's [MIT](../../LICENSE-MIT) or
[Apache-2.0](../../LICENSE-APACHE) licenses at your option.
