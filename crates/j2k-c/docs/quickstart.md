# j2k for C and C++

This package contains a CPU JPEG 2000/HTJ2K decoder and encoder, C and C++17
headers, and CMake and pkg-config integration. It is licensed under either of `LICENSE-MIT` or
`LICENSE-APACHE` at your option; `NOTICES.md` records source provenance, and
third-party licenses are included under `licenses/`.

## Build the examples

A binary package requires a C/C++ compiler and CMake 3.20 or newer. No Rust
toolchain is needed to consume it. In the unpacked package directory:

```sh
cmake -S examples -B build -DCMAKE_PREFIX_PATH="$PWD"
cmake --build build
./build/j2k_decode image.jp2
./build/j2k_decode_cpp image.jp2
```

The examples accept raw `.j2c`/`.j2k`, `.jp2`, and `.jph` inputs. They decode native
samples and print their dimensions and interpretation. Reading an SVS, TIFF, or
DICOM container requires a separate reader that supplies the encoded frames or
tiles; those file formats are not parsed by this library.

## Link your application

```cmake
find_package(J2K CONFIG REQUIRED)
target_link_libraries(your_application PRIVATE J2K::j2k)
```

Set `CMAKE_PREFIX_PATH` to the package directory. C callers include `j2k.h`;
C++17 callers may use the move-only `j2k::decoder` in `j2k.hpp`.

Other build systems use pkg-config. The file is relocatable; the library is
not on the default loader path, so also link with a runtime search path:

```sh
export PKG_CONFIG_PATH="/path/to/package/lib/pkgconfig"
cc app.c $(pkg-config --cflags --libs j2k) -Wl,-rpath,"$(pkg-config --variable=libdir j2k)"
```

```cpp
j2k::decoder decoder(encoded.data(), encoded.size());
auto info = decoder.output_info();
std::vector<uint8_t> pixels(static_cast<size_t>(info.buffer_bytes));
decoder.decode_into(pixels.data(), pixels.size(), static_cast<size_t>(info.row_bytes));
```

The sample type and significant bit depth come from `info`; do not assume RGB8.

For an image too large to hold at once, stream its rows instead:

```cpp
decoder.decode_rows([&](uint32_t y, const uint8_t *row, size_t row_bytes) {
    write_row(y, row, row_bytes); // your sink; return false to stop
    return true;
});
```

`j2k::encode_lossless(samples, length, options)` encodes interleaved samples
with a `j2k_encode_options` initialized by `J2K_ENCODE_OPTIONS_INIT`.
`j2k::encode(samples, length, options, params, &report)` adds coding parameters
from `J2K_ENCODE_PARAMS_INIT`: for example lossy coding to a rate target, with
an ICC profile when `J2K_ENCODE_FILE` writes a JP2/JPH file:

```cpp
const double bits_per_pixel[] = {2.0};
j2k_encode_params params = J2K_ENCODE_PARAMS_INIT;
params.mode = J2K_MODE_LOSSY;
params.rate = J2K_RATE_BITS_PER_PIXEL;
params.target_count = 1;
params.targets = bits_per_pixel;
params.progression = J2K_PROGRESSION_RPCL;
params.markers = J2K_MARKER_TLM | J2K_MARKER_PLT;
auto codestream = j2k::encode(samples, length, options, params);
```

`decoder.color_info()` and `decoder.icc_profile()` report the colour
specification decoding applies.
For region decode, initialize `j2k_decode_options` with
`J2K_DECODE_OPTIONS_INIT`, set x/y/width/height and optional reduction, and pass
the same options to `output_info` and `decode_into`.

The package is relocatable. CMake sets a build-time runtime search path for the
examples. Your application installer must deploy the versioned shared library
(`libj2k.so.1` or `libj2k.1.dylib`; the unversioned name is a link-time
symlink) and set an appropriate runtime search path. Keep the package's license
files and the included third-party notices with your distribution.

Read [the API contract](api.md) and [supported configurations](supported.md)
before integrating medical-image sample handling.
