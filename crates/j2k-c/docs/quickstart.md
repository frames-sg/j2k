# J2K for C and C++

This package holds a CPU JPEG 2000 and HTJ2K decoder and encoder, its C and
C++17 headers, and CMake and pkg-config files. It is licensed under either
`LICENSE-MIT` or `LICENSE-APACHE` at your option. `NOTICES.md` records where
the source comes from, and `licenses/` holds the third-party licenses.

## Download a prebuilt package

Each [GitHub release](https://github.com/frames-sg/j2k/releases) has one
package per platform. Pick the one for the machine your program runs on:

| Platform | Asset |
| --- | --- |
| Linux x86-64 with glibc 2.28 or newer | `j2k-c-VERSION-linux-x86_64.tar.gz` |
| Linux arm64 with glibc 2.28 or newer | `j2k-c-VERSION-linux-aarch64.tar.gz` |
| macOS 11 or newer on Apple silicon | `j2k-c-VERSION-macos-aarch64.tar.gz` |
| Windows x86-64, for MSVC | `j2k-c-VERSION-windows-x86_64.zip` |

Each archive has a `.sha256` file next to it, and `j2k-c-VERSION-SHA256SUMS`
lists them all. Check the archive before you unpack it:

```sh
sha256sum -c j2k-c-VERSION-linux-x86_64.tar.gz.sha256       # Linux
shasum -a 256 -c j2k-c-VERSION-macos-aarch64.tar.gz.sha256   # macOS
tar -xzf j2k-c-VERSION-linux-x86_64.tar.gz
```

On Windows, the hash PowerShell prints must equal the one in the `.sha256`
file:

```powershell
(Get-FileHash j2k-c-VERSION-windows-x86_64.zip -Algorithm SHA256).Hash.ToLower()
Get-Content j2k-c-VERSION-windows-x86_64.zip.sha256
Expand-Archive j2k-c-VERSION-windows-x86_64.zip -DestinationPath .
```

Each archive unpacks to one directory, `j2k-c-VERSION-PLATFORM`; the commands
below run from inside it. To build a package from source instead, see the
[C README](https://github.com/frames-sg/j2k/blob/main/crates/j2k-c/README.md).

## Build the examples

You need a C and C++ compiler and CMake 3.20 or newer; no Rust toolchain.
From the unpacked package directory:

```sh
cmake -S examples -B build -DCMAKE_PREFIX_PATH="$PWD"
cmake --build build
./build/j2k_decode image.jp2
./build/j2k_decode_cpp image.jp2
```

On Windows, Visual Studio puts the programs in `build\Release`, and the
example build copies `j2k.dll` next to them:

```powershell
cmake -S examples -B build -DCMAKE_PREFIX_PATH="$PWD"
cmake --build build --config Release
.\build\Release\j2k_decode.exe image.jp2
```

The examples read raw codestreams (`.j2c`, `.j2k`), `.jp2` and `.jph` files,
decode them, and print the dimensions and sample format. The library doesn't
parse SVS, TIFF or DICOM files: a separate reader has to hand it the encoded
frames or tiles.

## Link your application

```cmake
find_package(J2K CONFIG REQUIRED)
target_link_libraries(your_application PRIVATE J2K::j2k)
```

Point `CMAKE_PREFIX_PATH` at the package directory. C code includes `j2k.h`;
C++17 code can use the move-only `j2k::decoder` from `j2k.hpp`.

On Windows, `J2K::j2k` links the import library `lib/j2k.dll.lib`, and the
program loads `bin/j2k.dll` when it starts. Windows looks for the DLL in the
program's directory, so copy it there after each build:

```cmake
add_custom_command(TARGET your_application POST_BUILD COMMAND "${CMAKE_COMMAND}" -E
    copy_if_different "$<TARGET_FILE:J2K::j2k>" "$<TARGET_FILE_DIR:your_application>")
```

Other build systems on Linux and macOS can use pkg-config. The `.pc` file is
relocatable, but the library isn't on the default loader path, so link with a
runtime search path too:

```sh
export PKG_CONFIG_PATH="/path/to/package/lib/pkgconfig"
cc app.c $(pkg-config --cflags --libs j2k) -Wl,-rpath,"$(pkg-config --variable=libdir j2k)"
```

To link the static archive on macOS or Linux, define `J2K_STATIC` and pass
`lib/libj2k.a` explicitly. `pkg-config --static --libs j2k` reports the
archive's private dependencies but also includes `-lj2k`; remove that entry so
the linker does not select the shared library:

```sh
export PKG_CONFIG_PATH="/path/to/package/lib/pkgconfig"
cc -DJ2K_STATIC app.c /path/to/package/lib/libj2k.a \
    $(pkg-config --cflags j2k) \
    $(pkg-config --static --libs-only-l j2k | sed 's/-lj2k//')
```

CMake consumers can link the exported static target instead; it defines
`J2K_STATIC` and carries the archive's private link libraries:

```cmake
find_package(J2K CONFIG REQUIRED)
target_link_libraries(your_application PRIVATE J2K::j2k_static)
```

Windows packages have no pkg-config file and no static archive. Without CMake,
add `include` to the include path and link `lib/j2k.dll.lib`.

## Decode

```cpp
j2k::decoder decoder(encoded.data(), encoded.size());
auto info = decoder.output_info();
std::vector<uint8_t> pixels(static_cast<size_t>(info.buffer_bytes));
decoder.decode_into(pixels.data(), pixels.size(), static_cast<size_t>(info.row_bytes));
```

`info` gives the sample type, channel count and significant bits. Don't
assume 8-bit RGB. `examples/decode.c` does the same in C.

To decode a region or a reduced resolution, initialize a `j2k_decode_options`
with `J2K_DECODE_OPTIONS_INIT`, set `x`, `y`, `width`, `height` and
`reduction` (0 to 3), and pass the same options to `output_info` and
`decode_into`. For deeper reductions, up to 31, or 8-bit sRGB output, use
`plan_region` and `decode_region_into` with a `j2k_operation_options`.

An image too large to hold at once can be streamed row by row:

```cpp
decoder.decode_rows([&](uint32_t y, const uint8_t *row, size_t row_bytes) {
    write_row(y, row, row_bytes); // your sink
    return true;                  // false stops the decode
});
```

`decoder.color_info()` and `decoder.icc_profile()` report the color
specification that decoding applies.

## Encode

`j2k::encode_lossless(samples, length, options)` encodes interleaved samples
described by a `j2k_encode_options` initialized with `J2K_ENCODE_OPTIONS_INIT`.
`j2k::encode(samples, length, options, params, &report)` also takes coding
parameters from `J2K_ENCODE_PARAMS_INIT`. For example, lossy coding to
2 bits per pixel, written as a JP2 or JPH file when `options.flags` includes
`J2K_ENCODE_FILE`:

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

## Deploy

The package can be moved anywhere. On Linux and macOS, CMake gives the
examples a build-time runtime search path; your installer has to ship the
versioned shared library (`libj2k.so.1` or `libj2k.1.dylib`; the unversioned
name is only a link-time symlink) and set a runtime search path for it. On
Windows, install `j2k.dll` next to your executable. It needs the Microsoft
Visual C++ runtime (`vcruntime140.dll`), which the Visual C++ Redistributable
installs. Ship the package's license files and third-party notices with your
application.

Read the [API reference](api.md) and the
[supported configurations](supported.md) before you handle medical images:
they say exactly which samples come back and what is rejected. Changes are
listed in the [release notes](https://github.com/frames-sg/j2k/releases).
