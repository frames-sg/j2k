# Supported configurations

The library exposes the CPU codec of the `j2k` crate it is built and
versioned with, through the batch route and the core whole-image route.

| Area | Supported |
| --- | --- |
| Input | JPEG 2000 Part 1 and HTJ2K (Part 15) codestreams, and JP2 and JPH files holding a still image |
| Output | Interleaved gray, RGB or RGBA in native `U8`, `U16` or `I16` samples, with precision and alpha reported |
| Geometry | Full image, region, half, quarter or eighth resolution, and a region at a reduced resolution when the image has the levels |
| Codestream lifecycle | `j2k_decoder_feed` moves a decoder to a new codestream and keeps its threads, settings, limits and Metal session. `j2k_decoder_transfer_syntax` reports classic JPEG 2000 or HTJ2K, reversible or irreversible, or `J2K_SYNTAX_OTHER`; check the call's status before reading it |
| Whole-image route | Raw one- or three-component unsigned 8- to 16-bit images at full resolution, without uniform component subsampling, through the core whole-image decoder. Samples and stride match the batch route |
| Metal decode | Optional full-image, full-resolution decode on macOS through the `metal` feature, reusing one session per decoder. Strict, with no CPU fallback. Prebuilt packages are CPU-only; build from source with `--features metal` |
| Exact operations | Regions at any reduction from 0 to 31 that every component and tile supports, with reference-grid geometry |
| sRGB output | 8-bit gray, RGB or RGBA with straight alpha, converting supported ICC profiles, palettes, channel mapping and component sampling |
| Operation limits | Working-memory and output limits and cancellation, for native and sRGB operations, including the preparation an operation does itself. ICC conversion under a working limit needs a matrix/TRC or gray-TRC profile with an XYZ connection space |
| Row streaming | Full image at full resolution, for 8-bit unsigned and 9- to 16-bit samples, holding at most `max_rows_per_stripe` rows |
| Encoding | Lossless or lossy, classic or HTJ2K, 1- to 16-bit unsigned or signed interleaved samples. Bits-per-pixel, byte or PSNR targets, cumulative quality layers, HTJ2K Qfactor. Decomposition levels, progression order, tiles, tile-part packet limits, and SOP, EPH, TLM, PLT, PLM, PPM and PPT markers. Raw codestream, JP2 or JPH output, with an optional ICC profile |
| Color metadata | The first COLR box, which decoding applies: an enumerated `EnumCS`, ICC profile bytes, or an uninterpreted method |
| Validation | Strict by default; `J2K_OPEN_LENIENT` accepts the codec's JP2/JPH metadata recoveries and reports them |
| Partial input | `J2K_OPEN_PARTIAL` accepts a codestream that is cut off after its main header and decodes the reductions whose data is present |
| JPEG-LS | Decoding and encoding of 2- to 16-bit lossless and near-lossless codestreams, as in DICOM, with every interleave mode and restart intervals. Decoding also reads preset coding parameters |
| DICOM RLE | Decoding and encoding of RLE Lossless frames with 8, 16 or 32 bits allocated and up to 15 segments, into color-by-pixel or color-by-plane samples. Strict by default, with a lenient mode that reports what it accepted |
| Threads | The codec's shared pool, or a decoder's own pool of 1 to 64 threads |
| Platforms | Prebuilt packages for Linux x86-64 and arm64 with glibc 2.28 or newer, macOS 11 or newer on Apple silicon, and Windows x86-64 for MSVC. Each package runs only on its own platform. Linux and macOS packages also hold a static archive. A package you build yourself works on the host it was built on, and on Linux needs that host's glibc or newer |
| Consumers | C11 and C++17, through CMake (`J2K::j2k`, or `J2K::j2k_static` on Linux and macOS) or, on Linux and macOS, pkg-config (`j2k`). Packages need no Rust toolchain |

Native pixel decoding rejects ICC-tagged and palette-mapped images, mixed
precision or signedness, precision above 16 bits, unsupported colors or
component counts, and images whose components are all subsampled alike so
that none covers the image grid. Components with different supported
sampling factors are replicated onto the output grid. sRGB operations accept
more color representations through the general Rust decoder. They
un-premultiply premultiplied samples at the source precision before sYCC
conversion, and reject premultiplied CIE Lab and premultiplied subsampled or
signed input, except 8-bit mapped samples that need no color conversion.
Header inspection can succeed for an image that pixel decoding rejects.

JPEG-LS decoding rejects mapping tables, subsampled components, point
transforms, HP color transforms and images over 65535 pixels on a side. DICOM
RLE supports neither 1-bit frames nor bit widths other than 8, 16 or 32.

## Large images

`max_output_bytes` (512 MiB by default) bounds a request's output, including
Metal decode. The codec also enforces its own allocation budgets, so
`j2k_decoder_decode_into` can return `J2K_RESOURCE_LIMIT` ("image is too
large") after
`j2k_decoder_output_info` succeeds. In our measurements through this API,
full decodes of 8-bit HTJ2K images of 6144 x 6144 RGB, 8192 x 8192 gray, and
8192 x 8192 RGB with 2048 x 2048 tiles succeed. Decode larger images as
regions, at a reduced resolution, or row by row with
`j2k_decoder_decode_rows`, which needs no whole-image buffer. Peak memory of
`j2k_decoder_decode_into` includes the codec's own copy of the output as well
as your buffer.

## Not supported

GPU encoding, CUDA decode, submitting several images in one call, streaming
rows of a region or a reduced resolution, range I/O, prebuilt packages for
Intel Macs, Windows on Arm or musl-based Linux, MinGW import libraries, .NET
or Java bindings, and parsing slide containers. Metal decode requires a
source build with the `metal` feature and a macOS host. Python bindings are
the separate `j2k` package on PyPI.

The library's tests cover its ownership rules and decoded sample values.
They are not a new ISO conformance run or a medical-device validation. Metal
tests run on a Metal host when `J2K_REQUIRE_METAL_RUNTIME=1` and otherwise
skip. Each prebuilt package is unpacked and passes the same C and C++
consumer tests and examples on its own platform before it is released;
passing on one platform says nothing about another.

## Packaged libraries

The shared library is named after the C ABI revision, not the package
version. GNU/Linux packages contain `lib/libj2k.so.1` with SONAME
`libj2k.so.1`. macOS packages contain `lib/libj2k.1.dylib` with install name
`@rpath/libj2k.1.dylib`, compatibility version 1.0.0, and current version
equal to the package version. Each `lib/` also has an unversioned
`libj2k.so` or `libj2k.dylib` symlink for linking. Applications load the
versioned name, so deploy that file. Windows packages contain `bin/j2k.dll`,
its import library `lib/j2k.dll.lib`, and no pkg-config file; the DLL name
doesn't carry the ABI revision. The library is stripped to its exported
`j2k_*` symbols and embeds no build-machine paths.

Cargo also builds a static archive, `libj2k_c.a`, for source consumers that
link the C ABI statically. Linux and macOS packages ship it as `lib/libj2k.a`
beside the shared library.

Prebuilt GNU/Linux packages are built against glibc 2.28, so they run on
distributions with glibc 2.28 or newer, such as RHEL 8, Debian 10 and
Ubuntu 20.04. A package you build yourself needs at least the glibc of the
machine that built it. To find the newest glibc symbol version a package
uses, and compare it with `ldd --version` on the target machine:

```sh
objdump -T lib/libj2k.so.1 | grep -o 'GLIBC_[0-9][0-9.]*' | sort -uV | tail -n 1
```

Prebuilt macOS packages need macOS 11 or newer. A macOS package records its
minimum OS in `LC_BUILD_VERSION`, which follows `MACOSX_DEPLOYMENT_TARGET` at
build time:

```sh
otool -l lib/libj2k.1.dylib | grep -A4 LC_BUILD_VERSION
```
