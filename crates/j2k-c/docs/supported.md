# Supported configurations

The library targets the CPU native-integer batch contract of the `j2k` crate it is
built and versioned with.

| Surface | Scope |
| --- | --- |
| Input | JPEG 2000 Part 1 and HTJ2K Part 15 codestreams; JP2/JPH still-image wrappers |
| Output | Interleaved gray/RGB/RGBA, native U8/U16/I16 samples, explicit precision and alpha metadata |
| Geometry | Full image, region, half/quarter/eighth resolution, and combined region/reduction when the source supports it |
| Exact operations | Additive region API with exact 0–31 reductions, effective image-wide maximum, and reference-grid geometry |
| Explicit sRGB | Bounded gray/RGB/RGBA U8 normalization for supported ICC, palette/channel mapping, and component sampling; straight alpha |
| Operation controls | Native and raw sRGB working/output limits and cooperative cancellation, including source preparation an operation performs; ICC conversion under a working limit requires an XYZ-connection matrix/TRC or gray-TRC profile |
| Row streaming | Full image at full resolution, 8-bit unsigned and 9- to 16-bit samples, at most `max_rows_per_stripe` rows held |
| Encoding | Lossless, classic or HTJ2K, 1- to 16-bit unsigned or signed interleaved samples; raw codestream, JP2, or JPH |
| Validation | Strict by default; `J2K_OPEN_LENIENT` accepts the codec's JP2/JPH metadata recoveries and reports them |
| Threads | The codec's shared pool, or a decoder's own pool of 1–64 threads |
| Platforms | Native macOS and GNU/Linux shared-library packages; each binary is specific to its build target, and GNU/Linux packages need the build host's glibc or newer |
| Consumers | C11 and C++17; no Rust toolchain required for prebuilt packages |

Native pixel decode rejects ICC-tagged and palette-mapped inputs, mixed precision or
signedness, precision above 16 bits, unsupported colors/component counts, and
uniformly subsampled component grids that cannot represent the image grid.
J2K handles supported mixed sampling by replicating components onto the output
grid. Explicit sRGB operations support additional colour representations through
the general Rust decoder. Normalization of premultiplied samples unassociates
them at the source precision before sYCC conversion; premultiplied CIE Lab input,
and premultiplied subsampled or signed input other than eight-bit mapped samples
without colour conversion, are unsupported. Header inspection may still succeed for an image that
pixel decode rejects.

## Request size

`max_output_bytes` (512 MiB by default) bounds a request's output. Full-frame
decodes are no longer held to the fixed 512 MiB per-decode working-memory
ceiling of `j2k` 0.13.1: measured on 8-bit HTJ2K images through this ABI,
6144x6144 RGB, 8192x8192 gray, and 8192x8192 RGB (2048x2048 tiles) full-frame
decodes succeed with exact samples. The codec still enforces its own allocation
budgets, so a request can return `J2K_RESOURCE_LIMIT` ("image is too large")
from `j2k_decoder_decode_into` after `j2k_decoder_output_info` succeeds.
Decode such images as regions, at reduced resolution, or row by row with
`j2k_decoder_decode_rows`, which needs no whole-image buffer. Peak memory of
`j2k_decoder_decode_into` includes the codec-owned copy of the output in
addition to the caller's buffer.

GPU decode, lossy encoding, multi-image batch submission, row streaming of
regions or reduced resolutions, range I/O, Windows
packaging, .NET/Java bindings, and slide-container parsing are not exposed by
this release. Python bindings are the separate
`j2k-python` crate.

The library's tests cover its adapter/ownership contract and numerical samples. They
do not constitute a new ISO conformance run or medical-device validation.
Platform CI builds and runs the same C/C++ consumer tests against an assembled
package. Passing on one platform does not establish validation on another.

## Packaged libraries

The shared library is named by C ABI revision, not package version. GNU/Linux
packages contain `lib/libj2k.so.1` with SONAME `libj2k.so.1`. macOS
packages contain `lib/libj2k.1.dylib` with install name
`@rpath/libj2k.1.dylib`, compatibility version 1.0.0, and current version
equal to the package version. Each `lib/` also has an unversioned `libj2k.so` or
`libj2k.dylib` symlink for linking. Applications load the versioned name, so
deploy that file. The library is stripped to its exported `j2k_*` symbols
and embeds no build-machine paths.

A GNU/Linux package requires at least the glibc version of its build host; no
older glibc baseline is claimed. Find the newest glibc symbol version a package
needs, then compare it with `ldd --version` on the deployment host:

```sh
objdump -T lib/libj2k.so.1 | grep -o 'GLIBC_[0-9][0-9.]*' | sort -uV | tail -n 1
```

A package built by CI on GitHub's `ubuntu-latest` image needs that image's glibc
or newer. The image moves to newer Ubuntu releases over time; the CI log prints
the floor of each build. macOS packages record their minimum OS in
`LC_BUILD_VERSION` (`otool -l lib/libj2k.1.dylib | grep -A4 LC_BUILD_VERSION`),
which follows `MACOSX_DEPLOYMENT_TARGET` at build time.
