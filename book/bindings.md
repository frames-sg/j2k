# Python, C, and the command line

The bindings and the command-line tool are thin layers over the Rust crates,
so they decode the same images with the same results. Each has its own full
documentation; this page shows what each is for and how to start.

## Python

The `j2k` Python extension decodes JPEG 2000 and HTJ2K, and 8-bit JPEG in
batches, to NumPy arrays or PyTorch tensors of the image's own sample type
(`uint8`, `uint16`, or `int16`). It handles regions, reduced resolutions,
prepared batches for training loops, and wavelet coefficients, and it
encodes on the CPU. Built with CUDA, it decodes into PyTorch CUDA memory;
built with Metal, it returns `mps` tensors on Apple Silicon.

Install it from PyPI (CPython 3.11 or later; NumPy and PyTorch are optional):

```sh
pip install j2k            # or "j2k[torch]" / "j2k[numpy]"
```

The PyPI wheels cover Linux x86-64 on the CPU and Apple Silicon macOS with
Metal. For CUDA, or another platform, build the wheel from this repository
with Maturin and `--features cuda`; the Python README has the steps.

```python
from pathlib import Path
import j2k

data = Path("image.jp2").read_bytes()
print(j2k.inspect(data))
decoder = j2k.Decoder(layout="nchw")
image = decoder.decode(data).to_torch()               # [C, H, W]
small = decoder.decode(data, reduction=1).to_numpy()  # half resolution
```

The [Python README](../crates/j2k-python/README.md) covers batches, regions,
limits and cancellation, strict and lenient decoding, encoding, GPU output,
and tensor ownership.

## C and C++

`j2k-c` is a C ABI (`j2k.h`), a header-only C++17 wrapper (`j2k.hpp`), and a
relocatable package with the shared library, a static archive, and CMake and
pkg-config files. It inspects images; keeps one decoder across codestreams;
decodes full images, regions, and reduced resolutions into
your buffers as native U8, U16, or I16 samples; streams full images row by row;
and encodes losslessly or to a rate, size, or PSNR target. It runs on the CPU
by default; builds with the `metal` feature add a strict full-image Metal GPU
decode route through `j2k_decoder_metal_init` and
`j2k_decoder_metal_decode_into`.

Each [GitHub release](https://github.com/frames-sg/j2k/releases) has a
prebuilt package for Linux x86-64 and arm64 (glibc 2.28 or newer), macOS 11
or newer on Apple silicon, and Windows x86-64 (MSVC), with a SHA-256 checksum
for each. Programs that use a package need only a C or C++ compiler and CMake
or pkg-config, not a Rust toolchain. The
[quickstart](../crates/j2k-c/docs/quickstart.md#download-a-prebuilt-package)
says which archive to pick and how to check it. To build a package for your
own machine instead, from the repository root:

```sh
python3 crates/j2k-c/scripts/package.py --output dist/j2k-c
```

```cmake
find_package(J2K CONFIG REQUIRED)
target_link_libraries(your_application PRIVATE J2K::j2k)
```

```cpp
#include <j2k.hpp>

j2k::decoder decoder(encoded.data(), encoded.size());
auto info = decoder.output_info();
std::vector<uint8_t> pixels(static_cast<size_t>(info.buffer_bytes));
decoder.decode_into(pixels.data(), pixels.size(), static_cast<size_t>(info.row_bytes));
```

The sample type and bit depth come from `info`; don't assume RGB8. Start with
the [C README](../crates/j2k-c/README.md) and the
[quickstart](../crates/j2k-c/docs/quickstart.md), then read the [API
reference](../crates/j2k-c/docs/api.md) and [supported
configurations](../crates/j2k-c/docs/supported.md) before handling medical
images.

## Command line

`j2k-cli` installs the `j2k` command:

```bash
cargo install j2k-cli
j2k inspect image.jp2                                       # size, color space, depth, levels, tiles
j2k transcode input.jpg output.j2k --htj2k --lossless-53   # JPEG to HTJ2K
j2k decode -i image.jp2 -d 0,0,1024,1024 -r 2 -o tile.png  # opj_decompress options
```

`j2k decode` takes OpenJPEG `opj_decompress` options with the same meanings,
and the binary behaves as `opj_decompress` when run under that name (through
a symlink, for example), so image servers that call `opj_decompress` can use
it unchanged. `transcode` does the lossless conversion described in
[Transcoding](transcoding.md). The tool doesn't encode images; use the
library for that.

The [CLI README](../crates/j2k-cli/README.md) lists the options and how the
output differs from OpenJPEG's. [Serving IIIF images](iiif.md) shows how to
run it behind Cantaloupe.
