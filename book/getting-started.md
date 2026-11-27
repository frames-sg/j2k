# Getting started

## Install

Add the main crate to your project:

```bash
cargo add j2k
```

J2K needs Rust 1.97 or later. The `j2k` crate has no feature flags and no
C dependencies; GPU support comes from separate crates (see [Which crate do
I need?](#which-crate-do-i-need)).

## Decode an image

Read a JPEG 2000 file and decode it to 8-bit RGB (or gray or RGBA, depending
on the image):

```rust
use j2k::J2kDecoder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let bytes = std::fs::read("image.jp2")?;
    let mut decoder = J2kDecoder::new(&bytes)?;
    let (width, height) = decoder.info().dimensions;

    let image = decoder.decode_srgb8()?;
    println!("{width}x{height}, {:?}, {} bytes", image.layout(), image.data().len());
    Ok(())
}
```

`J2kDecoder::new` accepts JP2 and JPH files and raw `.j2k`, `.j2c`, and HTJ2K
codestreams. `decode_srgb8` applies the file's palette and color profile and
returns pixels ready to display. [Decoding](decoding.md) covers other output
formats, regions, reduced resolutions, and batches.

## Encode an image

Encode 8-bit RGB pixels as lossless HTJ2K:

```rust
use j2k::{encode_j2k_lossless, J2kBlockCodingMode, J2kLosslessEncodeOptions, J2kLosslessSamples};

let samples = J2kLosslessSamples::new(&rgb, width, height, 3, 8, false)?;
let options = J2kLosslessEncodeOptions::default()
    .with_block_coding_mode(J2kBlockCodingMode::HighThroughput);
let encoded = encode_j2k_lossless(samples, &options)?;
std::fs::write("image.j2c", &encoded.codestream)?;
```

`encoded.codestream` is a raw codestream; `wrap_j2k_codestream` puts it in a
`.jph` file. The encoder decodes its own output and compares it with the
input before returning, unless you turn that off. [Encoding](encoding.md)
covers lossy encoding, quality targets, and containers.

## Which crate do I need?

Most code only needs `j2k`. Add a GPU crate if you want GPU decoding.

| You want to | Use |
| --- | --- |
| Read, write, or convert JPEG 2000 / HTJ2K | `j2k` |
| Decode with an NVIDIA GPU | `j2k` + `j2k-cuda` |
| Decode with an Apple GPU | `j2k` + `j2k-metal` |
| Read or write regular JPEG | `j2k-jpeg` |
| Read or write JPEG-LS, as in DICOM | `j2k-jpegls` |
| Decode batches of JPEGs into dense RGB8 tensors | `j2k_jpeg::JpegBatchDecoder` |
| Convert JPEG to HTJ2K | `j2k-transcode` (+ `j2k-transcode-cuda` or `j2k-transcode-metal`) |
| JPEG on the GPU | `j2k-jpeg-cuda` or `j2k-jpeg-metal` |
| Feed decoded images to Burn | `j2k-ml` |
| Feed decoded images to MPSGraph | `j2k-mpsgraph` |
| Feed pixels or wavelet bands to PyTorch or NumPy | The `j2k` Python package (`pip install j2k`) |
| Extract selected wavelet coefficients in Rust | `j2k::J2kDecoder::decode_wavelets` |
| Decompress Deflate, Zstd, LZW, or PackBits tiles (as in TIFF), or decode and encode DICOM RLE frames | `j2k-tilecodec` |
| A command-line tool | `j2k-cli` |
| Decode or encode from C or C++ | `j2k-c`, built from source |

The other `j2k-*` crates are building blocks shared by the crates above; you
don't need to depend on them directly. [API stability](stability.md) lists
which crates are stable and which are experimental.

## Supported platforms

- `j2k` runs on the CPU on Linux, macOS, and Windows.
- `j2k-jpeg` and the crates built on it support `x86_64` and `aarch64` only;
  other targets fail at compile time.
- `j2k-cuda` and the other CUDA crates need Linux, an NVIDIA driver, and by
  default a GPU of compute capability 8.0 or newer. The CUDA kernels are
  built only on Linux hosts with the cuda-oxide toolchain. See [GPU decoding
  and encoding](gpu.md#cuda).
- `j2k-metal` and the other Metal crates need macOS. They build elsewhere,
  but Metal requests return an error.
- The Python extension needs CPython 3.11 or later. The C package is built
  for macOS and GNU/Linux. See [Python, C, and the command line](bindings.md).

## Runnable examples

The crates ship small examples that generate their own input:

```bash
cargo run -p j2k --example decode_generated
cargo run -p j2k-transcode --example jpeg_to_htj2k
cargo run -p j2k-metal --example decode_route_report   # macOS
```

The [API documentation](https://docs.rs/j2k) has the full reference for every
type mentioned in this book.
