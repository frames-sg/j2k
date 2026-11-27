# Transcoding JPEG to HTJ2K

`j2k-transcode` converts JPEG to HTJ2K without decoding to pixels. It reads
the JPEG's DCT coefficients, computes the wavelet coefficients directly from
them, and codes those as HTJ2K. This is how you move a large archive of JPEG
tiles, such as slide scans, to HTJ2K without the cost and the extra loss of
a full decode and re-encode.

```bash
cargo add j2k-transcode
```

To convert JPEG 2000 to HTJ2K instead, use
[`recode_j2k_to_htj2k_lossless`](encoding.md#converting-jpeg-2000-to-htj2k)
in `j2k`.

## Usage

```rust
use j2k_transcode::{jpeg_to_htj2k, JpegToHtj2kOptions};

let jpeg = std::fs::read("tile.jpg")?;
let encoded = jpeg_to_htj2k(&jpeg, &JpegToHtj2kOptions::lossless_53())?;
std::fs::write("tile.j2c", &encoded.codestream)?;

let report = &encoded.report;
println!("{}x{}, {} components", report.width, report.height, report.component_count);
```

There are two presets:

- `JpegToHtj2kOptions::lossless_53()`, the default, uses the reversible 5/3
  wavelet. The HTJ2K decodes to exactly the samples `j2k-jpeg`'s integer IDCT
  produces for each JPEG component, so the conversion adds no loss.
- `JpegToHtj2kOptions::lossy_97()` uses the irreversible 9/7 wavelet for
  smaller files. `encode_options.irreversible_quantization_scale` sets the
  size: lower values give larger files with higher quality.

Both presets write one wavelet decomposition level. If you want to decode the
result at reduced resolution, raise `encode_options.num_decomposition_levels`.

The command-line tool does the lossless conversion too: `j2k transcode
in.jpg out.j2k --htj2k --lossless-53` (see [the command
line](bindings.md#command-line)).

## What goes in and what comes out

The input can be 8-bit baseline JPEG (SOF0), including files whose
components are split across several scans, or 8-bit progressive JPEG (SOF2),
in grayscale, YCbCr, or RGB. Other JPEG types, including 8-bit extended
sequential (SOF1), return an error. The JPEG is parsed with
`j2k-jpeg`'s strict policy (see [JPEG](jpeg.md#strict-and-lenient-decoding)),
so damaged files are rejected rather than concealed.

The output is a raw HTJ2K codestream. It keeps the JPEG's components as they
are: color stays in the JPEG's own color space (usually YCbCr), without a
JPEG 2000 color transform, and subsampled chroma stays subsampled, signalled
through the component sampling factors instead of being upsampled. A raw
codestream doesn't record the color space, so whatever reads it has to know,
typically from the surrounding DICOM or TIFF metadata.

To make a `.jph` file, wrap the codestream with
[`wrap_j2k_codestream`](encoding.md#containers) and give the color space
explicitly. The default, `J2kFileColorSpec::Infer`, labels three components
as RGB, which is wrong for YCbCr data:

```rust
use j2k::{wrap_j2k_codestream, J2kFileColorSpec, J2kFileWrapOptions};
use j2k_core::Colorspace;

let color = J2kFileColorSpec::Enumerated(Colorspace::YCbCr); // written as sYCC
let jph = wrap_j2k_codestream(&encoded.codestream, J2kFileWrapOptions::jph().with_color(color))?;
```

## Many tiles

Keep one `JpegToHtj2kTranscoder` per thread and reuse it; it keeps its
buffers between calls. `transcode_batch` converts many tiles at once:

```rust
use j2k_transcode::{JpegTileBatchInput, JpegToHtj2kOptions, JpegToHtj2kTranscoder};

let inputs: Vec<JpegTileBatchInput<'_>> =
    tiles.iter().map(|bytes| JpegTileBatchInput { bytes }).collect();
let mut transcoder = JpegToHtj2kTranscoder::default();
let batch = transcoder.transcode_batch(&inputs, &JpegToHtj2kOptions::lossless_53())?;
for (index, tile) in batch.tiles.iter().enumerate() {
    match tile {
        Ok(encoded) => { /* encoded.codestream */ }
        Err(error) => eprintln!("tile {index}: {error}"),
    }
}
```

The CPU batch runs tiles in parallel on the current Rayon pool, in waves
bounded by the host-memory limit. Results and per-tile errors keep input
order. The stage times in `batch.report` add up per-tile time, so with
parallel tiles they can exceed the elapsed time.

## 9/7 precision

The 9/7 path computes in IEEE binary32 by default
(`TranscodePrecision::F32`). The IDCT, the wavelet lifting, and the quantizer
follow one fixed sequence of operations that the CPU, CUDA, and Metal
backends all evaluate, so the same JPEG and options give byte-identical
codestreams on every backend. The quantizer multiplies by the binary32
reciprocal of each step size.

`TranscodePrecision::F64` computes in binary64 on the CPU only, and its output
differs from `F32` and from every GPU backend. Use it only to reproduce
codestreams from older J2K releases, which computed in binary64. An explicit
GPU accelerator rejects it with `TranscodeStageError::Unsupported`; an `Auto`
accelerator leaves it on the CPU.

```rust
use j2k_transcode::{JpegToHtj2kOptions, TranscodePrecision};

let mut options = JpegToHtj2kOptions::lossy_97();
options.precision = TranscodePrecision::F64;
```

The 5/3 path is integer arithmetic and is unaffected.

## GPU acceleration

`j2k-transcode-cuda` and `j2k-transcode-metal` run the transform stages (and,
on CUDA, code-block preparation) on the GPU. The HTJ2K codestream is always
assembled on the CPU by `j2k-transcode`. Pass an accelerator to the
transcoder:

```rust
use j2k_transcode::{JpegToHtj2kOptions, JpegToHtj2kTranscoder};
use j2k_transcode_cuda::CudaDctToWaveletStageAccelerator;

let mut accelerator = CudaDctToWaveletStageAccelerator::for_auto();
let mut transcoder = JpegToHtj2kTranscoder::default();
let encoded = transcoder.transcode_with_accelerator(
    &jpeg,
    &JpegToHtj2kOptions::lossless_53(),
    &mut accelerator,
)?;
```

`MetalDctToWaveletStageAccelerator` works the same way, and
`transcode_batch_with_accelerator` takes either one for batches. An `Auto`
accelerator (`for_auto()`) runs a stage on the CPU when the GPU is
unavailable or the job is too small or unsupported. An explicit one
(`new_explicit()`) returns an error instead. The output bytes are the same
either way.

Under `Auto`:

- CUDA takes a single transform job from 224 x 224 component samples, and
  same-shape 5/3 and 9/7 batches from 32 jobs and 224 x 224 x 32 samples.
- Metal keeps single tiles on the CPU unless you lower
  `with_auto_reversible_min_samples` or `with_auto_dwt97_min_samples`.
  Same-shape batches go to Metal from 32 jobs and 224 x 224 x 32 samples,
  except that 9/7 batches with a tile side over 1024 samples stay on the
  CPU.
- `with_auto_reversible_batch_thresholds` and
  `with_auto_dwt97_batch_thresholds` change the batch thresholds on both.
  Lower them only if your own measurements justify it: meeting a threshold
  doesn't guarantee a speedup.

The CUDA accelerator needs the `cuda-runtime` feature and a Linux build with
the CUDA kernels; see [GPU decoding and encoding](gpu.md#cuda). Without it,
`for_auto()` stays on the CPU and `new_explicit()` returns an error.
`new_explicit_resident_ht_encode()` also keeps 9/7 code-block coefficients
on the GPU and HT-codes them there before the CPU packetizes them.

The Metal device constructors take a retained `objc2-metal`
`ProtocolObject<dyn MTLDevice>`. `jpeg_to_htj2k_with_metal_route` and its
batch form also report the requested backend, the backend each stage ran
on, the reason `Auto` stayed on the CPU, and the bytes transferred. Run
`cargo run -p j2k-transcode-metal --example jpeg_to_htj2k_route_report` to
see one.

To keep stages in GPU memory, `ResidentJpegDctGrid`, `ResidentDwtSubband`,
and `ResidentCodestreamBuffer` describe device buffers, checking their
metadata when built. On macOS,
`resident_codestream_buffer_from_metal_encoded_j2k` turns buffer-backed
`j2k-metal` encode output into a `ResidentCodestreamBuffer`.
