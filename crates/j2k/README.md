# j2k — Pure-Rust JPEG 2000 and HTJ2K Codec

The main J2K crate: JPEG 2000 Part 1 and HTJ2K Part 15 on the CPU.

It provides inspection, decode, owned batch decode, native wavelet-band
extraction (`J2kDecoder::decode_wavelets`), encode, lossless J2K-to-HTJ2K
recode, JP2/JPH container wrapping, and GPU-decode planning, all backed by the
native J2K engine.

The public API is safe Rust, and the CPU path is the reference implementation.
GPU paths are optional and are used only for shapes where they were measured to
be faster. Single-image HTJ2K encode to host memory runs on the CPU by default.

Supported inputs are raw J2K/J2C codestreams, JP2 files, raw HTJ2K
codestreams, and JPH files. JPX (JPEG 2000 Part 2) extensions are not
supported except where JP2/JPH decoding needs them.

The encode-stage types shared with the CUDA, Metal, and transcode adapters are
re-exported at the crate root but hidden from the rendered docs; they are not
the encode API for applications. GPU surfaces and runtimes live in the
`j2k-cuda` and `j2k-metal` crates.

For CPU JPEG 2000 / HTJ2K code, depend on this crate. For GPU decoding, also
depend on the adapter crate for your backend.

## Decode strictness

`j2k::DecodeSettings::default()` and `j2k_native::DecodeSettings::default()`
are strict. `J2kView::parse` and `J2kDecoder::new` use the default.

To accept some malformed files, pass `DecodeSettings::lenient()` to
`J2kView::parse_with_settings` or `J2kDecoder::new_with_settings`. Lenient mode
only relaxes a few metadata checks: it may ignore a malformed trailing
top-level box after the JP2/JPH header and codestream, ignore a malformed
trailing child box after `ihdr` and `colr`, ignore malformed optional `cdef` or
`pclr` metadata (keeping an earlier complete value), or infer one undeclared
alpha component. Codestream and entropy validation, bounds and overflow
checks, and allocation limits are the same as in strict mode.

A successful lenient decode reports `J2kDecodeWarning::LenientMetadataRecovery`
only when one of those recoveries was actually used.

## Partial codestreams

A viewer that fetches a file a piece at a time can show a low-resolution
image before the rest arrives. With
`DecodeSettings::with_partial_codestream(true)`, the decoder accepts input
that stops anywhere after the codestream's main header:

```rust
use j2k::{DecodeSettings, J2kDecoder, J2kScratchPool, PixelFormat, Rect};

let settings = DecodeSettings::strict().with_partial_codestream(true);
let mut decoder = J2kDecoder::new_with_settings(&bytes_so_far, settings)?;
if let Some(reduction) = decoder.codestream_completeness()?.min_exact_reduction() {
    let (width, height) = decoder.info().dimensions;
    let area = Rect { x: 0, y: 0, w: width, h: height };
    let plan = decoder.region_decode_plan(PixelFormat::Gray16, area, reduction)?;
    let mut pixels = vec![0_u8; plan.buffer_bytes];
    decoder.decode_region_scaled_pow2_into(
        &mut J2kScratchPool::new(), &mut pixels, plan.row_bytes, PixelFormat::Gray16,
        area, reduction,
    )?;
}
```

`codestream_completeness` reads the packet headers without decoding samples
and reports:

- `min_exact_reduction`: the fewest resolution levels to drop for a decode
  of the whole image that only reads data that is present. A decode at that
  reduction or deeper returns the same samples as the complete file decoded
  at the same reduction. It is `None` until the lowest resolution of every
  tile has arrived, and `Some(0)` once every packet is present.
- `tile_min_exact_reduction`: the same for each tile, so a region over tiles
  that have arrived can be decoded while others are still missing.
- `is_complete`: whether every packet is present. The final EOC marker may
  still be missing.

A decode that needs data that hasn't arrived fails with
`J2kError::IncompleteCodestream`, which counts as truncated input
(`is_truncated()`). Decoding never fills in missing data.

How quickly resolutions become usable depends on the progression order.
With a resolution-first order (RPCL or RLCP), each resolution level is
complete as soon as its bytes arrive, which makes RPCL the order to choose
for progressive delivery. With a layer-first order (LRCP), a resolution is
complete only once every quality layer of it has arrived.

The input can be a raw codestream or a JP2/JPH file that stops inside its
codestream box; the boxes before it must be complete. The JP2/JPH container
is otherwise read as strictly as without the setting, and so is every tile
part that is present in full: a malformed one is still an error. Partial
input is never offered for byte copying, and it decodes on the CPU only.

## Batch decode

`prepare_batch` and `CpuBatchDecoder::new` take `BatchDecodeOptions`, which is
`#[non_exhaustive]`. Start from `BatchDecodeOptions::default()` and assign
`layout`, `settings`, `workers`, or `max_working_memory_bytes`:

```rust
let mut options = j2k::BatchDecodeOptions::default();
options.layout = j2k::BatchLayout::Nhwc;
let decoder = j2k::CpuBatchDecoder::new(options);
```

`BatchLayout` is defined in `j2k-core` and re-exported here.

## Lossy encode behavior

- **Decomposition depth.** The requested depth is capped by the smallest tile
  component, counting partial edge tiles and component sampling. By default,
  images narrower than 64 samples are not decomposed. A quantization profile
  has `1 + 3 * depth` entries for the resolved depth, which COD also signals.
- **Component quantization profile.** Weights multiply the nominal step before
  it is rounded to a stored step, and a weighted step that cannot be stored is
  an error. A profile of all ones is the same as no profile, whose steps may
  saturate instead. For example, with 30-bit input and two guard bits the
  nominal step 0.25 clips to the finest stored step, 0.5: weights near one are
  rejected and weight two is accepted.
- **Byte and bits-per-pixel targets.** The encoder searches for a quantization
  scale; each probe is a full encode. It returns the first result within 1% of
  the target, or the finest admissible scale if that fits. After
  `psnr_iteration_budget` bracketed probes it returns the candidate closest to
  the target within the size allowance.
- **PSNR targets** are a minimum quality within `psnr_tolerance_db`. The
  search doubles, then bisects, and validates the codestream it returns. It
  does not enumerate every scale if quality is not monotonic in scale.

## Output sample conversion

For 1- to 16-bit sources:

- Native batch output (`U8`, `U16`, `I16`) keeps sample values and signedness.
- `Gray8`/`Rgb8`/`Rgba8` output scales the unsigned sample range to 0–255;
  negative samples clamp to zero.
- `Gray16`/`Rgb16`/`Rgba16` output widens unsigned sources of at most 8 bits
  to 0–65535 and keeps native values for 9- to 16-bit sources. Signed sources
  keep their values as 16-bit two's complement.
- Alpha the decoder synthesizes is the output maximum (above 8 bits, the
  native maximum). Real alpha is converted like any other component.
- Irreversible 9/7 output rounds ties to even before the level shift.
- CUDA and Metal decodes apply the same conversions.

## Links

- API docs: <https://docs.rs/j2k>
- [Pure-Rust JPEG 2000 codec documentation](https://frames-sg.com/j2k/rust-jpeg2000-codec/)
- Repository: <https://github.com/frames-sg/j2k>
- Support policy: <https://github.com/frames-sg/j2k/blob/main/docs/public-support.md>
