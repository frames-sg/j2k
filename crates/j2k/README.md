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
`layout`, `settings`, and `workers`:

```rust
let mut options = j2k::BatchDecodeOptions::default();
options.layout = j2k::BatchLayout::Nhwc;
let decoder = j2k::CpuBatchDecoder::new(options);
```

`BatchLayout` is defined in `j2k-core` and re-exported here.

## Encode depth, quantization, and output samples

The decomposition option selects a depth under the existing progression policy;
images narrower than 64 samples retain the undecomposed default. The encoder
then constrains that selection by the smallest tile component, including partial
edge tiles and component sampling. The effective-depth helper, transform plan,
profile row length (`1 + 3 * depth`), and COD signaling use that resolved depth.

An active component quantization profile weights the unrounded nominal step,
before stored-step rounding. It rejects unrepresentable weighted steps. An exact
all-ones profile is the explicit identity/disabled profile and preserves the
no-profile codestream, whose global steps may saturate. This distinction matters
at high precision: with 30-bit input and two guard bits, nominal 0.25 clips to the
finest stored step 0.5. Active weights near one remain below the representable
range; weight two reaches it. Applying weights to an already clipped rate-search
candidate would change that contract and the profile's ratios.

Rate search uses scales at or above one within the common admissible interval of
all weighted steps. Byte/BPP targets return the first candidate within 1% of the
target: each probe is a full encode, the first at a scale guessed from the
target bits per sample, the bit depth and whether the multi-component transform
applies, and the next scale is interpolated from
the measured log sizes, bisecting a bracket that fails to halve. The finest
admissible scale is returned when it fits. After
`psnr_iteration_budget` bracketed probes, the candidate closest to the target
within the size allowance is re-encoded. CPU searches return the selected probe
directly; accelerator searches re-encode the selected scale.
PSNR specifies a minimum quality within `psnr_tolerance_db`; its search uses
doubling and bounded bisection, evaluates an encountered admissible boundary,
retains a feasible candidate, and validates the returned codestream. It also probes
beyond an initial quality miss and checks the endpoint if no probe succeeds;
it does not exhaustively enumerate potentially non-monotonic quality islands.

Prepared native batches preserve significant sample values and signed storage.
For the supported 1–16-bit source routes, requested `Gray8`/`Rgb8`/`Rgba8`
output scales unsigned sample ranges to 0–255 (negative samples clamp to zero).
Requested `Gray16`/`Rgb16`/`Rgba16` output widens unsigned sources of at most
eight bits to 0–65535 and preserves native values for 9–16-bit sources. Signed
sources of every precision keep their values as 16-bit two's complement, as the
`I16` native samples hold them. Synthesized alpha uses the
output storage maximum (the native significant maximum above eight bits), while
genuine alpha is converted as a source component. Irreversible integer output
rounds the centered 9/7 sample ties-to-even before applying the unsigned level
shift. CUDA and Metal device decodes apply the same conversion to signed
grayscale and color sources. Supported formats and backend routing remain
API-specific.

## Links

- API docs: <https://docs.rs/j2k>
- [Pure-Rust JPEG 2000 codec documentation](https://frames-sg.com/j2k/rust-jpeg2000-codec/)
- Repository: <https://github.com/frames-sg/j2k>
- Support policy: <https://github.com/frames-sg/j2k/blob/main/docs/public-support.md>
