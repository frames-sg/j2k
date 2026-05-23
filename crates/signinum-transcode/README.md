# signinum-transcode

Experimental transcode primitives for coefficient-domain JPEG to HTJ2K work.

This crate is intentionally not a stable public conversion API yet. It starts
with constrained math proofs and keeps JPEG/HTJ2K codec coupling outside the
codec crates.

## Current Scope

The experimental path currently targets baseline sequential and progressive
JPEG DCT blocks, reversible 5/3 HTJ2K output, and an opt-in irreversible 9/7
float-linear path:

```text
JPEG bytes
  -> parsed headers and entropy-decoded quantized/dequantized DCT blocks
  -> direct DCT-domain 5/3 or 9/7 wavelet coefficients
  -> signinum-j2k-native precomputed-band HTJ2K encode
```

It preserves native component sampling for grayscale, 4:4:4, 4:2:2, and 4:2:0
inputs. Progressive JPEG coefficients are accumulated from all scans before
the DCT-to-wavelet stage. `FloatDirectLinear97` computes the first
irreversible 9/7 level directly from DCT blocks, recurses conventionally over
LL for additional levels, and encodes through the native precomputed lossy
HTJ2K boundary. RGB conversion and chroma upsample remain out of scope.

`JpegToHtj2kCoefficientPath::IntegerDirect53` is the default production path. It
computes the first reversible 5/3 level from DCT blocks without a full spatial
image plane, then recurses conventionally over LL for additional levels. The
floating-point 5/3 and 9/7 paths remain selectable for math-oracle validation
and lossy experiments.

Use `JpegToHtj2kTranscoder` when repeatedly transcoding tiles from a worker
thread. The `jpeg_to_htj2k` function remains a stateless convenience wrapper
over the same scalar path. Reusable scratch covers the float-linear validation
path and the default integer-direct block-local ISLOW sample cache.
`JpegToHtj2kTranscoder::transcode_with_accelerator` accepts an optional
`DctToWaveletStageAccelerator` for future SIMD/GPU backends; the default
accelerator always falls back to the scalar oracle.
Use `JpegToHtj2kOptions::lossless_53()` and `JpegToHtj2kOptions::lossy_97()`
instead of manually combining coefficient paths with reversible/irreversible
encoder settings. Contradictory options fail before JPEG parsing.

`TranscodeReport` includes the typed coefficient path and optional validation
classifications. When an oracle is enabled, metrics are classified as exact,
one-LSB-bounded at the 99.9% exact-match threshold, or outside threshold.

`htj2k_wavelet::WaveletImage53<i32>` and `WaveletImage97<f32>` can be validated
and converted into `signinum-j2k-native`'s precomputed HTJ2K representations,
including SIZ sampling checks against the caller-provided reference grid. This
keeps the cross-codec adapter in `signinum-transcode`.

## Promotion Gate

Do not expose this crate as a stable conversion API until all of the following
are true and documented with current evidence:

- Synthetic 1D and 2D DCT-to-5/3 tests pass.
- Real JPEG grayscale, 4:4:4, 4:2:2, and 4:2:0 transcode tests pass.
- Generated HTJ2K codestreams decode with the native decoder.
- At least one external decoder path accepts generated HTJ2K fixtures where
  tooling is available.
- Error histogram reporting is documented for committed fixtures and optional
  local WSI tiles.
- Unsupported JPEG modes fail loudly instead of silently falling back.
- `signinum-jpeg` and `signinum-j2k-native` remain independent; cross-codec
  coupling stays in `signinum-transcode`.

See [`../../docs/dct-to-htj2k-notes.md`](../../docs/dct-to-htj2k-notes.md) for
the current validation commands and optional WSI corpus environment variables.
