# signinum-transcode

Experimental transcode primitives for coefficient-domain JPEG to HTJ2K work.

This crate is intentionally not a stable public conversion API yet. It starts
with constrained math proofs and keeps JPEG/HTJ2K codec coupling outside the
codec crates.

## Current Scope

The experimental path currently targets baseline sequential JPEG DCT blocks and
reversible 5/3 HTJ2K output:

```text
JPEG bytes
  -> parsed headers and entropy-decoded dequantized DCT blocks
  -> direct DCT-domain 5/3 wavelet coefficients
  -> signinum-j2k-native precomputed-band HTJ2K encode
```

It preserves native component sampling for grayscale, 4:4:4, 4:2:2, and 4:2:0
inputs. Progressive JPEG, 9/7 lossy, RGB conversion, and chroma upsample remain
out of scope.

Use `JpegToHtj2kTranscoder` when repeatedly transcoding tiles from a worker
thread; it keeps DCT block conversion and 5/3 grid-projection scratch allocated
between calls. The `jpeg_to_htj2k` function remains a stateless convenience
wrapper over the same scalar path.

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
