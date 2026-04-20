# M8 — slidecodec-tilecodec Design

Status: approved implementation spec derived from the umbrella design.

## Goal

Add `slidecodec-tilecodec`, a byte-to-byte tile decompression crate that fits
the shared `TileDecompress` trait surface in `slidecodec-core` and covers the
four compression modes explicitly called out in the umbrella:

- Deflate
- Zstd
- LZW
- Uncompressed

The crate is not a new image codec. It is a WSI/TIFF-facing utility crate for
decompressing tile payloads into caller-owned buffers with reusable typed
scratch state.

## Scope

In scope:

- A new `crates/slidecodec-tilecodec` crate in the workspace
- Four public codec entry points:
  - `DeflateCodec`
  - `ZstdCodec`
  - `LzwCodec`
  - `UncompressedCodec`
- One typed pool per codec:
  - `DeflatePool`
  - `ZstdPool`
  - `LzwPool`
  - `NoPool`
- One shared typed error enum `TileCodecError`
- Behavior-focused tests for valid decode, undersized output buffers, and pool
  reuse
- A compare bench that measures throughput against the underlying reference
  libraries used in the implementation
- Bench/docs/CI wiring so the crate is part of the workspace release gate

Out of scope:

- TIFF container parsing
- Predictor undo, byte shuffling, or image-layout transforms above raw
  decompression
- LZMA, PackBits, or WebP/JXL image codecs

## Implementation choices

### Deflate

Use `flate2` with the C `zlib` backend and support both common payload shapes:

- zlib-wrapped Deflate
- raw Deflate

`DeflateCodec::decompress_into` tries zlib first, then retries raw Deflate when
the wrapped path reports invalid data. This is a narrowly scoped
wire-format normalization inside one codec, not higher-level codec
auto-detection: the container still dispatches explicitly to `DeflateCodec`,
but the wire payload may legitimately appear with or without a zlib wrapper.

### Zstd

Use the `zstd` crate’s reusable bulk decompressor stored inside `ZstdPool`.
This keeps the public API typed and scratch-oriented while delegating the heavy
lifting to the tuned `libzstd` backend.

### LZW

Use `weezl` for TIFF-style LZW decode. `LzwPool` owns a grow-only `Vec<u8>`
scratch buffer used as the staging area for decode output before validating and
copying into the caller’s buffer.

LZW is not a primary perf target in the umbrella contract, so correctness and
predictable API shape matter more than heroic optimization here.

### Uncompressed

`UncompressedCodec` is a checked `memcpy`:

- `expected_size(input)` returns `Some(input.len())`
- `decompress_into` copies into the caller’s buffer or returns
  `BufferError::OutputTooSmall`

## Public API

The crate exports:

- `TileCodecError`
- `DeflateCodec`
- `ZstdCodec`
- `LzwCodec`
- `UncompressedCodec`
- `DeflatePool`
- `ZstdPool`
- `LzwPool`
- `NoPool`

Each codec implements `slidecodec_core::TileDecompress`.

No auto-detection codec is added in M8. The higher-level WSI reader already
knows the container compression enum and should dispatch explicitly.

## Error model

`TileCodecError` composes the shared `slidecodec-core` sub-errors:

- `TileCodecError::Buffer(BufferError)`
- `TileCodecError::Input(InputError)`
- `TileCodecError::Unsupported(Unsupported)`
- `TileCodecError::Backend(&'static str or String)`

Invalid/truncated compressed payloads are surfaced as `InputError` where the
backend exposes a clear boundary; otherwise they become `Backend`.

`expected_size` behavior is fixed per codec:

- `UncompressedCodec`: `Some(input.len())`
- `DeflateCodec`: `None`
- `ZstdCodec`: `None` in M8
- `LzwCodec`: `None`

## Verification

M8 is complete when:

- `cargo test -p slidecodec-tilecodec`
- `cargo clippy -p slidecodec-tilecodec --all-targets -- -D warnings`
- `cargo bench -p slidecodec-tilecodec --bench compare --no-run`
- `cargo check --manifest-path crates/slidecodec-tilecodec/fuzz/Cargo.toml`

all pass, and the compare bench compiles with reference-library comparator
paths enabled on the local host.
