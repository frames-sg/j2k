# J2K for Python

> JPEG 2000 and HTJ2K pixels and wavelets for PyTorch and NumPy, from
> [J2K](https://github.com/frames-sg/j2k)

`j2k` decodes JPEG 2000 and HTJ2K images into tensors that keep the image's
own sample type (`uint8`, `uint16` or `int16`). It decodes batches for
training, regions at reduced resolution and raw wavelet coefficients, and it
encodes on the CPU. A GPU build decodes straight into PyTorch CUDA or MPS
memory. Batches can also contain 8-bit JPEG files. The codec is written in
Rust. NumPy and PyTorch are optional: you need them only for `to_numpy()` and
`to_torch()`.

## Install

```sh
pip install j2k            # or "j2k[torch]" / "j2k[numpy]"
```

PyPI has wheels for CPython 3.11 and later on Linux x86-64 (manylinux 2.28,
CPU) and Apple Silicon macOS (CPU and MPS). For CUDA or another platform,
build from source with Rust (rustup installs the pinned toolchain) and
maturin:

```sh
git clone https://github.com/frames-sg/j2k && cd j2k
pip install "maturin>=1.9,<2"
maturin build --release --manifest-path crates/j2k-python/Cargo.toml --out target/wheels
pip install --no-deps --no-index --find-links target/wheels j2k
```

Add `--features cuda` (Linux) or `--features metal` (Apple Silicon) to the
`maturin` command for a GPU build. CUDA also needs the build requirements in
the [GPU guide](https://github.com/frames-sg/j2k/blob/main/book/gpu.md), an
NVIDIA driver and a CUDA build of PyTorch. `j2k.cuda_enabled` and
`j2k.metal_enabled` say what a wheel was built with, and `j2k.__version__`
which release it is.

## Examples

```python
from pathlib import Path
import j2k

data = Path("image.jp2").read_bytes()
decoder = j2k.Decoder()                         # NCHW; layout="nhwc" also works
image = decoder.decode(data).to_torch()         # [C, H, W]
tile = decoder.decode(data, region=(1024, 512, 256, 256), reduction=2).to_numpy()

info = j2k.inspect(data)                        # header facts, no pixels
wavelets = j2k.decode_wavelets(data, resolutions=[0], bands=["LL"])
coarse = wavelets.components[0].bands[0].tensor.to_torch()   # int64 or float32

# samples: interleaved 8-bit RGB as bytes
codestream = j2k.encode(samples, width, height, components=3, bit_depth=8)
jph, report = j2k.encode(samples, width, height, mode="lossy", target_bpp=2.0,
                         file=True, return_report=True)
```

`region` is `(x, y, width, height)` in full-resolution coordinates;
`reduction=2` is quarter resolution. Nothing is resampled.

Batches are prepared once and can be decoded every epoch:

```python
inputs = [data, j2k.ImageRequest(data, reduction=1), data]
labels = [0, 1, 2]
prepared = decoder.prepare(inputs)
result = decoder.decode_prepared(prepared)
if result.errors or result.group_errors:
    raise RuntimeError((result.errors, result.group_errors))
for group in result.groups:
    images = group.tensor.to_torch()            # [N, C, H, W], integer samples
    batch_labels = [labels[i] for i in group.source_indices]
```

Images of different shapes or sample types come back in separate groups, in
no particular order. A group is not a training batch. `source_indices` maps
each row back to its input; `precision` and `signed` say how to scale it.

## GPU decoding

```python
decoder = j2k.Decoder(device="cuda:0")          # or "cuda", "mps"
image = decoder.decode(data).to_torch()         # never passes through host memory
```

GPU calls block until the GPU work and its status checks finish. A device the
wheel wasn't built with raises `ValueError`, and a GPU decoder never falls
back to the CPU. On CUDA the decoder holds the device's primary context and
decodes into a tensor PyTorch allocated; `"cuda"` binds PyTorch's current
device when the decoder is created. On MPS each group is decoded into a new
Metal buffer on the system default GPU and handed over as DLPack device
`(8, 0)`; `"mps:0"` means `"mps"`.

| | CPU | CUDA | MPS |
| --- | --- | --- | --- |
| `decode`, `decode_batch`, `prepare`, `decode_prepared` | Yes | Yes | Yes |
| Regions, reductions 0-31, prepared overrides | Yes | Yes | Yes |
| Both layouts; gray, RGB, RGBA; `uint8`, `uint16`, `int16` | Yes | Yes | Yes |
| Components with different sampling | Yes | Group error | Full image at full resolution only |
| JPEG inputs | Yes | Indexed error | Indexed error |
| `max_output_bytes`, cancellation | Yes | Yes | Yes |
| `max_working_memory_bytes` | Yes | Indexed error | Indexed error |
| `prepare(control=...)`, `plan_region`, `decode_srgb8`, `set_operation_cache_limit` | Yes | `ValueError` | `ValueError` |

GPU output has the CPU's rectangles and samples, for reversible and
irreversible images. A cancellation token is checked before each group is
submitted; submitted work runs to completion. `decode_wavelets` and encoding
always run on the CPU.

## Decoding

`Decoder(*, layout="nchw", workers=None, strict=True, device="cpu")`. One
image is `[C, H, W]` (NCHW) or `[H, W, C]` (NHWC); batches add a leading `N`.
`workers=None` uses every core.

- `decode(data, *, region=None, reduction=0)` returns one `Tensor` or raises
  `ValueError`; `data` can also be an `ImageRequest`.
  `decode_batch(inputs, ...)` and `prepare(inputs, ...)` report failures per
  input instead. Their `region` and `reduction` apply to every input; use
  `ImageRequest(data, *, region=None, reduction=0)` for per-input requests,
  but not both at once.
- `reduction` is an exact level count from 0 to 31 and can't exceed the
  image's decomposition levels.
- Decoding handles gray, RGB and RGBA up to 16 bits, unsigned or signed.
  Components with different sampling factors are replicated onto the image
  grid. No orientation, normalization, resizing, padding or alpha compositing
  is applied.
- Pixel decoding rejects ICC-tagged and palette images, images whose
  components are all subsampled alike, mixed precision or signedness, and
  samples deeper than 16 bits. `decode_srgb8` handles ICC and palette images,
  and `inspect` still describes the rest.

Every decode, inspection and planning call takes `bytes` or any
one-dimensional, C-contiguous buffer of unsigned bytes (`bytearray`,
`memoryview`, `mmap.mmap`, `array.array("B")`, a `uint8` NumPy array). Other
item formats and non-buffers raise `TypeError`; other layouts raise
`BufferError`. `inspect`, `plan_region`, `decode_srgb8`, `decode_wavelets` and
`decode_rgb` read `bytes` and read-only maps (`access=mmap.ACCESS_READ`, or a
`memoryview` of one) in place, and copy any other buffer first. `decode`,
batch inputs, `ImageRequest` and prepared batches keep their own copy. A
mapped file must not change or shrink during a call: that can cause errors or
wrong pixels, and truncation raises `SIGBUS`.

### JPEG input

`decode` and `decode_batch` also take complete 8-bit JPEG files, recognized by
their SOI marker, mixed with JPEG 2000. Output is always RGB `uint8`, so
grayscale JPEG gets three channels. JPEG is full-image only (a region or
reduction is an indexed error), `prepare` and `decode_prepared` don't take
it, and GPU decoders report it as an indexed error.

`ImageRequest(data, color_transform="auto")` carries a color hint from the
container, such as a slide tile's photometric interpretation: `"ycbcr"`
converts to RGB and `"none"` treats the components as RGB. JPEG 2000 inputs
reject a hint.

### Strict and lenient decoding

`strict=True`, the default, returns only images with all their data present,
with no concealment, substitution or unverified completion. `strict=False`
allows the codec's recoveries; unrecoverable errors and safety checks still
apply. The [JPEG guide](https://github.com/frames-sg/j2k/blob/main/book/jpeg.md)
lists what each mode accepts for JPEG, including missing EOI markers and
reserved Adobe APP14 transforms, which `color_transform` can resolve.

- A recovered JPEG comes back with `BatchGroup.recovered` set for its row,
  and `decode` emits `j2k.RecoveredImageWarning`.
- For JPEG 2000, lenient decoding tolerates JP2/JPH metadata errors
  (malformed trailing or optional boxes, an undeclared alpha channel) but
  never salvages entropy-coded data. The recovery is listed in
  `BatchGroup.warnings`, `recovered` stays false, and `decode` and
  `decode_srgb8` emit `RecoveredImageWarning`.
- Corruption that still forms a valid codestream can't always be detected.

### Batch results

A `Batch` has `groups`, `errors` (`(input_index, message)` pairs) and
`group_errors` (`([source_indices], message)` pairs). Successful groups are
returned even when others fail, so check both lists. A `BatchGroup` has
`tensor`, `source_indices`, `decoded_rects`, `warnings`, `recovered`,
`precision`, `signed`, `color` (`"Gray"`, `"Rgb"`, `"Rgba"`), `alpha`
(`"None"`, `"Straight"`, `"Premultiplied"`) and `layout`. A `PreparedBatch`
has `group_count` and `errors`.

Allocation and scheduling failures raise instead of returning a partial
result. JPEG output and the decoder's worker storage share the codec's 512 MiB
budget: an input that can't fit is an indexed error, and large groups are
split before allocation.

### Limits and cancellation

```python
token = j2k.CancellationToken()                 # token.cancel() from any thread
control = j2k.DecodeControl(max_working_memory_bytes=32 << 20,
                            max_output_bytes=8 << 20, cancellation=token)
batch = decoder.decode_prepared(prepared, region=(16, 16, 128, 128), reduction=4,
                                control=control)
```

- `decode_prepared(prepared, *, region=None, reduction=None, control=None)`
  overrides each prepared request with what you pass. The decoder needs the
  layout the batch was prepared with. `decode` and `decode_batch` take no
  control; prepare first.
- A `DecodeControl` covers one call. `prepare(control=...)` bounds
  preparation only, one input at a time; a failing input is an indexed error.
- `max_working_memory_bytes` must be positive and bounds the codec's plans,
  scratch and own output. It excludes the compressed input, the prepared
  source and outputs you already hold, so it isn't a process memory limit,
  and it can't raise the codec's own ceiling.
- `max_output_bytes` bounds the decoded output; 0 rejects any non-empty
  output.
- Cancellation is permanent for a token (`token.cancelled` reports it) and
  shows up in the indexed errors. A failed or cancelled input gets no tensor
  row.
- `set_operation_cache_limit(bytes)` bounds the plans a CPU decoder keeps
  between calls; 0 disables the cache.

### Region plans and sRGB output

`plan_region(data, *, region=None, reduction=0, interpretation="native")`
returns a dict describing a decode without running it or allocating pixels:
the source and output regions, `width` and `height`, the image origin,
`max_reduction`, sample metadata, and the output's `shape`, `strides`,
`row_bytes` and `buffer_bytes` in the decoder's layout. The decode can still
fail on a bad packet or a memory limit. `interpretation="srgb8"` plans
`decode_srgb8` output.

`decode_srgb8(data, *, region=None, reduction=0, control=None)` returns 8-bit
sRGB `uint8` with straight alpha, converting ICC profiles, palettes, channel
definitions and subsampled components. The
[decoding guide](https://github.com/frames-sg/j2k/blob/main/book/decoding.md)
lists the premultiplied inputs it rejects and the ICC profiles it converts
under a working-memory limit. With NCHW output, that limit counts both
layouts.

Both methods are CPU-only and JPEG 2000-only.

### Inspecting images

`inspect(data)` returns a dict read from the headers: `width`, `height`,
`components`, `bit_depth`, `signed`, `resolution_levels`,
`decomposition_levels`, `max_reduction`, the tile size and `tiles_x` by
`tiles_y` grid, the reference-grid origins (`image_x0`, `image_y0`,
`tile_x0`, `tile_y0`), the main header's `quality_layers`,
`progression_order` and code-block size, and per-component
`component_precision`, `component_signed` and `component_sampling`.
`color_specification` (`"none"`, `"enumerated"`, `"icc"`, `"unknown"`)
describes the first COLR box, which decoding applies, with its
`enumerated_colorspace` (16 is sRGB) or `icc_profile` bytes.

`max_reduction` is the deepest reduction every tile and component supports.
Tile-part COD/COC markers can make it lower than `decomposition_levels`, so
use it for IIIF scale factors, and take each size from `plan_region`, which
gives the exact output size even when the image origin shifts the reduced
grid:

```python
reductions = range(info["max_reduction"] + 1)
plans = [decoder.plan_region(data, reduction=r) for r in reversed(reductions)]
info_json = {"width": info["width"], "height": info["height"],
             "tiles": [{"width": info["tile_width"], "height": info["tile_height"],
                        "scaleFactors": [2**r for r in reductions]}],
             "sizes": [{"width": p["width"], "height": p["height"]} for p in plans]}
```

The [IIIF guide](https://github.com/frames-sg/j2k/blob/main/book/iiif.md)
covers the rest of `info.json` and the image requests.

### Tensors and DataLoader

- `Tensor.to_torch()`, `Tensor.to_numpy()` and `torch.from_dlpack(tensor)`
  hand the samples over through DLPack without copying, once: a second export
  raises `RuntimeError`. J2K keeps no view afterwards, so in-place operations
  are safe and the tensor outlives the decoder. A `Tensor` also has `shape`,
  `dtype` and `device`.
- DLPack `copy=True` and device conversion raise `BufferError`. `to_numpy()`
  rejects GPU storage before handing anything over, so `to_torch()` still
  works. NumPy usually exposes a read-only array. If a consumer rejects the
  capsule unconsumed, the `Tensor` gets its storage back.
- On the CPU, create one decoder lazily in each DataLoader worker with a
  small `workers` count. Decoding releases the GIL. Use the `spawn` start
  method: a decoder inherited across `fork` raises `RuntimeError`.
- On a GPU, have the workers return plain records (bytes, hints, coordinates,
  labels) and decode them with one persistent decoder in the training process
  or rank, not in a multi-worker collate function. Use `pin_memory=False` for
  MPS.
- `Decoder`, `PreparedBatch`, `ImageRequest` and `Tensor` can't be pickled.

## Wavelet coefficients

`decode_wavelets(data, *, tiles=None, components=None, resolutions=None,
bands=None)` returns a `WaveletImage` with a `rect` and a flat list of
`components` across tiles (tiles aren't stitched). It runs on the CPU for
JPEG 2000 and HTJ2K.

- A `WaveletComponent` has `tile_index`, `tile_rect`, `index`, `rect`,
  `precision`, `signed`, `sampling`, `transform` (`"5/3"` or `"9/7"`),
  `decomposition_levels`, `uses_color_transform` and `bands`. A
  `WaveletBand` has `tensor` (`[height, width]`), `kind`, `resolution`, `rect`
  and `quantization_step`.
- Reversible 5/3 coefficients are exact `int64`, which needs a codestream
  coded without quantization. Irreversible 9/7 coefficients are `float32`
  with the quantization step applied.
- Resolution 0 is the coarsest LL band; resolutions 1 to N hold `HL`, `LH`
  and `HH` from coarsest to finest. `HL` is high-pass in x and low-pass in y.
- Coefficients come before the inverse color transform, level shift, palette,
  channel mapping and ICC: with `uses_color_transform=True` the first three
  components are RCT or ICT, not RGB. A band is not a reduced image.
- An omitted or empty selector means all; an invalid index, or a selection
  matching no band, raises `ValueError`. Unselected bands skip entropy
  decoding, and no inverse transform runs. Packet headers are still parsed
  and coefficient storage is allocated per tile, so memory and I/O don't
  shrink in proportion.

## Encoding

`encode(data, width, height, **options)` takes interleaved samples as `bytes`
(one byte per sample up to 8 bits, two little-endian bytes above) and returns
a raw codestream, or a file with `file=True`. Invalid combinations raise
`ValueError`.

| Keyword | Meaning |
| --- | --- |
| `components`, `bit_depth`, `signed` | Sample description; defaults 3, 8, `False` |
| `codec` | `"htj2k"` (default) or `"j2k"` |
| `mode` | `"lossless"` (default) or `"lossy"` |
| `target_bpp`, `target_bytes`, `target_psnr` | Lossy target, at most one. A list gives cumulative quality layers |
| `qfactor` | HTJ2K quality factor, 1-100, instead of a target |
| `quality_layers` | Lossless layer count |
| `color_transform` | `"auto"` (default), or `"none"` to code three lossless components without the reversible color transform |
| `levels`, `progression`, `tile_size`, `tile_part_packets` | Coding parameters. `progression` is `"lrcp"` (default), `"rlcp"`, `"rpcl"`, `"pcrl"` or `"cprl"` |
| `markers` | Any of `"sop"`, `"eph"`, `"tlm"`, `"plt"`, `"plm"`, `"ppm"`, `"ppt"` |
| `file`, `icc_profile` | `file=True` writes JP2 (JPH with HTJ2K); `icc_profile` goes in its COLR box |
| `return_report` | Also return a dict of `codestream_bytes`, `bits_per_pixel`, `psnr_db` (or `None`) and `quality_layers` |

Three helpers work on 8-bit RGB:

- `encode_rgb(data, width, height, *, codec="htj2k", quality=None,
  target_bpp=None, levels=5, profile=None)` encodes baseline JPEG 4:4:4
  (`"jpeg"`), or JPEG 2000 (`"j2k"`) or HTJ2K with the irreversible color
  transform. `quality` is the JPEG quality (default 90) or the HTJ2K Qfactor;
  `target_bpp`, `levels` and a per-component quantization `profile` apply to
  the wavelet codecs. `quality` and `target_bpp` are mutually exclusive.
- `decode_rgb(data)` returns `(width, height, rgb_bytes)` for an 8-bit JPEG
  or a three-component 8-bit JPEG 2000 image.
- `forward_rgb97(data, width, height, *, levels=5, color="ict")` and
  `inverse_rgb97(bands, width, height, *, levels=5, color="ict")` run the
  float 9/7 transform on little-endian `float32` interleaved RGB of up to
  4,194,304 pixels, with (`"ict"`) or without (`"rgb"`) the color transform.
  Bands are `(component, level, orientation, width, height, bytes)` out and
  `(component, level, orientation, bytes)` in. Level 0 is LL, level 1 the
  finest detail level and `levels` the coarsest; `decode_wavelets` numbers
  detail resolutions the other way.

`encode`, `encode_rgb` and `forward_rgb97` take `bytes` only.

## Tests and versioning

From a clone, `python -m unittest discover -s crates/j2k-python/tests -v`
runs the tests. `J2K_REQUIRE_CUDA=1` or `J2K_REQUIRE_METAL=1` makes the GPU
tests fail instead of skip when the build or device is missing.

Patch releases keep this API. Before 1.0, a minor release may change it; the
[release notes](https://github.com/frames-sg/j2k/releases) say how.

## License

`j2k` is licensed under MIT or Apache-2.0.

<!-- j2k-crate-footer -->

---

Part of [J2K](https://github.com/frames-sg/j2k), a JPEG 2000 and HTJ2K codec
for Rust. See the [J2K Book](https://github.com/frames-sg/j2k/tree/main/book).
