# J2K for Python: pixels and wavelets for PyTorch and NumPy

`j2k` is a Python extension written in Rust. It decodes JPEG 2000 and HTJ2K
(and 8-bit JPEG in batches) to tensors of the image's own sample type, encodes
on the CPU, extracts wavelet coefficients, and can decode straight into
PyTorch CUDA or MPS memory.

It is experimental and is built from this repository as the `j2k` wheel; a
wheel on PyPI is not implied. The import name and the distribution are both
`j2k`; the workspace crate is `j2k-python` and is not published to crates.io.
NumPy and PyTorch are optional: install them only if you call `to_numpy()` or
`to_torch()`.

## Build and install

You need the workspace's pinned Rust toolchain and CPython 3.11 or later (the
wheel uses the stable ABI from 3.11). NumPy 2 and PyTorch 2.12 are the tested
consumers. From the workspace root:

```sh
uvx maturin build --manifest-path crates/j2k-python/Cargo.toml --release --out target/wheels
python -m pip install --no-deps --no-index --find-links target/wheels j2k
```

- **CUDA** (Linux): add `--features cuda` to the Maturin command. It needs the
  CUDA build requirements of `j2k-cuda`, an NVIDIA driver, and a CUDA-enabled
  PyTorch. `j2k.cuda_enabled` reports whether the wheel has it.
- **MPS** (Apple Silicon macOS): add `--features metal`. `j2k.metal_enabled`
  reports whether the wheel has it.

A wheel without a backend rejects that `device`. An explicit GPU request
never falls back to the CPU.

## Decode

```python
from pathlib import Path
import j2k

data = Path("image.jp2").read_bytes()
print(j2k.inspect(data))
decoder = j2k.Decoder(layout="nchw", workers=1)
image = decoder.decode(data).to_torch()               # [C,H,W], uint8/uint16/int16
small = decoder.decode(data, reduction=1).to_numpy()  # half resolution
```

`Decoder(layout="nchw", workers=None, strict=True, device="cpu")`. A single
NHWC image is `[H,W,C]`; batches add a leading `N`.

### Requests

- `region=(x, y, width, height)` is in full-resolution image coordinates and
  can be combined with `reduction`.
- `reduction` is an exact count from 0 through 31: 1 is half resolution, 2 a
  quarter. It cannot exceed the image's decomposition levels; a deeper request
  raises `ValueError` (or an indexed batch error). Nothing is resampled.
- `decode_batch` and `prepare` take the same `region` and `reduction` for
  every input, or per-input `ImageRequest(data, region=..., reduction=...)`
  values, but not both.

### What decodes

- Gray, RGB, or RGBA with unsigned samples up to 16 bits or signed samples up
  to 16 bits, as `uint8`, `uint16`, or `int16`.
- Components with different sampling factors are replicated onto the image
  grid.
- Rejected in pixel decoding: ICC-tagged and palette images, images whose
  components are all subsampled alike, mixed precision or signedness, and
  samples deeper than 16 bits. `decode_srgb8` (below) handles ICC and palette
  images. `inspect` still reports per-component precision, signedness, and
  sampling for images that pixel decoding rejects.
- No orientation, display normalization, resizing, padding, or alpha
  compositing is applied.

### JPEG

`decode` and `decode_batch` also take complete 8-bit JPEG files, recognized
by their SOI marker, in the same call as JPEG 2000.

- Output is always RGB `uint8`; grayscale is expanded to three channels.
  Grayscale JPEG 2000 stays one channel. There is no `channels` option.
- Full image only. A region or reduction is an indexed error.
- `prepare` and `decode_prepared` are JPEG 2000 only.
- CUDA and MPS decoders report JPEG inputs as indexed errors.
- `ImageRequest(data, color_transform="auto")` carries a color hint from the
  container: `ycbcr` converts to RGB, `none` treats the components as RGB.
  A slide reader that knows a tile's photometric interpretation should pass
  it this way. Hints are rejected for JPEG 2000.

### Encoded input

Every decode, inspection, and planning call takes `bytes` or any other
one-dimensional, C-contiguous buffer of unsigned bytes: `bytearray`,
`memoryview`, `mmap.mmap`, `array.array("B")`, or a `uint8` NumPy array.
Other buffer layouts raise `BufferError`, and non-buffers raise `TypeError`.

- `inspect`, `plan_region`, `decode_srgb8`, `decode_wavelets`, and
  `decode_rgb` read `bytes` and read-only memory maps
  (`mmap.mmap(..., access=mmap.ACCESS_READ)`, or a `memoryview` of either) in
  place.
- Any other buffer is copied once before decoding, so a write during the call
  cannot change what the codec reads.
- Batch inputs, `ImageRequest`, and prepared batches always keep their own
  copy.
- A mapped file must not change or shrink while a call reads it. A writer can
  cause a decode error or wrong pixels, and truncation raises `SIGBUS`. Pass
  `bytes` if you cannot guarantee that.

### Strict and lenient

`strict=True` is the default. A successful strict decode has all the required
image data, with no damage concealment, substituted data, or unverified
completion. A violation is an indexed error with no tensor row.

`strict=False` allows the supported recoveries. It does not suppress
unrecoverable errors, safety checks, or device rules.

- **JPEG.** A recovered image is returned with `BatchGroup.recovered` set for
  its row and no error entry. `decode` emits `j2k.RecoveredImageWarning`
  instead; use `decode_batch([request])` for per-row status. A missing EOI is
  accepted only when completeness is verified: a progressive file without one
  needs every coefficient band at its final approximation. A file that does
  end with EOI may stop its progressive script early. A
  reserved Adobe APP14 transform byte is an error when strict, unless
  `color_transform` resolves a three-component image; lenient decoding reads
  it as libjpeg-turbo does (YCbCr for three components, YCCK for four).
- **JPEG 2000.** Lenient decoding tolerates selected JP2/JPH metadata errors
  and never salvages entropy data. `recovered` stays false; the recovery is
  listed in `BatchGroup.warnings`, and `decode` and `decode_srgb8` emit
  `RecoveredImageWarning` for it.

Detection has limits: corruption that still forms a valid codestream cannot
always be detected.

## Batches

```python
inputs = [data, j2k.ImageRequest(data, reduction=1), data]
labels = [0, 1, 2]
prepared = decoder.prepare(inputs)

for epoch in range(2):
    result = decoder.decode_prepared(prepared)
    if result.errors or result.group_errors:
        raise RuntimeError((result.errors, result.group_errors))
    for group in result.groups:
        images = group.tensor.to_torch()  # [N,C,H,W]
        ordered_labels = [labels[i] for i in group.source_indices]
        # Example scaling; choose what your model needs.
        images = images.float()
        if not group.signed:
            images = images - 2 ** (group.precision - 1)
        images = images / 2 ** (group.precision - int(group.signed))
```

- Images of different shapes or sample types come back in separate groups. A
  group is not a training batch. Group order is unspecified.
- `group.source_indices` gives each row's position in the original input
  list, including when other inputs fail.
- `errors` is a list of `(input_index, message)` and `group_errors` a list of
  `([source_indices], message)`. Successful groups are still returned. Check
  both lists.
- Each group has `precision`, `signed`, `color`, `alpha`, `layout`,
  `decoded_rects`, `warnings`, and `recovered`.
- Allocation and scheduling failures raise instead of returning a partial
  result.
- JPEG output and the session's worker storage share the codec's 512 MiB
  budget. An input that cannot fit is an indexed error, and large groups are
  split before allocation.

## Limits and cancellation

```python
prepared = decoder.prepare([data], control=j2k.DecodeControl(
    max_working_memory_bytes=64 << 20))
token = j2k.CancellationToken()
control = j2k.DecodeControl(
    max_working_memory_bytes=32 << 20,
    max_output_bytes=8 << 20,
    cancellation=token,
)
batch = decoder.decode_prepared(prepared, region=(16, 16, 128, 128),
                                reduction=4, control=control)
# Another thread can call token.cancel() while the decode runs.
```

- `decode_prepared` takes optional `region`, `reduction`, and `control`
  overrides. Omitted overrides keep each prepared request. The decoder must
  have the layout the batch was prepared with.
- A `DecodeControl` applies to one call. `prepare(control=...)` bounds
  preparation only, one input at a time; a failing input is an indexed
  preparation error.
- `max_working_memory_bytes` bounds the codec's plans, scratch, and its own
  output for the operation. It must be positive and cannot raise the engine's
  ceiling. Compressed input, the prepared source, and outputs you already
  hold are not counted, so it is not a process memory limit.
- `max_output_bytes` bounds the decoded output; 0 rejects any non-empty
  output.
- Cancellation is permanent for a token and is reported in the indexed error
  messages. A failed or cancelled item returns no tensor row.
- `decode` and `decode_batch` take no control: prepare first.
- `Decoder.set_operation_cache_limit(bytes)` bounds the derived plans a CPU
  session keeps between calls; 0 disables that cache.

## Region plans and sRGB output

`Decoder.plan_region(data, region=..., reduction=..., interpretation="native")`
returns a dict describing a decode without running it: source and output
rectangles, the image origin, the maximum reduction, sample metadata, shape,
strides, row bytes, and total bytes, in the decoder's layout. It allocates no
pixels. The decode can still fail later on a bad packet or a working-memory
limit. `interpretation="srgb8"` plans the output of `decode_srgb8`.

`Decoder.decode_srgb8(data, region=..., reduction=..., control=...)` returns a
CPU `uint8` tensor in 8-bit sRGB, converting gray and RGB, ICC profiles,
palettes, channel definitions, and subsampled components. RGBA output has
straight alpha.

- Premultiplied color is un-premultiplied at the source precision before
  conversion. Premultiplied CIE Lab input is rejected, as is premultiplied
  input with subsampled or signed components unless it is 8-bit mapped
  samples that need no color conversion.
- With a working-memory limit, ICC conversion needs a matrix/TRC or gray-TRC
  profile with an XYZ connection space, and NCHW output counts both layouts
  against the limit.

Both methods are CPU-only and JPEG 2000-only.

## Image-server metadata (IIIF)

`inspect(data)` reports what an IIIF `info.json` needs, without decoding
pixels:

- `width`, `height`, `resolution_levels`;
- `max_reduction`, the deepest reduction every tile and component supports.
  Tile-part COD/COC markers can make it lower than the main header's
  `decomposition_levels`, so use it for scale factors;
- `tile_width`, `tile_height`, and the `tiles_x` by `tiles_y` grid;
- `image_x0`, `image_y0`, `tile_x0`, `tile_y0`, the reference-grid origins;
- the main-header `quality_layers`, `progression_order`, `code_block_width`,
  and `code_block_height`;
- `color_specification` (`"none"`, `"enumerated"`, `"icc"`, or `"unknown"`),
  for the first COLR box, which decoding applies, with its
  `enumerated_colorspace` (such as 16 for sRGB) or `icc_profile` bytes.

Take each size from `plan_region`, which gives the exact output of a reduced
decode, including for images whose origin shifts the reduced grid:

```python
info = j2k.inspect(data)
decoder = j2k.Decoder()
reductions = range(info["max_reduction"] + 1)
sizes = []
for reduction in reversed(reductions):
    plan = decoder.plan_region(data, reduction=reduction)
    sizes.append({"width": plan["width"], "height": plan["height"]})
info_json = {
    "@context": "http://iiif.io/api/image/3/context.json",
    "id": identifier,
    "type": "ImageService3",
    "protocol": "http://iiif.io/api/image",
    "profile": "level1",
    "width": info["width"],
    "height": info["height"],
    "tiles": [{"width": info["tile_width"], "height": info["tile_height"],
               "scaleFactors": [2**reduction for reduction in reductions]}],
    "sizes": sizes,
}
```

`tests/test_iiif.py` runs this against tiled and offset images. The image
requests themselves are covered in
[docs/iiif.md](https://github.com/frames-sg/j2k/blob/main/docs/iiif.md).

## Using it in a DataLoader

- **CPU.** Create one CPU decoder lazily inside each DataLoader worker, with
  a small explicit `workers` count, and return ordinary CPU torch tensors.
  The decoder releases the interpreter while it decodes. Use the `spawn`
  start method: a session inherited across `fork` is rejected.
- **CUDA or MPS.** Have reader workers return plain Python records (bytes,
  hints, coordinates, labels). Decode them with one persistent GPU decoder in
  the training process or distributed rank, not in a multiworker collate
  function. Use `pin_memory=False` for MPS.
- `Decoder`, `PreparedBatch`, `ImageRequest`, and `Tensor` have no
  multiprocessing serialization support. Do not send them between processes.

## Tensor ownership

- `Tensor.to_torch()`, `Tensor.to_numpy()`, and `torch.from_dlpack(tensor)`
  hand the samples to the consumer through DLPack without copying. Each
  `j2k.Tensor` can be exported once; a second export raises `RuntimeError`.
  Make further views with the framework tensor.
- After export, Rust keeps no view of the samples, so in-place operations are
  safe, and the framework tensor outlives the decoder.
- DLPack `copy=True` and device conversion are rejected, not performed.
- `to_numpy()` rejects GPU storage before transferring ownership, so
  `to_torch()` still works afterwards. NumPy usually exposes a read-only
  array.
- If a consumer rejects the capsule without consuming it, the storage goes
  back to the `j2k.Tensor`, which can be exported again.

## Wavelet coefficients

```python
wavelets = j2k.decode_wavelets(data)
coarse = j2k.decode_wavelets(data, resolutions=[0], bands=["LL"])
details = j2k.decode_wavelets(data, components=[0], resolutions=[1], bands=["HL", "LH", "HH"])

for component in coarse.components:
    print(component.tile_index, component.index, component.transform)
    for band in component.bands:
        values = band.tensor.to_torch()  # [band_height, band_width]
```

- CPU only, for classic JPEG 2000 and HTJ2K.
- Reversible 5/3 coefficients are exact `int64`. Irreversible 9/7
  coefficients are `float32` with `quantization_step` already applied.
  Reversible output needs a codestream coded without quantization.
- Resolution 0 is the coarsest LL band; resolutions 1 through N are detail
  bands from coarsest to finest. `HL` is high-pass in x and low-pass in y;
  `LH` is the reverse. Each band has its own `(x, y, w, h)` rectangle.
- `components` is a flat list across tiles. Each entry has its tile index and
  rectangle, component index, precision, signedness, sampling, transform, and
  decomposition count. Tiles are not stitched.
- These are codestream components before the inverse color transform, level
  shift, palette or channel mapping, and ICC. `uses_color_transform=True`
  means the first three components are in the RCT/ICT domain, not RGB. A
  wavelet band is not a display-ready reduced image.
- `tiles`, `components`, `resolutions`, and `bands` select what is returned;
  an omitted or empty selector means all. An invalid index, or a selection
  that matches no band, is an error.
- Selection skips the entropy decoding of unrequested bands and all inverse
  transform, color, and packing work. Packet headers are still parsed and
  coefficient storage is still allocated for the tile, so memory and I/O do
  not shrink in proportion.

## Encoding

```python
codestream = j2k.encode(samples, width, height, components=3, bit_depth=8)
jph, report = j2k.encode(samples, width, height, mode="lossy", target_bpp=2.0,
                         file=True, return_report=True)
```

`encode` takes interleaved samples as `bytes` on the CPU: one byte per sample
up to 8 bits, two little-endian bytes above. Defaults are `codec="htj2k"`,
`mode="lossless"`, `progression="lrcp"`, and `color_transform="auto"`.

| Keyword | Meaning |
| --- | --- |
| `components`, `bit_depth`, `signed` | Sample description |
| `codec` | `"j2k"` or `"htj2k"` |
| `mode` | `"lossless"` or `"lossy"` |
| `target_bpp`, `target_bytes`, `target_psnr` | Lossy target; at most one. A list gives cumulative quality layers |
| `qfactor` | HTJ2K quality factor, instead of a target |
| `quality_layers` | Lossless layer count |
| `color_transform` | `"none"` codes three components without the reversible color transform |
| `levels`, `progression`, `tile_size`, `tile_part_packets` | Coding parameters; `progression` is `"lrcp"`, `"rlcp"`, `"rpcl"`, `"pcrl"`, or `"cprl"` |
| `markers` | Any of `"sop"`, `"eph"`, `"tlm"`, `"plt"`, `"plm"`, `"ppm"`, `"ppt"` |
| `file`, `icc_profile` | `file=True` writes JP2 (JPH with HTJ2K); `icc_profile` goes in its COLR box |
| `return_report` | Also return `codestream_bytes`, `bits_per_pixel`, `psnr_db` (or `None`), and `quality_layers` |

The keywords map one to one onto the `j2k` crate's encode options, which
validate every combination. An invalid combination raises `ValueError`.

### RGB helpers

- `encode_rgb(data, width, height, codec="htj2k")` encodes interleaved 8-bit
  RGB as baseline JPEG 4:4:4 (`"jpeg"`), classic JPEG 2000 (`"j2k"`), or
  HTJ2K with the ICT. `quality` is the JPEG quality or the HTJ2K Qfactor; the
  wavelet codecs also take `target_bpp`, `levels`, and a per-component
  quantization `profile`. `quality` and `target_bpp` are mutually exclusive.
- `decode_rgb(data)` returns `(width, height, rgb_bytes)` for an 8-bit JPEG
  or a three-component 8-bit JPEG 2000 image.
- `forward_rgb97` and `inverse_rgb97` run the float 9/7 wavelet transform on
  little-endian `float32` interleaved RGB, with (`color="ict"`) or without
  (`"rgb"`) the ICT. Here level 1 is the finest detail level, the opposite of
  `decode_wavelets`.

`encode`, `encode_rgb`, and `forward_rgb97` take `bytes` only.

## GPU output

```python
decoder = j2k.Decoder(device="cuda:0")   # or "mps"
image = decoder.decode(data).to_torch()
assert image.device.type == "cuda"
```

Both GPU decoders are blocking: a call returns after the GPU work and the
codec's status checks finish. Decoded pixels never pass through host memory.

- **CUDA.** The decoder keeps the device's primary context, allocates a
  PyTorch tensor, and decodes into it. `device="cuda"` binds PyTorch's current
  CUDA device when the decoder is created; `"cuda:N"` is explicit. PyTorch
  owns the tensor afterwards.
- **MPS** (Apple Silicon). The decoder decodes each group into a new private
  Metal buffer on the system default GPU and hands it to PyTorch as DLPack
  device `(8, 0)`. `"mps"` and `"mps:0"` are the same device.

| | CPU | CUDA | MPS |
| --- | --- | --- | --- |
| `decode`, `decode_batch`, `prepare`, `decode_prepared` | Yes | Yes | Yes |
| Regions, reductions 0-31, prepared overrides | Yes | Yes | Yes |
| Both layouts; Gray, RGB, RGBA; `uint8`, `uint16`, `int16` | Yes | Yes | Yes |
| Components with different sampling | Yes | Group error | Full image at full resolution only |
| JPEG inputs | Yes | Indexed error | Indexed error |
| `max_output_bytes` and cancellation | Yes | Yes | Yes |
| `max_working_memory_bytes` | Yes | Indexed error | Indexed error |
| `prepare(control=...)`, `plan_region`, `decode_srgb8`, `set_operation_cache_limit` | Yes | No | No |

GPU output matches the CPU's rectangles and samples, for reversible and
irreversible 9/7 images. On a GPU a cancellation token is checked before each
group is submitted; submitted work runs to completion. `decode_wavelets` and
encoding always run on the CPU.

## Tests

```sh
cargo test -p j2k --test wavelets
python -m unittest discover -s crates/j2k-python/tests -v
```

Set `J2K_REQUIRE_CUDA=1` or `J2K_REQUIRE_METAL=1` to make the GPU tests fail,
rather than skip, when the build or device is missing. The MPS path is tested
with PyTorch 2.12.1 on Apple Silicon.

Patch releases keep this API. A pre-1.0 minor release may change it, with
migration notes.

## Licenses

`j2k` is licensed under MIT or Apache-2.0. Its JPEG codec includes code
derived from libjpeg-turbo, covered by the licenses in `LICENSE.libjpeg-turbo`
and `README.ijg`. This software is based in part on the work of the
Independent JPEG Group.

## Publishing

The `python-bindings.yml` workflow builds and tests CPU wheels for Linux
(manylinux) and CPU/MPS wheels for Apple Silicon macOS. MPS tests skip on
runners without GPU access, and CUDA builds are validation-only.

Publishing runs only on a manual dispatch from `main` with `publish` enabled,
after the wheel tests and any requested CUDA checks pass. It uses the `pypi`
GitHub environment and short-lived OIDC credentials. To set it up, configure a
PyPI pending Trusted Publisher with project `j2k`, owner `frames-sg`,
repository `j2k`, workflow `python-bindings.yml`, and environment `pypi`. The
first upload creates the PyPI project; a pending publisher does not reserve
the name. Each release needs a new workspace version.
