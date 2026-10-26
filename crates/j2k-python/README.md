# J2K pixels and wavelets in PyTorch

`j2k-python` implements the `j2k` native extension entirely in Rust. It
exposes JPEG 2000/HTJ2K inspection, CPU pixel batches, selected native wavelet
bands, and optional direct CUDA and MPS output for PyTorch. NumPy and PyTorch are
optional consumers. Bare `device="cuda"` imports PyTorch at construction to bind
the current CUDA device for the decoder lifetime; explicit `cuda:N` remains explicit.

The extension is experimental and is built as the `j2k` wheel. This does not
mean a wheel has been published to PyPI. Ordinary Rust builds do not activate
PyO3; the extension is enabled only by its `python`, `extension-module`, or
`cuda`/`metal` Cargo features. The Rust crate is not published separately on crates.io.

The distribution and Python import are both named `j2k`; the workspace crate
remains `j2k-python`. All codec and adapter implementation is Rust. The only
Python source maintained here is the consumer integration tests; `j2k.pyi`
provides type information, and Maturin generates the wheel's import loader.

## Build and install

Use the workspace's pinned Rust toolchain and an existing Python environment.
Maturin builds the native extension; the wheel uses CPython's stable ABI from
3.10 onward. NumPy 2 and PyTorch 2.12 are the initial consumer targets.

From the workspace root:

```sh
uvx maturin build --manifest-path crates/j2k-python/Cargo.toml --release --out target/wheels
python -m pip install --no-deps --no-index --find-links target/wheels j2k
```

Install the appropriate NumPy/PyTorch distribution in that environment if it
is not already available. Runtime checks are exercised against the installed
consumer versions; the wheel ABI tag alone does not establish every
Python/PyTorch/platform combination as hardware-validated.

On a supported Linux CUDA build host, add `--features cuda` to the Maturin
command. Follow the existing CUDA codec build requirements. CUDA wheels require
the driver, supported hardware, and a CUDA-enabled PyTorch installation. A CPU
wheel rejects `device="cuda"`; an explicit GPU request never silently uses CPU.

On Apple Silicon macOS, add `--features metal` for `device="mps"` or `device="mps:0"` support.
The macOS wheel workflow enables this feature. `j2k.metal_enabled` reports
whether the extension includes it; `j2k.cuda_enabled` reports CUDA build support.

## Pixels and native sample types

```python
from pathlib import Path
import j2k

data = Path("image.jp2").read_bytes()
print(j2k.inspect(data))
decoder = j2k.Decoder(layout="nchw", workers=1)
image = decoder.decode(data).to_torch()  # [C,H,W], U8/U16/I16
small = decoder.decode(data, reduction=1).to_numpy()  # [C,H,W], half resolution
```

`reduction` is 0, 1, 2, or 3 for full, half, quarter, or eighth resolution.
Each halving comes from the codestream's wavelet ladder, so `reduction` cannot
exceed the decomposition levels; a deeper request raises `ValueError` (or an
indexed batch error) rather than resampling or cropping a shallower reduction.
`region=(x,y,width,height)` uses full-resolution source coordinates and can be
combined with reduction. NHWC single-image output is `[H,W,C]`. Actual batch
rectangles are returned with each group.

Pixel decoding uses the existing homogeneous batch contract: uniform Gray,
RGB, or RGBA with U8, U16, or I16 output. ICC-tagged, palette-mapped,
component-subsampled, mixed-precision/signedness, and above-16-bit pixel-batch
inputs are rejected. The general Rust decoder supports additional cases.
Inspection reports per-component precision, signedness, and sampling. The
adapter does not apply orientation, ICC transforms, resizing, padding, alpha
compositing, or normalization.

`decode` and `decode_batch` also accept complete 8-bit JPEG files, detected by
their SOI marker, in the same call as JPEG 2000 inputs. JPEG groups are always
RGB `uint8` (grayscale is expanded to RGB) and are grouped by size;
`source_indices` and errors refer to positions in the original input list.
JPEG inputs support full-image decode only: a region or reduction is reported
as an indexed error. `prepare` and `decode_prepared` remain JPEG 2000 only, and
a CUDA or MPS decoder reports JPEG inputs as indexed errors rather than decoding them
on the CPU.

JPEG tensor output remains RGB uint8 by default, with grayscale expanded to three identical channels. Grayscale JPEG 2000 and HTJ2K remain one channel. Output metadata describes the returned tensor. A later explicit `channels="native"` option may preserve grayscale JPEG as one channel without changing existing defaults. Source component counts, when exposed, are per input row and remain aligned with source_indices.

`Decoder(strict=True)` is the default for every supported codec. A normal success must have completely decoded the required image data without damage concealment, missing-data substitution or unverified completion, under the requested interpretation and output contract. Detected violations produce indexed batch errors and no tensor row for the affected image. Proven non-pixel-affecting irregularities may remain warnings; missing EOI is harmless only when completeness is established, and arbitrary stray-byte skipping is not automatically harmless. `strict=False` permits supported salvage with explicit per-index recovery status and warnings. It does not suppress unrecoverable errors, safety checks, precision requirements or device policy. A recovered row is not also an error entry.

`BatchGroup.recovered` aligns with `source_indices` and tensor rows. Cosmetic
warnings leave it false. Single-image lenient recovery emits
`j2k.RecoveredImageWarning`; use `decode_batch([request])` for structured status.
Strict progressive EOF acceptance requires every coefficient band of every
component through final approximation. An actual EOI may terminate a legal
partial script. The current J2K lenient path tolerates selected metadata errors
and performs no entropy salvage. The guarantee is detection based; corruption
that still forms a valid, indistinguishable codestream cannot always be detected.
There is no `channels` option yet; the names `default`, `native`, and `rgb` are
reserved for a future explicit policy.

`ImageRequest(data, color_transform="auto")` accepts `auto`, `ycbcr`, or `none`
to carry a JPEG color hint from the slide container. `ycbcr` forces conversion to RGB;
`none` treats the components as RGB. Overrides are rejected for JPEG 2000.
For `wsi_rs.CompressedTile` values, pass each tile's `data` and
`color_transform` to an `ImageRequest`; this preserves the reader's effective
photometric interpretation even when a JPEG header is ambiguous.

## Batching, labels, and reuse

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
        # Example native-range centering; choose the policy your model requires.
        images = images.float()
        if not group.signed:
            images = images - 2 ** (group.precision - 1)
        images = images / 2 ** (group.precision - int(group.signed))
```

Different shapes produce separate groups; grouping is not training-batch
collation. `source_indices` refers to the original submission, including when
other inputs fail. `errors` contains `(input_index, message)` and `group_errors`
contains `([source_indices], message)`. Successful groups remain available.
Group order is unspecified. `source_indices` is authoritative for submission
order and label alignment. Allocation or scheduling failures abort the entire
call with an exception instead of returning partial results. Dense JPEG output and the session's worker storage share the 512 MiB codec
budget. Inputs that cannot fit individually are indexed errors. Groups are
split into bounded subgroups before allocation. Retaining returned groups
retains their owned outputs, independently of future session calls.

Each group preserves precision, signedness, color/alpha interpretation,
warnings, and decoded rectangles. Do not discard the error lists silently.

For CPU decoding, create one CPU decoder lazily inside each DataLoader worker,
with a small explicit worker count, and convert outputs to ordinary CPU torch
tensors before returning them. The decoder releases the interpreter during CPU
work and locks its session only after detaching. Use spawn; sessions inherited
across fork are rejected.

For CUDA or MPS, reader workers return built-in Python records containing bytes,
hints, coordinates and labels. One persistent GPU decoder in each training
process or distributed rank constructs ImageRequest values and decodes the
received batch there. Keep GPU decode out of multiworker collate functions.
Use pin_memory=False for MPS. j2k Decoder, PreparedBatch, ImageRequest and Tensor
owners have no documented multiprocessing serialization contract.

## Tensor ownership

`Tensor.to_torch()`, `Tensor.to_numpy()`, and `torch.from_dlpack(tensor)` consume
the output owner's DLPack export exactly once. CPU data is shared with the
consumer without copying decoded samples. Rust retains no readable or mutable
view of exported samples, so consumer in-place operations are safe. The
returned framework tensor outlives the decoder and preparation objects.

For a second view, use the returned framework tensor's APIs. Re-exporting the
original `j2k.Tensor` raises `RuntimeError`. DLPack `copy=True` and device
conversion are rejected rather than performed implicitly. NumPy's consumer
normally exposes a read-only array. Convert or copy explicitly if necessary.
`to_numpy()` rejects GPU storage before transferring ownership, so a rejected
call can still be followed by `to_torch()`. A foreign consumer that rejects an unconsumed capsule returns storage to the
original live `j2k.Tensor`, which can then be imported successfully. If the
producer has already been destroyed, capsule destruction frees storage. Once a
consumer consumes the capsule, the consumer owns storage until its deleter runs.

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

Extraction currently runs on CPU, for classic JPEG 2000 and HTJ2K. Reversible
5/3 coefficients use exact `int64` storage; irreversible 9/7 values use
dequantized `float32`, including midpoint reconstruction of incomplete coding
passes. `quantization_step` is already applied. No model normalization is
performed. Reversible coefficients require no-quantization coding; other
native codec precision/style limits still apply.

Resolution 0 is the coarsest LL band. Resolutions 1 through N contain detail
bands from coarsest to finest. `HL` is high-pass in x and low-pass in y; `LH`
is low-pass in x and high-pass in y. Each band retains its own rectangle, so
odd sizes and tile-origin phase are explicit. Rectangle tuples are `(x,y,w,h)`.
The flattened component list retains tile index/rectangle, component index,
precision, signedness, sampling, transform, and decomposition count.

These are codestream components before inverse color conversion, unsigned
level restoration, JP2 palette/channel mapping, or ICC processing.
`uses_color_transform=True` means the first three components are in the
RCT/ICT domain, not RGB. Tile boundaries are retained, never stitched silently.
Neither a wavelet band nor its LL plane is promised to equal a display-ready
reduced-resolution RGB image.

Empty selector lists mean all. Invalid indices and selectors matching no bands
are errors. Selection skips unrequested Tier-1 entropy decoding and all inverse
DWT/color/pixel-packing work. Packet headers are still parsed and coefficient
storage is still allocated for the active tile. It does not promise proportional
I/O or peak-memory savings, or incremental refinement across separate calls.

## Direct CUDA-to-PyTorch output

```python
decoder = j2k.Decoder(device="cuda:0")
image = decoder.decode(data).to_torch()
assert image.device.type == "cuda"
```

The adapter retains that device's CUDA primary context, allocates a fresh
contiguous PyTorch tensor, and decodes directly into its validated allocation.
It orders allocation-stream work before codec access and waits for decode and
entropy-status validation before exposing the tensor. There is no application
pixel readback, host staging, or intermediate decoded-pixel upload. PyTorch
owns subsequent model-use lifetimes and allocation reuse. A completion failure
that cannot establish safety quarantines the unpublished allocation.

This first interface is blocking. It does not claim host-asynchronous model
overlap. Direct CUDA output inherits the codec's input coverage and supports
the same layout/native integer types as the adapter. Wavelet output remains
CPU-only.

## Direct Metal-to-PyTorch MPS output

```python
decoder = j2k.Decoder(device="mps")
image = decoder.decode(data).to_torch()
assert image.device.type == "mps"
```

On Apple Silicon, the adapter decodes each homogeneous group into a fresh
private Metal buffer on the system default GPU, then transfers ownership to
PyTorch using DLPack's Metal device `(8, 0)`. It waits for GPU completion and
codec-status validation before returning. Decoded pixels stay on the GPU;
there is no host pixel staging or upload into a second framework allocation.

The buffer is exclusively owned by the consumer after export, so in-place
PyTorch operations are safe and tensors survive decoder reuse or destruction.
Both layouts, native U8/U16/I16 samples, Gray/RGB/RGBA, and region/reduction
requests use the shared batch contract. JPEG on MPS is not yet supported.
Metal DLPack exports accept no stream argument. Allocations start at byte
offset zero; importing external buffers or suballocations is not exposed.
This interface is blocking and supports only Apple Silicon macOS.

## Validation and compatibility

```sh
cargo test -p j2k --test wavelets
cargo test -p j2k-native --lib selected_ll_bands_skip_entropy_work
python -m unittest discover -s crates/j2k-python/tests -v
```

CPU contract tests use existing independent OpenJPH fixtures and exercise exact
native dtypes, source alignment, ROI/reduction, ownership, and spawned workers.
Rust tests reconstruct both wavelet families, check odd multi-tile origins and
29-bit precision, and verify selection actually skips entropy work. CUDA tests
exercise default/nondefault streams, native dtypes, prepared reuse, and lifetime.
Run CUDA tests with `J2K_REQUIRE_CUDA=1`; a missing CUDA build/device then fails
instead of skipping. Local compile-only checks are not CUDA hardware validation.

MPS tests cover native types, layouts, region/reduction, consumer mutation,
prepared reuse, export-once semantics, and destruction from a worker thread.
Run with `J2K_REQUIRE_METAL=1` to fail if the Metal build or MPS device is
missing. The direct path is exercised with PyTorch 2.12.1 on Apple Silicon.

Patch releases preserve the adapter's documented API; pre-1.0 minor changes
require migration notes. Compatibility claims should name the consumer and
hardware combinations actually exercised. Normalization and labels remain
application responsibilities.

## Publishing

The `python-bindings.yml` workflow builds and tests CPU wheels for Linux and
CPU/MPS wheels for Apple Silicon macOS. MPS tests skip on runners without GPU
access. Linux release wheels use manylinux containers and PyPI-compatible tags.
Publishing runs only for an explicit manual dispatch from `main` with `publish`
enabled by the repository maintainer, after the wheel tests and any requested
CUDA checks succeed. CUDA builds are validation-only in this workflow.

Configure a PyPI pending Trusted Publisher with project `j2k`, GitHub owner
`frames-sg`, repository `j2k`, workflow `python-bindings.yml`, and environment
`pypi`. The publish job uses that GitHub environment and short-lived OIDC
credentials. The first successful upload creates the PyPI project; a pending
publisher does not reserve the name. The workflow and package changes must be
on `main` before publishing. Each release needs a new workspace version.
