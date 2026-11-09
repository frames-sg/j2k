# SPDX-License-Identifier: MIT OR Apache-2.0

from collections.abc import Sequence
from mmap import mmap
from typing import Any, Literal, overload
import numpy as np
import torch

__version__: str
cuda_enabled: bool
metal_enabled: bool
Region = tuple[int, int, int, int]
# Encoded input: bytes, or any other one-dimensional C-contiguous buffer of
# unsigned bytes. Other layouts raise BufferError; non-buffers raise TypeError.
BytesLike = bytes | bytearray | memoryview | mmap

class Tensor:
    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def dtype(self) -> str: ...
    @property
    def device(self) -> tuple[int, int]: ...
    def __dlpack_device__(self) -> tuple[int, int]: ...
    def __dlpack__(self, stream: int | None = None, *, max_version: tuple[int, int] | None = None, dl_device: tuple[int, int] | None = None, copy: bool | None = None) -> Any: ...
    def to_torch(self) -> torch.Tensor: ...
    def to_numpy(self) -> np.ndarray: ...

class RecoveredImageWarning(UserWarning): ...

class CancellationToken:
    def __init__(self) -> None: ...
    def cancel(self) -> None: ...
    @property
    def cancelled(self) -> bool: ...

class DecodeControl:
    def __init__(self, *, max_working_memory_bytes: int | None = None, max_output_bytes: int | None = None, cancellation: CancellationToken | None = None) -> None: ...
    @property
    def max_working_memory_bytes(self) -> int | None: ...
    @property
    def max_output_bytes(self) -> int | None: ...

class ImageRequest:
    def __init__(self, data: BytesLike, *, region: Region | None = None, reduction: int = 0, color_transform: Literal["auto", "ycbcr", "none"] = "auto") -> None: ...

class PreparedBatch:
    @property
    def group_count(self) -> int: ...
    @property
    def errors(self) -> list[tuple[int, str]]: ...

class BatchGroup:
    tensor: Tensor
    source_indices: list[int]
    decoded_rects: list[Region]
    warnings: list[list[str]]
    recovered: list[bool]
    precision: int
    signed: bool
    color: Literal["Rgb", "Gray", "Rgba"]
    alpha: Literal["None", "Straight", "Premultiplied"]
    layout: Literal["nchw", "nhwc"]

class Batch:
    groups: list[BatchGroup]
    errors: list[tuple[int, str]]
    group_errors: list[tuple[list[int], str]]

class Decoder:
    def __init__(self, *, layout: Literal["nchw", "nhwc"] = "nchw", workers: int | None = None, strict: bool = True, device: str = "cpu") -> None: ...
    def prepare(self, inputs: list[BytesLike | ImageRequest], *, region: Region | None = None, reduction: int = 0, control: DecodeControl | None = None) -> PreparedBatch: ...
    def decode(self, data: BytesLike | ImageRequest, *, region: Region | None = None, reduction: int = 0) -> Tensor: ...
    def decode_batch(self, inputs: list[BytesLike | ImageRequest], *, region: Region | None = None, reduction: int = 0) -> Batch: ...
    def decode_prepared(self, prepared: PreparedBatch, *, region: Region | None = None, reduction: int | None = None, control: DecodeControl | None = None) -> Batch: ...
    def plan_region(self, data: BytesLike, *, region: Region | None = None, reduction: int = 0, interpretation: Literal["native", "srgb8"] = "native") -> dict[str, Any]: ...
    def decode_srgb8(self, data: BytesLike, *, region: Region | None = None, reduction: int = 0, control: DecodeControl | None = None) -> Tensor: ...
    def set_operation_cache_limit(self, bytes: int) -> None: ...

class WaveletBand:
    tensor: Tensor
    kind: Literal["LL", "HL", "LH", "HH"]
    resolution: int
    rect: Region
    quantization_step: float

class WaveletComponent:
    tile_index: int
    tile_rect: Region
    index: int
    rect: Region
    precision: int
    signed: bool
    sampling: tuple[int, int]
    uses_color_transform: bool
    transform: Literal["5/3", "9/7"]
    decomposition_levels: int
    bands: list[WaveletBand]

class WaveletImage:
    rect: Region
    components: list[WaveletComponent]

# Keys include color_specification ("none", "enumerated", "icc", "unknown": the
# first COLR box, which decoding applies), enumerated_colorspace (its EnumCS, or
# None), and icc_profile (the applied profile's bytes, or None).
# Geometry for image servers: resolution_levels, max_reduction (the reduction
# every tile and component supports, which tile COD/COC markers can lower),
# image_x0/image_y0 and tile_x0/tile_y0 (reference-grid origins),
# tile_width/tile_height, tiles_x/tiles_y. Main-header COD defaults:
# decomposition_levels, quality_layers, progression_order (a Progression), and
# code_block_width/code_block_height.
def inspect(data: BytesLike) -> dict[str, Any]: ...
def decode_wavelets(data: BytesLike, *, tiles: list[int] | None = None, components: list[int] | None = None, resolutions: list[int] | None = None, bands: list[Literal["LL", "HL", "LH", "HH"]] | None = None) -> WaveletImage: ...

def forward_rgb97(data: bytes, width: int, height: int, *, levels: int = 5, color: Literal["rgb", "ict"] = "ict") -> list[tuple[int, int, str, int, int, bytes]]: ...
def inverse_rgb97(bands: list[tuple[int, int, str, bytes]], width: int, height: int, *, levels: int = 5, color: Literal["rgb", "ict"] = "ict") -> bytes: ...

Progression = Literal["lrcp", "rlcp", "rpcl", "pcrl", "cprl"]
Marker = Literal["sop", "eph", "tlm", "plt", "plm", "ppm", "ppt"]

# Interleaved samples on the CPU: lossless by default, or mode="lossy" with at
# most one of target_bpp, target_bytes, target_psnr (a list gives cumulative
# quality layers). Samples of up to 8 bits take one byte; deeper ones two, little-endian.
# return_report=True also returns codestream_bytes, bits_per_pixel, psnr_db
# (None when not measured), and quality_layers.
@overload
def encode(data: bytes, width: int, height: int, *, components: int = 3, bit_depth: int = 8, signed: bool = False, codec: Literal["j2k", "htj2k"] = "htj2k", mode: Literal["lossless", "lossy"] = "lossless", levels: int | None = None, progression: Progression = "lrcp", tile_size: tuple[int, int] | None = None, tile_part_packets: int | None = None, markers: Sequence[Marker] = (), quality_layers: int | None = None, color_transform: Literal["auto", "none"] = "auto", qfactor: int | None = None, target_bpp: float | Sequence[float] | None = None, target_bytes: int | Sequence[int] | None = None, target_psnr: float | Sequence[float] | None = None, file: bool = False, icc_profile: bytes | None = None, return_report: Literal[False] = False) -> bytes: ...
@overload
def encode(data: bytes, width: int, height: int, *, components: int = 3, bit_depth: int = 8, signed: bool = False, codec: Literal["j2k", "htj2k"] = "htj2k", mode: Literal["lossless", "lossy"] = "lossless", levels: int | None = None, progression: Progression = "lrcp", tile_size: tuple[int, int] | None = None, tile_part_packets: int | None = None, markers: Sequence[Marker] = (), quality_layers: int | None = None, color_transform: Literal["auto", "none"] = "auto", qfactor: int | None = None, target_bpp: float | Sequence[float] | None = None, target_bytes: int | Sequence[int] | None = None, target_psnr: float | Sequence[float] | None = None, file: bool = False, icc_profile: bytes | None = None, return_report: Literal[True]) -> tuple[bytes, dict[str, Any]]: ...

# RGB research encoding uses CPU, JPEG 4:4:4 and the wavelet codec's ICT.
def encode_rgb(data: bytes, width: int, height: int, *, codec: Literal["jpeg", "j2k", "htj2k"] = "htj2k", quality: int | None = None, target_bpp: float | None = None, levels: int = 5, profile: list[list[float]] | None = None) -> bytes: ...
def decode_rgb(data: BytesLike) -> tuple[int, int, bytes]: ...
