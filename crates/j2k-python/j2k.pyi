# SPDX-License-Identifier: MIT OR Apache-2.0

from typing import Any, Literal
import numpy as np
import torch

__version__: str
cuda_enabled: bool
Region = tuple[int, int, int, int]

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

class ImageRequest:
    def __init__(self, data: bytes, *, region: Region | None = None, reduction: int = 0) -> None: ...

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
    precision: int
    signed: bool
    color: str
    alpha: str
    layout: Literal["nchw", "nhwc"]

class Batch:
    groups: list[BatchGroup]
    errors: list[tuple[int, str]]
    group_errors: list[tuple[list[int], str]]

class Decoder:
    def __init__(self, *, layout: Literal["nchw", "nhwc"] = "nchw", workers: int | None = None, strict: bool = True, device: str = "cpu") -> None: ...
    def prepare(self, inputs: list[bytes | ImageRequest], *, region: Region | None = None, reduction: int = 0) -> PreparedBatch: ...
    def decode(self, data: bytes, *, region: Region | None = None, reduction: int = 0) -> Tensor: ...
    def decode_batch(self, inputs: list[bytes | ImageRequest], *, region: Region | None = None, reduction: int = 0) -> Batch: ...
    def decode_prepared(self, prepared: PreparedBatch) -> Batch: ...

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

def inspect(data: bytes) -> dict[str, Any]: ...
def decode_wavelets(data: bytes, *, tiles: list[int] | None = None, components: list[int] | None = None, resolutions: list[int] | None = None, bands: list[Literal["LL", "HL", "LH", "HH"]] | None = None) -> WaveletImage: ...
