# j2k

JPEG 2000 Part 1 and HTJ2K Part 15 public decoder/encoder crate for J2K.

This crate exposes inspect, decode, encode, recode, device-surface, and
encode-stage adapter contracts backed by the native J2K engine and optional
device adapters.

The public support boundary is raw J2K/J2C codestreams, JP2 still-image files,
raw HTJ2K codestreams, and JPH still-image files. JPX / JPEG 2000 Part 2
extensions are not part of this crate's support claim unless required for
standard JP2/JPH still-image correctness.

The encode-stage adapter module is a backend SPI for CUDA, Metal, and transcode
integration. It is not the primary end-user encode API.

For JPEG 2000 / HTJ2K application code, including CPU and supported GPU-backed
paths, use this crate directly.
