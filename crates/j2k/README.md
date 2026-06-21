# j2k

JPEG 2000 and HTJ2K public decoder/encoder crate for J2K.

This crate exposes inspect, decode, encode, recode, device-surface, and
encode-stage adapter contracts backed by the native J2K engine and optional
device adapters.

The encode-stage adapter module is a backend SPI for CUDA, Metal, and transcode
integration. It is not the primary end-user encode API.

For JPEG 2000 / HTJ2K application code, including CPU and supported GPU-backed
paths, use this crate directly.
