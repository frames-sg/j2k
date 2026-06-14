# signinum-j2k

JPEG 2000 and HTJ2K public codec crate for Signinum.

This crate exposes inspect, decode, encode, recode, device-surface, and
encode-stage adapter contracts backed by the native J2K engine and optional
device adapters.

The encode-stage adapter module is a backend SPI for CUDA, Metal, and transcode
integration. It is not the primary end-user encode API.

For application code, prefer the `signinum` facade.
