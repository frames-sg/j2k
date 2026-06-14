# signinum-j2k-native

Native JPEG 2000 / HTJ2K engine used by Signinum J2K APIs and adapters.

This crate owns codestream parsing, native encode/decode helpers, packetization
support, HTJ2K table helpers, and header inspection helpers used by higher-level
crates.

Most application code should use `signinum` or `signinum-j2k` instead.
