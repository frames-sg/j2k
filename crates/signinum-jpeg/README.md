# signinum-jpeg

Pure-Rust JPEG inspect/decode crate for Signinum.

CPU decode is the correctness baseline. Supported JPEG classes are covered by
tests and capability reports; unsupported classes return structured errors.
The crate also contains fixture/fallback baseline encode support used by tests
and adapters.

For application code, prefer the `signinum` facade.
