# j2k-jpeg

Pure-Rust JPEG inspect/decode crate for J2K transcode and codec pipelines.

CPU decode is the correctness baseline. Supported JPEG classes are covered by
tests and capability reports; unsupported classes return structured errors.
The crate also contains fixture/fallback baseline encode support used by tests
and adapters.

Use this crate directly for JPEG input; use `j2k` for JPEG 2000 / HTJ2K and
`j2k-transcode` for JPEG-to-J2K/HTJ2K transcode paths.
