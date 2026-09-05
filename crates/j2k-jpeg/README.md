# j2k-jpeg

Pure-Rust JPEG inspect, decode, and baseline encode crate for pathology,
transcode, and general codec pipelines in the j2k workspace.

CPU decode is the correctness baseline. Supported JPEG classes are covered by
tests and capability reports; unsupported classes return structured errors.
The crate also provides the portable baseline encoder used by CPU integrations;
the sibling Metal and CUDA adapter crates provide accelerated baseline paths.

Pathology-oriented integrations can use `prepare_tiff_jpeg_tile` to normalize
TIFF `JPEGTables` plus abbreviated strip/tile payloads, including zero-SOF
dimension repair from container metadata. `extract_icc_profile`,
`insert_icc_profile`, and `set_icc_profile` provide ordered APP2 ICC assembly
and replacement without putting color management policy inside the codec.

Use this crate directly for JPEG input; use `j2k` for JPEG 2000 / HTJ2K and
`j2k-transcode` for JPEG-to-HTJ2K coefficient-domain transcode paths.

## Links

- API docs: <https://docs.rs/j2k-jpeg>
- Repository: <https://github.com/frames-sg/j2k>
- Support policy: <https://github.com/frames-sg/j2k/blob/main/docs/public-support.md>
