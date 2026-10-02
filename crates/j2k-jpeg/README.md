# j2k-jpeg

Pure-Rust JPEG inspect, decode, and baseline encode crate for pathology,
transcode, and general codec pipelines in the j2k workspace.

The CPU decoder is the reference implementation. Unsupported JPEG types return
an error. The crate also has a portable baseline encoder; the Metal and CUDA
adapter crates provide GPU baseline encode.

Pathology-oriented integrations can use `prepare_tiff_jpeg_tile` to normalize
TIFF `JPEGTables` plus abbreviated strip/tile payloads, including zero-SOF
dimension repair from container metadata. `extract_icc_profile`,
`insert_icc_profile`, and `set_icc_profile` provide ordered APP2 ICC assembly
and replacement; color management is left to the caller.

Use this crate directly for JPEG input; use `j2k` for JPEG 2000 / HTJ2K and
`j2k-transcode` for JPEG-to-HTJ2K coefficient-domain transcode paths.

## Scaled decoding

Sequential and progressive DCT JPEGs use reduced IDCTs at 1/2, 1/4, and 1/8
scale. Component-specific IDCT sizes and chroma interpolation follow
libjpeg-turbo, including subsampled color, small chroma planes, and cropped
regions. The committed libjpeg-turbo 3.1.4.1 reference matrix covers these paths,
restart intervals, progressive scans, and supported 12-bit inputs.

Unsupported sampling layouts return `NotImplemented` rather than being
converted.

## Links

- API docs: <https://docs.rs/j2k-jpeg>
- Repository: <https://github.com/frames-sg/j2k>
- Support policy: <https://github.com/frames-sg/j2k/blob/main/docs/public-support.md>
