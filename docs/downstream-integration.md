# Downstream Integration

Signinum provides codec primitives. It does not own container parsing, pyramid
policy, cache policy, or DICOM writing.

Use `statumen` before signinum when a workflow needs whole-slide container
parsing. Use `wsi-dicom` after signinum when a workflow needs DICOM VL Whole
Slide Microscopy output.

The local smoke command validates the examples that exercise this boundary:

```sh
cargo xtask downstream-smoke
```

That command compiles and runs the facade examples plus the transcode example.
It is not a full cross-repository integration test; it verifies that the public
codec boundary remains usable by downstream container and DICOM layers.

Adoption examples should keep these responsibilities separate:

- container parsing and tile lookup: downstream container crate
- compressed tile passthrough decision: signinum codec view plus downstream
  metadata validation
- pixel decode, ROI, scaling, row streaming, and tile batch decode: signinum
- DICOM VL Whole Slide Microscopy metadata and frame writing: downstream DICOM
  crate

