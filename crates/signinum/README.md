# signinum

Facade crate for the `signinum` pathology image codec workspace.

The default build exposes CPU-portable JPEG, JPEG 2000 / HTJ2K, shared core
contracts, and tile decompression APIs. Runtime backend selection defaults to
`Auto` / `ACCELERATED`: codecs use CPU for CPU-shaped stages and Metal/CUDA for
benchmark-approved device-shaped stages when the matching facade feature is
enabled. Explicit Metal/CUDA requests remain strict proof paths. Metal is
available through the `metal` or `gpu` features; CUDA is available through the
`cuda`, `cuda-runtime`, or `gpu` features.

Install:

```sh
cargo add signinum
```

Use this crate when an application wants one import surface for Signinum codec
primitives. Use `statumen` for whole-slide container parsing and `wsi-dicom`
for DICOM VL Whole Slide Microscopy export.
