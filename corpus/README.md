# Corpus

Small committed test fixtures, and metadata for larger corpora that stay
outside the repository.

- `conformance/`: JPEG reference fixtures made with libjpeg-turbo.
  `manifest.json` lists each file with its SHA-256. `generate.sh` regenerates
  them and runs `lossless_matrix.py`, `scaled_matrix.py` and, when DCMTK is
  installed, `dicom_matrix.py`, which write the matrices under
  `crates/j2k-test-support/fixtures/`. That crate's `fixtures/conformance/`
  holds identical copies, and the xtask repository lint checks both against
  the manifest.
- `j2k-conformance/`: the ITU-T T.803 decoder inventory, the encoder test
  matrices and the feature-support inventory. See its README.
- `wsi-samples/`: where `scripts/parity-corpus-fetch.sh` downloads whole-slide
  images listed in a manifest you supply.
- `fuzz-seeds/` and `regressions/` hold only a `.gitkeep`.

Larger WSI, JPEG, JPEG 2000 and HTJ2K inputs are supplied through
environment variables or external locations. Don't commit them. To see what
is here, read the manifests and directory listings rather than this file.
