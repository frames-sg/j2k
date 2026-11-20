# JPEG 2000 conformance manifests

This directory holds manifests we wrote, not test data. The official
ISO/IEC 15444-4 / ITU-T T.803 v3 electronic attachment is copyrighted and
stays external: don't commit it, copy it into a package, or upload it as a CI
artifact.

`t803-v3.toml` is the Part 1 and Part 15 decoder inventory. It pins the
attachment's URL, archive size and SHA-256, and the hash of every selected
codestream and reference: all entries in the five selected Part 1 decoder
tables, the nine Annex G JP2 files, 26 HTJ2K BSETs, and ten Annex G JPH
BSETs. `cargo xtask t803 fetch` downloads and verifies the corpus into
`target/t803/`, and `cargo xtask t803 run` fails if that corpus is missing or
altered.

The deterministic encoder tests are defined by:

- `encoder-ics-cpu.toml`
- `encoder-ics-cuda.toml`
- `encoder-ics-metal.toml`
- `encoder-matrix-v2.toml`

Under T.803, these Annex D/F results are informative; they are not decoder
conformance results. Supported cases are decoded by the vendored OpenJPEG
2.5.3 T.804 decoder. OpenJPEG rejects HT code blocks that carry RGN before
decoding them, so the HT+RGN case is decoded by a pinned, unmodified
OpenHTJ2K 0.23.4 executable instead, as a Part 15 interoperability check.
Run `scripts/prepare-openhtj2k-reference.sh` before the CPU matrix. The
report records the OpenHTJ2K source revision and executable hash, and marks
the case as not a T.804 result.

`support-inventory.tsv` is the feature-support table that
`cargo xtask public-support` reads. It names no corpus paths, the T.803
runner doesn't use it, and it is not a conformance result.

The results are the JSON and Markdown reports the runner writes, not any
summary of them. The [conformance page](../../book/conformance.md) explains
what they cover.
