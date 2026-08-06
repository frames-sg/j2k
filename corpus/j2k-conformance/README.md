# J2K Conformance Metadata

This directory contains authored metadata only. The official ISO/IEC 15444-4 /
ITU-T T.803 v3 copyrighted electronic attachment remains external. It must not be committed,
copied into a package, or uploaded as a CI artifact.

The blocking Part 1 decoder and Annex G inventory is `t803-v3.toml`. It pins the
official attachment URL, archive size and SHA-256, every selected codestream and
reference hash, all entries in the five selected decoder tables, and all nine
Annex G JP2 files. `cargo xtask t803 fetch` materializes the verified corpus
only under `target/t803/`; `run` fails when that corpus is absent or altered.

The deterministic encoder procedure is described by:

- `encoder-ics-cpu.toml`
- `encoder-ics-cuda.toml`
- `encoder-ics-metal.toml`
- `encoder-matrix-v1.toml`

Those Annex D/F results are informative under T.803 and are not decoder
conformance claims. `support-inventory.tsv` is the feature-support ledger used
by `cargo xtask public-support`; it contains no corpus paths, is not consumed by
the T.803 runner, and must not be cited as exact-reference evidence.

Generated JSON and Markdown reports, not narrative summaries, are the release
evidence. The claim policy and current blocker are documented in
[`docs/t803-conformance.md`](../../docs/t803-conformance.md).
