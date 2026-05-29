# Public API Documentation

This document tracks stable crate rustdoc coverage for the `0.4.x` line. The
workspace already gates `cargo xtask doc` with rustdoc warnings denied; the
remaining adoption work is a missing_docs ratchet for the stable public API.

Stable crates covered by the ratchet:

- `signinum`
- `signinum-core`
- `signinum-jpeg`
- `signinum-j2k`
- `signinum-tilecodec`

## Policy

- New public items in stable crates must include rustdoc before merge.
- Existing undocumented public items are paid down crate-by-crate.
- A crate should enable `missing_docs` only after its current public surface can
  pass with `RUSTDOCFLAGS="-D warnings" cargo xtask doc`.
- Experimental crates may document promotion gates before every public item is
  fully documented, but stable facade-facing types should still carry examples
  when practical.

## Current ratchet

- `signinum`: facade docs and runnable examples are the landing surface.
- `signinum-core`: next target for module/type-level missing_docs cleanup.
- `signinum-jpeg`: broadest stable surface; document parser, row, ROI, scaled,
  batch, passthrough, and scratch/context APIs before enabling the lint.
- `signinum-j2k`: document encode options, recode reports, adapter planning,
  and error fields before enabling the lint.
- `signinum-tilecodec`: small enough to promote first once codec structs, pools,
  and error variants have item docs.

