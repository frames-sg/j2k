# `block` 0.1.6 patch provenance

This directory is derived from the crates.io `block` 0.1.6 release published
from `SSheldon/rust-block` tag `0.1.6`.

Pinned SHA-256 digests:

- crates.io `block-0.1.6.crate` archive: `0d8c1fef690941d3e7788d328517591fecc684c084084702d6ff1641e993699a`
- upstream `block-0.1.6/src/lib.rs`: `eb31678adf63b53109d9b94eba23699fd5f9ebfdb950f6e1a57ad51bb6a146fa`
- patched local `src/lib.rs`: `bf799f4d01bb497fdcffe7a5e28d998e721ed45c1be866ed1b454df39ce876a9`

Documented ABI deltas from upstream are intentionally limited to:

1. Replace the uninhabited opaque `enum Class {}` used for
   `_NSConcreteStackBlock` with a `#[repr(C)]` opaque zero-sized `Class` struct.
   This preserves pointer opacity while avoiding the upstream future-incompatible
   extern-static type.
2. Spell the C ABI explicitly as `extern "C"` on the extern block, block invoke
   function pointers, generated invoke shims, copy/dispose callbacks, and their
   descriptor fields. Upstream used implicit `extern`, which means the C ABI;
   this patch makes that existing ABI contract explicit.

There are no documented algorithmic or ownership changes. The repository lint
pins this document and recomputes the patched local source digest offline.

## Release approval

- Reviewer identity: `greg`
- Approval date: `2026-07-12`

These structured fields record the maintainer who reviewed the pinned source
and documented ABI-only delta for the 0.7 release candidate. The release
integrity gate rejects placeholders and requires a real reviewer identity plus
a calendar-valid approval date before publication.
