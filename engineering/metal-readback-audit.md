# Metal Readback Audit

Status: **complete as of 2026-07-09**

This note records the July 2026 review and completed migration of CPU-visible
Metal buffer access in the J2K Metal adapters.

## Result

- `crates/j2k-metal/src/compute/direct_buffers.rs` is the adapter's centralized
  typed buffer-access boundary. Its readback wrappers require `GpuAbi` and
  delegate bounds, offset arithmetic, alignment, CPU-visibility, and allocation
  checks to `j2k-metal-support`.
- The migrated compute status, profile, token, scratch, and output readbacks use
  `checked_buffer_read`, `checked_buffer_slice`, or
  `checked_buffer_slice_at`. Those wrappers return owned values after the
  producing command buffer has completed.
- Public host views in `surface.rs` and `encode/encoded.rs` use
  `j2k_metal_support::checked_buffer_read_vec` rather than constructing slices
  directly from a Metal pointer.
- Upload and zero-initialization paths use the shared checked write/fill
  helpers. Caller-owned no-copy wrappers have explicit unsafe lifetime and
  exclusivity contracts.
- The only direct `buffer.contents()` use under `crates/j2k-metal/src` is the
  `buffer_is_cpu_visible` predicate in `direct_buffers.rs`; it checks visibility
  and does not dereference or expose the pointer.
- Static inspection found no public or caller-provided Metal buffer byte view on
  an unchecked readback path in `j2k-metal`.

## Completion status

The migrated implementation compiles, its library tests pass, the unsafe
inventory is current, and the required strict clippy command is green. Remaining
uploads, zeroing, visibility tests, and explicitly unsafe caller-owned no-copy
wrappers are not unchecked readback slices.

## Verification commands

- `rg -n '\.contents\(\)' crates/j2k-metal/src --glob '*.rs'`
- `cargo test -p j2k-metal --lib`
- `cargo test -p j2k-metal-support --lib`
- `cargo xtask unsafe-audit`
- `cargo clippy -p j2k-metal-support -p j2k-metal --all-targets --all-features -- -D warnings`

2026-07-09 result: `cargo test -p j2k-metal-support --lib` passed 10 tests. At
the SAFE-001 checkpoint, `cargo test -p j2k-metal --lib` passed 200 runnable
tests; after the later GPU ordering/API hardening, the candidate passed 204
runnable tests with 18 explicit hardware-lane tests ignored by the default
invocation. The raw-`contents()` search returned only the visibility predicate.
The final candidate must rerun `cargo xtask unsafe-audit` after every structural
move so the inventory names each current source. The strict Clippy command
passed at both checkpoints.
