# Metal Readback Audit

This note records the July 2026 review of CPU-visible Metal `contents()`
readbacks in the J2K Metal adapters.

## Result

- Public byte views now use `j2k_metal_support::checked_buffer_contents_slice`
  for bounds, alignment, null-contents, and offset arithmetic checks:
  - `crates/j2k-metal/src/surface.rs`
  - `crates/j2k-metal/src/encode/encoded.rs`
  - existing encode input staging in `crates/j2k-metal/src/encode.rs`
  - existing direct surface pack staging in
    `crates/j2k-metal/src/compute/direct_surface_pack.rs`
- The remaining raw `contents()` call sites are private kernel status,
  profile, token, and scratch readbacks. They are paired with one of:
  - buffers allocated in the same function from `size_of::<T>() * count`
  - counts converted before allocation and before readback
  - status buffers with fixed single-status allocation
  - command-buffer completion before CPU access
- No public or caller-provided Metal buffer byte view remains on a direct
  unchecked `contents()` slice path in `j2k-metal`.

## Deferred Follow-up

The private status/profile paths should be migrated incrementally to a typed
checked readback helper once the internal Metal ABI structs implement the shared
`GpuAbi` layout contract. That is a follow-up hardening pass, not a currently
known bounds bug.

## Verification

- `cargo test -p j2k-metal --lib`
- `cargo test -p xtask --test repo_lint`
- `cargo clippy -p j2k-native -p j2k -p j2k-metal -p xtask --all-targets --all-features -- -D warnings`
