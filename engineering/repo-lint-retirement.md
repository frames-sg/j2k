# Repository lint retirement record

This record explains the repository-policy tests removed in the 2026 repo-lint
teardown. A removed test either has a mechanism-backed replacement or asserted
only the spelling, location, or size of source code and therefore did not
protect behavior.

## Replacement mechanisms

| Property | Mechanism |
| --- | --- |
| Move-only encode/decode owner graphs | Crate-local compile-time trait-ambiguity assertions. Adding `Clone`, including a handwritten implementation, makes a normal build fail. |
| Host allocation behavior | The serial `j2k-alloc-probe` package measures the process-wide allocator, warms the retained Rayon pool, checks zero-allocation workspace reuse where the API supports it, bounds intentional transient allocation, and cross-checks the public encode allocation-ledger boundary. |
| Infallible production allocation APIs | Workspace Clippy `disallowed-methods` and `disallowed-macros` configuration provides a fast local diagnostic. The allocator probe remains the behavioral gate. |
| GitHub Actions supply-chain and permission policy | Parsed YAML rules validate every workflow structurally, without copying action SHAs, tool versions, job scripts, or checksum literals into Rust. |
| Native-backend public-API isolation | The stable-API collector rejects `j2k_native` in both ordinary and rustdoc-hidden inventories for every collected package other than `j2k-native` itself. |
| Encode/decode call ordering | The retained `rust_function_policy` Syn visitor checks actual call expressions and their order. |
| Release, stable API, semver, clone, panic, coverage, and unsafe inventories | Their repository-owned xtask commands and command/unit tests remain the source of truth. |

## Removed allocation text proxies

The following policies searched production source for allocator spellings,
helper names, owner names, or test function names. They could be green while
the measured allocation path changed, and many explicitly allowed the real
`BudgetedVec` allocation path:

- `j2k_batch_allocation_policy.rs`
- `j2k_container_allocation_policy.rs`
- `j2k_scratch_allocation_policy.rs`
- `jpeg_batch_allocation_policy.rs`
- `jpeg_decode_allocation_policy.rs`
- `jpeg_header_allocation_policy.rs`
- `jpeg_segment_allocation_policy.rs`
- `jpeg_transcode_allocation_policy/**`
- `native_decode_allocation_policy/**`
- `native_decode_context_reuse_policy.rs`
- `native_encode_allocation_policy/**`
- `profile_allocation_policy.rs`
- allocation and ownership checks below `gpu_adapter_policy/**`

The move-only assertions and allocation probe above are stronger replacements.
Policies that only asserted module size or the existence of a similarly named
unit test were removed without replacement because neither fact establishes an
allocation property.

## Removed source-shape policies

The following modules asserted function/type/module names, exact error-mapping
spellings, line ceilings, or the presence of regression-test names:

- `encode_compare_structure_policy.rs`
- `encode_stage_error_policy.rs`
- `fixture_compare_structure_policy/**`
- `gpu_adapter_policy/**`
- `gpu_device_structure_policy.rs`
- `j2k_component_handoff_policy.rs`
- `j2k_decode_structure_policy.rs`
- `j2k_encode_validation_policy.rs`
- `j2k_error_source_policy.rs`
- `jpeg_dct_reemit_policy.rs`
- `jpeg_decoder_structure_policy/**`
- `jpeg_encoder_structure_policy.rs`
- `jpeg_metal_resource_safety_policy.rs`
- `jpeg_prepared_table_policy.rs`
- `jpeg_restart_policy/**`
- `jpeg_simd_boundary_policy.rs`
- `metal_buffer_pool_policy.rs`
- `metal_compute_structure_policy/**`
- `metal_compute_symbol_policy.rs`
- `metal_direct_plan_structure_policy.rs`
- `metal_resource_construction_policy.rs`
- `native_direct_plan_structure_policy.rs`
- `native_tile_metadata_policy.rs`
- `public_typed_helper_error_policy.rs`
- `shader_policy.rs`
- `tilecodec_error_policy.rs`
- `transcode_api_policy.rs`
- `transcode_cuda_policy/**`
- `transcode_structure_policy/**`

Rust type checking, Clippy, the real unit/integration suites, and the retained
Syn call-order visitor cover enforceable properties here. Tests that merely
required a production test name or a particular module split were not
replaced: renaming or reorganizing correct code is not a regression.

The one non-compiler Metal hazard in this family remains independently
enforced: raw `MTLBuffer::contents()` access is confined to its reviewed file
inventory. The attacker-controlled JPEG entropy paths also retain their
strict no-`unwrap`/no-`expect` inventory.

## Removed self-referential and command-proxy policies

- `audit_integrity_policy.rs` duplicated constants and implementation names
  from the real `clone-audit` and `panic-surface` commands.
- `coverage_structure_policy/**` asserted the coverage implementation's file
  layout and ratchet spellings; `cargo xtask coverage host` remains the gate.
- `xtask_main_structure_policy/**` asserted dispatch-arm and help-text
  spellings, keeping unused commands alive.
- `docs_and_workflows_policy/workflow_coverage_policy.rs`,
  `stable_api_governance.rs`, and the non-evidence documentation policies
  copied command names, versions, and workflow fragments already exercised by
  their real commands.
- `docs_and_workflows_policy/structural_ratchets/policy_line_limits.rs`,
  `policy_ownership/**`, and all other `.lines().count()` ownership ceilings
  governed the lint suite or source layout rather than behavior.
- `release_policy.rs` copied publish lists, script fragments, and release
  documentation. `cargo xtask release-integrity`, package command tests, and
  the structural workflow policy remain authoritative.
- `workflow_policy.rs` copied workflow SHAs, versions, scripts, and job names.
  Parsed structural rules replace its supply-chain and permission properties.

The retired `downstream-smoke`, `nextest`, `bench-report`, `typos`, and `deny`
xtask dispatch aliases had no repository callers outside the source-text
policies. The underlying gates remain where applicable: ordinary workspace
tests, the GitHub typos action, and the cargo-deny action.

## Removed editorial and benchmark-layout checks

The repository-hygiene policy no longer blacklists former product or vendor
names, agent-specific filenames, or selected prose claims. Those are editorial
review concerns rather than code properties. The independently checkable home
path and missing-script scans remain.

The `j2k-ml` benchmark policy no longer requires particular module, function,
type, or constant names, nor does it duplicate the workspace suppression
policy. Cargo metadata still checks the supported benchmark/feature matrix, and
the Syn visitor still verifies that persistent decoder sessions are created
after each workload is materialized.

## Narrow retained policies

The teardown intentionally preserves checks with an independent source of
truth:

- exact manifest/source suppression inventories;
- the SHA-256 path-patch and conformance-corpus inventories;
- the cargo-metadata versus architecture-document symmetric diff;
- environment-variable documentation set comparison;
- local-home-path, missing-script, and package include-path escape scans;
- stable-API review evidence cross-comparison;
- the `j2k-ml` Cargo metadata matrix and Syn session-lifecycle check;
- structural workflow YAML rules;
- the Metal raw-buffer access inventory;
- the entropy-decoder panic hotspot inventory; and
- the Syn function-call ordering policy.

These checks may inspect text as an input format, but they compare it with a
separate inventory, parsed structure, filesystem state, compiler property, or
measured behavior rather than treating a substring as proof of a code
property.
