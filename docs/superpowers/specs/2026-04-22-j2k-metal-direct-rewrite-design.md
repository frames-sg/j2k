# J2K / HTJ2K MetalDirect Rewrite Design

## Summary

Rewrite the J2K / HTJ2K Metal backend as a true GPU-owned decode pipeline.

The current architecture is CPU-owned decode with Metal hooks over host buffers. That model is good enough for parity experiments and hybrid fallback, but it is structurally incapable of producing a hard Metal win. The direct backend must instead make the host a planner and scheduler only, while the GPU owns codeblock decode, coefficient buffers, IDWT, MCT/store, and final output pack.

This rewrite is explicitly for classic J2K and HTJ2K. It is not another tuning pass on the existing hybrid path.

## Hard Invariants

- `CpuOnly` remains a strict, separate path.
- `MetalDirect` remains a strict, separate path.
- `MetalDirect` may not call any CPU-upload or host-plane fallback helpers.
- `BackendRequest::Metal` means direct Metal execution or an unsupported/unavailable error.
- `BackendRequest::Auto` may choose CPU, but it must make that choice before execution begins. It may not start on Metal and silently fall back inside the Metal path.
- The new native planning path must not depend on host `channel_data`, host `idwt_output`, or host-side component staging as primary pipeline state.

## Goals

- Make J2K and HTJ2K first-class direct-Metal decoders.
- Move the decode boundary so the host only parses and plans.
- Keep all meaningful intermediate state on device for the direct path.
- Preserve a clean CPU baseline and fallback path.
- Reach a point where Metal wins are possible on the right J2K / HTJ2K workloads instead of being blocked by architecture.

## Non-Goals

- No CUDA work in this rewrite.
- No public API expansion in the first phase unless required by proven runtime constraints.
- No attempt to make `MetalDirect` win every workload immediately.
- No hybrid behavior inside `MetalDirect`.

## Current Problems

### Wrong ownership boundary

The native J2K decoder still owns the pipeline:

- codeblock payloads are parsed and decoded into host coefficient buffers
- IDWT is driven over host-owned slices
- store writes into host component buffers
- Metal participates only as optional stage hooks

This means the direct path still pays for CPU-owned pipeline structure even when kernels exist.

### Hybrid contamination

The current Metal surface path can be competitive only because it defaults to CPU decode plus Metal upload. That is useful as a fallback policy, but it prevents the direct backend from becoming a true GPU decoder if it remains part of the same execution path.

### Wrong optimization target

Pack-only tuning and upload tuning help at the margins, but they cannot produce a hard Metal win when codeblock decode, coefficient ownership, and transform ownership are still CPU-first.

## Architecture

### Execution strategies

The runtime keeps three internal strategies:

- `CpuOnly`
- `MetalDirect`
- `Auto`

`Auto` is a selector only. It is not its own execution engine.

`HybridCpuMetal` is deliberately removed from the J2K / HTJ2K direct rewrite target. If a hybrid path is retained elsewhere for compatibility or transition, it must not share internal execution code with `MetalDirect`.

### CpuOnly

`CpuOnly` keeps the current native decoder behavior:

- host-owned codeblock decode
- host-owned coefficient storage
- host-owned IDWT
- host-owned MCT/store
- host or device upload after decode as needed

This path remains the correctness baseline, fallback, and benchmark reference.

### MetalDirect

`MetalDirect` becomes a GPU-owned tile pipeline:

1. Host parses codestream structure and target tile / resolution selection.
2. Host builds a device plan describing the requested decode.
3. Metal executes the plan without host coefficient or channel staging.

The GPU owns:

- classic J2K codeblock decode
- HTJ2K codeblock decode
- device coefficient buffers
- device IDWT scratch and outputs
- device inverse MCT when applicable
- device store into output planes
- final packed output surface

The host owns:

- codestream parsing
- validation
- tile / resolution / region selection
- command scheduling
- fallback policy

## Native Device Plan

### Purpose

The native crate must expose a hidden planning path that builds a direct-execution plan without decoding into `channel_data`.

### First plan scope

The first supported plan is intentionally narrow:

- grayscale only
- full decode only
- single tile only
- classic J2K and HTJ2K

This removes MCT and multi-tile scheduling from the first boundary move.

### Plan contents

The first grayscale plan should contain:

- output dimensions
- output bit depth
- ordered sub-band decode jobs
- codeblock payloads and placement inside each sub-band
- decomposition graph for IDWT levels
- final store window for the grayscale plane
- final pack format metadata

The plan must be sufficient for Metal to execute the full decode without asking the CPU to materialize coefficients or planes.

## MetalDirect Executor

### Inputs

- native grayscale direct plan
- pixel format target
- backend request

### Outputs

- Metal-backed `Surface`

### Execution order

1. Allocate device coefficient buffers per sub-band.
2. Decode classic or HTJ2K codeblocks directly into those device buffers.
3. Apply device-side IDWT level by level until the final component plane exists on device.
4. Apply device-side store into the final grayscale output plane if needed.
5. Pack the final Metal surface.

No host coefficient arrays or host component planes are part of this hot path.

## Rewrite Order

### Phase 1: planning seam

Add the hidden direct grayscale device-plan builder in `slidecodec-j2k-native`.

Requirements:

- no host `channel_data` dependency for plan construction
- no direct execution in the plan builder
- no public API changes

### Phase 2: direct grayscale executor

Add a new Metal executor that consumes the grayscale plan and returns a Metal surface.

Requirements:

- classic J2K supported
- HTJ2K supported
- full decode supported
- single-tile supported

### Phase 3: strict strategy split

Refactor `slidecodec-j2k-metal` strategy dispatch so:

- `CpuOnly` uses the existing CPU path
- `MetalDirect` uses only the new direct executor
- explicit `Metal` never routes through CPU-upload

### Phase 4: broader direct coverage

After the grayscale full-decode path is stable and benchmarked:

- grayscale region
- grayscale scaled decode
- multi-tile batch
- RGB and inverse MCT
- broader output format coverage

## Performance Priorities

The direct rewrite is about moving the ownership boundary first. After that, the highest-value runtime improvements are:

- global Metal runtime instead of thread-local runtime
- precompiled `.metallib` instead of runtime shader compilation
- buffer pooling for hot-path allocations
- batched command submission without per-tile waits

These are important, but they are not substitutes for the pipeline rewrite.

## Testing

### Correctness

For every phase:

- direct Metal grayscale output must match CPU output exactly
- classic J2K and HTJ2K both need parity coverage
- explicit `BackendRequest::Metal` must fail rather than silently route through CPU paths when unsupported

### Structural tests

Add tests that prove:

- `MetalDirect` does not invoke CPU-upload helpers
- `MetalDirect` does not rely on host component staging
- explicit `Metal` and explicit `Cpu` remain distinct execution paths

### Benchmarking

Use real compare benches, not tiny synthetic fixtures, to judge wins.

The first direct-path success condition is not “beat every CPU baseline immediately.” It is:

- the direct path is fully GPU-owned for the supported grayscale slice
- the direct path is correct
- the direct path is benchmarkable as an independent executor

Once that is true, performance work on batch execution, runtime reuse, and broader coverage becomes meaningful.

## Risks

### Complexity explosion

If RGB, MCT, batch, region, and scaled decode are all included in the first cut, the rewrite will turn into another half-finished hybrid compromise. The first cut must stay grayscale-only.

### Accidental fallback contamination

If the new direct executor shares fallback codepaths with CPU-upload helpers, the architecture regresses immediately. Strategy boundaries must stay strict.

### Premature tuning

Optimizing uploads, pack kernels, or scheduling before the ownership boundary is fixed will waste time and obscure whether the direct path is actually real.

## Expected Outcome

After this rewrite, the project will have:

- a strict CPU-only J2K / HTJ2K path
- a strict MetalDirect J2K / HTJ2K path
- a hidden native device-plan seam that lets the host plan while the GPU executes

That is the first architecture in this repo that can plausibly deliver a hard Metal win for J2K / HTJ2K.
