# Adaptive J2K RCA Gates Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add adaptive J2K / HTJ2K RCA instrumentation and gate evidence scaffolding while keeping CUDA work last and preventing premature GPU default promotion.

**Architecture:** Route planning stays in `signinum-j2k`; Metal and CUDA remain adapter-only. Adaptive routing keeps CPU selected unless both stage and end-to-end gates pass. Instrumentation is additive and diagnostic; it must not change CPU correctness or strict device failure behavior.

**Tech Stack:** Rust 2021, Cargo workspace, Criterion benches, self-hosted Metal/CUDA runners, `signinum-core` backend traits, `signinum-j2k-native` encode/decode engine.

---

## File Map

- `crates/signinum-j2k/src/adapter/adaptive_route.rs`: adaptive planner evidence selection and stage gate status reporting.
- `crates/signinum-j2k/tests/adaptive_route.rs`: behavior tests for candidate stage reporting and exact RCA reclassification.
- `crates/signinum-j2k-metal/src/encode.rs`: public resident encode batch stats and host readback timing propagation.
- `crates/signinum-j2k-metal/src/compute.rs`: internal resident HTJ2K encode timing buckets.
- `crates/signinum-j2k-metal/benches/encode_stages.rs`: profile-visible resident Metal encode stats rows.
- `crates/signinum/benches/facade.rs`: RGB/RGBA 512/1024 facade gate rows.
- `crates/signinum/tests/bench_harness.rs`: source guards for facade benchmark row names and require variables.
- `docs/adaptive-j2k-gates.md`: evidence recording structure and final measured gate decisions.
- `crates/signinum-j2k-cuda/src/profile.rs`: CUDA detailed decode profile report fields and profile-row emission.
- `crates/signinum-j2k-cuda/src/decoder.rs`: CUDA decode wall timing, upload/download attribution, and per-stage dispatch counts.
- `crates/signinum-j2k-cuda/src/surface.rs`: profiled CUDA download helper.
- `crates/signinum-j2k-cuda/benches/htj2k_decode.rs`: CUDA Gray/RGB/RGBA strict resident decode bench matrix.
- `crates/signinum-j2k-cuda/tests/bench_harness.rs`: CUDA bench source guards.

## Commit Plan

Commit after each task:

1. `j2k: preserve adaptive gate candidate evidence`
2. `metal: expose resident encode rca timings`
3. `bench: expand facade htj2k gate rows`
4. `docs: scaffold adaptive gate evidence recording`
5. `cuda: add detailed htj2k decode profiling`
6. `cuda: add rgb rgba htj2k decode benches`
7. `docs: record adaptive j2k gate reruns`

CUDA tasks are deliberately last.

---

### Task 1: Planner Candidate Evidence Reporting

**Files:**
- Modify: `crates/signinum-j2k/src/adapter/adaptive_route.rs`
- Modify: `crates/signinum-j2k/tests/adaptive_route.rs`

- [ ] **Step 1: Add failing adaptive route tests**

Append these tests to `crates/signinum-j2k/tests/adaptive_route.rs`:

```rust
fn metal_stage_candidate_benchmarks_for(stage: J2kAdaptiveStage) -> J2kAdaptiveBenchmarks {
    let mut benchmarks = J2kAdaptiveBenchmarks::default();
    benchmarks.push_stage(J2kAdaptiveBenchmarkEvidence::stage(
        stage,
        BackendKind::Metal,
        100_000,
        70_000,
        1.0,
    ));
    benchmarks
}

#[test]
fn stage_candidate_remains_cpu_when_end_to_end_gate_is_missing() {
    let workload = rgb_wsi_htj2k_encode();
    let benchmarks = metal_stage_candidate_benchmarks_for(J2kAdaptiveStage::Dwt);

    let report = J2kAdaptiveRoutePlanner::new(metal_caps())
        .plan(workload, J2kAdaptiveBackendRequest::Accelerated, &benchmarks)
        .expect("route should plan with stage evidence only");

    let dwt = report.stage(J2kAdaptiveStage::Dwt).expect("DWT decision");
    assert_eq!(report.route_kind, J2kAdaptiveRouteKind::CpuOnly);
    assert_eq!(report.selected_device, None);
    assert_eq!(dwt.logical_owner, J2kAdaptiveStageOwner::Gpu);
    assert_eq!(dwt.selected_backend, BackendKind::Cpu);
    assert_eq!(
        dwt.gate_status,
        J2kAdaptiveStageGateStatus::EndToEndGateBlocked
    );
    assert!(
        dwt.improvement_percent.is_some(),
        "stage candidate evidence should remain visible for RCA"
    );
}

#[test]
fn stage_candidate_remains_cpu_when_end_to_end_gate_fails() {
    let workload = rgb_wsi_htj2k_encode();
    let mut benchmarks = metal_stage_candidate_benchmarks_for(J2kAdaptiveStage::HtBlockCoding);
    benchmarks.push_end_to_end(J2kAdaptiveBenchmarkEvidence::end_to_end(
        BackendKind::Metal,
        2_000_000,
        1_950_000,
        1.0,
    ));

    let report = J2kAdaptiveRoutePlanner::new(metal_caps())
        .plan(workload, J2kAdaptiveBackendRequest::Accelerated, &benchmarks)
        .expect("route should plan with a failing end-to-end gate");

    let ht = report
        .stage(J2kAdaptiveStage::HtBlockCoding)
        .expect("HT block decision");
    assert_eq!(report.route_kind, J2kAdaptiveRouteKind::CpuOnly);
    assert_eq!(report.selected_device, None);
    assert_eq!(ht.selected_backend, BackendKind::Cpu);
    assert_eq!(
        ht.gate_status,
        J2kAdaptiveStageGateStatus::EndToEndGateBlocked
    );
    assert!(ht.improvement_percent.is_some());
}

#[test]
fn rca_reclassification_is_exact_to_stage_and_backend() {
    let workload = rgb_wsi_htj2k_encode();
    let mut benchmarks = approved_metal_benchmarks_for(workload);
    benchmarks.push_stage(J2kAdaptiveBenchmarkEvidence::stage(
        J2kAdaptiveStage::Dwt,
        BackendKind::Metal,
        100_000,
        96_000,
        1.0,
    ));

    let report = J2kAdaptiveRoutePlanner::new(metal_caps())
        .with_rca_finding(J2kAdaptiveRcaFinding::reclassify_cpu(
            J2kAdaptiveStage::HtBlockCoding,
            BackendKind::Metal,
            J2kAdaptiveRcaReason::TransferSyncOverhead,
        ))
        .plan(workload, J2kAdaptiveBackendRequest::Accelerated, &benchmarks)
        .expect("route should plan with non-matching RCA");

    let dwt = report.stage(J2kAdaptiveStage::Dwt).expect("DWT decision");
    assert_eq!(dwt.gate_status, J2kAdaptiveStageGateStatus::BlockedNeedsRca);
    assert_eq!(dwt.selected_backend, BackendKind::Cpu);
    assert!(report.has_unresolved_rca());
}
```

- [ ] **Step 2: Run the tests and confirm the first two fail**

Run:

```sh
cargo test -p signinum-j2k --test adaptive_route stage_candidate_remains_cpu_when_end_to_end_gate_is_missing stage_candidate_remains_cpu_when_end_to_end_gate_fails rca_reclassification_is_exact_to_stage_and_backend
```

Expected: the stage-candidate tests fail because current planner reports `BenchmarkGateMissing` or hides the stage evidence when no approved end-to-end backend exists.

- [ ] **Step 3: Add candidate evidence selection to the planner**

In `crates/signinum-j2k/src/adapter/adaptive_route.rs`, add these methods to `impl J2kAdaptiveBenchmarks`:

```rust
    fn has_evidence_for(&self, backend: BackendKind) -> bool {
        self.end_to_end_for(backend).is_some()
            || self.stage.iter().any(|evidence| evidence.backend == backend)
    }

    fn best_observed_ns_for(&self, backend: BackendKind) -> Option<u64> {
        let end_to_end = self.end_to_end_for(backend).map(|evidence| evidence.accelerated_ns);
        let stage = self
            .stage
            .iter()
            .rev()
            .find(|evidence| evidence.backend == backend)
            .map(|evidence| evidence.accelerated_ns);
        end_to_end.or(stage)
    }
```

Then replace the start of `accelerated_report` with:

```rust
        let Some(backend) = self.best_candidate_device(benchmarks) else {
            return self.gated_cpu_report(workload, request, None, benchmarks);
        };

        let end_to_end_passed = benchmarks
            .end_to_end_for(backend)
            .is_some_and(|evidence| evidence.passes(self.policy));
        if !end_to_end_passed {
            return self.gated_cpu_report(workload, request, Some(backend), benchmarks);
        }
```

Replace `best_approved_device` with this implementation and update its call sites to `best_candidate_device`:

```rust
    fn best_candidate_device(&self, benchmarks: &J2kAdaptiveBenchmarks) -> Option<BackendKind> {
        [BackendKind::Metal, BackendKind::Cuda]
            .into_iter()
            .filter(|backend| self.supports_backend(*backend))
            .filter(|backend| benchmarks.has_evidence_for(*backend))
            .min_by_key(|backend| benchmarks.best_observed_ns_for(*backend).unwrap_or(u64::MAX))
    }
```

Do not change `stage_decision`; it already emits `EndToEndGateBlocked` when called with `end_to_end_passed = false`.

- [ ] **Step 4: Run the adaptive route tests**

Run:

```sh
cargo test -p signinum-j2k --test adaptive_route
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 1**

```sh
git add crates/signinum-j2k/src/adapter/adaptive_route.rs crates/signinum-j2k/tests/adaptive_route.rs
git commit -m "j2k: preserve adaptive gate candidate evidence"
```

---

### Task 2: Metal Resident Encode RCA Timings

**Files:**
- Modify: `crates/signinum-j2k-metal/src/encode.rs`
- Modify: `crates/signinum-j2k-metal/src/compute.rs`
- Modify: `crates/signinum-j2k-metal/benches/encode_stages.rs`

- [ ] **Step 1: Extend the zero-default stage stats test**

In `crates/signinum-j2k-metal/src/encode.rs`, update `resident_lossless_stage_stats_default_to_zero` so it checks the new fields:

```rust
        assert_eq!(stats.stage_stats.coefficient_prep_duration, Duration::ZERO);
        assert_eq!(stats.stage_stats.deinterleave_rct_duration, Duration::ZERO);
        assert_eq!(stats.stage_stats.dwt53_duration, Duration::ZERO);
        assert_eq!(stats.stage_stats.coefficient_extract_duration, Duration::ZERO);
        assert_eq!(stats.stage_stats.ht_block_encode_duration, Duration::ZERO);
        assert_eq!(stats.stage_stats.packet_block_prep_duration, Duration::ZERO);
        assert_eq!(stats.stage_stats.packetization_duration, Duration::ZERO);
        assert_eq!(stats.stage_stats.codestream_assembly_duration, Duration::ZERO);
        assert_eq!(stats.stage_stats.sync_wait_duration, Duration::ZERO);
        assert_eq!(stats.stage_stats.host_readback_duration, Duration::ZERO);
```

- [ ] **Step 2: Run the focused test and confirm it fails to compile**

Run:

```sh
cargo test -p signinum-j2k-metal resident_lossless_stage_stats_default_to_zero
```

Expected: compile failure for missing fields.

- [ ] **Step 3: Add public Metal stats fields and accumulation**

In `MetalLosslessEncodeStageStats`, add these fields:

```rust
    /// Time spent preparing resident encode coefficients.
    pub coefficient_prep_duration: Duration,
    /// Time spent in fused deinterleave plus RCT work when used.
    pub deinterleave_rct_duration: Duration,
    /// Time spent in forward 5/3 DWT work.
    pub dwt53_duration: Duration,
    /// Time spent extracting coefficient buffers for resident encode.
    pub coefficient_extract_duration: Duration,
    /// Time spent encoding HT code blocks.
    pub ht_block_encode_duration: Duration,
    /// Time spent preparing packet block metadata.
    pub packet_block_prep_duration: Duration,
    /// Time spent encoding packet bodies.
    pub packetization_duration: Duration,
    /// Time spent assembling codestream bytes.
    pub codestream_assembly_duration: Duration,
    /// Time spent waiting for resident codestream completion.
    pub sync_wait_duration: Duration,
    /// Time spent materializing buffer-backed codestream bytes into host bytes.
    pub host_readback_duration: Duration,
```

Update `has_timings()` by adding these checks:

```rust
            || self.coefficient_prep_duration > Duration::ZERO
            || self.deinterleave_rct_duration > Duration::ZERO
            || self.dwt53_duration > Duration::ZERO
            || self.coefficient_extract_duration > Duration::ZERO
            || self.ht_block_encode_duration > Duration::ZERO
            || self.packet_block_prep_duration > Duration::ZERO
            || self.packetization_duration > Duration::ZERO
            || self.codestream_assembly_duration > Duration::ZERO
            || self.sync_wait_duration > Duration::ZERO
            || self.host_readback_duration > Duration::ZERO
```

Update `add_assign()` with:

```rust
        self.coefficient_prep_duration = self
            .coefficient_prep_duration
            .saturating_add(other.coefficient_prep_duration);
        self.deinterleave_rct_duration = self
            .deinterleave_rct_duration
            .saturating_add(other.deinterleave_rct_duration);
        self.dwt53_duration = self.dwt53_duration.saturating_add(other.dwt53_duration);
        self.coefficient_extract_duration = self
            .coefficient_extract_duration
            .saturating_add(other.coefficient_extract_duration);
        self.ht_block_encode_duration = self
            .ht_block_encode_duration
            .saturating_add(other.ht_block_encode_duration);
        self.packet_block_prep_duration = self
            .packet_block_prep_duration
            .saturating_add(other.packet_block_prep_duration);
        self.packetization_duration = self
            .packetization_duration
            .saturating_add(other.packetization_duration);
        self.codestream_assembly_duration = self
            .codestream_assembly_duration
            .saturating_add(other.codestream_assembly_duration);
        self.sync_wait_duration = self.sync_wait_duration.saturating_add(other.sync_wait_duration);
        self.host_readback_duration = self
            .host_readback_duration
            .saturating_add(other.host_readback_duration);
```

- [ ] **Step 4: Mirror stats in `compute.rs`**

Extend `J2kResidentEncodeStageStats` with the same fields, excluding host readback:

```rust
    pub(crate) coefficient_prep_duration: Duration,
    pub(crate) deinterleave_rct_duration: Duration,
    pub(crate) dwt53_duration: Duration,
    pub(crate) coefficient_extract_duration: Duration,
    pub(crate) ht_block_encode_duration: Duration,
    pub(crate) packet_block_prep_duration: Duration,
    pub(crate) packetization_duration: Duration,
    pub(crate) codestream_assembly_duration: Duration,
```

Update `impl From<compute::J2kResidentEncodeStageStats> for MetalLosslessEncodeStageStats`:

```rust
            coefficient_prep_duration: stats.coefficient_prep_duration,
            deinterleave_rct_duration: stats.deinterleave_rct_duration,
            dwt53_duration: stats.dwt53_duration,
            coefficient_extract_duration: stats.coefficient_extract_duration,
            ht_block_encode_duration: stats.ht_block_encode_duration,
            packet_block_prep_duration: stats.packet_block_prep_duration,
            packetization_duration: stats.packetization_duration,
            codestream_assembly_duration: stats.codestream_assembly_duration,
```

- [ ] **Step 5: Populate split HT batch timing buckets**

In `submit_lossless_codestream_buffers_from_prepared_ht_batch`, split timing around the existing command encoder sections:

```rust
        let ht_block_encode_started = profile_stages.then(Instant::now);
        if tier1_job_count > 0 {
            let encoder = command_buffer.new_compute_command_encoder();
            label_compute_encoder(encoder, "HTJ2K Tier-1 encode");
            let pipeline = kernel.pipeline(runtime)?;
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(0, Some(&coefficient_buffer), 0);
            encoder.set_buffer(1, Some(&tier1_output_buffer), 0);
            encoder.set_buffer(2, Some(&tier1_job_buffer), 0);
            encoder.set_buffer(3, Some(&runtime.ht_vlc_encode_table0), 0);
            encoder.set_buffer(4, Some(&runtime.ht_vlc_encode_table1), 0);
            encoder.set_buffer(5, Some(&runtime.ht_uvlc_encode_table), 0);
            encoder.set_buffer(6, Some(&tier1_status_buffer), 0);
            encoder.set_bytes(7, size_of::<u32>() as u64, (&raw const tier1_job_count).cast());
            kernel.dispatch(encoder, pipeline, tier1_job_count);
            encoder.end_encoding();
        }
        if let Some(started) = ht_block_encode_started {
            stage_stats.ht_block_encode_duration = started.elapsed();
        }
```

Apply the same pattern around:

```rust
label_compute_encoder(encoder, "HTJ2K packet block prep");
```

and store elapsed time in `packet_block_prep_duration`.

Apply the same pattern around:

```rust
label_compute_encoder(encoder, "HTJ2K packetization");
```

and store elapsed time in `packetization_duration`.

Apply the same pattern around:

```rust
label_compute_encoder(encoder, "HTJ2K codestream assembly");
```

and store elapsed time in `codestream_assembly_duration`.

Keep `ht_command_encode_duration` by setting it to the sum of the split command-encode buckets:

```rust
        stage_stats.ht_command_encode_duration = stage_stats
            .ht_block_encode_duration
            .saturating_add(stage_stats.packet_block_prep_duration)
            .saturating_add(stage_stats.packetization_duration)
            .saturating_add(stage_stats.codestream_assembly_duration);
```

- [ ] **Step 6: Populate sync/wait and host readback timing**

In `wait_submitted_resident_lossless_buffer_encode_batch`, after measuring `wait_started`, also set:

```rust
                    submitted.stats.stage_stats.sync_wait_duration = submitted
                        .stats
                        .stage_stats
                        .sync_wait_duration
                        .saturating_add(started.elapsed());
```

In the host-byte conversion path that calls `MetalEncodedJ2k::to_encoded_j2k()`, measure the call:

```rust
let host_readback_started = compute::metal_profile_stages_enabled().then(Instant::now);
let encoded = buffer_outcome.encoded.to_encoded_j2k()?;
let host_readback_duration = host_readback_started.map_or(Duration::ZERO, |started| started.elapsed());
```

Then pass `host_readback_duration` into the returned `MetalLosslessEncodeOutcome`.

- [ ] **Step 7: Add profile output in the Metal encode bench**

In `crates/signinum-j2k-metal/benches/encode_stages.rs`, after each resident report is produced, print stats only when profile stages are enabled:

```rust
fn emit_resident_stats(label: &str, stats: signinum_j2k_metal::MetalLosslessEncodeBatchStats) {
    if std::env::var_os("SIGNINUM_J2K_METAL_PROFILE_STAGES").is_none() {
        return;
    }
    eprintln!(
        "signinum_profile codec=j2k op=encode path=metal_resident label={label} plan_us={} prepare_submit_us={} ht_block_encode_us={} packet_block_prep_us={} packetization_us={} codestream_assembly_us={} sync_wait_us={} tile_count={} code_block_count={}",
        stats.stage_stats.plan_duration.as_micros(),
        stats.stage_stats.prepare_submit_duration.as_micros(),
        stats.stage_stats.ht_block_encode_duration.as_micros(),
        stats.stage_stats.packet_block_prep_duration.as_micros(),
        stats.stage_stats.packetization_duration.as_micros(),
        stats.stage_stats.codestream_assembly_duration.as_micros(),
        stats.stage_stats.sync_wait_duration.as_micros(),
        stats.stage_stats.tile_count,
        stats.stage_stats.code_block_count,
    );
}
```

Call this helper for resident HTJ2K RGB8 512/1024 rows and resident RPCL batch rows that already return batch stats.

- [ ] **Step 8: Run Metal tests and bench build**

Run:

```sh
cargo test -p signinum-j2k-metal resident_lossless_stage_stats_default_to_zero
cargo bench -p signinum-j2k-metal --bench encode_stages --no-run
```

Expected: tests pass and the bench compiles.

- [ ] **Step 9: Commit Task 2**

```sh
git add crates/signinum-j2k-metal/src/encode.rs crates/signinum-j2k-metal/src/compute.rs crates/signinum-j2k-metal/benches/encode_stages.rs
git commit -m "metal: expose resident encode rca timings"
```

---

### Task 3: Facade Gate Row Scaffolding

**Files:**
- Modify: `crates/signinum/benches/facade.rs`
- Modify: `crates/signinum/tests/bench_harness.rs`

- [ ] **Step 1: Add failing bench harness guards**

Extend `facade_bench_exposes_cpu_and_hybrid_encode_surfaces` in `crates/signinum/tests/bench_harness.rs` with these expected strings:

```rust
        "cpu_rgb8_1024_htj2k_external",
        "adaptive_rgb8_1024_htj2k_perf_gate_external",
        "strict_metal_rgb8_1024_htj2k_external",
        "strict_cuda_rgb8_1024_htj2k_external",
        "cpu_rgba8_512_htj2k_external",
        "adaptive_rgba8_512_htj2k_perf_gate_external",
        "strict_metal_rgba8_512_htj2k_external",
        "strict_cuda_rgba8_512_htj2k_external",
        "cpu_rgba8_1024_htj2k_external",
        "adaptive_rgba8_1024_htj2k_perf_gate_external",
        "strict_metal_rgba8_1024_htj2k_external",
        "strict_cuda_rgba8_1024_htj2k_external",
```

- [ ] **Step 2: Run the harness and confirm it fails**

Run:

```sh
cargo test -p signinum --test bench_harness
```

Expected: missing benchmark string failure.

- [ ] **Step 3: Add local facade bench case helpers**

In `crates/signinum/benches/facade.rs`, add this helper near `patterned_rgb8` usage:

```rust
fn patterned_rgba8(width: u32, height: u32) -> Vec<u8> {
    let rgb = patterned_rgb8(width, height);
    let mut rgba = Vec::with_capacity(width as usize * height as usize * 4);
    for chunk in rgb.chunks_exact(3) {
        rgba.extend_from_slice(chunk);
        rgba.push(255);
    }
    rgba
}

struct FacadeMatrixCase {
    label: &'static str,
    width: u32,
    height: u32,
    components: u8,
    pixels: Vec<u8>,
}
```

- [ ] **Step 4: Replace `bench_facade_backend_speed_matrix` with case loops**

Use this structure so existing 512 RGB labels remain unchanged and new labels are added:

```rust
    let cases = [
        FacadeMatrixCase {
            label: "rgb8_512",
            width: 512,
            height: 512,
            components: 3,
            pixels: patterned_rgb8(512, 512),
        },
        FacadeMatrixCase {
            label: "rgb8_1024",
            width: 1024,
            height: 1024,
            components: 3,
            pixels: patterned_rgb8(1024, 1024),
        },
        FacadeMatrixCase {
            label: "rgba8_512",
            width: 512,
            height: 512,
            components: 4,
            pixels: patterned_rgba8(512, 512),
        },
        FacadeMatrixCase {
            label: "rgba8_1024",
            width: 1024,
            height: 1024,
            components: 4,
            pixels: patterned_rgba8(1024, 1024),
        },
    ];

    let mut group = c.benchmark_group("facade_j2k_htj2k_encode_backend_speed_matrix");
    for case in &cases {
        let cpu_name = format!("cpu_{}_htj2k_external", case.label);
        group.bench_function(cpu_name.as_str(), |b| {
            b.iter(|| {
                let samples = J2kLosslessSamples::new(
                    black_box(case.pixels.as_slice()),
                    case.width,
                    case.height,
                    case.components,
                    8,
                    false,
                )
                .expect("valid facade matrix samples");
                let encoded = facade_encode_j2k_lossless(samples, &cpu_options)
                    .expect("CPU HTJ2K encode");
                black_box(encoded.codestream.len());
            });
        });

        let adaptive_name = format!("adaptive_{}_htj2k_perf_gate_external", case.label);
        group.bench_function(adaptive_name.as_str(), |b| {
            b.iter(|| {
                let samples = J2kLosslessSamples::new(
                    black_box(case.pixels.as_slice()),
                    case.width,
                    case.height,
                    case.components,
                    8,
                    false,
                )
                .expect("valid facade matrix samples");
                let encoded = facade_encode_j2k_lossless(samples, &adaptive_options)
                    .expect("adaptive HTJ2K encode");
                black_box((encoded.backend, encoded.codestream.len()));
            });
        });
    }
```

Then add strict Metal/CUDA loops in their existing `#[cfg]` blocks using the same label pattern:

```rust
        let strict_name = format!("strict_metal_{}_htj2k_external", case.label);
```

and:

```rust
        let strict_name = format!("strict_cuda_{}_htj2k_external", case.label);
```

Update `metal_htj2k_encode_available` and `cuda_htj2k_encode_available` to accept `case: &FacadeMatrixCase` and build samples from `case.pixels`, `case.width`, `case.height`, and `case.components`.

- [ ] **Step 5: Run facade bench harness and bench build**

Run:

```sh
cargo test -p signinum --test bench_harness
cargo bench -p signinum --bench facade --no-run
```

Expected: harness passes and facade bench compiles.

- [ ] **Step 6: Commit Task 3**

```sh
git add crates/signinum/benches/facade.rs crates/signinum/tests/bench_harness.rs
git commit -m "bench: expand facade htj2k gate rows"
```

---

### Task 4: Documentation Evidence Scaffolding

**Files:**
- Modify: `docs/adaptive-j2k-gates.md`
- Modify: `docs/bench.md`

- [ ] **Step 1: Add documentation structure without measured numbers**

Add this section to `docs/adaptive-j2k-gates.md` after the gate policy:

```markdown
## Evidence Recording Template

New adaptive gate reruns must record:

- Commit SHA.
- Host, OS, architecture, CPU/GPU, memory, driver/runtime versions.
- Rust compiler version.
- Exact command and required environment variables.
- CPU-only Criterion interval.
- Adaptive Criterion interval.
- Strict device Criterion interval when available.
- Stage Criterion intervals for every device-shaped stage under consideration.
- Gate decision: `approved`, `candidate`, `blocked`, or `reclassified-cpu`.
- RCA reason for every blocked logical GPU-shaped stage.

Do not copy numbers into this file from a different host class. Apple Metal
numbers and CUDA runner numbers are separate evidence sets.
```

Add this bullet under the benchmark behavior section in `docs/bench.md`:

```markdown
- Gate documentation may include internal RCA evidence, but it is not a public
  speed claim. Public claims still require `cargo xtask bench-report` and the
  raw benchmark output bundle.
```

- [ ] **Step 2: Run doc whitespace check**

Run:

```sh
git diff --check
```

Expected: no output.

- [ ] **Step 3: Commit Task 4**

```sh
git add docs/adaptive-j2k-gates.md docs/bench.md
git commit -m "docs: scaffold adaptive gate evidence recording"
```

---

### Task 5: CUDA Detailed Decode Profiling

**Files:**
- Modify: `crates/signinum-j2k-cuda/src/profile.rs`
- Modify: `crates/signinum-j2k-cuda/src/decoder.rs`
- Modify: `crates/signinum-j2k-cuda/src/surface.rs`

CUDA starts here.

- [ ] **Step 1: Add failing profile tests**

In `crates/signinum-j2k-cuda/src/profile.rs`, add tests beside existing profile tests:

```rust
    #[test]
    fn detailed_decode_profile_separates_wall_and_stage_sum() {
        let mut report = CudaHtj2kProfileReport {
            parse_us: 1,
            plan_us: 2,
            flatten_us: 3,
            h2d_us: 4,
            ht_cleanup_us: 5,
            ht_refine_us: 5,
            dequant_us: 6,
            idwt_us: 7,
            mct_us: 8,
            store_us: 9,
            total_us: 0,
            block_count: 10,
            payload_bytes: 11,
            dispatch_count: 12,
            residency: SurfaceResidency::CudaResidentDecode,
            detail: CudaHtj2kDecodeProfileDetail::default(),
        };
        report.detail.wall_total_us = 100;
        report.detail.table_upload_us = 13;
        report.detail.payload_upload_us = 17;
        report.detail.ht_dispatch_count = 2;
        finalize_decode_total_us(&mut report);

        assert_eq!(report.detail.wall_total_us, 100);
        assert_eq!(report.detail.stage_sum_us, report.total_us);
        assert_eq!(report.detail.ht_dispatch_count, 2);
    }
```

- [ ] **Step 2: Run the profile test and confirm it fails to compile**

Run:

```sh
cargo test -p signinum-j2k-cuda detailed_decode_profile_separates_wall_and_stage_sum
```

Expected: compile failure for missing `CudaHtj2kDecodeProfileDetail` or `detail`.

- [ ] **Step 3: Add the detailed profile struct**

In `profile.rs`, add:

```rust
/// Detailed route-overhead timings for strict CUDA HTJ2K decode.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CudaHtj2kDecodeProfileDetail {
    pub wall_total_us: u128,
    pub stage_sum_us: u128,
    pub table_upload_us: u128,
    pub payload_upload_us: u128,
    pub job_upload_us: u128,
    pub status_d2h_us: u128,
    pub output_d2h_us: u128,
    pub ht_dispatch_count: usize,
    pub dequant_dispatch_count: usize,
    pub idwt_dispatch_count: usize,
    pub mct_dispatch_count: usize,
    pub store_dispatch_count: usize,
}
```

Add this field to `CudaHtj2kProfileReport`:

```rust
    /// Detailed route-overhead profile for RCA.
    pub detail: CudaHtj2kDecodeProfileDetail,
```

Update every `CudaHtj2kProfileReport` literal in `decoder.rs` and `profile.rs` with:

```rust
detail: CudaHtj2kDecodeProfileDetail::default(),
```

Update `finalize_decode_total_us`:

```rust
    report.total_us = [
        report.parse_us,
        report.plan_us,
        report.flatten_us,
        report.h2d_us,
        report.ht_cleanup_us,
        report.dequant_us,
        report.idwt_us,
        report.mct_us,
        report.store_us,
    ]
    .into_iter()
    .fold(0u128, u128::saturating_add);
    report.detail.stage_sum_us = report.total_us;
```

Do not include `ht_refine_us` in the summed total while it is fused with cleanup.

- [ ] **Step 4: Emit detailed profile fields**

Extend `emit_htj2k_profile_row` with these string fields:

```rust
    let wall_total_us = report.detail.wall_total_us.to_string();
    let stage_sum_us = report.detail.stage_sum_us.to_string();
    let table_upload_us = report.detail.table_upload_us.to_string();
    let payload_upload_us = report.detail.payload_upload_us.to_string();
    let job_upload_us = report.detail.job_upload_us.to_string();
    let status_d2h_us = report.detail.status_d2h_us.to_string();
    let output_d2h_us = report.detail.output_d2h_us.to_string();
    let ht_dispatch_count = report.detail.ht_dispatch_count.to_string();
    let dequant_dispatch_count = report.detail.dequant_dispatch_count.to_string();
    let idwt_dispatch_count = report.detail.idwt_dispatch_count.to_string();
    let mct_dispatch_count = report.detail.mct_dispatch_count.to_string();
    let store_dispatch_count = report.detail.store_dispatch_count.to_string();
```

Add them to the emitted field array with names matching the variable names.

- [ ] **Step 5: Add wall timing in decoder entry points**

In `decode_to_cuda_resident_surface_with_profile_impl`, start an outer timer:

```rust
    let wall_started = profile::profile_now(true);
```

Before returning each profiled report, set:

```rust
    report.detail.wall_total_us = profile::elapsed_us(wall_started);
    profile::finalize_decode_total_us(&mut report);
```

For color decode, set `color.report.detail.wall_total_us` the same way before finalizing.

- [ ] **Step 6: Track upload and dispatch detail**

Where table/resource upload is measured, add the elapsed value to both `h2d_us` and `detail.table_upload_us`.

Where payload upload is measured for color decode, add the elapsed value to both `h2d_us` and `detail.payload_upload_us`.

Where subband decode output stage timings are read, set:

```rust
        timings.ht_cleanup = timings
            .ht_cleanup
            .saturating_add(stage_timings.ht_cleanup_us);
        timings.dequant = timings.dequant.saturating_add(stage_timings.dequant_us);
```

When component timings are added to the report, also update:

```rust
        report.detail.ht_dispatch_count = report
            .detail
            .ht_dispatch_count
            .saturating_add(component.decode_dispatches);
```

For IDWT, MCT, and store sections that already have stats, add their dispatch counts to `detail.idwt_dispatch_count`, `detail.mct_dispatch_count`, and `detail.store_dispatch_count`.

- [ ] **Step 7: Add profiled download helper**

In `crates/signinum-j2k-cuda/src/surface.rs`, add:

```rust
impl Surface {
    /// Download the surface and return elapsed host copy time in microseconds.
    pub fn download_into_profiled(&self, out: &mut [u8], stride: usize) -> Result<u128, Error> {
        let started = std::time::Instant::now();
        self.download_into(out, stride)?;
        Ok(started.elapsed().as_micros())
    }
}
```

This helper does not change strict decode behavior.

- [ ] **Step 8: Run CUDA profile tests without requiring runtime**

Run:

```sh
cargo test -p signinum-j2k-cuda detailed_decode_profile_separates_wall_and_stage_sum
cargo check -p signinum-j2k-cuda --features cuda-runtime --tests
```

Expected: test passes and CUDA-runtime build checks.

- [ ] **Step 9: Commit Task 5**

```sh
git add crates/signinum-j2k-cuda/src/profile.rs crates/signinum-j2k-cuda/src/decoder.rs crates/signinum-j2k-cuda/src/surface.rs
git commit -m "cuda: add detailed htj2k decode profiling"
```

---

### Task 6: CUDA RGB/RGBA Decode Benches

**Files:**
- Modify: `crates/signinum-j2k-cuda/benches/htj2k_decode.rs`
- Create: `crates/signinum-j2k-cuda/tests/bench_harness.rs`

- [ ] **Step 1: Add failing CUDA bench harness**

Create `crates/signinum-j2k-cuda/tests/bench_harness.rs`:

```rust
// SPDX-License-Identifier: Apache-2.0

#[test]
fn cuda_htj2k_decode_bench_exposes_gray_rgb_rgba_rows() {
    let bench = include_str!("../benches/htj2k_decode.rs");

    for expected in [
        "cpu_gray8",
        "cuda_gray8",
        "cpu_rgb8",
        "cuda_rgb8",
        "cpu_rgba8",
        "cuda_rgba8",
        "j2k_cuda_htj2k_full_tile_decode",
        "j2k_cuda_htj2k_roi_decode",
        "j2k_cuda_htj2k_scaled_decode",
        "j2k_cuda_htj2k_roi_scaled_decode",
        "j2k_cuda_htj2k_tile_batch_decode",
        "SIGNINUM_REQUIRE_CUDA_BENCH",
    ] {
        assert!(
            bench.contains(expected),
            "CUDA HTJ2K decode benchmark is missing `{expected}`"
        );
    }
}
```

- [ ] **Step 2: Run the harness and confirm it fails**

Run:

```sh
cargo test -p signinum-j2k-cuda --test bench_harness
```

Expected: missing RGB/RGBA row string failure.

- [ ] **Step 3: Add bench cases and generated RGB fixture**

In `htj2k_decode.rs`, add:

```rust
use signinum_core::{BackendKind, DeviceSurface};

struct DecodeBenchCase {
    id: &'static str,
    fixture: Vec<u8>,
    fmt: PixelFormat,
    cuda_available: bool,
}
```

Replace the fixture setup in `bench_htj2k_decode` with:

```rust
    let gray_fixture = htj2k_gray8_fixture(TILE_DIM, TILE_DIM);
    let rgb_fixture = htj2k_rgb8_fixture(TILE_DIM, TILE_DIM);
    let cases = vec![
        DecodeBenchCase {
            id: "gray8",
            cuda_available: cuda_decode_available("gray8", &gray_fixture, PixelFormat::Gray8),
            fixture: gray_fixture,
            fmt: PixelFormat::Gray8,
        },
        DecodeBenchCase {
            id: "rgb8",
            cuda_available: cuda_decode_available("rgb8", &rgb_fixture, PixelFormat::Rgb8),
            fixture: rgb_fixture.clone(),
            fmt: PixelFormat::Rgb8,
        },
        DecodeBenchCase {
            id: "rgba8",
            cuda_available: cuda_decode_available("rgba8", &rgb_fixture, PixelFormat::Rgba8),
            fixture: rgb_fixture,
            fmt: PixelFormat::Rgba8,
        },
    ];
```

Add the RGB fixture helper:

```rust
fn htj2k_rgb8_fixture(width: u32, height: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity(width as usize * height as usize * 3);
    for idx in 0..width * height {
        pixels.push(u8::try_from((idx * 17 + idx / 3) & 0xff).expect("masked red fits"));
        pixels.push(u8::try_from((idx * 29 + 7) & 0xff).expect("masked green fits"));
        pixels.push(u8::try_from((idx * 43 + 19) & 0xff).expect("masked blue fits"));
    }
    let options = EncodeOptions {
        reversible: true,
        use_ht_block_coding: true,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    encode_htj2k(&pixels, width, height, 3, 8, false, &options).expect("encode RGB HTJ2K fixture")
}
```

- [ ] **Step 4: Parameterize every decode bench by case**

Change each bench function signature to accept `cases: &[DecodeBenchCase]`.

For CPU full tile rows, use:

```rust
    for case in cases {
        let cpu_id = format!("cpu_{}", case.id);
        group.bench_with_input(BenchmarkId::new(cpu_id, TILE_DIM), case, |b, case| {
            b.iter(|| {
                let mut decoder = J2kDecoder::new(black_box(case.fixture.as_slice())).expect("decoder");
                let stride = TILE_DIM as usize * case.fmt.bytes_per_pixel();
                let mut out = vec![0u8; stride * TILE_DIM as usize];
                decoder
                    .decode_into(&mut out, stride, case.fmt)
                    .expect("CPU HTJ2K decode");
                black_box(out)
            });
        });
    }
```

For CUDA full tile rows, use:

```rust
        if case.cuda_available {
            let cuda_id = format!("cuda_{}", case.id);
            group.bench_with_input(BenchmarkId::new(cuda_id, TILE_DIM), case, |b, case| {
                b.iter(|| {
                    let mut decoder = J2kDecoder::new(black_box(case.fixture.as_slice())).expect("decoder");
                    let surface = decoder
                        .decode_to_device(case.fmt, BackendRequest::Cuda)
                        .expect("strict CUDA HTJ2K decode");
                    assert_cuda_resident_decode(&surface);
                    black_box(surface)
                });
            });
        }
```

Apply the same pattern to ROI, scaled, ROI-scaled, and tile-batch rows. Use `scaled.w as usize * case.fmt.bytes_per_pixel()` for scaled stride and `roi.w as usize * case.fmt.bytes_per_pixel()` for ROI stride.

- [ ] **Step 5: Add strict CUDA residency assertion helper**

Add:

```rust
fn assert_cuda_resident_decode(surface: &signinum_j2k_cuda::Surface) {
    assert_eq!(surface.backend_kind(), BackendKind::Cuda);
    assert_eq!(surface.residency(), SurfaceResidency::CudaResidentDecode);
    assert!(surface.as_host_bytes().is_none());
    let cuda = surface.cuda_surface().expect("cuda surface");
    assert_ne!(cuda.device_ptr(), 0);
    assert_eq!(cuda.stats().copy_kernel_dispatches(), 0);
    assert!(cuda.stats().decode_kernel_dispatches() > 0);
}
```

Update `cuda_decode_available`:

```rust
fn cuda_decode_available(label: &str, fixture: &[u8], fmt: PixelFormat) -> bool {
    let mut session = CudaSession::default();
    let result = J2kDecoder::new(fixture)
        .and_then(|mut decoder| decoder.decode_to_device_with_session(fmt, &mut session));
    match result {
        Ok(surface) if surface.residency() == SurfaceResidency::CudaResidentDecode => true,
        Ok(_) if std::env::var_os("SIGNINUM_REQUIRE_CUDA_BENCH").is_some() => {
            panic!("SIGNINUM_REQUIRE_CUDA_BENCH is set but {label} decode was not CUDA resident")
        }
        Ok(_) => {
            eprintln!("skipping CUDA HTJ2K {label} decode benches: strict CUDA resident path unavailable");
            false
        }
        Err(error) if std::env::var_os("SIGNINUM_REQUIRE_CUDA_BENCH").is_some() => {
            panic!("SIGNINUM_REQUIRE_CUDA_BENCH is set but {label} CUDA decode failed: {error}")
        }
        Err(error) => {
            eprintln!("skipping CUDA HTJ2K {label} decode benches: {error}");
            false
        }
    }
}
```

- [ ] **Step 6: Run CUDA bench guards and bench build**

Run:

```sh
cargo test -p signinum-j2k-cuda --test bench_harness
cargo bench -p signinum-j2k-cuda --bench htj2k_decode --features cuda-runtime --no-run
```

Expected: guard passes and benchmark compiles.

- [ ] **Step 7: Commit Task 6**

```sh
git add crates/signinum-j2k-cuda/benches/htj2k_decode.rs crates/signinum-j2k-cuda/tests/bench_harness.rs
git commit -m "cuda: add rgb rgba htj2k decode benches"
```

---

### Task 7: Timed Gate Runs and Evidence Docs

**Files:**
- Modify: `docs/adaptive-j2k-gates.md`

- [ ] **Step 1: Run baseline local checks**

Run:

```sh
cargo fmt --all
git diff --check
cargo test -p signinum-core --test repo_integrity public_docs_describe_facade_auto_and_cuda_runtime_surface_scope
cargo test -p signinum-j2k --test adaptive_route
cargo test -p signinum --test bench_harness
```

Expected: all commands pass.

- [ ] **Step 2: Run Metal gate commands on Apple Silicon Metal host**

Run:

```sh
SIGNINUM_REQUIRE_METAL_BENCH=1 SIGNINUM_J2K_METAL_PROFILE_STAGES=1 \
  cargo bench -p signinum-j2k-metal --bench encode_stages -- \
  --noplot --sample-size 10 --warm-up-time 1 --measurement-time 2

SIGNINUM_REQUIRE_METAL_BENCH=1 \
  cargo bench -p signinum --bench facade --features metal -- \
  facade_j2k_htj2k_encode_backend_speed_matrix \
  --noplot --sample-size 10 --warm-up-time 1 --measurement-time 2
```

Expected: strict Metal rows run or fail loudly. Record Criterion intervals and profile rows.

- [ ] **Step 3: Run CUDA checks last on the CUDA runner**

Run:

```sh
cargo check -p signinum-j2k-cuda --features cuda-runtime --benches --tests
cargo clippy -p signinum-j2k-cuda --features cuda-runtime --all-targets -- -D warnings
SIGNINUM_REQUIRE_CUDA_RUNTIME=1 SIGNINUM_REQUIRE_CUDA_HTJ2K_STRICT=1 \
  cargo test -p signinum-j2k-cuda --features cuda-runtime --test host_surface
cargo bench -p signinum-j2k-cuda --features cuda-runtime --bench htj2k_decode --no-run
SIGNINUM_REQUIRE_CUDA_BENCH=1 \
  cargo bench -p signinum-j2k-cuda --features cuda-runtime --bench htj2k_decode -- \
  --noplot --sample-size 10 --warm-up-time 1 --measurement-time 2
SIGNINUM_REQUIRE_CUDA_BENCH=1 \
  cargo bench -p signinum --bench facade --features cuda-runtime -- \
  facade_j2k_htj2k_encode_backend_speed_matrix \
  --noplot --sample-size 10 --warm-up-time 1 --measurement-time 2
```

Expected: strict CUDA rows run or fail loudly. Record Criterion intervals and profile rows.

- [ ] **Step 4: Update gate documentation with measured decisions**

Add dated subsections to `docs/adaptive-j2k-gates.md` using this shape:

```markdown
## 2026-05-31 Metal Resident HTJ2K Encode RCA Rerun

Evidence:

- Commit: output from `git rev-parse --short HEAD`
- Host: output from `uname -a`, GPU name from `system_profiler SPDisplaysDataType`
  on macOS or `nvidia-smi --query-gpu=name,driver_version --format=csv,noheader`
  on CUDA Linux.
- Rust: first line of `rustc -Vv`
- Commands:
  - Copy each exact command that produced the recorded Criterion output.

Decision:

- Use one of `approved`, `candidate`, `blocked`, or `reclassified-cpu`, then
  state the measured reason.

RCA:

- For each stage, record the Criterion interval or profile row that supports
  the decision.
```

Use the same structure for CUDA decode RGB/RGBA rows. Do not add values that were not measured.

- [ ] **Step 5: Commit Task 7**

```sh
git add docs/adaptive-j2k-gates.md
git commit -m "docs: record adaptive j2k gate reruns"
```

---

## Final Verification

Run at minimum before final handoff:

```sh
cargo fmt --all
git diff --check
cargo test -p signinum-core --test repo_integrity public_docs_describe_facade_auto_and_cuda_runtime_surface_scope
cargo test -p signinum-j2k --test adaptive_route
cargo test -p signinum --test bench_harness
```

Run the Metal and CUDA commands from Task 7 on matching hardware before claiming gate decisions.
