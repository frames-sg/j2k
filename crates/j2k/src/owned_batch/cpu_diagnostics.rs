// SPDX-License-Identifier: MIT OR Apache-2.0

//! Observable reuse and retained-memory counters for CPU batch sessions.

/// Aggregate reuse counters for native workspaces retained by a CPU batch session.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CpuBatchWorkspaceStats {
    pub(super) preparation_calls: u64,
    pub(super) preparation_worker_reuses: u64,
    pub(super) decode_calls: u64,
    pub(super) prepared_plan_decode_calls: u64,
    pub(super) component_owner_reuses: u64,
    pub(super) tier1_owner_reuses: u64,
    pub(super) idwt_owner_reuses: u64,
    pub(super) scratch_capacity_retries: u64,
    pub(super) retained_component_bytes: usize,
    pub(super) retained_tier1_bytes: usize,
    pub(super) retained_idwt_bytes: usize,
    pub(super) retained_prepared_plan_classic_workspace_bytes: usize,
    pub(super) retained_prepared_plan_ht_workspace_bytes: usize,
    pub(super) flattened_group_plans: u64,
    pub(super) flattened_payload_jobs: u64,
    pub(super) flattened_cleanup_jobs: u64,
    pub(super) flattened_sigprop_jobs: u64,
    pub(super) flattened_magref_jobs: u64,
    pub(super) flattened_classic_jobs: u64,
    pub(super) entropy_job_dispatches: u64,
    pub(super) cross_image_entropy_windows: u64,
    pub(super) compressed_arena_reuses: u64,
    pub(super) retained_compressed_arena_bytes: usize,
    pub(super) output_group_allocations: u64,
    pub(super) output_compaction_copied_samples: u64,
}

impl CpuBatchWorkspaceStats {
    /// Number of image plan preparations performed by retained session workers.
    #[must_use]
    pub const fn preparation_calls(self) -> u64 {
        self.preparation_calls
    }

    /// Number of image preparations performed by a worker that had already
    /// completed at least one earlier preparation.
    #[must_use]
    pub const fn preparation_worker_reuses(self) -> u64 {
        self.preparation_worker_reuses
    }

    /// Number of native image decode calls across retained workers.
    #[must_use]
    pub const fn decode_calls(self) -> u64 {
        self.decode_calls
    }

    /// Number of image decodes executed from retained codec geometry without
    /// constructing or reparsing a native image decoder.
    #[must_use]
    pub const fn prepared_plan_decode_calls(self) -> u64 {
        self.prepared_plan_decode_calls
    }

    /// Number of image decodes that reused an existing component owner.
    #[must_use]
    pub const fn component_owner_reuses(self) -> u64 {
        self.component_owner_reuses
    }

    /// Number of image decodes that began with retained Tier-1 owners.
    #[must_use]
    pub const fn tier1_owner_reuses(self) -> u64 {
        self.tier1_owner_reuses
    }

    /// Number of image decodes that began with retained IDWT owners.
    #[must_use]
    pub const fn idwt_owner_reuses(self) -> u64 {
        self.idwt_owner_reuses
    }

    /// Number of retained-scratch evictions followed by a fresh retry.
    #[must_use]
    pub const fn scratch_capacity_retries(self) -> u64 {
        self.scratch_capacity_retries
    }

    /// Total retained decoded-component capacity across workers.
    #[must_use]
    pub const fn retained_component_bytes(self) -> usize {
        self.retained_component_bytes
    }

    /// Total retained classic and HT Tier-1 capacity across workers.
    #[must_use]
    pub const fn retained_tier1_bytes(self) -> usize {
        self.retained_tier1_bytes
    }

    /// Total retained floating-point and exact-integer IDWT capacity across workers.
    #[must_use]
    pub const fn retained_idwt_bytes(self) -> usize {
        self.retained_idwt_bytes
    }

    /// Total HT code-block workspace retained by parse-free prepared-plan workers.
    #[must_use]
    pub const fn retained_prepared_plan_ht_workspace_bytes(self) -> usize {
        self.retained_prepared_plan_ht_workspace_bytes
    }

    /// Total classic code-block workspace retained by parse-free prepared-plan workers.
    #[must_use]
    pub const fn retained_prepared_plan_classic_workspace_bytes(self) -> usize {
        self.retained_prepared_plan_classic_workspace_bytes
    }

    /// Number of homogeneous prepared groups flattened into one compressed arena.
    #[must_use]
    pub const fn flattened_group_plans(self) -> u64 {
        self.flattened_group_plans
    }

    /// Number of source-indexed code-block payload jobs flattened across images.
    #[must_use]
    pub const fn flattened_payload_jobs(self) -> u64 {
        self.flattened_payload_jobs
    }

    /// Number of flattened one-pass HT cleanup jobs.
    #[must_use]
    pub const fn flattened_cleanup_jobs(self) -> u64 {
        self.flattened_cleanup_jobs
    }

    /// Number of flattened two-pass HT `SigProp` jobs.
    #[must_use]
    pub const fn flattened_sigprop_jobs(self) -> u64 {
        self.flattened_sigprop_jobs
    }

    /// Number of flattened three-or-more-pass HT `MagRef` jobs.
    #[must_use]
    pub const fn flattened_magref_jobs(self) -> u64 {
        self.flattened_magref_jobs
    }

    /// Number of flattened classic JPEG 2000 code-block jobs.
    #[must_use]
    pub const fn flattened_classic_jobs(self) -> u64 {
        self.flattened_classic_jobs
    }

    /// Number of flattened code blocks submitted to retained CPU workers.
    #[must_use]
    pub const fn entropy_job_dispatches(self) -> u64 {
        self.entropy_job_dispatches
    }

    /// Number of bounded entropy windows containing more than one image.
    #[must_use]
    pub const fn cross_image_entropy_windows(self) -> u64 {
        self.cross_image_entropy_windows
    }

    /// Number of group plans that reused the existing compressed arena capacity.
    #[must_use]
    pub const fn compressed_arena_reuses(self) -> u64 {
        self.compressed_arena_reuses
    }

    /// Compressed payload arena capacity retained by the session.
    #[must_use]
    pub const fn retained_compressed_arena_bytes(self) -> usize {
        self.retained_compressed_arena_bytes
    }

    /// Number of successful final typed group allocations.
    #[must_use]
    pub const fn output_group_allocations(self) -> u64 {
        self.output_group_allocations
    }

    /// Samples moved only to compact successful images after indexed decode failures.
    #[must_use]
    pub const fn output_compaction_copied_samples(self) -> u64 {
        self.output_compaction_copied_samples
    }

    /// Total retained lifetime-free decode scratch across workers.
    #[must_use]
    pub const fn retained_scratch_bytes(self) -> usize {
        self.retained_tier1_bytes
            .saturating_add(self.retained_idwt_bytes)
    }
}
