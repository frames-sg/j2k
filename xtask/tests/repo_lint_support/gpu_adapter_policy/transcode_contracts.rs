// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use crate::repo_lint_support::{assert_pattern_checks, repo_root, PatternCheck};

#[test]
fn transcode_gpu_auto_threshold_policy_is_documented() {
    let root = repo_root();
    let cuda = fs::read_to_string(root.join("crates/j2k-transcode-cuda/src/lib.rs"))
        .expect("read CUDA transcode adapter");
    let metal_root = fs::read_to_string(root.join("crates/j2k-transcode-metal/src/lib.rs"))
        .expect("read Metal transcode adapter");
    let metal_accelerator =
        fs::read_to_string(root.join("crates/j2k-transcode-metal/src/accelerator.rs"))
            .expect("read Metal transcode accelerator");
    let metal = format!("{metal_root}\n{metal_accelerator}");
    let cuda_readme = fs::read_to_string(root.join("crates/j2k-transcode-cuda/README.md"))
        .expect("read CUDA transcode README");
    let metal_readme = fs::read_to_string(root.join("crates/j2k-transcode-metal/README.md"))
        .expect("read Metal transcode README");

    let shared_auto_batch_thresholds = [
        "const DEFAULT_AUTO_REVERSIBLE_BATCH_MIN_JOBS: usize = 32;",
        "const DEFAULT_AUTO_REVERSIBLE_BATCH_MIN_SAMPLES: usize = 224 * 224 * 32;",
        "const DEFAULT_AUTO_DWT97_BATCH_MIN_JOBS: usize = 32;",
        "const DEFAULT_AUTO_DWT97_BATCH_MIN_SAMPLES: usize = 224 * 224 * 32;",
    ];
    assert_pattern_checks(&[
        PatternCheck::new("CUDA transcode Auto batch thresholds", &cuda)
            .required(&shared_auto_batch_thresholds),
        PatternCheck::new("Metal transcode Auto batch thresholds", &metal)
            .required(&shared_auto_batch_thresholds),
        PatternCheck::new("CUDA transcode Auto threshold rationale", &cuda)
            .required(&["Batch thresholds below intentionally match Metal"]),
        PatternCheck::new("CUDA transcode README threshold rationale", &cuda_readme).required(&[
            "shared `224 * 224` component-sample floor",
            "defaults are routing policy, not a speedup promise",
        ]),
        PatternCheck::new("Metal transcode Auto threshold policy", &metal).required(&[
            "single-job Auto dispatch is disabled",
            "const DEFAULT_AUTO_DWT97_MIN_SAMPLES: usize = usize::MAX;",
            "const DEFAULT_AUTO_REVERSIBLE_MIN_SAMPLES: usize = usize::MAX;",
            "const MAX_AUTO_DWT97_STAGED_BATCH_AXIS: usize = 1024;",
        ]),
        PatternCheck::new("Metal transcode README staged-axis policy", &metal_readme).required(&[
            "either tile axis exceeds 1024 samples",
            "defaults are routing policy, not a speedup promise",
        ]),
    ]);
}

#[test]
fn transcode_stage_counters_are_shared_between_gpu_adapters() {
    let root = repo_root();
    let accelerator =
        fs::read_to_string(root.join("crates/j2k-transcode/src/accelerator_contracts.rs"))
            .expect("read transcode accelerator contracts");
    let cuda = fs::read_to_string(root.join("crates/j2k-transcode-cuda/src/lib.rs"))
        .expect("read CUDA transcode adapter");
    let metal_root = fs::read_to_string(root.join("crates/j2k-transcode-metal/src/lib.rs"))
        .expect("read Metal transcode adapter");
    let metal_accelerator =
        fs::read_to_string(root.join("crates/j2k-transcode-metal/src/accelerator.rs"))
            .expect("read Metal transcode accelerator");
    let metal_dispatch =
        fs::read_to_string(root.join("crates/j2k-transcode-metal/src/accelerator/dispatch.rs"))
            .expect("read Metal transcode dispatch implementation");
    let metal = format!("{metal_root}\n{metal_accelerator}\n{metal_dispatch}");

    assert_pattern_checks(&[PatternCheck::new(
        "j2k-transcode accelerator shared counters",
        &accelerator,
    )
    .required(&[
        "pub struct DctToWaveletStageCounters",
        "pub enum DctToWaveletStageCounterEvent",
        "pub enum TranscodeStageDispatchMode",
        "pub const fn unavailable<T>",
        "pub fn recover<T, E>",
        "pub fn record(&mut self, event: DctToWaveletStageCounterEvent, count: usize)",
        "DctToWaveletStageCounterEvent::Htj2k97CodeblockBatchAttempt",
        "DctToWaveletStageCounterEvent::Htj2k97CodeblockBatchDispatch",
    ])]);

    for (label, source) in [("CUDA", cuda.as_str()), ("Metal", metal.as_str())] {
        let check_name = format!("{label} transcode shared counters and dispatch policy");
        assert_pattern_checks(&[PatternCheck::new(&check_name, source)
            .required(&[
                "DctToWaveletStageCounterEvent as CounterEvent",
                "counters: DctToWaveletStageCounters",
                "self.counters.record(CounterEvent::",
                "mode: TranscodeStageDispatchMode",
                "self.mode.unavailable()",
            ])
            .forbidden(&[
                "reversible_dwt53_attempts: usize",
                "dwt53_attempts: usize",
                "dwt97_attempts: usize",
                "htj2k97_codeblock_batch_attempts: usize",
                "enum CudaDispatchMode",
                "enum MetalDispatchMode",
                "fn unavailable<T>(&self)",
                "MetalTranscodeError::MetalUnavailable | MetalTranscodeError::UnsupportedJob(_)",
            ])]);
    }

    assert_pattern_checks(&[
        PatternCheck::new("CUDA transcode shared recovery policy", &cuda)
            .required(&[".recover(error, CudaTranscodeError::is_recoverable)"]),
        PatternCheck::new("Metal transcode shared recovery policy", &metal)
            .required(&[".recover(error, MetalTranscodeError::is_recoverable)"]),
    ]);
}
