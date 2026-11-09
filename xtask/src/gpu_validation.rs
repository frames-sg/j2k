// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared command-line policy for fail-closed GPU validation modes.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ValidationMode {
    Quick,
    Full,
}

impl ValidationMode {
    pub(crate) fn parse(mut args: impl Iterator<Item = String>) -> Result<Self, String> {
        let Some(argument) = args.next() else {
            return Ok(Self::Full);
        };
        if argument != "--mode" {
            return Err(format!(
                "unknown GPU validation argument `{argument}`; expected --mode quick|full"
            ));
        }
        let value = args
            .next()
            .ok_or_else(|| "--mode requires quick or full".to_string())?;
        if let Some(extra) = args.next() {
            return Err(format!(
                "unexpected GPU validation argument `{extra}`; expected only --mode quick|full"
            ));
        }
        match value.as_str() {
            "quick" => Ok(Self::Quick),
            "full" => Ok(Self::Full),
            _ => Err(format!(
                "unknown GPU validation mode `{value}`; expected quick or full"
            )),
        }
    }

    pub(crate) const fn cargo_profile_args(self) -> &'static [&'static str] {
        match self {
            Self::Quick => &["--profile", "gpu-quick"],
            Self::Full => &["--release"],
        }
    }
}

/// Names `j2k-test-support`'s per-skip log for the GPU-gated tests a hosted
/// command runs, then reports how many skipped: on stdout and, under GitHub
/// Actions, in the job summary. It never changes a command's outcome.
pub(crate) struct GpuSkipLog {
    path: std::path::PathBuf,
}

/// The variable `j2k_test_support::GPU_TEST_SKIP_LOG_ENV` reads.
pub(crate) const GPU_TEST_SKIP_LOG_ENV: &str = "J2K_GPU_SKIP_LOG";

impl GpuSkipLog {
    pub(crate) fn new(label: &str) -> Result<Self, String> {
        let directory = std::env::temp_dir().join("j2k-gpu-test-skips");
        std::fs::create_dir_all(&directory)
            .map_err(|error| format!("failed to create {}: {error}", directory.display()))?;
        let path = directory.join(format!("{label}-{}.log", std::process::id()));
        match std::fs::remove_file(&path) {
            Err(error) if error.kind() != std::io::ErrorKind::NotFound => {
                return Err(format!("failed to reset {}: {error}", path.display()));
            }
            _ => {}
        }
        Ok(Self { path })
    }

    pub(crate) fn path(&self) -> &str {
        self.path.to_str().unwrap_or_default()
    }

    /// Print and summarize the skips recorded so far, then forget them.
    pub(crate) fn report(&self, label: &str) -> Result<(), String> {
        let log = match std::fs::read_to_string(&self.path) {
            Ok(log) => log,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => String::new(),
            Err(error) => return Err(format!("failed to read {}: {error}", self.path.display())),
        };
        let line = summarize_gpu_skips(label, &log);
        println!("{line}");
        // Only real skips reach the job summary; hermetic command-plan tests
        // record none.
        let summary = std::env::var_os("GITHUB_STEP_SUMMARY").filter(|_| !log.is_empty());
        if let Some(summary) = summary {
            use std::io::Write as _;
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&summary)
                .and_then(|mut file| writeln!(file, "{line}"))
                .map_err(|error| format!("failed to append the GPU skip summary: {error}"))?;
        }
        let _ = std::fs::remove_file(&self.path);
        Ok(())
    }
}

/// One line: how many GPU-gated tests skipped, per require gate.
pub(crate) fn summarize_gpu_skips(label: &str, log: &str) -> String {
    let mut per_gate = std::collections::BTreeMap::<&str, usize>::new();
    for gate in log.lines().filter_map(|line| line.split('\t').next()) {
        *per_gate.entry(gate).or_default() += 1;
    }
    let total: usize = per_gate.values().sum();
    if total == 0 {
        return format!("{label}: no GPU-gated tests skipped");
    }
    let gates = per_gate
        .iter()
        .map(|(gate, count)| format!("{count} without {gate}"))
        .collect::<Vec<_>>()
        .join(", ");
    format!("{label}: {total} GPU-gated tests skipped ({gates}); set the gate to 1 on a GPU host to run them")
}

#[cfg(test)]
mod skip_summary_tests {
    use super::summarize_gpu_skips;

    #[test]
    fn skip_summary_counts_skips_per_gate() {
        let log = "J2K_REQUIRE_METAL_RUNTIME\ta\nJ2K_REQUIRE_CUDA_RUNTIME\tb\nJ2K_REQUIRE_METAL_RUNTIME\tc\n";
        assert_eq!(
            summarize_gpu_skips("xtask test", log),
            "xtask test: 3 GPU-gated tests skipped (1 without J2K_REQUIRE_CUDA_RUNTIME, \
             2 without J2K_REQUIRE_METAL_RUNTIME); set the gate to 1 on a GPU host to run them"
        );
        assert_eq!(
            summarize_gpu_skips("xtask test", ""),
            "xtask test: no GPU-gated tests skipped"
        );
    }
}
