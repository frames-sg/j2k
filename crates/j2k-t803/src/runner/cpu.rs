// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{fs, path::Path, path::PathBuf, process::Command};

use crate::{PlatformIdentity, T803Suite};

use super::{cases, encoder, execute};

const PART1_CLAIM: &str =
    "Profile-1 Cclass-1; Profile-1 Cclass-1HF; Annex G JP2 reader (candidate evidence)";
const PART15_CLAIM: &str = "HTJ2K DS1-HM Cclass-1h, MMAGB 15, including DS1-HT, DS0-HM, and DS0-HT subset evidence; HTJ2K Cclass-1HFh, MMAGB 20; Annex G JPH reader at MMAGB 15 (candidate evidence)";
const ALL_CLAIM: &str = "Profile-1 Cclass-1; Profile-1 Cclass-1HF; Annex G JP2 reader; HTJ2K DS1-HM Cclass-1h, MMAGB 15, including DS1-HT, DS0-HM, and DS0-HT subset evidence; HTJ2K Cclass-1HFh, MMAGB 20; Annex G JPH reader at MMAGB 15 (candidate evidence)";

pub(super) fn run(
    cache_dir: &Path,
    output_dir: Option<PathBuf>,
    development: bool,
    suite: T803Suite,
) -> Result<(), String> {
    let features = Vec::from(["cpu".to_string()]);
    execute::run(
        cache_dir,
        output_dir,
        development,
        suite,
        execute::IutConfig {
            name: "j2k",
            claim: match suite {
                T803Suite::Part1 => PART1_CLAIM,
                T803Suite::Part15 => PART15_CLAIM,
                T803Suite::All => ALL_CLAIM,
            },
            report_stem: "cpu",
            features,
            platform: cpu_platform(),
        },
        encoder::run_cpu,
        |input, reduction_levels| cases::decode_cpu(&input, reduction_levels),
    )
}

fn cpu_platform() -> PlatformIdentity {
    PlatformIdentity {
        os: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
        hardware: cpu_hardware(),
        driver: "not-applicable".to_string(),
    }
}

fn cpu_hardware() -> String {
    if cfg!(target_os = "macos") {
        for key in ["machdep.cpu.brand_string", "hw.model"] {
            if let Ok(output) = Command::new("sysctl").args(["-n", key]).output() {
                if output.status.success() {
                    let value = String::from_utf8_lossy(&output.stdout).trim().to_string();
                    if !value.is_empty() {
                        return value;
                    }
                }
            }
        }
    }
    if cfg!(target_os = "linux") {
        if let Ok(cpuinfo) = fs::read_to_string("/proc/cpuinfo") {
            if let Some(value) = cpuinfo.lines().find_map(|line| {
                line.split_once(':')
                    .filter(|(key, _)| matches!(key.trim(), "model name" | "Hardware"))
                    .map(|(_, value)| value.trim())
            }) {
                if !value.is_empty() {
                    return value.to_string();
                }
            }
        }
    }
    std::env::var("PROCESSOR_IDENTIFIER").unwrap_or_else(|_| "unknown-cpu".to_string())
}
