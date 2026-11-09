// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared GPU test gates.

use std::io::Write as _;

/// Marker emitted when an optional GPU test skips because its require gate is unset.
pub const GPU_TEST_SKIP_MARKER: &str = "J2K_GPU_TEST_SKIPPED";

/// Environment variable naming a file that collects one `gate<TAB>test` line
/// per skipped GPU test, so a runner can report how many did not run.
pub const GPU_TEST_SKIP_LOG_ENV: &str = "J2K_GPU_SKIP_LOG";

/// Returns true when a GPU-gated test should run.
///
/// If `required` is false, the test should return after calling this helper.
/// The skip message is intentionally stable so CI can count skipped GPU tests.
pub fn gpu_test_gate(required: bool, gate: &'static str, context: &str) -> bool {
    if required {
        return true;
    }
    report_gpu_test_skip(gate, context, None);
    false
}

/// Handles a missing GPU device/runtime for a test that already attempted setup.
///
/// Returns true when the caller should skip. Panics when the matching require
/// gate is set, so self-hosted validation fails instead of skipping.
///
/// # Panics
///
/// Panics when `required` is true, because a required GPU runtime was missing.
pub fn gpu_device_unavailable_is_skip(required: bool, gate: &'static str, context: &str) -> bool {
    assert!(
        !required,
        "{gate} is set but GPU device/runtime is unavailable for {context}"
    );
    report_gpu_test_skip(gate, context, Some("device-unavailable"));
    true
}

/// Report one skipped GPU test three ways: the stable machine marker the
/// release lanes reject, a `SKIPPED:` line that stays visible for passing
/// tests, and a line in the [`GPU_TEST_SKIP_LOG_ENV`] file when one is named.
pub(crate) fn report_gpu_test_skip(gate: &str, context: &str, reason: Option<&str>) {
    let reason_field = reason.map_or_else(String::new, |reason| format!(" reason={reason}"));
    eprintln!("{GPU_TEST_SKIP_MARKER} gate={gate} context={context}{reason_field}");
    // libtest names each test's thread after the test.
    let thread = std::thread::current();
    let test = thread
        .name()
        .filter(|name| *name != "main")
        .unwrap_or(context);
    // libtest captures `eprintln!` output of passing tests, which is what made
    // skips silent; a direct stderr write is not captured. A failed write to
    // stderr has nowhere else to be reported.
    let _ = writeln!(
        std::io::stderr().lock(),
        "{}",
        skip_line(test, gate, reason)
    );
    if let Some(path) = std::env::var_os(GPU_TEST_SKIP_LOG_ENV) {
        let appended = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .and_then(|mut log| writeln!(log, "{gate}\t{test}"));
        if let Err(error) = appended {
            let _ = writeln!(
                std::io::stderr().lock(),
                "warning: could not record the GPU test skip in {}: {error}",
                std::path::Path::new(&path).display()
            );
        }
    }
}

/// The human-readable skip line, for example
/// `SKIPPED: decode::tests::rgb (no Metal runtime; set J2K_REQUIRE_METAL_RUNTIME=1 to require it)`.
pub(crate) fn skip_line(test: &str, gate: &str, reason: Option<&str>) -> String {
    let what = gate_requirement(gate);
    let why = match reason {
        Some("device-unavailable") => format!("{what} unavailable"),
        Some("oxide-ptx-not-built") => "cuda-oxide PTX not built".to_string(),
        Some(other) => format!("no {what}: {other}"),
        None => format!("no {what}"),
    };
    format!("SKIPPED: {test} ({why}; set {gate}=1 to require it)")
}

fn gate_requirement(gate: &str) -> &'static str {
    match gate {
        "J2K_REQUIRE_METAL_RUNTIME" => "Metal runtime",
        "J2K_REQUIRE_CUDA_RUNTIME" => "CUDA runtime",
        "J2K_REQUIRE_CUDA_OXIDE_BUILD" => "cuda-oxide build",
        "J2K_REQUIRE_CUDA_JPEG_HARDWARE_DECODE" => "CUDA JPEG hardware decode",
        _ => "GPU runtime",
    }
}

#[cfg(test)]
mod tests {
    use super::{gpu_test_gate, skip_line};

    #[test]
    fn skip_lines_name_the_test_the_missing_backend_and_the_gate() {
        assert_eq!(
            skip_line("decode::rgb", "J2K_REQUIRE_METAL_RUNTIME", None),
            "SKIPPED: decode::rgb (no Metal runtime; set J2K_REQUIRE_METAL_RUNTIME=1 to require it)"
        );
        assert_eq!(
            skip_line(
                "store::gray",
                "J2K_REQUIRE_CUDA_RUNTIME",
                Some("device-unavailable")
            ),
            "SKIPPED: store::gray (CUDA runtime unavailable; set J2K_REQUIRE_CUDA_RUNTIME=1 to require it)"
        );
        assert_eq!(
            skip_line(
                "ptx",
                "J2K_REQUIRE_CUDA_OXIDE_BUILD",
                Some("oxide-ptx-not-built")
            ),
            "SKIPPED: ptx (cuda-oxide PTX not built; set J2K_REQUIRE_CUDA_OXIDE_BUILD=1 to require it)"
        );
    }

    #[test]
    fn a_required_gate_runs_without_reporting_a_skip() {
        assert!(gpu_test_gate(true, "J2K_REQUIRE_METAL_RUNTIME", "context"));
        assert!(!gpu_test_gate(
            false,
            "J2K_REQUIRE_METAL_RUNTIME",
            "context"
        ));
    }
}
