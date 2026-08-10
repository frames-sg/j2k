// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(target_os = "macos")]
#[path = "auto_routing/decode.rs"]
mod decode;
#[cfg(target_os = "macos")]
#[path = "auto_routing/encode.rs"]
mod encode;
#[cfg(target_os = "macos")]
#[path = "auto_routing/runner.rs"]
mod runner;

#[cfg(not(target_os = "macos"))]
fn main() {
    assert!(
        std::env::var_os("J2K_REQUIRE_METAL_BENCH").is_none(),
        "J2K Metal Auto-routing benchmark requires macOS"
    );
    eprintln!("J2K Metal Auto-routing benchmark skipped outside macOS");
}

#[cfg(target_os = "macos")]
fn main() {
    runner::run();
}

#[cfg(target_os = "macos")]
fn assert_output_parity(
    case_id: &str,
    operation: j2k_test_support::AutoRoutingOperation,
    cpu: &[u8],
    hybrid: &[u8],
) {
    if cpu == hybrid {
        return;
    }
    let first_difference = cpu
        .iter()
        .zip(hybrid)
        .position(|(cpu, hybrid)| cpu != hybrid)
        .map(|index| (index, cpu[index], hybrid[index]));
    panic!(
        "Metal {} output differs for {case_id}: cpu_len={}, hybrid_len={}, first_difference={first_difference:?}",
        j2k_test_support::auto_routing_operation_label(operation),
        cpu.len(),
        hybrid.len(),
    );
}
