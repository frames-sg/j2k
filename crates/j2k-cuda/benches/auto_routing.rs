// SPDX-License-Identifier: MIT OR Apache-2.0

#[path = "auto_routing/decode.rs"]
mod decode;
#[path = "auto_routing/encode.rs"]
mod encode;
#[path = "auto_routing/runner.rs"]
mod runner;

fn main() {
    runner::run();
}

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
    let (mismatch_count, max_delta) = cpu.iter().zip(hybrid).fold(
        (0_usize, 0_u8),
        |(mismatch_count, max_delta), (&cpu, &hybrid)| {
            (
                mismatch_count + usize::from(cpu != hybrid),
                max_delta.max(cpu.abs_diff(hybrid)),
            )
        },
    );
    panic!(
        "CUDA {} output differs for {case_id}: cpu_len={}, hybrid_len={}, mismatch_count={mismatch_count}, max_delta={max_delta}, first_difference={first_difference:?}",
        j2k_test_support::auto_routing_operation_label(operation),
        cpu.len(),
        hybrid.len(),
    );
}
