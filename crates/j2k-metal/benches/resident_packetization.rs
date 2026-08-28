// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(target_os = "macos")]
#[path = "resident_packetization/batch_compare.rs"]
mod batch_compare;
#[cfg(target_os = "macos")]
#[path = "resident_packetization/packetization.rs"]
mod packetization;
#[cfg(target_os = "macos")]
#[path = "resident_packetization/support.rs"]
mod support;

#[cfg(not(target_os = "macos"))]
fn main() {
    assert!(
        std::env::var_os("J2K_REQUIRE_METAL_BENCH").is_none(),
        "J2K Metal resident packetization benchmark requires macOS"
    );
    eprintln!("J2K Metal resident packetization benchmark skipped outside macOS");
}

#[cfg(target_os = "macos")]
fn main() {
    use std::time::Duration;

    let mut criterion = criterion::Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3))
        .configure_from_args();
    packetization::bench(&mut criterion);
    batch_compare::bench(&mut criterion);
    criterion.final_summary();
}
