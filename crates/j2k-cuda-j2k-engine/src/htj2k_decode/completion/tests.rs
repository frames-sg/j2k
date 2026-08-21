// SPDX-License-Identifier: MIT OR Apache-2.0

#[test]
fn ht_status_timing_excludes_pool_release() {
    let source = include_str!("../completion.rs");
    let status_copy = source
        .find("status_buffer.copy_to_host")
        .expect("HT status copy");
    let status_timing = source
        .find("let status_d2h_us")
        .expect("HT status timing result");
    let pool_release = source
        .find("let release_result = pending_pool_reuse")
        .expect("HT pool release");
    assert!(status_copy < status_timing && status_timing < pool_release);
}
