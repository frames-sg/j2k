// SPDX-License-Identifier: MIT OR Apache-2.0

pub(super) fn assert_retention_precedes_launch(source: &str, signature: &str, launch_marker: &str) {
    let function = source
        .split_once(signature)
        .unwrap_or_else(|| panic!("missing queued launcher `{signature}`"))
        .1;
    let retention = function
        .find("let mut queued_resources = host_budget.try_vec_with_capacity(1)?;")
        .unwrap_or_else(|| panic!("`{signature}` must allocate queued retention"));
    let push = function
        .find("queued_resources.push(jobs_buffer);")
        .unwrap_or_else(|| panic!("`{signature}` must populate queued retention"));
    let launch = function
        .find(launch_marker)
        .unwrap_or_else(|| panic!("`{signature}` is missing launch marker `{launch_marker}`"));
    assert!(
        retention < push && push < launch,
        "`{signature}` must allocate and populate retention before CUDA launch"
    );
}
