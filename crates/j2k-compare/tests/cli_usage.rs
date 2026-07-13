// SPDX-License-Identifier: MIT OR Apache-2.0

use std::process::{Command, Output};

#[test]
fn encode_compare_help_prints_the_cli_contract() {
    let program = env!("CARGO_BIN_EXE_jp2k_encode_compare");
    let output = Command::new(program)
        .arg("--help")
        .output()
        .expect("run encode comparator help");

    assert_help_output(
        &output,
        &format!(
            "usage: {program} [case-name-filter ...]\n\
             {spaces}{program} --encode-one --input FILE.pnm --output FILE.jp2\n\
             Runs CLI-style lossless classic JPEG 2000 encoder benchmarks.\n",
            spaces = "       "
        ),
    );
}

#[test]
fn fixture_compare_short_help_prints_the_cli_contract() {
    let program = env!("CARGO_BIN_EXE_jp2k_fixture_compare");
    let output = Command::new(program)
        .arg("-h")
        .output()
        .expect("run fixture comparator help");

    assert_help_output(
        &output,
        &format!(
            "usage: {program} [case-name-filter ...]\n\
             Runs J2K/OpenJPEG/Grok decode benchmarks over the same named fixture bytes; set J2K_INCLUDE_OPENJPH=1 for optional OpenJPH HTJ2K CLI rows or J2K_INCLUDE_KAKADU=1 for optional Kakadu CLI rows.\n"
        ),
    );
}

fn assert_help_output(output: &Output, expected_stderr: &str) {
    assert!(
        output.status.success(),
        "help command failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stdout.is_empty(), "help wrote unexpected stdout");
    assert_eq!(String::from_utf8_lossy(&output.stderr), expected_stderr);
}
