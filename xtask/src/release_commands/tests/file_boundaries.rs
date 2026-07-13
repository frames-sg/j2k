// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{
    validate_publish_script, validate_publish_workflow, validate_release_docs,
    validate_release_metadata, ReleaseIntegrityMode,
};
use crate::test_command::RecordingProgram;

const CASE_ENV: &str = "XTASK_TEST_RELEASE_FILE_BOUNDARY_CASE";

#[test]
fn missing_release_contract_files_fail_with_path_context() {
    if let Ok(case) = std::env::var(CASE_ENV) {
        let error = match case.as_str() {
            "workflow" => validate_publish_workflow(&mut Vec::new()),
            "script" => validate_publish_script(&mut Vec::new()),
            "docs" => validate_release_docs(&mut Vec::new()),
            "changelog" => validate_release_metadata(
                "0.7.0",
                ReleaseIntegrityMode::PreCandidate,
                &mut Vec::new(),
            ),
            "provenance" => {
                validate_release_metadata("0.7.0", ReleaseIntegrityMode::Publish, &mut Vec::new())
            }
            other => panic!("unknown file boundary case {other}"),
        }
        .expect_err("missing release contract file must reject");
        let expected = match case.as_str() {
            "workflow" => ".github/workflows/publish.yml",
            "script" => "scripts/publish-crate.sh",
            "docs" => "docs/release.md",
            "changelog" => "CHANGELOG.md",
            "provenance" => "PATCH_PROVENANCE.md",
            _ => unreachable!("validated case"),
        };
        assert!(error.contains(expected), "unexpected error: {error}");
        return;
    }

    for case in ["workflow", "script", "docs", "changelog", "provenance"] {
        let fixture = RecordingProgram::new("release-file-boundary", "");
        let release_root = fixture
            .program()
            .parent()
            .expect("recording program parent")
            .join("release-root");
        std::fs::create_dir_all(&release_root).expect("create release boundary root");
        if case == "provenance" {
            std::fs::write(
                release_root.join("CHANGELOG.md"),
                "# Changelog\n\n## [0.7.0] - 2026-07-12\n",
            )
            .expect("write final changelog fixture");
        }
        let output = std::process::Command::new(std::env::current_exe().expect("test binary"))
            .arg("release_commands::tests::file_boundaries::missing_release_contract_files_fail_with_path_context")
            .arg("--exact")
            .arg("--nocapture")
            .current_dir(&release_root)
            .env(CASE_ENV, case)
            .output()
            .expect("run release file boundary child");
        assert!(
            output.status.success(),
            "boundary child failed for {case}:\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    }
}
