// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use super::{existing_ran_step, existing_steps};
use crate::adoption_benchmark::{
    options::AdoptionBenchmarkOptions,
    summary::{AdoptionStep, StepStatus},
};

struct TestDirectory(PathBuf);

static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(0);

impl TestDirectory {
    fn new(name: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "j2k-adoption-existing-{name}-{}-{}",
            std::process::id(),
            NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).expect("create test directory");
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).expect("remove test directory");
    }
}

fn options(out_dir: PathBuf, cuda: bool, metal: bool) -> AdoptionBenchmarkOptions {
    AdoptionBenchmarkOptions {
        out_dir,
        input_dirs: None,
        manifest: None,
        encode_input_dirs: None,
        encode_manifest: None,
        cuda_decode_batch_sizes: None,
        include_generated: true,
        quick: false,
        cuda,
        metal,
        openjph: false,
        kakadu: false,
        require_cuda: false,
        require_metal: false,
        require_openjph: false,
        require_kakadu: false,
        finalize_existing: true,
    }
}

fn write_artifacts(out_dir: &Path, names: &[&str]) {
    for name in names {
        fs::write(out_dir.join(format!("{name}.out")), "completed\n")
            .expect("write stdout artifact");
        fs::write(out_dir.join(format!("{name}.err")), "").expect("write stderr artifact");
    }
}

fn step<'a>(steps: &'a [AdoptionStep], name: &str) -> &'a AdoptionStep {
    steps
        .iter()
        .find(|candidate| candidate.name == name)
        .expect("named adoption step")
}

#[test]
fn existing_step_requires_nonempty_stdout_and_a_regular_stderr_file() {
    let directory = TestDirectory::new("artifact-contracts");

    let missing_stdout = existing_ran_step("cpu-fixture-compare", None, directory.path())
        .expect_err("missing stdout must fail closed");
    assert!(missing_stdout.contains("requires completed cpu-fixture-compare stdout"));

    fs::write(directory.path().join("cpu-fixture-compare.out"), "").expect("write empty stdout");
    let empty_stdout = existing_ran_step("cpu-fixture-compare", None, directory.path())
        .expect_err("empty stdout must fail closed");
    assert!(empty_stdout.contains("found empty cpu-fixture-compare stdout"));

    fs::write(
        directory.path().join("cpu-fixture-compare.out"),
        "completed\n",
    )
    .expect("write completed stdout");
    let missing_stderr = existing_ran_step("cpu-fixture-compare", None, directory.path())
        .expect_err("missing stderr must fail closed");
    assert!(missing_stderr.contains("requires cpu-fixture-compare stderr"));

    fs::create_dir(directory.path().join("cpu-fixture-compare.err"))
        .expect("create invalid stderr directory");
    let stderr_directory = existing_ran_step("cpu-fixture-compare", None, directory.path())
        .expect_err("a directory is not a completed stderr artifact");
    assert!(stderr_directory.contains("requires cpu-fixture-compare stderr"));
}

#[test]
fn existing_step_preserves_artifact_and_criterion_paths() {
    let directory = TestDirectory::new("successful-step");
    write_artifacts(directory.path(), &["cpu-public-api-encode"]);
    let target = directory.path().join("cargo-target/cpu-public-api-encode");

    let step = existing_ran_step("cpu-public-api-encode", Some(&target), directory.path())
        .expect("valid existing step");

    assert_eq!(
        step.command,
        "existing artifact reused by --finalize-existing"
    );
    assert_eq!(
        step.stdout,
        directory.path().join("cpu-public-api-encode.out")
    );
    assert_eq!(
        step.stderr,
        directory.path().join("cpu-public-api-encode.err")
    );
    assert_eq!(step.criterion_root, Some(target.join("criterion")));
    assert!(matches!(step.status, StepStatus::Ran));
}

#[test]
fn existing_steps_marks_unrequested_accelerators_skipped() {
    let directory = TestDirectory::new("cpu-only");
    write_artifacts(
        directory.path(),
        &[
            "cpu-fixture-compare",
            "cpu-encode-compare",
            "cpu-public-api-encode",
            "cpu-public-api-decode",
        ],
    );
    let options = options(directory.path().to_path_buf(), false, false);

    let steps = existing_steps(&options).expect("CPU-only existing steps");

    assert_eq!(steps.len(), 9);
    for name in [
        "cuda-htj2k-decode",
        "cuda-htj2k-encode",
        "metal-decode-benchmark",
        "metal-encode-auto-routing",
        "metal-transcode-benchmark",
    ] {
        let StepStatus::Skipped { reason } = &step(&steps, name).status else {
            panic!("{name} must be skipped")
        };
        assert!(reason.contains("not requested"));
    }
}

#[test]
fn existing_steps_reuses_requested_accelerator_artifacts() {
    let directory = TestDirectory::new("accelerators");
    let names = [
        "cpu-fixture-compare",
        "cpu-encode-compare",
        "cpu-public-api-encode",
        "cpu-public-api-decode",
        "cuda-htj2k-decode",
        "cuda-htj2k-encode",
        "metal-decode-benchmark",
        "metal-encode-auto-routing",
        "metal-transcode-benchmark",
    ];
    write_artifacts(directory.path(), &names);
    let options = options(directory.path().to_path_buf(), true, true);

    let steps = existing_steps(&options).expect("accelerator existing steps");

    assert_eq!(steps.len(), names.len());
    assert!(steps
        .iter()
        .all(|candidate| matches!(candidate.status, StepStatus::Ran)));
    assert!(step(&steps, "cuda-htj2k-decode")
        .criterion_root
        .as_ref()
        .is_some_and(|path| path.ends_with("cuda-htj2k-decode/criterion")));
    assert!(step(&steps, "metal-transcode-benchmark")
        .criterion_root
        .as_ref()
        .is_some_and(|path| path.ends_with("metal-transcode-benchmark/criterion")));
    assert!(step(&steps, "metal-decode-benchmark")
        .criterion_root
        .is_none());
}

#[test]
fn existing_steps_propagates_the_exact_missing_artifact() {
    let names = [
        "cpu-fixture-compare",
        "cpu-encode-compare",
        "cpu-public-api-encode",
        "cpu-public-api-decode",
        "cuda-htj2k-decode",
        "cuda-htj2k-encode",
        "metal-decode-benchmark",
        "metal-encode-auto-routing",
        "metal-transcode-benchmark",
    ];

    for missing in names {
        let directory = TestDirectory::new(missing);
        let present = names
            .iter()
            .copied()
            .filter(|name| *name != missing)
            .collect::<Vec<_>>();
        write_artifacts(directory.path(), &present);
        let options = options(directory.path().to_path_buf(), true, true);

        let error = existing_steps(&options).expect_err("missing artifact must fail closed");

        assert!(error.contains(missing), "missing {missing}: {error}");
    }
}
