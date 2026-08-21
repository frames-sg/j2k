use std::{ffi::OsString, fs, panic::catch_unwind, path::Path};

use super::{
    codec_math_crate_path, compile_project, copy_file_as, fallback_codec_math_crate_path,
    j2k_types_crate_path, skip_project, stage_cuda_oxide_dependency_config, CudaOxideProject,
    CODEC_MATH_MANIFEST_DIR_ENV,
};

struct TestDir(std::path::PathBuf);

impl TestDir {
    fn new(label: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "j2k-cuda-build-support-{label}-{}",
            std::process::id()
        ));
        fs::create_dir_all(&path).expect("create test directory");
        Self(path)
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

struct EnvironmentRestore {
    key: &'static str,
    previous: Option<OsString>,
}

impl EnvironmentRestore {
    fn set(key: &'static str, value: &Path) -> Self {
        let previous = std::env::var_os(key);
        std::env::set_var(key, value);
        Self { key, previous }
    }
}

impl Drop for EnvironmentRestore {
    fn drop(&mut self) {
        if let Some(previous) = self.previous.take() {
            std::env::set_var(self.key, previous);
        } else {
            std::env::remove_var(self.key);
        }
    }
}

#[test]
fn required_cuda_oxide_skip_uses_explicit_error_path() {
    let outcome = catch_unwind(|| {
        skip_project(
            Path::new("unused.ptx"),
            true,
            "test CUDA-Oxide project",
            "required compiler is unavailable",
        )
    });

    let error = outcome
        .expect("required build failure must not panic")
        .expect_err("required build failure must return an error");
    assert_eq!(error.to_string(), "required compiler is unavailable");
}

#[test]
fn optional_cuda_oxide_skip_preserves_placeholder_ptx() {
    let dir = TestDir::new("placeholder");
    let output = dir.0.join("placeholder.ptx");

    assert!(
        !skip_project(&output, false, "test project", "not available")
            .expect("optional skip writes placeholder")
    );
    assert_eq!(
        fs::read(output).expect("read placeholder"),
        b".version 7.0\n.target sm_52\n.address_size 64\n\0"
    );
}

#[test]
fn template_staging_reports_source_context() {
    let dir = TestDir::new("missing-template");
    let source_dir = dir.0.join("source");
    let project_dir = dir.0.join("project");
    fs::create_dir_all(&source_dir).expect("create source directory");

    let error = copy_file_as(
        &source_dir,
        &project_dir,
        Path::new("missing.toml.in"),
        Path::new("Cargo.toml"),
        &dir.0,
    )
    .expect_err("missing template returns an error");
    assert!(error.to_string().contains("missing.toml.in"));
}

#[test]
fn packaged_codec_math_fallback_uses_the_versioned_extraction_directory() {
    let dir = TestDir::new("versioned-codec-math");
    let packaged = dir.0.join("packaged");
    let runtime = packaged.join("j2k-cuda-runtime-0.10.0");
    let codec_math = packaged.join("j2k-codec-math-0.10.0");
    fs::create_dir_all(&runtime).expect("create packaged runtime");
    fs::create_dir_all(&codec_math).expect("create packaged codec math");
    fs::write(
        codec_math.join("Cargo.toml"),
        "[package]\nname='j2k-codec-math'\n",
    )
    .expect("write codec math manifest");

    assert_eq!(
        fallback_codec_math_crate_path(&runtime, "0.10.0").expect("locate versioned codec math"),
        codec_math
    );
}

#[test]
fn packaged_codec_math_fallback_reports_both_missing_sibling_locations() {
    let dir = TestDir::new("missing-codec-math");
    let runtime = dir.0.join("j2k-cuda-runtime-0.10.0");
    fs::create_dir_all(&runtime).expect("create packaged runtime");

    let error = fallback_codec_math_crate_path(&runtime, "0.10.0")
        .expect_err("missing codec-math siblings must be rejected");
    let message = error.to_string();
    assert!(message.contains(dir.0.join("j2k-codec-math").to_string_lossy().as_ref()));
    assert!(message.contains(
        dir.0
            .join("j2k-codec-math-0.10.0")
            .to_string_lossy()
            .as_ref()
    ));
}

#[test]
fn explicit_codec_math_metadata_must_identify_a_crate_manifest() {
    let dir = TestDir::new("invalid-codec-math-metadata");
    let invalid_codec_math = dir.0.join("not-a-codec-math-crate");
    fs::create_dir_all(&invalid_codec_math).expect("create invalid codec-math directory");
    let _restore = EnvironmentRestore::set(CODEC_MATH_MANIFEST_DIR_ENV, &invalid_codec_math);

    let error = codec_math_crate_path(Path::new("unused-manifest"))
        .expect_err("metadata without a crate manifest must be rejected");
    assert!(error.to_string().contains(
        format!("{CODEC_MATH_MANIFEST_DIR_ENV} does not identify a j2k-codec-math crate").as_str()
    ));
    assert!(error
        .to_string()
        .contains(invalid_codec_math.to_string_lossy().as_ref()));
}

#[test]
fn packaged_j2k_types_lookup_reports_both_missing_sibling_locations() {
    let dir = TestDir::new("missing-j2k-types");
    let codec_math = dir.0.join("j2k-codec-math-0.10.0");
    fs::create_dir_all(&codec_math).expect("create packaged codec math");

    let error = j2k_types_crate_path(&codec_math, "0.10.0")
        .expect_err("missing j2k-types siblings must be rejected");
    let message = error.to_string();
    assert!(message.contains(dir.0.join("j2k-types").to_string_lossy().as_ref()));
    assert!(message.contains(dir.0.join("j2k-types-0.10.0").to_string_lossy().as_ref()));
}

#[test]
fn non_linux_cuda_oxide_build_preserves_the_optional_placeholder_contract() {
    let dir = TestDir::new("non-linux-placeholder");
    let output_name = "test-kernel.ptx";
    let project = CudaOxideProject {
        feature_env: "CARGO_FEATURE_UNUSED_TEST_PROJECT",
        source_dir: "unused-source",
        output_name,
        artifact_name: "unused-artifact.ptx",
        display_name: "test CUDA-Oxide project",
        extra_sources: &[],
        built_cfg: "test_cuda_oxide_built",
    };

    assert!(!compile_project(
        Path::new("unused-manifest"),
        &dir.0,
        "aarch64-apple-darwin",
        Path::new("unused-codec-math"),
        project,
        false,
    )
    .expect("optional non-Linux build writes a placeholder"));
    assert_eq!(
        fs::read(dir.0.join(output_name)).expect("read placeholder"),
        b".version 7.0\n.target sm_52\n.address_size 64\n\0"
    );
}

#[test]
fn staged_cuda_oxide_config_patches_the_versioned_j2k_types_package() {
    let dir = TestDir::new("versioned-j2k-types");
    let packaged = dir.0.join("packaged");
    let codec_math = packaged.join("j2k-codec-math-0.10.0");
    let j2k_types = packaged.join("j2k-types-0.10.0");
    let project = dir.0.join("staged-project");
    fs::create_dir_all(&codec_math).expect("create packaged codec math");
    fs::create_dir_all(&j2k_types).expect("create packaged j2k types");
    fs::write(
        j2k_types.join("Cargo.toml"),
        "[package]\nname='j2k-types'\n",
    )
    .expect("write j2k types manifest");

    let resolved = j2k_types_crate_path(&codec_math, "0.10.0").expect("locate versioned j2k types");
    stage_cuda_oxide_dependency_config(&project, &resolved)
        .expect("stage CUDA-Oxide dependency patch");

    assert_eq!(resolved, j2k_types);
    let config =
        fs::read_to_string(project.join(".cargo/config.toml")).expect("read staged Cargo config");
    assert_eq!(
        config,
        format!(
            "[patch.crates-io]\nj2k-types = {{ path = {:?} }}\n",
            j2k_types.to_string_lossy()
        )
    );
}
