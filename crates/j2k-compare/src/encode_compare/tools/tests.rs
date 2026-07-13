// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{
    fs,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use super::{
    command_template, command_version_label, dimensions_label, discover_command,
    extract_version_line, path_lookup, run_encoder_once, samples_label, selected_encoders_label,
    start_with_executable_busy_retry, tool_available, tool_command, tool_version,
    tool_version_available, version_line_by_priority, EXECUTABLE_BUSY_RETRY_DELAYS,
};
use crate::encode_compare::{EncoderKind, EncoderTool, ImageCase};

static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(0);

fn temp_dir(label: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!(
        "j2k-encode-tools-{label}-{}-{}",
        std::process::id(),
        NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&root).expect("create tool test directory");
    root
}

fn executable(path: &Path, source: &str) {
    let staging = path.with_extension("staging");
    fs::write(&staging, source).expect("write staged fake executable");
    let mut permissions = fs::metadata(&staging)
        .expect("fake executable metadata")
        .permissions();
    permissions.set_mode(0o700);
    fs::set_permissions(&staging, permissions).expect("make staged fake executable runnable");
    fs::rename(staging, path).expect("publish closed fake executable");
}

#[test]
fn executable_busy_retry_is_bounded_and_preserves_other_start_errors() {
    let mut transient_attempts = 0;
    let value = start_with_executable_busy_retry(|| {
        transient_attempts += 1;
        if transient_attempts < 3 {
            Err(std::io::Error::from(std::io::ErrorKind::ExecutableFileBusy))
        } else {
            Ok("started")
        }
    })
    .expect("transient executable-busy error is retried");
    assert_eq!(value, "started");
    assert_eq!(transient_attempts, 3);

    let mut permanent_attempts = 0;
    let error = start_with_executable_busy_retry::<()>(|| {
        permanent_attempts += 1;
        Err(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "denied",
        ))
    })
    .expect_err("non-transient start error is preserved");
    assert_eq!(error.kind(), std::io::ErrorKind::PermissionDenied);
    assert_eq!(permanent_attempts, 1);

    let mut exhausted_attempts = 0;
    let error = start_with_executable_busy_retry::<()>(|| {
        exhausted_attempts += 1;
        Err(std::io::Error::from(std::io::ErrorKind::ExecutableFileBusy))
    })
    .expect_err("persistent executable-busy error remains visible");
    assert_eq!(error.kind(), std::io::ErrorKind::ExecutableFileBusy);
    assert_eq!(exhausted_attempts, EXECUTABLE_BUSY_RETRY_DELAYS.len() + 1);
}

fn image_case(root: &Path) -> ImageCase {
    ImageCase {
        name: "tool-fixture".to_string(),
        input_source: "external:fixture".to_string(),
        corpus_category: "natural-image".to_string(),
        corpus_name: "fixture".to_string(),
        license_status: "cc0".to_string(),
        source_command: "fixture".to_string(),
        manifest_status: "covered".to_string(),
        source_format: "pgm".to_string(),
        width: 1,
        height: 1,
        components: 1,
        pixels: vec![7],
        pnm_path: root.join("fixture.pgm"),
    }
}

fn tool(kind: EncoderKind, program: &Path, available: bool) -> EncoderTool {
    EncoderTool {
        kind,
        program: program.to_path_buf(),
        available,
    }
}

#[test]
fn encoder_runner_builds_each_cli_contract_and_reports_process_failures() {
    let root = temp_dir("runner");
    let log = root.join("args.log");
    let program = root.join("encoder.sh");
    executable(
        &program,
        &format!(
            "#!/bin/sh\nprintf '%s|OPJ_NUM_THREADS=%s\\n' \"$*\" \"${{OPJ_NUM_THREADS-unset}}\" >> '{}'\n",
            log.display()
        ),
    );
    let case = image_case(&root);
    for kind in [
        EncoderKind::J2k,
        EncoderKind::OpenJpeg,
        EncoderKind::Grok,
        EncoderKind::Kakadu,
    ] {
        let output = run_encoder_once(&case, &tool(kind, &program, true), &root, kind.label())
            .expect("fake encoder succeeds");
        assert_eq!(
            output,
            root.join(format!(
                "{}_tool-fixture_{}.jp2",
                kind.label(),
                kind.label()
            ))
        );
    }
    let log = fs::read_to_string(log).expect("encoder argument log");
    assert!(log.contains("--encode-one --input"));
    assert!(log.contains("-n 3 -b 64,64 -p LRCP -threads 1|OPJ_NUM_THREADS=1"));
    assert!(log.contains("-n 3 -b 64,64 -p LRCP -H 1"));
    assert!(log.contains("Creversible=yes Clevels=2 Cblk={64,64} Corder=LRCP -rate -"));

    let failing = root.join("failing.sh");
    executable(&failing, "#!/bin/sh\nexit 7\n");
    let error = run_encoder_once(
        &case,
        &tool(EncoderKind::Grok, &failing, true),
        &root,
        "failure",
    )
    .expect_err("nonzero encoder");
    assert!(error.contains("grok encoder exited with exit status: 7"));
    let error = run_encoder_once(
        &case,
        &tool(EncoderKind::OpenJpeg, &root.join("missing"), true),
        &root,
        "missing",
    )
    .expect_err("missing executable");
    assert!(error.contains("start openjpeg"));
}

#[test]
fn version_commands_parse_stdout_stderr_fallbacks_and_missing_lines() {
    let root = temp_dir("versions");
    let openjpeg = root.join("openjpeg.sh");
    executable(
        &openjpeg,
        "#!/bin/sh\nprintf 'OpenJPEG fallback 1.0\\nCompiled against OpenJP2 version 2.5.3\\n'\n",
    );
    assert_eq!(
        command_version_label(&tool(EncoderKind::OpenJpeg, &openjpeg, true)),
        Ok("Compiled against OpenJP2 version 2.5.3".to_string())
    );

    let grok = root.join("grok.sh");
    executable(
        &grok,
        "#!/bin/sh\nif [ \"$1\" = --help ]; then echo 'generic help'; else echo 'Grok version 10.0' >&2; fi\n",
    );
    assert_eq!(
        command_version_label(&tool(EncoderKind::Grok, &grok, true)),
        Ok("Grok version 10.0".to_string())
    );

    let kakadu = root.join("kakadu.sh");
    executable(&kakadu, "#!/bin/sh\necho 'usage without a product name'\n");
    assert_eq!(
        command_version_label(&tool(EncoderKind::Kakadu, &kakadu, true)),
        Ok("available-version-not-reported-by-kdu_compress".to_string())
    );
    assert_eq!(
        command_version_label(&tool(EncoderKind::J2k, &kakadu, true)),
        Ok(env!("CARGO_PKG_VERSION").to_string())
    );
    assert!(
        command_version_label(&tool(EncoderKind::Grok, &root.join("missing"), true))
            .unwrap_err()
            .contains("grok:")
    );
}

#[test]
fn version_line_priority_prefers_compiled_identity_then_product_fallback() {
    let text = "OpenJPEG command line\n  Compiled against OpenJP2 2.5\n";
    assert_eq!(
        extract_version_line(EncoderKind::OpenJpeg, text),
        Some("Compiled against OpenJP2 2.5".to_string())
    );
    assert_eq!(
        version_line_by_priority(EncoderKind::OpenJpeg, text, false),
        Some("OpenJPEG command line".to_string())
    );
    assert_eq!(
        extract_version_line(EncoderKind::Grok, "Compiled against libgrok 10"),
        Some("Compiled against libgrok 10".to_string())
    );
    assert_eq!(
        extract_version_line(EncoderKind::Kakadu, "kdu_compress from Kakadu v8"),
        Some("kdu_compress from Kakadu v8".to_string())
    );
    assert_eq!(extract_version_line(EncoderKind::J2k, "j2k 0.7"), None);
}

#[test]
fn discovery_and_tool_labels_handle_present_missing_and_unavailable_entries() {
    let root = temp_dir("discovery");
    let fallback = root.join("fallback-tool");
    fs::write(&fallback, []).expect("write fallback tool");
    let fallback_label = fallback.to_str().expect("UTF-8 fallback");
    assert_eq!(
        discover_command(
            "TEST_ENCODE_COMPARE_TOOL_PATH_UNSET",
            "__j2k_compare_missing_program__",
            &[fallback_label]
        ),
        Some(fallback.clone())
    );
    assert_eq!(path_lookup("__j2k_compare_missing_program__"), None);

    let tools = vec![
        tool(EncoderKind::J2k, &fallback, true),
        tool(EncoderKind::OpenJpeg, &fallback, false),
    ];
    assert!(tool_available(&tools, EncoderKind::J2k));
    assert!(!tool_available(&tools, EncoderKind::OpenJpeg));
    assert!(!tool_available(&tools, EncoderKind::Grok));
    assert_eq!(
        tool_command(&tools, EncoderKind::J2k),
        fallback.display().to_string()
    );
    assert_eq!(tool_command(&tools, EncoderKind::Grok), "not found");
    assert_eq!(tool_version(&tools, EncoderKind::OpenJpeg), "unavailable");
    assert_eq!(tool_version(&tools, EncoderKind::Grok), "not found");
    assert!(!tool_version_available(&tools, EncoderKind::OpenJpeg));
    assert!(!tool_version_available(&tools, EncoderKind::Grok));
    assert_eq!(selected_encoders_label(&tools), "j2k,openjpeg");
}

#[test]
fn command_and_measurement_labels_cover_every_encoder_variant() {
    assert!(command_template(EncoderKind::J2k).contains("--encode-one"));
    assert!(command_template(EncoderKind::OpenJpeg).contains("OPJ_NUM_THREADS=1"));
    assert!(command_template(EncoderKind::Grok).contains("-H 1"));
    assert!(command_template(EncoderKind::Kakadu).contains("Creversible=yes"));
    assert_eq!(samples_label(&[1.0, 2.3456]), "1.000,2.346");
    assert_eq!(samples_label(&[]), "");
    assert_eq!(dimensions_label(640, 480), "640x480");
}
