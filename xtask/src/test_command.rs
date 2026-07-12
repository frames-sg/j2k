// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{
    fs,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(0);

pub(crate) struct RecordingProgram {
    root: PathBuf,
    program: PathBuf,
    log: PathBuf,
}

impl RecordingProgram {
    pub(crate) fn new(label: &str, script_body: &str) -> Self {
        let root = std::env::temp_dir().join(format!(
            "j2k-{label}-{}-{}",
            std::process::id(),
            NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&root).expect("create recording-program test directory");
        let program = root.join("program.sh");
        let log = root.join("program.log");
        fs::write(
            &program,
            format!(
                "#!/bin/sh\nprintf '%s|RUSTDOCFLAGS=%s|RUST_TEST_THREADS=%s\\n' \"$*\" \"${{RUSTDOCFLAGS-unset}}\" \"${{RUST_TEST_THREADS-unset}}\" >> '{}'\n{script_body}\n",
                log.display()
            ),
        )
        .expect("write recording test program");
        let mut permissions = fs::metadata(&program)
            .expect("recording-program metadata")
            .permissions();
        permissions.set_mode(0o700);
        fs::set_permissions(&program, permissions).expect("make recording test program executable");
        Self { root, program, log }
    }

    pub(crate) fn program(&self) -> &Path {
        &self.program
    }

    pub(crate) fn log(&self) -> String {
        fs::read_to_string(&self.log).expect("read recording-program log")
    }
}

impl Drop for RecordingProgram {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.root).expect("remove recording-program test directory");
    }
}
