// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{
    fs,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    process::{Command, Output},
    sync::atomic::{AtomicU64, Ordering},
};

static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(0);

pub(crate) struct Harness {
    root: PathBuf,
    cargo: PathBuf,
    log: PathBuf,
    path: String,
}

impl Harness {
    pub(crate) fn new() -> Self {
        let root = std::env::temp_dir().join(format!(
            "j2k-xtask-orchestration-{}-{}",
            std::process::id(),
            NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&root).expect("create orchestration test directory");
        let cargo = root.join("cargo.sh");
        let log = root.join("cargo.log");
        let real_cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
        fs::write(
            &cargo,
            format!(
                "#!/bin/sh\nprintf '%s|RUSTDOCFLAGS=%s|RUST_TEST_THREADS=%s\\n' \"$*\" \"${{RUSTDOCFLAGS-unset}}\" \"${{RUST_TEST_THREADS-unset}}\" >> '{}'\nif [ \"$1\" = metadata ]; then exec \"{}\" \"$@\"; fi\nif [ \"$1\" = clippy ]; then printf '%s\\n' '{{\"reason\":\"build-finished\",\"success\":true}}'; exit 0; fi\nif [ \"$1\" = test ]; then printf 'test result: ok. 100 passed; 0 failed;\\n'; exit 0; fi\nif [ \"$1\" = -V ]; then printf 'cargo 1.96.0\\n'; fi\n",
                log.display(),
                real_cargo
            ),
        )
        .expect("write fake Cargo");
        make_executable(&cargo, "fake Cargo");
        for tool in ["typos", "cargo-machete"] {
            let program = root.join(tool);
            fs::write(
                &program,
                format!(
                    "#!/bin/sh\nprintf '%s %s\\n' \"$(basename \"$0\")\" \"$*\" >> '{}'\n",
                    log.display()
                ),
            )
            .expect("write fake external tool");
            make_executable(&program, "fake external tool");
        }
        let git = root.join("git");
        let real_git = find_program("git");
        fs::write(
            &git,
            format!(
                "#!/bin/sh\nprintf 'git %s\\n' \"$*\" >> '{}'\nif [ \"$1\" = status ]; then exit 0; fi\nif [ \"$1\" = config ] && [ \"$2\" = --get ] && [ \"$3\" = remote.origin.url ]; then printf '%s\\n' 'git@example.invalid:frames-sg/j2k.git'; exit 0; fi\nexec \"{}\" \"$@\"\n",
                log.display(),
                real_git.display()
            ),
        )
        .expect("write fake git wrapper");
        make_executable(&git, "fake git");
        let rustup = root.join("rustup");
        fs::write(
            &rustup,
            format!(
                "#!/bin/sh\nprintf 'rustup %s\\n' \"$*\" >> '{}'\ncase \"$*\" in\n  *'public-api --version'*) printf 'cargo-public-api 0.52.0\\n' ;;\n  *' cargo public-api '*) case \"${{RUSTDOCFLAGS-}}\" in *document-hidden*) printf 'pub struct SyntheticHidden\\n' ;; *) printf 'pub struct Synthetic\\n' ;; esac ;;\n  *'semver-checks --version'*) printf 'cargo-semver-checks 0.48.0\\n' ;;\nesac\n",
                log.display()
            ),
        )
        .expect("write fake rustup");
        make_executable(&rustup, "fake rustup");
        let python = root.join("python3");
        fs::write(
            &python,
            format!(
                "#!/bin/sh\nprintf 'python3 %s\\n' \"$*\" >> '{}'\n",
                log.display()
            ),
        )
        .expect("write fake Python");
        make_executable(&python, "fake Python");
        let path = format!(
            "{}:{}",
            root.display(),
            std::env::var("PATH").unwrap_or_default()
        );
        Self {
            root,
            cargo,
            log,
            path,
        }
    }

    pub(crate) fn run(&self, args: &[&str]) -> Output {
        self.run_with_env(args, &[])
    }

    pub(crate) fn run_with_env(&self, args: &[&str], envs: &[(&str, &str)]) -> Output {
        self.run_in_with_env(workspace_root(), args, envs)
    }

    pub(crate) fn run_in(&self, directory: &Path, args: &[&str]) -> Output {
        self.run_in_with_env(directory, args, &[])
    }

    fn run_in_with_env(&self, directory: &Path, args: &[&str], envs: &[(&str, &str)]) -> Output {
        let mut command = Command::new(env!("CARGO_BIN_EXE_xtask"));
        command
            .args(args)
            .current_dir(directory)
            .env("CARGO", &self.cargo)
            .env("PATH", &self.path);
        for key in [
            "CARGO_BUILD_TARGET",
            "CARGO_ENCODED_RUSTDOCFLAGS",
            "CARGO_ENCODED_RUSTFLAGS",
            "DOCS_RS",
            "GH_TOKEN",
            "GITHUB_REPOSITORY",
            "GITHUB_TOKEN",
            "MACOSX_DEPLOYMENT_TARGET",
            "RUSTC",
            "RUSTC_BOOTSTRAP",
            "RUSTC_WRAPPER",
            "RUSTC_WORKSPACE_WRAPPER",
            "RUSTDOC",
            "RUSTDOCFLAGS",
            "RUSTFLAGS",
        ] {
            command.env_remove(key);
        }
        for (key, value) in envs {
            command.env(key, value);
        }
        command.output().expect("run xtask child process")
    }

    pub(crate) fn path(&self, name: &str) -> PathBuf {
        self.root.join(name)
    }

    pub(crate) fn log(&self) -> String {
        fs::read_to_string(&self.log).expect("read fake Cargo log")
    }
}

impl Drop for Harness {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.root).expect("remove orchestration test directory");
    }
}

pub(crate) fn assert_success(output: &Output, task: &str) {
    assert!(
        output.status.success(),
        "{task} failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn workspace_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask manifest has workspace parent")
}

fn find_program(name: &str) -> PathBuf {
    std::env::split_paths(&std::env::var_os("PATH").unwrap_or_default())
        .map(|directory| directory.join(name))
        .find(|candidate| candidate.is_file())
        .unwrap_or_else(|| panic!("failed to find real {name} on PATH"))
}

fn make_executable(path: &Path, label: &str) {
    let mut permissions = fs::metadata(path).expect(label).permissions();
    permissions.set_mode(0o700);
    fs::set_permissions(path, permissions).expect(label);
}
