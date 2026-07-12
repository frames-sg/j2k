use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::process::{self, cargo};

pub(super) fn ensure_clean_worktree() -> Result<(), String> {
    let status = process::command_output_os(OsString::from("git"), &["status", "--porcelain"])?;
    if status.trim().is_empty() {
        Ok(())
    } else {
        Err(format!(
            "working tree must be clean before packaging:\n{status}"
        ))
    }
}

pub(super) fn run_cargo(args: &[&str]) -> Result<(), String> {
    run_cargo_with_env(args, &[])
}

pub(super) fn run_cargo_with_env(args: &[&str], envs: &[(&str, &str)]) -> Result<(), String> {
    run_program(cargo(), args, envs)
}

pub(super) fn run_cargo_test_with_pass_floor(
    args: &[&str],
    envs: &[(&str, &str)],
    min_passed: usize,
    label: &str,
) -> Result<(), String> {
    let mut test_args = args.to_vec();
    test_args.extend_from_slice(&["--", "--nocapture"]);
    let env_display = envs
        .iter()
        .map(|(key, value)| format!("{key}={value}"))
        .collect::<Vec<_>>()
        .join(" ");
    let display_prefix = if env_display.is_empty() {
        String::new()
    } else {
        format!("{env_display} ")
    };
    eprintln!(
        "+ {display_prefix}{} {}",
        cargo().to_string_lossy(),
        test_args.join(" ")
    );

    let mut command = Command::new(cargo());
    command.args(&test_args);
    for (key, value) in envs {
        command.env(key, value);
    }
    let output = command
        .output()
        .map_err(|err| format!("failed to start `{}`: {err}", cargo().to_string_lossy()))?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    print!("{stdout}");
    eprint!("{stderr}");
    if !output.status.success() {
        return Err(format!(
            "`{}` exited with {} while running {label}",
            cargo().to_string_lossy(),
            output.status
        ));
    }

    let passed = passed_test_count(&format!("{stdout}\n{stderr}"));
    if passed < min_passed {
        return Err(format!(
            "{label} executed {passed} tests, expected at least {min_passed}; \
             check required comparator tools and skip gates"
        ));
    }
    Ok(())
}

pub(super) fn passed_test_count(output: &str) -> usize {
    output
        .lines()
        .filter_map(|line| {
            let rest = line.trim().strip_prefix("test result: ok.")?;
            if !rest.contains(" passed") {
                return None;
            }
            rest.split_whitespace().next()?.parse::<usize>().ok()
        })
        .sum()
}

pub(super) fn run_nightly_cargo(args: &[&str]) -> Result<(), String> {
    let mut rustup_args = vec!["run", "nightly", "cargo"];
    rustup_args.extend_from_slice(args);
    run_program(OsString::from("rustup"), &rustup_args, &[])
}

pub(super) fn run_nightly_cargo_in_dir_owned(dir: &str, args: &[String]) -> Result<(), String> {
    let mut rustup_args = vec![
        "run".to_string(),
        "nightly".to_string(),
        "cargo".to_string(),
    ];
    rustup_args.extend_from_slice(args);
    run_program_in_dir_owned_with_program(OsString::from("rustup"), dir, &rustup_args, &[])
}

pub(super) fn run_program(
    program: OsString,
    args: &[&str],
    envs: &[(&str, &str)],
) -> Result<(), String> {
    process::run_command(program, args, process::CommandContext::new().envs(envs))
}

pub(super) fn run_program_in_dir_owned_with_program(
    program: OsString,
    dir: &str,
    args: &[String],
    envs: &[(&str, &str)],
) -> Result<(), String> {
    process::run_command_owned(
        program,
        args,
        process::CommandContext::new()
            .current_dir(Path::new(dir))
            .envs(envs),
    )
}

pub(super) fn command_output(program: &str, args: &[&str]) -> Result<String, String> {
    command_output_os(OsString::from(program), args)
}

pub(super) fn command_output_allow_failure(program: &str, args: &[&str]) -> Result<String, String> {
    process::command_output_allow_failure(program, args)
}

pub(super) fn command_output_os(program: OsString, args: &[&str]) -> Result<String, String> {
    process::command_output_os(program, args)
}

pub(super) fn command_output_os_detailed_with_env(
    program: OsString,
    args: &[&str],
    envs: &[(&str, &str)],
) -> Result<String, String> {
    let display = format!("{} {}", program.to_string_lossy(), args.join(" "));
    let output = process::command_output(program, args, process::CommandContext::new().envs(envs))?;
    if output.status.success() {
        return String::from_utf8(output.stdout)
            .map(|stdout| stdout.trim().to_string())
            .map_err(|err| format!("`{display}` emitted non-UTF-8 stdout: {err}"));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let combined = format!("{stdout}\n{stderr}");
    Err(format!(
        "`{display}` exited with {}:\n{}",
        output.status,
        combined.trim()
    ))
}

pub(super) fn workspace_version() -> Result<String, String> {
    let manifest = fs::read_to_string("Cargo.toml")
        .map_err(|err| format!("failed to read Cargo.toml: {err}"))?;
    manifest
        .lines()
        .find_map(|line| {
            let line = line.trim();
            line.strip_prefix("version")
                .and_then(|rest| rest.split('"').nth(1))
                .map(str::to_string)
        })
        .ok_or_else(|| "failed to find workspace package version".to_string())
}

pub(super) fn rust_sources(root: &Path) -> Result<Vec<PathBuf>, String> {
    let mut out = Vec::new();
    collect_rust_sources(root, &mut out)?;
    Ok(out)
}

fn collect_rust_sources(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    for entry in
        fs::read_dir(dir).map_err(|err| format!("failed to read {}: {err}", dir.display()))?
    {
        let entry =
            entry.map_err(|err| format!("failed to read {} entry: {err}", dir.display()))?;
        let path = entry.path();
        if path.is_dir() {
            collect_rust_sources(&path, out)?;
        } else if path.extension().and_then(|ext| ext.to_str()) == Some("rs") {
            out.push(path);
        }
    }
    Ok(())
}
