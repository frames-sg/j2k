use std::{
    env,
    ffi::{OsStr, OsString},
    path::Path,
    process::{Command, Output},
};

/// Path to the cargo binary driving this xtask invocation.
pub(crate) fn cargo() -> OsString {
    env::var_os("CARGO").unwrap_or_else(|| OsString::from("cargo"))
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct CommandContext<'a> {
    current_dir: Option<&'a Path>,
    envs: &'a [(&'a str, &'a str)],
    target_dir: Option<&'a Path>,
}

impl<'a> CommandContext<'a> {
    pub(crate) const fn new() -> Self {
        Self {
            current_dir: None,
            envs: &[],
            target_dir: None,
        }
    }

    pub(crate) const fn current_dir(mut self, current_dir: &'a Path) -> Self {
        self.current_dir = Some(current_dir);
        self
    }

    pub(crate) const fn envs(mut self, envs: &'a [(&'a str, &'a str)]) -> Self {
        self.envs = envs;
        self
    }

    pub(crate) const fn target_dir(mut self, target_dir: &'a Path) -> Self {
        self.target_dir = Some(target_dir);
        self
    }
}

pub(crate) fn run_command(
    program: impl AsRef<OsStr>,
    args: &[&str],
    context: CommandContext<'_>,
) -> Result<(), String> {
    let program = program.as_ref();
    let display = display_command(program, args, context);
    eprintln!("+ {display}");
    let status = configured_command(program, args, context)
        .status()
        .map_err(|err| format!("failed to start `{}`: {err}", program.to_string_lossy()))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!(
            "`{}` exited with {status}",
            program.to_string_lossy()
        ))
    }
}

pub(crate) fn run_command_owned(
    program: impl AsRef<OsStr>,
    args: &[String],
    context: CommandContext<'_>,
) -> Result<(), String> {
    let borrowed = args.iter().map(String::as_str).collect::<Vec<_>>();
    run_command(program, &borrowed, context)
}

pub(crate) fn command_output(
    program: impl AsRef<OsStr>,
    args: &[&str],
    context: CommandContext<'_>,
) -> Result<Output, String> {
    let program = program.as_ref();
    configured_command(program, args, context)
        .output()
        .map_err(|err| format!("failed to start `{}`: {err}", program.to_string_lossy()))
}

pub(crate) fn command_output_os(
    program: impl AsRef<OsStr>,
    args: &[&str],
) -> Result<String, String> {
    let program = program.as_ref();
    let output = command_output(program, args, CommandContext::new())?;
    if !output.status.success() {
        return Err(format!(
            "`{}` exited with {}",
            program.to_string_lossy(),
            output.status
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

pub(crate) fn command_output_allow_failure(program: &str, args: &[&str]) -> Result<String, String> {
    let output = command_output(OsString::from(program), args, CommandContext::new())?;
    let mut text = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    if text.is_empty() {
        text = stderr;
    } else if !stderr.is_empty() {
        text.push('\n');
        text.push_str(&stderr);
    }
    if text.is_empty() {
        Err(format!(
            "`{program}` exited with {} and no output",
            output.status
        ))
    } else {
        Ok(text)
    }
}

fn configured_command(program: &OsStr, args: &[&str], context: CommandContext<'_>) -> Command {
    let mut command = Command::new(program);
    command.args(args);
    if let Some(current_dir) = context.current_dir {
        command.current_dir(current_dir);
    }
    if let Some(target_dir) = context.target_dir {
        command.env("CARGO_TARGET_DIR", target_dir);
    }
    for (key, value) in context.envs {
        command.env(key, value);
    }
    command
}

fn display_command(program: &OsStr, args: &[&str], context: CommandContext<'_>) -> String {
    let mut parts = Vec::new();
    if let Some(current_dir) = context.current_dir {
        parts.push(format!("cd {} &&", current_dir.display()));
    }
    if let Some(target_dir) = context.target_dir {
        parts.push(format!("CARGO_TARGET_DIR={}", target_dir.display()));
    }
    for (key, value) in context.envs {
        parts.push(format!("{key}={value}"));
    }
    parts.push(program.to_string_lossy().into_owned());
    parts.extend(args.iter().map(|arg| (*arg).to_string()));
    parts.join(" ")
}
