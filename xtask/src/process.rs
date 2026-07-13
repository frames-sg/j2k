use std::{
    env,
    ffi::{OsStr, OsString},
    io,
    path::Path,
    process::{Command, Output},
    thread,
    time::Duration,
};

#[cfg(all(test, unix))]
use std::{cell::RefCell, marker::PhantomData, rc::Rc};

#[cfg(all(test, unix))]
thread_local! {
    static TEST_CARGO_PROGRAM: RefCell<Option<OsString>> = const { RefCell::new(None) };
}

const EXECUTABLE_BUSY_RETRY_DELAYS: [u64; 8] = [1, 2, 4, 8, 16, 32, 64, 128];

pub(crate) fn retry_executable_busy<T>(mut start: impl FnMut() -> io::Result<T>) -> io::Result<T> {
    for delay_ms in EXECUTABLE_BUSY_RETRY_DELAYS {
        match start() {
            Err(error) if is_executable_busy(&error) => {
                thread::sleep(Duration::from_millis(delay_ms));
            }
            result => return result,
        }
    }
    start()
}

fn is_executable_busy(error: &io::Error) -> bool {
    error.kind() == io::ErrorKind::ExecutableFileBusy
}

/// Path to the cargo binary driving this xtask invocation.
pub(crate) fn cargo() -> OsString {
    #[cfg(all(test, unix))]
    if let Some(program) = TEST_CARGO_PROGRAM.with(|program| program.borrow().clone()) {
        return program;
    }
    env::var_os("CARGO").unwrap_or_else(|| OsString::from("cargo"))
}

#[cfg(all(test, unix))]
pub(crate) struct TestCargoProgramGuard {
    previous: Option<OsString>,
    _thread_bound: PhantomData<Rc<()>>,
}

#[cfg(all(test, unix))]
pub(crate) fn use_test_cargo_program(program: OsString) -> TestCargoProgramGuard {
    let previous = TEST_CARGO_PROGRAM.with(|slot| slot.replace(Some(program)));
    TestCargoProgramGuard {
        previous,
        _thread_bound: PhantomData,
    }
}

#[cfg(all(test, unix))]
impl Drop for TestCargoProgramGuard {
    fn drop(&mut self) {
        TEST_CARGO_PROGRAM.with(|slot| {
            slot.replace(self.previous.take());
        });
    }
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
    let mut command = configured_command(program, args, context);
    let status = retry_executable_busy(|| command.status())
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
    let mut command = configured_command(program, args, context);
    retry_executable_busy(|| command.output())
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

#[cfg(all(test, unix))]
mod tests {
    use super::{
        cargo, retry_executable_busy, use_test_cargo_program, EXECUTABLE_BUSY_RETRY_DELAYS,
    };
    use std::{ffi::OsString, io};

    #[test]
    fn executable_busy_retry_is_bounded_and_preserves_other_start_errors() {
        let mut transient_attempts = 0;
        let value = retry_executable_busy(|| {
            transient_attempts += 1;
            if transient_attempts < 3 {
                Err(io::Error::from(io::ErrorKind::ExecutableFileBusy))
            } else {
                Ok("started")
            }
        })
        .expect("transient executable-busy error is retried");
        assert_eq!(value, "started");
        assert_eq!(transient_attempts, 3);

        let mut permanent_attempts = 0;
        let error = retry_executable_busy::<()>(|| {
            permanent_attempts += 1;
            Err(io::Error::new(io::ErrorKind::PermissionDenied, "denied"))
        })
        .expect_err("non-transient start error is preserved");
        assert_eq!(error.kind(), io::ErrorKind::PermissionDenied);
        assert_eq!(permanent_attempts, 1);

        let mut exhausted_attempts = 0;
        let error = retry_executable_busy::<()>(|| {
            exhausted_attempts += 1;
            Err(io::Error::from(io::ErrorKind::ExecutableFileBusy))
        })
        .expect_err("persistent executable-busy error remains visible");
        assert_eq!(error.kind(), io::ErrorKind::ExecutableFileBusy);
        assert_eq!(exhausted_attempts, EXECUTABLE_BUSY_RETRY_DELAYS.len() + 1);
    }

    #[test]
    fn cargo_test_override_is_thread_local_nested_and_transactional() {
        let inherited = cargo();
        let outer_program = OsString::from("j2k-test-cargo-outer");
        let inner_program = OsString::from("j2k-test-cargo-inner");

        let outer = use_test_cargo_program(outer_program.clone());
        assert_eq!(cargo(), outer_program);
        assert_eq!(
            std::thread::spawn(cargo).join().expect("Cargo thread"),
            inherited
        );

        {
            let _inner = use_test_cargo_program(inner_program.clone());
            assert_eq!(cargo(), inner_program);
        }
        assert_eq!(cargo(), outer_program);

        drop(outer);
        assert_eq!(cargo(), inherited);
    }
}
