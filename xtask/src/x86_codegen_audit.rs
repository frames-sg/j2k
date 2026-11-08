// SPDX-License-Identifier: MIT OR Apache-2.0

//! Audit of the baseline x86-64 machine code of `j2k-native` and of the `j2k`
//! facade, which converts decoded samples to output pixels, for two
//! regressions the source cannot show:
//!
//! - A floor, round or fused multiply-add on scalar `f32` outside a runtime
//!   SIMD dispatch. Baseline x86-64 (SSE2) has no instruction for these, so
//!   each one is a C library call per sample (see `j2k_native::math`).
//! - A generic SIMD helper instantiated for AVX2 or AVX-512 but compiled out
//!   of line, such as a function item's `Fn` shim. Out of line it lacks the
//!   dispatch's target features, so every vector operation it performs is a
//!   call.
//!
//! Code generated for the SSE2, SSE4.2 and scalar fallback levels may call
//! the library: those levels serve CPUs without the instructions. So may a
//! function that calls into the AVX2 level, since its other levels are
//! inlined into it.

use std::io::{BufRead, BufReader};
use std::path::Path;
use std::process::{Command, Stdio};

use crate::command_support::{run_cargo_with_env, run_program};
use crate::process::retry_executable_busy;

const TARGET: &str = "x86_64-unknown-linux-gnu";
const TARGET_DIR: &str = "target/x86-codegen-audit";

/// The audited crates and their library files.
const CRATES: [(&str, &str); 2] = [("j2k-native", "libj2k_native.rlib"), ("j2k", "libj2k.rlib")];

/// Library routines baseline x86-64 calls for scalar rounding and fused
/// multiply-add.
const LIBM_ROUTINES: [&str; 16] = [
    "floorf",
    "ceilf",
    "truncf",
    "roundf",
    "rintf",
    "nearbyintf",
    "roundevenf",
    "fmaf",
    "floor",
    "ceil",
    "trunc",
    "round",
    "rint",
    "nearbyint",
    "roundeven",
    "fma",
];

/// Code for these SIMD levels serves CPUs without the instructions.
const OLD_CPU_LEVELS: [&str; 6] = [
    "vectorize_sse2",
    "vectorize_sse4_2",
    "vectorize_fallback",
    "sse2::Sse2",
    "sse4_2::Sse4_2",
    "fallback::Fallback",
];

const OBJDUMP_ARGS: [&str; 4] = ["-d", "-r", "-C", "--no-show-raw-insn"];

/// The levels whose operations must be inlined into their dispatch.
const WIDE_LEVELS: [(&str, &str); 2] = [
    ("vectorize_avx2", "13fearless_simd9generated4avx2"),
    ("vectorize_avx512", "13fearless_simd9generated6avx512"),
];

pub(crate) fn x86_codegen_audit(args: impl Iterator<Item = String>) -> Result<(), String> {
    let arguments = args.collect::<Vec<_>>();
    if !arguments.is_empty() {
        return Err(format!(
            "x86-codegen-audit accepts no arguments; received {}",
            arguments.join(" ")
        ));
    }
    run_program("rustup".into(), &["target", "add", TARGET], &[])?;
    build_audited_libraries()?;
    let mut violations = Vec::new();
    for (package, library) in CRATES {
        let library = Path::new(TARGET_DIR)
            .join(TARGET)
            .join("release")
            .join(library);
        let found = audit_library(&library)?;
        if !found.is_empty() {
            violations.push(format!("{package}:"));
            violations.extend(found);
        }
    }
    if violations.is_empty() {
        eprintln!("x86-codegen-audit: no scalar library calls or out-of-line wide SIMD operations");
        return Ok(());
    }
    Err(format!(
        "x86-codegen-audit found violations in baseline x86-64 code:\n{}",
        violations.join("\n")
    ))
}

fn build_audited_libraries() -> Result<(), String> {
    let mut build = vec!["build"];
    for (package, _) in CRATES {
        build.extend(["-p", package]);
    }
    build.extend(["--lib", "--release", "--target", TARGET]);
    // Without LTO each library keeps object code, not only bitcode.
    run_cargo_with_env(
        &build,
        &[
            ("CARGO_PROFILE_RELEASE_LTO", "false"),
            ("CARGO_TARGET_DIR", TARGET_DIR),
            // Cargo prefers encoded flags over inherited plain and target flags.
            (
                "CARGO_ENCODED_RUSTFLAGS",
                "-C\x1ftarget-cpu=x86-64\x1f-D\x1fwarnings",
            ),
        ],
    )
}

/// Disassemble `library` with `objdump` (or `$OBJDUMP`) and audit it.
fn audit_library(library: &Path) -> Result<Vec<String>, String> {
    let objdump = std::env::var_os("OBJDUMP").unwrap_or_else(|| "objdump".into());
    let mut command = Command::new(&objdump);
    command
        .args(OBJDUMP_ARGS)
        .arg(library)
        .stdout(Stdio::piped());
    eprintln!(
        "+ {} {} {}",
        objdump.to_string_lossy(),
        OBJDUMP_ARGS.join(" "),
        library.display()
    );
    let mut child = retry_executable_busy(|| command.spawn())
        .map_err(|err| format!("failed to start `{}`: {err}", objdump.to_string_lossy()))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "objdump stdout was not captured".to_string())?;
    let mut audit = Audit::default();
    for line in BufReader::new(stdout).lines() {
        audit.line(&line.map_err(|err| format!("failed to read objdump output: {err}"))?);
    }
    let status = child
        .wait()
        .map_err(|err| format!("failed to wait for objdump: {err}"))?;
    if !status.success() {
        return Err(format!(
            "`{}` exited with {status}",
            objdump.to_string_lossy()
        ));
    }
    let report = audit.finish();
    if report.functions == 0 {
        return Err(format!(
            "objdump listed no functions in {}",
            library.display()
        ));
    }
    Ok(report.violations)
}

#[derive(Debug, Default)]
struct Function {
    name: String,
    libm_calls: Vec<String>,
    wide_operation_calls: Vec<String>,
    dispatches: bool,
}

#[derive(Debug, Default)]
struct Audit {
    current: Option<Function>,
    functions: usize,
    violations: Vec<String>,
}

#[derive(Debug, PartialEq, Eq)]
struct Report {
    functions: usize,
    violations: Vec<String>,
}

impl Audit {
    fn line(&mut self, line: &str) {
        let line = line.trim();
        if let Some(name) = function_header(line) {
            self.close();
            self.current = Some(Function {
                name: name.to_string(),
                ..Function::default()
            });
            self.functions += 1;
            return;
        }
        let Some(function) = self.current.as_mut() else {
            return;
        };
        if WIDE_LEVELS.iter().any(|(entry, _)| line.contains(entry)) {
            function.dispatches = true;
        }
        let Some(symbol) = relocation_symbol(line) else {
            return;
        };
        if LIBM_ROUTINES.contains(&symbol) {
            function.libm_calls.push(symbol.to_string());
        } else if !symbol.contains("vectorize")
            && symbol.contains("10simd_trait4Simd")
            && WIDE_LEVELS.iter().any(|(_, level)| symbol.contains(level))
        {
            function.wide_operation_calls.push(symbol.to_string());
        }
    }

    fn close(&mut self) {
        let Some(function) = self.current.take() else {
            return;
        };
        let old_cpu = OLD_CPU_LEVELS
            .iter()
            .any(|level| function.name.contains(level));
        let in_wide_level = WIDE_LEVELS
            .iter()
            .any(|(entry, _)| function.name.contains(entry));
        // A function that calls into a wide level inlines its other levels.
        let dispatcher = function.dispatches && !in_wide_level;
        if !function.libm_calls.is_empty() && !old_cpu && !dispatcher {
            self.violations.push(format!(
                "  library call ({}) outside a SIMD dispatch: {}",
                function.libm_calls.join(", "),
                function.name
            ));
        }
        if !function.wide_operation_calls.is_empty() && !in_wide_level {
            self.violations.push(format!(
                "  {} wide SIMD operation call(s) compiled out of line: {}",
                function.wide_operation_calls.len(),
                function.name
            ));
        }
    }

    fn finish(mut self) -> Report {
        self.close();
        Report {
            functions: self.functions,
            violations: self.violations,
        }
    }
}

/// The demangled name of an `objdump -d` function header such as
/// `0000000000000000 <name>:`.
fn function_header(line: &str) -> Option<&str> {
    let (address, rest) = line.split_once(' ')?;
    if address.is_empty() || !address.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return None;
    }
    rest.strip_prefix('<')?.strip_suffix(">:")
}

/// The symbol of an x86-64 call or address relocation line such as
/// `0000000000000012:  R_X86_64_PLT32  floorf-0x4`, without the addend.
fn relocation_symbol(line: &str) -> Option<&str> {
    let (_, rest) = line.split_once("R_X86_64_")?;
    let symbol = rest.split_whitespace().nth(1)?;
    let symbol = symbol.split(['-', '+']).next().unwrap_or(symbol);
    Some(symbol.strip_prefix(".text.").unwrap_or(symbol))
}

#[cfg(test)]
mod tests {
    use super::{Audit, Report};

    #[cfg(unix)]
    #[test]
    fn audit_build_replaces_inherited_cpu_and_feature_flags() {
        use crate::{command_support::use_test_cargo_program, test_command::RecordingProgram};
        use std::{env, fs, process::Command};

        const CHILD_ENV: &str = "XTASK_TEST_X86_AUDIT_FLAGS_CHILD";
        const PLAIN_FLAGS: &str = "-C target-cpu=native -C target-feature=+avx2 -A warnings";
        const ENCODED_FLAGS: &str =
            "-C\x1ftarget-cpu=x86-64-v4\x1f-C\x1ftarget-feature=+avx512f\x1f-A\x1fwarnings";

        if env::var_os(CHILD_ENV).is_none() {
            let output = Command::new(env::current_exe().expect("current test binary"))
                .args([
                    "x86_codegen_audit::tests::audit_build_replaces_inherited_cpu_and_feature_flags",
                    "--exact",
                    "--nocapture",
                ])
                .env(CHILD_ENV, "1")
                .env("RUSTFLAGS", PLAIN_FLAGS)
                .env("CARGO_ENCODED_RUSTFLAGS", ENCODED_FLAGS)
                .output()
                .expect("run audit command test with isolated inherited flags");
            assert!(
                output.status.success(),
                "audit command child failed:\nstdout:\n{}\nstderr:\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr),
            );
            return;
        }

        let recording = RecordingProgram::new(
            "x86-audit-flags-test",
            "printf '%s' \"$CARGO_ENCODED_RUSTFLAGS\" > \"${0}.flags\"",
        );
        let _cargo = use_test_cargo_program(recording.program().as_os_str().to_owned());
        super::build_audited_libraries().expect("audit build command");

        let flags = fs::read_to_string(recording.program().with_extension("sh.flags"))
            .expect("recorded build flags");
        assert_eq!(
            flags.split('\x1f').collect::<Vec<_>>(),
            ["-C", "target-cpu=x86-64", "-D", "warnings"]
        );
        assert!(recording.log().starts_with(
            "build -p j2k-native -p j2k --lib --release --target x86_64-unknown-linux-gnu|"
        ));
        assert_eq!(env::var("RUSTFLAGS").unwrap(), PLAIN_FLAGS);
        assert_eq!(env::var("CARGO_ENCODED_RUSTFLAGS").unwrap(), ENCODED_FLAGS);
    }

    fn audit(disassembly: &str) -> Report {
        let mut audit = Audit::default();
        for line in disassembly.lines() {
            audit.line(line);
        }
        audit.finish()
    }

    const AVX2_ADD: &str = ".text._RNvNvNvXs2_NtNtCs6I8Lur5InzC_13fearless_simd9generated4avx2NtB9_4Avx2NtNtBb_10simd_trait4Simd9add_f32x8";
    const AVX2_ENTRY: &str = "_RNvXs2_NtNtCs6I8Lur5InzC_13fearless_simd9generated4avx2NtB5_4Avx2NtNtB7_10simd_trait4Simd9vectorize14vectorize_avx2";

    #[test]
    fn scalar_library_calls_fail_only_outside_old_cpu_levels_and_dispatches() {
        let report = audit(&format!(
            "0000000000000000 <j2k_native::store::copy_rows>:\n\
             \x20      12: callq 0x17 <j2k_native::store::copy_rows+0x17>\n\
             \x20      0000000000000013:  R_X86_64_PLT32\tfloorf-0x4\n\
             0000000000000000 <j2k_native::store::dispatching>:\n\
             \x20      0000000000000013:  R_X86_64_PLT32\troundf-0x4\n\
             \x20      0000000000000020:  R_X86_64_PLT32\t{AVX2_ENTRY}-0x4\n\
             0000000000000000 <<Sse2 as Simd>::vectorize::vectorize_sse2::<j2k_native::x::{{closure#0}}, ()>>:\n\
             \x20      0000000000000013:  R_X86_64_PLT32\tfmaf-0x4\n\
             0000000000000000 <j2k_native::idwt::lift::<fearless_simd::generated::sse2::Sse2>>:\n\
             \x20      0000000000000013:  R_X86_64_PLT32\tfloorf-0x4\n\
             0000000000000000 <<Avx2 as Simd>::vectorize::vectorize_avx2::<j2k_native::y::{{closure#0}}, ()>>:\n\
             \x20      12: jmp 0x17 <<Avx2 as Simd>::vectorize::vectorize_avx2::<j2k_native::y::{{closure#0}}, ()>+0x17>\n\
             \x20      0000000000000013:  R_X86_64_PLT32\troundf-0x4\n"
        ));
        assert_eq!(report.functions, 5);
        assert_eq!(
            report.violations,
            [
                "  library call (floorf) outside a SIMD dispatch: j2k_native::store::copy_rows",
                "  library call (roundf) outside a SIMD dispatch: <Avx2 as Simd>::vectorize::vectorize_avx2::<j2k_native::y::{closure#0}, ()>",
            ]
        );
    }

    #[test]
    fn wide_operations_fail_only_out_of_line() {
        let report = audit(&format!(
            "0000000000000000 <<j2k_native::banded::lift_53_low_simd<fearless_simd::generated::avx2::Avx2> as core::ops::function::Fn<()>>::call>:\n\
             \x20      0000000000000013:  R_X86_64_PLT32\t{AVX2_ADD}-0x4\n\
             0000000000000000 <<Avx2 as Simd>::vectorize::vectorize_avx2::<j2k_native::x::{{closure#0}}, ()>>:\n\
             \x20      0000000000000013:  R_X86_64_PLT32\t{AVX2_ADD}-0x4\n\
             0000000000000000 <j2k_native::store::dispatching>:\n\
             \x20      0000000000000020:  R_X86_64_PLT32\t{AVX2_ENTRY}-0x4\n"
        ));
        assert_eq!(report.functions, 3);
        assert_eq!(
            report.violations,
            ["  1 wide SIMD operation call(s) compiled out of line: <j2k_native::banded::lift_53_low_simd<fearless_simd::generated::avx2::Avx2> as core::ops::function::Fn<()>>::call"]
        );
    }

    #[test]
    fn headers_and_relocations_parse_from_objdump_lines() {
        assert_eq!(
            super::function_header("0000000000000000 <a::b<c>>:"),
            Some("a::b<c>")
        );
        assert_eq!(
            super::function_header("      12: callq 0x17 <a+0x17>"),
            None
        );
        assert_eq!(
            super::relocation_symbol("0000000000000013:  R_X86_64_PLT32\tfloorf-0x4"),
            Some("floorf")
        );
        assert_eq!(
            super::relocation_symbol("  R_X86_64_GOTPCRELX\tfmaf"),
            Some("fmaf")
        );
        assert_eq!(super::relocation_symbol("      12: callq 0x17"), None);
    }
}
