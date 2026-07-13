// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{fs, path::PathBuf, sync::atomic::Ordering};

#[cfg(unix)]
use std::{cell::RefCell, marker::PhantomData, os::unix::fs::PermissionsExt, path::Path, rc::Rc};

use super::{
    cleanup_cli_staging, cleanup_cli_temp, command_is_runnable, kakadu_temp_dir,
    openjph_input_extension, openjph_output_extension, openjph_temp_dir, read_cli_pnm_output,
    reduce_factor, Container, Downscale, PixelFormat, OPENJPH_TEMP_COUNTER,
};
#[cfg(unix)]
use super::{
    decode_kakadu_once, decode_openjph_once, kakadu_command_label, kakadu_version_label,
    openjph_command_label, openjph_version_label, FixtureCase,
};
#[cfg(unix)]
use crate::fixture_compare::{Codec, Operation};

#[cfg(unix)]
thread_local! {
    static TEST_COMPARATOR_PROGRAMS: RefCell<Option<TestComparatorPrograms>> = const {
        RefCell::new(None)
    };
}

#[cfg(unix)]
struct TestComparatorPrograms {
    openjph: PathBuf,
    kakadu: PathBuf,
}

#[cfg(unix)]
pub(super) fn test_openjph_program() -> Option<PathBuf> {
    TEST_COMPARATOR_PROGRAMS.with(|programs| {
        programs
            .borrow()
            .as_ref()
            .map(|programs| programs.openjph.clone())
    })
}

#[cfg(unix)]
pub(super) fn test_kakadu_program() -> Option<PathBuf> {
    TEST_COMPARATOR_PROGRAMS.with(|programs| {
        programs
            .borrow()
            .as_ref()
            .map(|programs| programs.kakadu.clone())
    })
}

#[cfg(unix)]
struct TestComparatorProgramsGuard {
    previous: Option<TestComparatorPrograms>,
    _thread_bound: PhantomData<Rc<()>>,
}

#[cfg(unix)]
fn use_test_comparator_programs(openjph: PathBuf, kakadu: PathBuf) -> TestComparatorProgramsGuard {
    let previous = TEST_COMPARATOR_PROGRAMS
        .with(|programs| programs.replace(Some(TestComparatorPrograms { openjph, kakadu })));
    TestComparatorProgramsGuard {
        previous,
        _thread_bound: PhantomData,
    }
}

#[cfg(unix)]
impl Drop for TestComparatorProgramsGuard {
    fn drop(&mut self) {
        TEST_COMPARATOR_PROGRAMS.with(|programs| {
            programs.replace(self.previous.take());
        });
    }
}

fn temp_dir(label: &str) -> PathBuf {
    let token = OPENJPH_TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let root = std::env::temp_dir().join(format!(
        "j2k-fixture-comparator-{label}-{}-{token}",
        std::process::id()
    ));
    fs::create_dir_all(&root).expect("create comparator test directory");
    root
}

#[cfg(unix)]
fn executable(path: &Path, source: &str) {
    fs::write(path, source).expect("write fake comparator executable");
    let mut permissions = fs::metadata(path)
        .expect("fake comparator metadata")
        .permissions();
    permissions.set_mode(0o700);
    fs::set_permissions(path, permissions).expect("make fake comparator executable runnable");
}

#[cfg(unix)]
fn fixture_case(format: PixelFormat, operation: Operation) -> FixtureCase {
    FixtureCase {
        name: "comparator-test".to_string(),
        input_source: "generated:test".to_string(),
        corpus_category: "generated".to_string(),
        corpus_name: "unit".to_string(),
        license_status: "generated".to_string(),
        encode_command: "unit".to_string(),
        manifest_status: "generated".to_string(),
        source_fnv1a64: None,
        codec: Codec::Htj2k,
        container: Container::Jph,
        bytes: Vec::new(),
        dimensions: (2, 1),
        format,
        operation,
    }
}

#[cfg(unix)]
fn argument_values(log: &str, flag: &str) -> Vec<PathBuf> {
    let args = log
        .lines()
        .filter_map(|line| line.strip_prefix("ARG="))
        .collect::<Vec<_>>();
    args.windows(2)
        .filter(|pair| pair[0] == flag)
        .map(|pair| PathBuf::from(pair[1]))
        .collect()
}

#[cfg(unix)]
fn fake_comparator(label: &str) -> (PathBuf, PathBuf, PathBuf) {
    let root = temp_dir(label);
    let log = root.join("arguments.log");
    let program = root.join("expand.sh");
    executable(
        &program,
        &format!(
            r#"#!/bin/sh
input=''
output=''
while [ "$#" -gt 0 ]; do
  printf 'ARG=%s\n' "$1" >> '{}'
  case "$1" in
    -i) shift; printf 'ARG=%s\n' "$1" >> '{}'; input=$1 ;;
    -o) shift; printf 'ARG=%s\n' "$1" >> '{}'; output=$1 ;;
  esac
  shift
done
case "$(cat "$input")" in
  fail)
    printf 'synthetic comparator failure\n' >&2
    exit 7
    ;;
  leak)
    rm "$input"
    mkdir "$input"
    printf 'P5\n2 1\n255\n\001\376' > "$output"
    ;;
  invalid)
    printf 'not a portable anymap' > "$output"
    ;;
  *)
    case "$output" in
      *.pgm) printf 'P5\n2 1\n255\n\001\376' > "$output" ;;
      *.ppm) printf 'P6\n1 1\n255\n\001\002\003' > "$output" ;;
      *.pnm) printf 'P6\n1 1\n255\n\001\002\003' > "$output" ;;
      *) exit 8 ;;
    esac
    ;;
esac
"#,
            log.display(),
            log.display(),
            log.display()
        ),
    );
    (root, log, program)
}

#[test]
fn comparator_extensions_and_reduce_factors_cover_supported_contracts() {
    assert_eq!(openjph_input_extension(Container::RawCodestream), "j2c");
    assert_eq!(openjph_input_extension(Container::Jp2), "jp2");
    assert_eq!(openjph_input_extension(Container::Jph), "jph");
    assert_eq!(openjph_input_extension(Container::Jhc), "jhc");
    assert_eq!(openjph_output_extension(PixelFormat::Gray8), "pgm");
    assert_eq!(openjph_output_extension(PixelFormat::Rgb8), "ppm");
    assert_eq!(openjph_output_extension(PixelFormat::Rgba8), "pnm");
    assert_eq!(reduce_factor(Downscale::None).unwrap(), 0);
    assert_eq!(reduce_factor(Downscale::Half).unwrap(), 1);
    assert_eq!(reduce_factor(Downscale::Quarter).unwrap(), 2);
    assert_eq!(reduce_factor(Downscale::Eighth).unwrap(), 3);
}

#[test]
fn cli_pnm_readback_validates_format_and_cleanup() {
    let directory = openjph_temp_dir().expect("OpenJPH temp dir");
    assert_eq!(
        directory,
        kakadu_temp_dir()
            .expect("Kakadu temp dir")
            .with_file_name("j2k-openjph-expand")
    );
    let token = OPENJPH_TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let gray_path = directory.join(format!("unit-{token}.pgm"));
    let rgb_path = directory.join(format!("unit-{token}.ppm"));
    fs::write(&gray_path, b"P5\n2 1\n255\n\x01\xFE").expect("gray PNM");
    fs::write(&rgb_path, b"P6\n1 1\n255\n\x01\x02\x03").expect("RGB PNM");

    assert_eq!(
        read_cli_pnm_output("unit", &gray_path, PixelFormat::Gray8).unwrap(),
        [1, 254]
    );
    assert_eq!(
        read_cli_pnm_output("unit", &rgb_path, PixelFormat::Rgb8).unwrap(),
        [1, 2, 3]
    );
    assert!(read_cli_pnm_output("unit", &gray_path, PixelFormat::Rgba8)
        .unwrap_err()
        .contains("unsupported"));
    assert!(
        read_cli_pnm_output("unit", &directory.join("missing.pgm"), PixelFormat::Gray8,)
            .unwrap_err()
            .contains("open unit output")
    );

    cleanup_cli_temp(&gray_path, true).expect("clean gray PNM");
    cleanup_cli_temp(&rgb_path, true).expect("clean RGB PNM");
    cleanup_cli_temp(&gray_path, true).expect("missing cleanup is harmless");
    assert!(!command_is_runnable(
        &directory.join("definitely-missing-program")
    ));
}

#[test]
fn staging_cleanup_reports_both_failures_after_attempting_both_paths() {
    let root = temp_dir("cleanup-errors");
    let input = root.join("input");
    let output = root.join("output");
    fs::create_dir(&input).expect("create synthetic input directory");
    fs::create_dir(&output).expect("create synthetic output directory");

    cleanup_cli_staging(&input, &output, false)
        .expect("best-effort cleanup preserves primary error");
    assert!(input.exists());
    assert!(output.exists());
    let error = cleanup_cli_staging(&input, &output, true).expect_err("directory cleanup fails");
    assert!(error.contains(&input.display().to_string()));
    assert!(error.contains(&output.display().to_string()));
    assert!(input.exists());
    assert!(output.exists());

    fs::remove_dir(input).expect("remove synthetic input directory");
    fs::remove_dir(output).expect("remove synthetic output directory");
    fs::remove_dir(root).expect("remove cleanup test directory");
}

#[test]
#[cfg(unix)]
fn comparator_commands_decode_and_report_process_errors() {
    let (root, log, program) = fake_comparator("commands");
    let _programs = use_test_comparator_programs(program.clone(), program.clone());

    let openjph_case = fixture_case(PixelFormat::Gray8, Operation::Scaled(Downscale::Half));
    assert_eq!(
        decode_openjph_once(&openjph_case, b"success").expect("OpenJPH decode"),
        [1, 254]
    );
    let kakadu_case = fixture_case(PixelFormat::Rgb8, Operation::Scaled(Downscale::Quarter));
    assert_eq!(
        decode_kakadu_once(&kakadu_case, b"success").expect("Kakadu decode"),
        [1, 2, 3]
    );
    assert_eq!(openjph_command_label(), program.display().to_string());
    assert_eq!(kakadu_command_label(), program.display().to_string());
    assert_eq!(
        openjph_version_label(),
        "available-version-not-reported-by-ojph_expand"
    );
    assert_eq!(
        kakadu_version_label(),
        "available-version-not-reported-by-kdu_expand"
    );

    let error = decode_openjph_once(&openjph_case, b"fail").expect_err("nonzero comparator");
    assert!(error.contains("exit status: 7"));
    assert!(error.contains("synthetic comparator failure"));
    let error = decode_openjph_once(&openjph_case, b"invalid").expect_err("invalid PNM output");
    assert!(error.contains("decode OpenJPH output"));
    let unsupported_case = fixture_case(PixelFormat::Rgba8, Operation::Full);
    let error =
        decode_openjph_once(&unsupported_case, b"success").expect_err("unsupported readback");
    assert!(error.contains("OpenJPH output format Rgba8 is unsupported"));

    let command_log = fs::read_to_string(&log).expect("comparator argument log");
    assert!(command_log.contains("ARG=-skip_res\nARG=1,1"));
    assert!(command_log.contains("ARG=-reduce\nARG=2"));
    for path in argument_values(&command_log, "-i")
        .iter()
        .chain(argument_values(&command_log, "-o").iter())
    {
        assert!(
            !path.exists(),
            "staged path was not cleaned: {}",
            path.display()
        );
    }
    fs::remove_file(&program).expect("remove fake comparator before spawn failure");
    let error = decode_kakadu_once(&kakadu_case, b"success").expect_err("missing comparator");
    assert!(error.contains("start kdu_expand"));
    fs::remove_dir_all(root).expect("remove comparator test directory");
}

#[test]
#[cfg(unix)]
fn comparator_cleanup_attempts_output_after_input_failure() {
    let (root, log, program) = fake_comparator("cleanup-sequencing");
    let _programs = use_test_comparator_programs(program.clone(), program);
    let openjph_case = fixture_case(PixelFormat::Gray8, Operation::Scaled(Downscale::Half));
    let error = decode_openjph_once(&openjph_case, b"leak").expect_err("input cleanup failure");
    assert!(error.contains("remove temp file"));
    let command_log = fs::read_to_string(&log).expect("comparator argument log");
    assert!(command_log.contains("ARG=-skip_res\nARG=1,1"));
    let inputs = argument_values(&command_log, "-i");
    let outputs = argument_values(&command_log, "-o");
    let leaked_input = inputs.last().expect("leak input");
    let leaked_output = outputs.last().expect("leak output");
    let output_was_left_behind = leaked_output.exists();
    fs::remove_dir(leaked_input).expect("remove synthetic input directory");
    if output_was_left_behind {
        fs::remove_file(leaked_output).expect("remove leaked output after regression check");
    }
    assert!(
        !output_was_left_behind,
        "output cleanup must still run after input cleanup fails"
    );
    fs::remove_dir_all(root).expect("remove comparator test directory");
}
