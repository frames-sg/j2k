// SPDX-License-Identifier: MIT OR Apache-2.0

use std::path::PathBuf;

use crate::T803Suite;

#[cfg(feature = "cuda-runner")]
use super::cuda;
#[cfg(feature = "metal-runner")]
use super::metal;
use super::{cache, cpu, evidence};

const DEFAULT_CACHE_DIR: &str = "target/t803";

pub fn run_cli(args: impl IntoIterator<Item = String>) -> Result<(), String> {
    let mut args = args.into_iter();
    let command = args.next().ok_or_else(usage)?;
    let options = parse_options(args)?;
    match command.as_str() {
        "fetch" => cache::fetch(&options.cache_dir),
        "run" => {
            let iut = options
                .iut
                .as_deref()
                .ok_or_else(|| "t803 run requires --iut cpu|cuda|metal".to_string())?;
            match iut {
                "cpu" => cpu::run(
                    &options.cache_dir,
                    options.output_dir,
                    options.development,
                    options.suite,
                ),
                "cuda" => run_cuda(&options),
                "metal" => run_metal(&options),
                _ => Err(format!("unknown T.803 IUT {iut:?}")),
            }
        }
        "verify" => {
            evidence::verify_reports(
                &options.cache_dir,
                &options.reports,
                options.candidate_sha.as_deref(),
                evidence::EvidenceScope::parse(options.scope.as_deref().ok_or_else(|| {
                    "t803 verify requires --scope cpu|cuda|metal|all".to_string()
                })?)?,
            )
        }
        "help" | "-h" | "--help" => Err(usage()),
        other => Err(format!("unknown T.803 command {other:?}\n{}", usage())),
    }
}

#[cfg(feature = "cuda-runner")]
fn run_cuda(options: &Options) -> Result<(), String> {
    cuda::run(
        &options.cache_dir,
        options.output_dir.clone(),
        options.development,
        options.suite,
    )
}

#[cfg(not(feature = "cuda-runner"))]
fn run_cuda(_options: &Options) -> Result<(), String> {
    Err("cuda T.803 adapter runner is not available in this build".to_string())
}

#[cfg(feature = "metal-runner")]
fn run_metal(options: &Options) -> Result<(), String> {
    metal::run(
        &options.cache_dir,
        options.output_dir.clone(),
        options.development,
        options.suite,
    )
}

#[cfg(not(feature = "metal-runner"))]
fn run_metal(_options: &Options) -> Result<(), String> {
    Err("metal T.803 adapter runner is not available in this build".to_string())
}

#[derive(Debug)]
struct Options {
    cache_dir: PathBuf,
    output_dir: Option<PathBuf>,
    iut: Option<String>,
    development: bool,
    reports: Vec<PathBuf>,
    candidate_sha: Option<String>,
    scope: Option<String>,
    suite: T803Suite,
}

fn parse_options(args: impl IntoIterator<Item = String>) -> Result<Options, String> {
    let mut options = Options {
        cache_dir: PathBuf::from(DEFAULT_CACHE_DIR),
        output_dir: None,
        iut: None,
        development: false,
        reports: Vec::new(),
        candidate_sha: None,
        scope: None,
        suite: T803Suite::All,
    };
    let mut args = args.into_iter();
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--cache-dir" => options.cache_dir = PathBuf::from(next_value(&mut args, &argument)?),
            "--out-dir" => {
                options.output_dir = Some(PathBuf::from(next_value(&mut args, &argument)?));
            }
            "--iut" => options.iut = Some(next_value(&mut args, &argument)?),
            "--development" => options.development = true,
            "--report" => options
                .reports
                .push(PathBuf::from(next_value(&mut args, &argument)?)),
            "--candidate-sha" => options.candidate_sha = Some(next_value(&mut args, &argument)?),
            "--scope" => options.scope = Some(next_value(&mut args, &argument)?),
            "--suite" => {
                options.suite = match next_value(&mut args, &argument)?.as_str() {
                    "part1" => T803Suite::Part1,
                    "part15" => T803Suite::Part15,
                    "all" => T803Suite::All,
                    other => {
                        return Err(format!(
                            "unknown T.803 suite {other:?}; expected part1|part15|all"
                        ));
                    }
                };
            }
            "-h" | "--help" => return Err(usage()),
            other => return Err(format!("unknown T.803 argument {other:?}\n{}", usage())),
        }
    }
    Ok(options)
}

fn next_value(args: &mut impl Iterator<Item = String>, option: &str) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("{option} requires a value"))
}

fn usage() -> String {
    "usage: cargo xtask t803 fetch [--cache-dir DIR]\n       cargo xtask t803 run --iut cpu|cuda|metal [--suite part1|part15|all] [--out-dir DIR] [--development] [--cache-dir DIR]\n       cargo xtask t803 verify --scope cpu|cuda|metal|all --candidate-sha SHA --report FILE [--report FILE...] [--cache-dir DIR]".to_string()
}
