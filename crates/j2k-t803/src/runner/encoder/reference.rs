// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{
    ffi::OsStr,
    fs,
    path::{Path, PathBuf},
    process::Command,
    sync::atomic::{AtomicU64, Ordering},
};

use j2k_compare::openjpeg::{OpenJpegDecodedComponent, OpenJpegDecodedImage};

use crate::{parse_pgx, EncoderSupplementalReferenceIdentity};

use super::input::GeneratedInput;
use crate::encoder::EncoderCase;
use crate::runner::archive::sha256_file;

pub(crate) const OPENHTJ2K_IMPLEMENTATION: &str = "OpenHTJ2K";
pub(crate) const OPENHTJ2K_SCOPE: &str = "independent Part 15 interoperability evidence; not T.804";
pub(crate) const OPENHTJ2K_SOURCE_COMMIT: &str = "e0f7ae853220d1e359c438b0bb6ad6cb2b3899db";
pub(crate) const OPENHTJ2K_SOURCE_URL: &str = "https://github.com/osamu620/OpenHTJ2K";
pub(crate) const OPENHTJ2K_VERSION: &str = "0.19.0";

const EXECUTABLE_ENV: &str = "J2K_OPENHTJ2K_DEC_BIN";
const SOURCE_ENV: &str = "J2K_OPENHTJ2K_SOURCE_DIR";
static RUN_SEQUENCE: AtomicU64 = AtomicU64::new(0);

pub(in crate::runner) fn verify_report_identity(
    iut_name: &str,
    selected: bool,
    identities: &[EncoderSupplementalReferenceIdentity],
) -> Result<(), String> {
    match identities {
        [identity] if selected => {
            if identity.decoder != crate::EncoderReferenceDecoder::OpenHtj2k
                || identity.scope != OPENHTJ2K_SCOPE
                || identity.implementation != OPENHTJ2K_IMPLEMENTATION
                || identity.version != OPENHTJ2K_VERSION
                || identity.source_url != OPENHTJ2K_SOURCE_URL
                || identity.source_commit != OPENHTJ2K_SOURCE_COMMIT
            {
                return Err(format!(
                    "{iut_name} encoder supplemental decoder is not the pinned OpenHTJ2K build"
                ));
            }
            Ok(())
        }
        [] if !selected => Ok(()),
        _ => Err(format!(
            "{iut_name} encoder supplemental decoder inventory differs from the matrix"
        )),
    }
}

pub(super) struct OpenHtj2kDecoder {
    executable: PathBuf,
    identity: EncoderSupplementalReferenceIdentity,
}

impl OpenHtj2kDecoder {
    pub(super) fn from_environment() -> Result<Self, String> {
        let executable = std::env::var_os(EXECUTABLE_ENV)
            .ok_or_else(|| format!("{EXECUTABLE_ENV} is required by the HT+RGN matrix case"))?;
        let source = std::env::var_os(SOURCE_ENV)
            .ok_or_else(|| format!("{SOURCE_ENV} is required by the HT+RGN matrix case"))?;
        Self::from_paths(Path::new(&executable), Path::new(&source))
    }

    pub(super) fn identity(&self) -> EncoderSupplementalReferenceIdentity {
        self.identity.clone()
    }

    pub(super) fn decode_components(
        &self,
        codestream: &[u8],
        case: &EncoderCase,
        expected: &GeneratedInput,
    ) -> Result<OpenJpegDecodedImage, String> {
        let work = ReferenceWorkDir::create()?;
        let input_path = work.path.join("input.j2c");
        let output_path = work.path.join("decoded.pgx");
        fs::write(&input_path, codestream)
            .map_err(|error| format!("write {}: {error}", input_path.display()))?;
        let output = Command::new(&self.executable)
            .args([OsStr::new("-i"), input_path.as_os_str()])
            .args([OsStr::new("-o"), output_path.as_os_str()])
            .args([OsStr::new("-num_threads"), OsStr::new("1")])
            .output()
            .map_err(|error| format!("start OpenHTJ2K decoder: {error}"))?;
        if !output.status.success() {
            return Err(format!(
                "OpenHTJ2K decoder exited with {}: {}",
                output.status,
                command_diagnostic(&output.stdout, &output.stderr)
            ));
        }

        let mut components = Vec::new();
        components
            .try_reserve_exact(expected.components.len())
            .map_err(|_| "allocate OpenHTJ2K component results".to_string())?;
        for (index, expected_component) in expected.components.iter().enumerate() {
            let path = work.path.join(format!("decoded_{index:02}.pgx"));
            let maximum_bytes = expected_component
                .samples
                .len()
                .checked_mul(4)
                .and_then(|bytes| bytes.checked_add(256))
                .ok_or_else(|| "OpenHTJ2K output size bound overflow".to_string())?;
            let metadata = fs::metadata(&path)
                .map_err(|error| format!("inspect {}: {error}", path.display()))?;
            if !metadata.is_file()
                || usize::try_from(metadata.len()).map_or(true, |bytes| bytes > maximum_bytes)
            {
                return Err(format!(
                    "OpenHTJ2K component {} exceeds its expected output bound",
                    path.display()
                ));
            }
            let pgx_bytes =
                fs::read(&path).map_err(|error| format!("read {}: {error}", path.display()))?;
            let pgx = parse_pgx(&pgx_bytes)
                .map_err(|error| format!("parse {}: {error}", path.display()))?;
            let samples = pgx
                .samples
                .into_iter()
                .map(|sample| {
                    i32::try_from(sample)
                        .map_err(|_| "OpenHTJ2K PGX sample exceeds the i32 evidence range")
                })
                .collect::<Result<Vec<_>, _>>()
                .map_err(str::to_string)?;
            components.push(OpenJpegDecodedComponent {
                dimensions: (pgx.width, pgx.height),
                sampling: (
                    infer_sampling(case.width, pgx.width)?,
                    infer_sampling(case.height, pgx.height)?,
                ),
                bit_depth: pgx.bit_depth,
                signed: pgx.signed,
                samples,
            });
        }
        let unexpected = work
            .path
            .join(format!("decoded_{:02}.pgx", expected.components.len()));
        if unexpected.exists() {
            return Err("OpenHTJ2K returned more components than the encoder input".to_string());
        }
        let first = components
            .first()
            .ok_or_else(|| "OpenHTJ2K returned no components".to_string())?;
        if first.sampling != (1, 1) {
            return Err(
                "OpenHTJ2K PGX output does not identify a full-grid first component".to_string(),
            );
        }
        Ok(OpenJpegDecodedImage {
            dimensions: first.dimensions,
            components,
        })
    }

    fn from_paths(executable: &Path, source: &Path) -> Result<Self, String> {
        let executable = executable
            .canonicalize()
            .map_err(|error| format!("resolve {}: {error}", executable.display()))?;
        let source = source
            .canonicalize()
            .map_err(|error| format!("resolve {}: {error}", source.display()))?;
        if !executable.is_file() || !source.is_dir() {
            return Err("OpenHTJ2K executable or source checkout has the wrong type".to_string());
        }
        let commit = git_output(&source, &["rev-parse", "HEAD"])?;
        let tag = git_output(&source, &["describe", "--tags", "--exact-match", "HEAD"])?;
        let remote = git_output(&source, &["remote", "get-url", "origin"])?;
        let dirty = git_output(&source, &["status", "--porcelain", "--untracked-files=no"])?;
        if commit != OPENHTJ2K_SOURCE_COMMIT
            || tag != format!("v{OPENHTJ2K_VERSION}")
            || !matches!(
                remote.as_str(),
                "https://github.com/osamu620/OpenHTJ2K"
                    | "https://github.com/osamu620/OpenHTJ2K.git"
            )
            || !dirty.is_empty()
        {
            return Err(
                "OpenHTJ2K source is not the clean, pinned official v0.19.0 checkout".to_string(),
            );
        }
        let executable_sha256 = sha256_file(&executable).map_err(|error| error.to_string())?;
        Ok(Self {
            executable,
            identity: EncoderSupplementalReferenceIdentity {
                decoder: crate::EncoderReferenceDecoder::OpenHtj2k,
                scope: OPENHTJ2K_SCOPE.to_string(),
                implementation: OPENHTJ2K_IMPLEMENTATION.to_string(),
                version: OPENHTJ2K_VERSION.to_string(),
                source_url: OPENHTJ2K_SOURCE_URL.to_string(),
                source_commit: OPENHTJ2K_SOURCE_COMMIT.to_string(),
                executable_sha256,
            },
        })
    }
}

struct ReferenceWorkDir {
    path: PathBuf,
}

impl ReferenceWorkDir {
    fn create() -> Result<Self, String> {
        let sequence = RUN_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "j2k-t803-openhtj2k-{}-{sequence}",
            std::process::id()
        ));
        fs::create_dir(&path).map_err(|error| format!("create {}: {error}", path.display()))?;
        Ok(Self { path })
    }
}

impl Drop for ReferenceWorkDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn git_output(source: &Path, args: &[&str]) -> Result<String, String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(source)
        .args(args)
        .output()
        .map_err(|error| format!("start git {}: {error}", args.join(" ")))?;
    if !output.status.success() {
        return Err(format!(
            "git {} exited with {}: {}",
            args.join(" "),
            output.status,
            command_diagnostic(&output.stdout, &output.stderr)
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn infer_sampling(reference: u32, component: u32) -> Result<u32, String> {
    (1..=255)
        .filter(|sampling| reference.div_ceil(*sampling) == component)
        .reduce(|_, _| 0)
        .filter(|sampling| *sampling != 0)
        .ok_or_else(|| {
            format!(
                "OpenHTJ2K PGX dimensions do not determine one sampling factor for {reference}/{component}"
            )
        })
}

fn command_diagnostic(stdout: &[u8], stderr: &[u8]) -> String {
    let combined = format!(
        "{} {}",
        String::from_utf8_lossy(stdout).trim(),
        String::from_utf8_lossy(stderr).trim()
    );
    combined.chars().take(2_048).collect()
}
