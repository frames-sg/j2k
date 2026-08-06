// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::Write as _,
    fs::{self, File, OpenOptions},
    io::{Read, Seek},
    path::{Component, Path, PathBuf},
};

use sha2::{Digest, Sha256};
use thiserror::Error;
use zip::ZipArchive;

use crate::CorpusFile;

/// Resource limits applied before extracting an external T.803 archive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArchiveLimits {
    /// Maximum number of ZIP entries, including directories.
    pub max_entries: usize,
    /// Maximum uncompressed size of one entry.
    pub max_entry_bytes: u64,
    /// Maximum aggregate uncompressed size.
    pub max_total_bytes: u64,
}

impl Default for ArchiveLimits {
    fn default() -> Self {
        Self {
            max_entries: 2_048,
            max_entry_bytes: 64 * 1024 * 1024,
            max_total_bytes: 512 * 1024 * 1024,
        }
    }
}

/// Error returned by fail-closed corpus acquisition and validation.
#[derive(Debug, Error)]
pub enum RunnerError {
    /// Filesystem operation failed.
    #[error("{operation} {}: {source}", path.display())]
    Io {
        /// Operation being attempted.
        operation: &'static str,
        /// Affected path.
        path: PathBuf,
        /// Underlying I/O failure.
        #[source]
        source: std::io::Error,
    },
    /// ZIP structure or decompression failed.
    #[error("invalid T.803 ZIP archive: {0}")]
    Zip(#[from] zip::result::ZipError),
    /// Archive or extracted inventory violated the pinned contract.
    #[error("invalid T.803 corpus: {0}")]
    Validation(String),
}

/// Validate an archive completely, then extract only the pinned files.
pub fn extract_selected_archive<R: Read + Seek>(
    reader: R,
    output: &Path,
    required: &[CorpusFile],
    limits: ArchiveLimits,
) -> Result<(), RunnerError> {
    ensure_empty_directory(output)?;
    let mut archive = ZipArchive::new(reader)?;
    if archive.len() > limits.max_entries {
        return validation(format!(
            "archive contains {} entries, limit is {}",
            archive.len(),
            limits.max_entries
        ));
    }

    let required_paths = required
        .iter()
        .map(|file| (file.path.as_str(), file))
        .collect::<BTreeMap<_, _>>();
    let mut entries = BTreeMap::new();
    let mut seen = BTreeSet::new();
    let mut total_bytes = 0_u64;
    for index in 0..archive.len() {
        let mut entry = archive.by_index(index)?;
        let name = validate_entry_name(&entry)?;
        if !seen.insert(name.clone()) {
            return validation(format!("archive contains duplicate entry {name:?}"));
        }
        if entry.is_symlink() {
            return validation(format!("archive entry {name:?} is a symlink"));
        }
        if entry.encrypted() {
            return validation(format!("archive entry {name:?} is encrypted"));
        }
        if entry.size() > limits.max_entry_bytes {
            return validation(format!("archive entry {name:?} is too large"));
        }
        total_bytes = total_bytes
            .checked_add(entry.size())
            .ok_or_else(|| RunnerError::Validation("archive size overflow".to_string()))?;
        if total_bytes > limits.max_total_bytes {
            return validation("archive uncompressed contents are too large");
        }

        if let Some(required_file) = required_paths.get(name.as_str()) {
            if entry.is_dir() {
                return validation(format!("required file {name:?} is a directory"));
            }
            let observed = sha256_reader(&mut entry)?;
            if observed != required_file.sha256 {
                return validation(format!(
                    "required file {name:?} SHA-256 is {observed}, expected {}",
                    required_file.sha256
                ));
            }
            entries.insert(name, index);
        }
    }

    for file in required {
        if !entries.contains_key(&file.path) {
            return validation(format!("required file {:?} is missing", file.path));
        }
    }

    for file in required {
        let index = entries[&file.path];
        let mut entry = archive.by_index(index)?;
        let destination = output.join(path_from_manifest(&file.path)?);
        let parent = destination.parent().ok_or_else(|| {
            RunnerError::Validation(format!("required file {:?} has no parent", file.path))
        })?;
        fs::create_dir_all(parent).map_err(|source| RunnerError::Io {
            operation: "create corpus directory",
            path: parent.to_path_buf(),
            source,
        })?;
        let mut destination_file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&destination)
            .map_err(|source| RunnerError::Io {
                operation: "create extracted corpus file",
                path: destination.clone(),
                source,
            })?;
        std::io::copy(&mut entry, &mut destination_file).map_err(|source| RunnerError::Io {
            operation: "extract corpus file",
            path: destination,
            source,
        })?;
    }
    verify_corpus(output, required)
}

/// Verify hashes and require the extracted tree to contain no extra files.
pub fn verify_corpus(root: &Path, required: &[CorpusFile]) -> Result<(), RunnerError> {
    let mut observed = BTreeSet::new();
    collect_files(root, root, &mut observed)?;
    let expected = required
        .iter()
        .map(|file| file.path.as_str())
        .collect::<BTreeSet<_>>();
    let observed_refs = observed.iter().map(String::as_str).collect::<BTreeSet<_>>();
    if observed_refs != expected {
        let missing = expected
            .difference(&observed_refs)
            .copied()
            .collect::<Vec<_>>();
        let extra = observed_refs
            .difference(&expected)
            .copied()
            .collect::<Vec<_>>();
        return validation(format!(
            "extracted inventory mismatch; missing {missing:?}, extra {extra:?}"
        ));
    }

    for file in required {
        let path = root.join(path_from_manifest(&file.path)?);
        let observed_hash = sha256_file(&path)?;
        if observed_hash != file.sha256 {
            return validation(format!(
                "{} SHA-256 is {observed_hash}, expected {}",
                file.path, file.sha256
            ));
        }
    }
    Ok(())
}

fn ensure_empty_directory(path: &Path) -> Result<(), RunnerError> {
    let mut entries = fs::read_dir(path).map_err(|source| RunnerError::Io {
        operation: "read extraction directory",
        path: path.to_path_buf(),
        source,
    })?;
    if entries
        .next()
        .transpose()
        .map_err(|source| RunnerError::Io {
            operation: "read extraction directory entry",
            path: path.to_path_buf(),
            source,
        })?
        .is_some()
    {
        return validation(format!(
            "extraction directory {} is not empty",
            path.display()
        ));
    }
    Ok(())
}

fn validate_entry_name<R: Read + ?Sized>(
    entry: &zip::read::ZipFile<'_, R>,
) -> Result<String, RunnerError> {
    let name = entry.name();
    if name.contains('\\') || entry.enclosed_name().is_none() {
        return validation(format!("archive entry {name:?} has an unsafe path"));
    }
    let normalized = name.strip_suffix('/').unwrap_or(name);
    if normalized.is_empty() {
        return validation("archive contains an empty path");
    }
    let path = Path::new(normalized);
    if path
        .components()
        .any(|component| !matches!(component, Component::Normal(_)))
    {
        return validation(format!("archive entry {name:?} has an unsafe path"));
    }
    manifest_path(path)
}

fn path_from_manifest(path: &str) -> Result<PathBuf, RunnerError> {
    if path.is_empty() || path.contains('\\') {
        return validation(format!("manifest path {path:?} is invalid"));
    }
    let parsed = Path::new(path);
    if parsed
        .components()
        .any(|component| !matches!(component, Component::Normal(_)))
    {
        return validation(format!("manifest path {path:?} is invalid"));
    }
    Ok(parsed.to_path_buf())
}

fn collect_files(
    root: &Path,
    directory: &Path,
    files: &mut BTreeSet<String>,
) -> Result<(), RunnerError> {
    let entries = fs::read_dir(directory).map_err(|source| RunnerError::Io {
        operation: "read extracted corpus directory",
        path: directory.to_path_buf(),
        source,
    })?;
    for entry in entries {
        let entry = entry.map_err(|source| RunnerError::Io {
            operation: "read extracted corpus entry",
            path: directory.to_path_buf(),
            source,
        })?;
        let path = entry.path();
        let metadata = fs::symlink_metadata(&path).map_err(|source| RunnerError::Io {
            operation: "inspect extracted corpus entry",
            path: path.clone(),
            source,
        })?;
        if metadata.file_type().is_symlink() {
            return validation(format!("extracted entry {} is a symlink", path.display()));
        }
        if metadata.is_dir() {
            collect_files(root, &path, files)?;
        } else if metadata.is_file() {
            let relative = path.strip_prefix(root).map_err(|_| {
                RunnerError::Validation("extracted file escaped corpus root".to_string())
            })?;
            files.insert(manifest_path(relative)?);
        } else {
            return validation(format!(
                "extracted entry {} is not a regular file",
                path.display()
            ));
        }
    }
    Ok(())
}

fn manifest_path(path: &Path) -> Result<String, RunnerError> {
    path.components()
        .map(|component| match component {
            Component::Normal(value) => value
                .to_str()
                .map(str::to_string)
                .ok_or_else(|| RunnerError::Validation("corpus path is not UTF-8".to_string())),
            _ => validation("corpus path is not relative and normalized"),
        })
        .collect::<Result<Vec<_>, _>>()
        .map(|components| components.join("/"))
}

pub(super) fn sha256_file(path: &Path) -> Result<String, RunnerError> {
    let mut file = File::open(path).map_err(|source| RunnerError::Io {
        operation: "open corpus file",
        path: path.to_path_buf(),
        source,
    })?;
    sha256_reader(&mut file)
}

fn sha256_reader(reader: &mut impl Read) -> Result<String, RunnerError> {
    let mut hasher = Sha256::new();
    let mut buffer = Vec::new();
    buffer
        .try_reserve_exact(64 * 1024)
        .map_err(|_| RunnerError::Validation("cannot allocate hash buffer".to_string()))?;
    buffer.resize(64 * 1024, 0);
    loop {
        let count = reader.read(&mut buffer).map_err(|source| RunnerError::Io {
            operation: "hash corpus data",
            path: PathBuf::from("<stream>"),
            source,
        })?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    let digest = hasher.finalize();
    let mut hex = String::new();
    hex.try_reserve_exact(64)
        .map_err(|_| RunnerError::Validation("cannot allocate SHA-256 text".to_string()))?;
    for byte in digest {
        let _ = write!(hex, "{byte:02x}");
    }
    Ok(hex)
}

fn validation<T>(message: impl Into<String>) -> Result<T, RunnerError> {
    Err(RunnerError::Validation(message.into()))
}
