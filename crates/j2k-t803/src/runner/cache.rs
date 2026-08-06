// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{
    fs::{self, File},
    path::{Path, PathBuf},
    process::Command,
};

use crate::T803Manifest;

use super::archive::{extract_selected_archive, sha256_file, verify_corpus, ArchiveLimits};

const ARCHIVE_NAME: &str = "t803-v3.zip";

pub(super) fn load_manifest() -> Result<T803Manifest, String> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("corpus/j2k-conformance/t803-v3.toml");
    let text =
        fs::read_to_string(&path).map_err(|error| format!("read {}: {error}", path.display()))?;
    T803Manifest::parse(&text).map_err(|error| error.to_string())
}

pub(super) fn archive_path(cache_dir: &Path) -> PathBuf {
    cache_dir.join(ARCHIVE_NAME)
}

pub(super) fn corpus_path(cache_dir: &Path, manifest: &T803Manifest) -> PathBuf {
    cache_dir.join(format!("corpus-{}", manifest.source.archive_sha256))
}

pub(super) fn verify_cached(cache_dir: &Path) -> Result<(T803Manifest, PathBuf), String> {
    let manifest = load_manifest()?;
    let archive = archive_path(cache_dir);
    if !archive.is_file() {
        return Err(format!(
            "pinned T.803 archive is absent at {}; run `cargo xtask t803 fetch`",
            archive.display()
        ));
    }
    verify_archive(&archive, &manifest)?;
    let corpus = corpus_path(cache_dir, &manifest);
    if !corpus.is_dir() {
        return Err(format!(
            "verified T.803 corpus is absent at {}; run `cargo xtask t803 fetch`",
            corpus.display()
        ));
    }
    verify_corpus(&corpus, &manifest.files).map_err(|error| error.to_string())?;
    Ok((manifest, corpus))
}

pub(super) fn fetch(cache_dir: &Path) -> Result<(), String> {
    let manifest = load_manifest()?;
    fs::create_dir_all(cache_dir)
        .map_err(|error| format!("create {}: {error}", cache_dir.display()))?;
    let archive = archive_path(cache_dir);
    if archive.exists() {
        verify_archive(&archive, &manifest)?;
    } else {
        download_archive(cache_dir, &archive, &manifest)?;
    }

    let corpus = corpus_path(cache_dir, &manifest);
    if corpus.exists() {
        verify_corpus(&corpus, &manifest.files).map_err(|error| error.to_string())?;
        return Ok(());
    }
    let staging = cache_dir.join(format!(
        ".corpus-{}-extracting-{}",
        manifest.source.archive_sha256,
        std::process::id()
    ));
    if staging.exists() {
        return Err(format!(
            "stale T.803 extraction directory exists at {}",
            staging.display()
        ));
    }
    fs::create_dir(&staging).map_err(|error| format!("create {}: {error}", staging.display()))?;
    let extraction = File::open(&archive)
        .map_err(|error| format!("open {}: {error}", archive.display()))
        .and_then(|file| {
            extract_selected_archive(file, &staging, &manifest.files, ArchiveLimits::default())
                .map_err(|error| error.to_string())
        });
    if let Err(error) = extraction {
        let _ = fs::remove_dir_all(&staging);
        return Err(error);
    }
    fs::rename(&staging, &corpus).map_err(|error| {
        format!(
            "publish extracted corpus {} as {}: {error}",
            staging.display(),
            corpus.display()
        )
    })?;
    verify_corpus(&corpus, &manifest.files).map_err(|error| error.to_string())
}

fn verify_archive(path: &Path, manifest: &T803Manifest) -> Result<(), String> {
    let metadata =
        fs::metadata(path).map_err(|error| format!("inspect {}: {error}", path.display()))?;
    if !metadata.is_file() {
        return Err(format!(
            "T.803 archive {} is not a regular file",
            path.display()
        ));
    }
    if metadata.len() != manifest.source.archive_bytes {
        return Err(format!(
            "T.803 archive size is {}, expected {}",
            metadata.len(),
            manifest.source.archive_bytes
        ));
    }
    let observed = sha256_file(path).map_err(|error| error.to_string())?;
    if observed != manifest.source.archive_sha256 {
        return Err(format!(
            "T.803 archive SHA-256 is {observed}, expected {}",
            manifest.source.archive_sha256
        ));
    }
    Ok(())
}

fn download_archive(
    cache_dir: &Path,
    destination: &Path,
    manifest: &T803Manifest,
) -> Result<(), String> {
    let partial = cache_dir.join(format!(".{ARCHIVE_NAME}.download-{}", std::process::id()));
    let headers = cache_dir.join(format!(".{ARCHIVE_NAME}.headers-{}", std::process::id()));
    for path in [&partial, &headers] {
        if path.exists() {
            return Err(format!(
                "stale T.803 download file exists at {}",
                path.display()
            ));
        }
    }
    let output = Command::new("curl")
        .args([
            "--fail",
            "--location",
            "--silent",
            "--show-error",
            "--proto",
            "=https",
            "--max-redirs",
            "5",
            "--dump-header",
        ])
        .arg(&headers)
        .arg("--output")
        .arg(&partial)
        .arg("--write-out")
        .arg("%{url_effective}")
        .arg(&manifest.source.url)
        .output()
        .map_err(|error| format!("start curl for official T.803 attachment: {error}"))?;
    if !output.status.success() {
        cleanup_download(&partial, &headers);
        return Err(format!(
            "official T.803 attachment download failed with {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let effective_url = String::from_utf8(output.stdout)
        .map_err(|error| format!("curl returned a non-UTF-8 effective URL: {error}"))?;
    let header_text = fs::read_to_string(&headers)
        .map_err(|error| format!("read {}: {error}", headers.display()))?;
    if let Err(error) = validate_redirects(&header_text, effective_url.trim()) {
        cleanup_download(&partial, &headers);
        return Err(error);
    }
    if let Err(error) = verify_archive(&partial, manifest) {
        cleanup_download(&partial, &headers);
        return Err(error);
    }
    fs::rename(&partial, destination).map_err(|error| {
        format!(
            "publish verified T.803 archive {} as {}: {error}",
            partial.display(),
            destination.display()
        )
    })?;
    let _ = fs::remove_file(headers);
    Ok(())
}

fn validate_redirects(headers: &str, effective_url: &str) -> Result<(), String> {
    if !approved_itu_https_url(effective_url) {
        return Err(format!(
            "T.803 download ended outside approved ITU domains: {effective_url:?}"
        ));
    }
    for line in headers.lines() {
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        if name.eq_ignore_ascii_case("location") {
            let location = value.trim();
            if is_absolute_or_network_url(location) && !approved_itu_redirect(location) {
                return Err(format!(
                    "T.803 download redirects outside approved ITU domains: {location:?}"
                ));
            }
        }
    }
    Ok(())
}

fn approved_itu_https_url(url: &str) -> bool {
    let Some(authority_and_path) = url.strip_prefix("https://") else {
        return false;
    };
    approved_itu_authority(authority_and_path)
}

fn approved_itu_redirect(url: &str) -> bool {
    url.strip_prefix("https://")
        .or_else(|| url.strip_prefix("//"))
        .is_some_and(approved_itu_authority)
}

fn approved_itu_authority(authority_and_path: &str) -> bool {
    let authority = authority_and_path.split('/').next().unwrap_or_default();
    matches!(authority, "handle.itu.int" | "www.itu.int")
}

fn is_absolute_or_network_url(url: &str) -> bool {
    url.contains("://") || url.starts_with("//")
}

fn cleanup_download(partial: &Path, headers: &Path) {
    let _ = fs::remove_file(partial);
    let _ = fs::remove_file(headers);
}

#[cfg(test)]
mod tests {
    use super::validate_redirects;

    #[test]
    fn redirect_validation_accepts_only_itu_https_targets() {
        validate_redirects(
            "HTTP/2 302\r\nlocation: //www.itu.int/rec/T-REC-T.803\r\n",
            "https://www.itu.int/rec/T-REC-T.803",
        )
        .expect("approved protocol-relative redirect");
        validate_redirects(
            "HTTP/2 200\r\n",
            "https://www.itu.int/wftp3/public/t/testsignal/SpeImage/T803/v2024_02/T.803v3_15444-4ed4-ElecAtt-codestreams.zip",
        )
        .expect("official test-signal attachment URL");

        for target in [
            "http://www.itu.int/attachment.zip",
            "https://www.itu.int.example/attachment.zip",
            "//example.test/attachment.zip",
        ] {
            let headers = format!("HTTP/2 302\r\nlocation: {target}\r\n");
            let error = validate_redirects(&headers, "https://www.itu.int/final")
                .expect_err("unapproved redirect must fail");
            assert!(error.contains("outside approved ITU domains"));
        }
    }
}
