// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{io::Cursor, path::PathBuf};

use super::integrity::complete_publishable_metadata;

pub(super) fn packaged_metadata() -> (serde_json::Value, PathBuf) {
    let mut metadata = complete_publishable_metadata();
    let packages = metadata["packages"]
        .as_array()
        .expect("package records")
        .iter()
        .filter_map(|package| {
            Some((
                package["name"].as_str()?.to_string(),
                package["version"].as_str()?.to_string(),
            ))
        })
        .collect::<Vec<_>>();
    let target =
        std::env::temp_dir().join(format!("j2k-release-package-target-{}", std::process::id()));
    metadata["target_directory"] = serde_json::Value::String(target.to_string_lossy().into_owned());
    let package_dir = target.join("package");
    std::fs::create_dir_all(&package_dir).expect("create package fixture directory");
    for (package, version) in packages {
        write_package_archive(&package_dir, &package, &version);
    }
    (metadata, target)
}

fn write_package_archive(package_dir: &std::path::Path, package: &str, version: &str) {
    let archive_path = package_dir.join(format!("{package}-{version}.crate"));
    let file = std::fs::File::create(archive_path).expect("create package fixture");
    let encoder = flate2::write::GzEncoder::new(file, flate2::Compression::default());
    let mut archive = tar::Builder::new(encoder);
    let contents = format!("[package]\nname = \"{package}\"\nversion = \"{version}\"\n");
    let mut header = tar::Header::new_gnu();
    header.set_size(contents.len() as u64);
    header.set_mode(0o644);
    header.set_cksum();
    archive
        .append_data(
            &mut header,
            format!("{package}-{version}/Cargo.toml"),
            Cursor::new(contents),
        )
        .expect("append package fixture");
    archive.finish().expect("finish package fixture");
}
