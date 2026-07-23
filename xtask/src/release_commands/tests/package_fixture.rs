// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{io::Cursor, path::PathBuf};

use super::integrity::complete_publishable_metadata;

pub(super) fn packaged_metadata() -> (serde_json::Value, PathBuf) {
    let mut metadata = complete_publishable_metadata();
    let version = metadata["packages"]
        .as_array()
        .expect("package records")
        .iter()
        .find(|package| package["name"] == "j2k-ml")
        .and_then(|package| package["version"].as_str())
        .expect("j2k-ml package version")
        .to_string();
    let target =
        std::env::temp_dir().join(format!("j2k-release-package-target-{}", std::process::id()));
    metadata["target_directory"] = serde_json::Value::String(target.to_string_lossy().into_owned());
    let archive_path = target
        .join("package")
        .join(format!("j2k-ml-{version}.crate"));
    std::fs::create_dir_all(archive_path.parent().expect("package fixture parent"))
        .expect("create package fixture directory");
    let file = std::fs::File::create(archive_path).expect("create package fixture");
    let encoder = flate2::write::GzEncoder::new(file, flate2::Compression::default());
    let mut archive = tar::Builder::new(encoder);
    let contents = format!("[package]\nname = \"j2k-ml\"\nversion = \"{version}\"\n");
    let mut header = tar::Header::new_gnu();
    header.set_size(contents.len() as u64);
    header.set_mode(0o644);
    header.set_cksum();
    archive
        .append_data(
            &mut header,
            format!("j2k-ml-{version}/Cargo.toml"),
            Cursor::new(contents),
        )
        .expect("append package fixture");
    archive.finish().expect("finish package fixture");
    (metadata, target)
}
