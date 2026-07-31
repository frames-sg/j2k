// SPDX-License-Identifier: MIT OR Apache-2.0

//! Native-backend isolation for collected public API inventories.

use super::PackageApiInventory;

pub(super) fn validate(package: &str, inventory: &PackageApiInventory) -> Result<(), String> {
    if package == "j2k-native" {
        return Ok(());
    }

    let ordinary = inventory
        .ordinary
        .iter()
        .filter(|line| mentions_native_backend(line))
        .collect::<Vec<_>>();
    let hidden = inventory
        .hidden
        .iter()
        .filter(|line| mentions_native_backend(line))
        .collect::<Vec<_>>();
    if ordinary.is_empty() && hidden.is_empty() {
        return Ok(());
    }

    Err(format!(
        "public API for package `{package}` exposes the private native backend; \
         ordinary items: {ordinary:#?}; rustdoc-hidden items: {hidden:#?}"
    ))
}

fn mentions_native_backend(line: &str) -> bool {
    line.match_indices("j2k_native").any(|(start, name)| {
        let before = line[..start].chars().next_back();
        let after = line[start + name.len()..].chars().next();
        before.is_none_or(|character| character != '_' && !character.is_alphanumeric())
            && after.is_none_or(|character| character != '_' && !character.is_alphanumeric())
    })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::validate;
    use crate::stable_api::{public_api_line_set, PackageApiInventory};

    #[test]
    fn non_native_packages_cannot_expose_native_backend_types() {
        let clean = PackageApiInventory {
            ordinary: public_api_line_set("pub fn j2k::decode(&[u8]) -> j2k::Image\n"),
            hidden: BTreeSet::new(),
        };
        assert!(validate("j2k", &clean).is_ok());
        assert!(validate("j2k-native", &clean).is_ok());

        let ordinary_leak = PackageApiInventory {
            ordinary: public_api_line_set("pub fn j2k::decode(&[u8]) -> j2k_native::RawBitmap\n"),
            hidden: BTreeSet::new(),
        };
        let error = validate("j2k", &ordinary_leak).expect_err("ordinary native type leak");
        assert!(error.contains("ordinary"));
        assert!(error.contains("j2k_native::RawBitmap"));
        validate("j2k-jpeg-cuda", &ordinary_leak)
            .expect_err("new packages must not bypass the native boundary");

        let hidden_leak = PackageApiInventory {
            ordinary: public_api_line_set("pub struct j2k_metal::Decoder\n"),
            hidden: public_api_line_set(
                "pub fn j2k_metal::Decoder::native() -> j2k_native::Image\n",
            ),
        };
        let error = validate("j2k-metal", &hidden_leak).expect_err("hidden native type leak");
        assert!(error.contains("rustdoc-hidden"));
        assert!(error.contains("j2k_native::Image"));
    }

    #[test]
    fn native_backend_detection_uses_identifier_boundaries() {
        let similarly_named_items = PackageApiInventory {
            ordinary: public_api_line_set(
                "pub struct j2k::not_j2k_native_adapter\n\
                 pub struct j2k_nativeish::Image\n",
            ),
            hidden: BTreeSet::new(),
        };
        assert!(validate("j2k", &similarly_named_items).is_ok());
    }
}
