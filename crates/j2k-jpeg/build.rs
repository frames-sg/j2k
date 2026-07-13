// SPDX-License-Identifier: MIT OR Apache-2.0

use std::env;
use std::process::Command;

const LIBJPEG_TURBO_PKG_CONFIG_ARGS: [&str; 2] = ["--libs", "libturbojpeg"];
const LIBJPEG_TURBO_VERSION_ARGS: [&str; 2] = ["--modversion", "libturbojpeg"];

fn is_v3(version: &[u8]) -> bool {
    std::str::from_utf8(version)
        .ok()
        .and_then(|version| version.trim().split('.').next())
        .and_then(|major| major.parse::<u32>().ok())
        .is_some_and(|major| major >= 3)
}

fn main() {
    println!("cargo:rustc-check-cfg=cfg(has_libjpeg_turbo)");
    println!("cargo:rustc-check-cfg=cfg(has_libjpeg_turbo_v3)");
    println!("cargo:rerun-if-changed=build.rs");

    // Probing pkg-config and linking system libjpeg-turbo is exclusively for
    // the comparison benches; library consumers must never pick up a system
    // JPEG link just because pkg-config can find one.
    if env::var_os("CARGO_FEATURE_BENCH_LIBJPEG_TURBO").is_none() {
        return;
    }
    println!("cargo:rerun-if-env-changed=PKG_CONFIG_PATH");

    let Ok(output) = Command::new("pkg-config")
        .args(LIBJPEG_TURBO_PKG_CONFIG_ARGS)
        .output()
    else {
        return;
    };
    if !output.status.success() {
        return;
    }

    println!("cargo:rustc-cfg=has_libjpeg_turbo");
    if Command::new("pkg-config")
        .args(LIBJPEG_TURBO_VERSION_ARGS)
        .output()
        .is_ok_and(|version| version.status.success() && is_v3(&version.stdout))
    {
        println!("cargo:rustc-cfg=has_libjpeg_turbo_v3");
    }
    let flags = String::from_utf8_lossy(&output.stdout);
    for token in flags.split_whitespace() {
        if let Some(path) = token.strip_prefix("-L") {
            println!("cargo:rustc-link-search=native={path}");
        } else if let Some(lib) = token.strip_prefix("-l") {
            println!("cargo:rustc-link-lib={lib}");
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn turbojpeg_major_version_selects_the_supported_ffi() {
        let _entrypoint: fn() = super::main;
        assert!(!super::is_v3(b"2.1.5\n"));
        assert!(super::is_v3(b"3.0.0\n"));
        assert!(super::is_v3(b"3.1.4.1\n"));
        assert!(!super::is_v3(b"not-a-version\n"));
    }

    #[test]
    fn turbojpeg_probe_requests_only_the_turbojpeg_api_package() {
        assert_eq!(
            super::LIBJPEG_TURBO_PKG_CONFIG_ARGS,
            ["--libs", "libturbojpeg"]
        );
    }
}
