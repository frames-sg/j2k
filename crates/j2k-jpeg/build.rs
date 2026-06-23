// SPDX-License-Identifier: MIT OR Apache-2.0

use std::env;
use std::process::Command;

fn main() {
    println!("cargo:rustc-check-cfg=cfg(has_libjpeg_turbo)");
    println!("cargo:rerun-if-changed=build.rs");

    // Probing pkg-config and linking system libjpeg-turbo is exclusively for
    // the comparison benches; library consumers must never pick up a system
    // JPEG link just because pkg-config can find one.
    if env::var_os("CARGO_FEATURE_BENCH_LIBJPEG_TURBO").is_none() {
        return;
    }
    println!("cargo:rerun-if-env-changed=PKG_CONFIG_PATH");

    let Ok(output) = Command::new("pkg-config")
        .args(["--libs", "libturbojpeg", "libjpeg"])
        .output()
    else {
        return;
    };
    if !output.status.success() {
        return;
    }

    println!("cargo:rustc-cfg=has_libjpeg_turbo");
    let flags = String::from_utf8_lossy(&output.stdout);
    for token in flags.split_whitespace() {
        if let Some(path) = token.strip_prefix("-L") {
            println!("cargo:rustc-link-search=native={path}");
        } else if let Some(lib) = token.strip_prefix("-l") {
            println!("cargo:rustc-link-lib={lib}");
        }
    }
}
