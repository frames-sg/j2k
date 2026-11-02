// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(not(test))]
use std::env;
#[cfg(not(test))]
use std::process::Command;

const LIBJPEG_TURBO_PKG_CONFIG_ARGS: [&str; 2] = ["--libs", "libturbojpeg"];
#[cfg(not(test))]
const LIBJPEG_TURBO_VERSION_ARGS: [&str; 2] = ["--modversion", "libturbojpeg"];
#[cfg(not(test))]
const REQUIRE_LIBJPEG_TURBO_ENV: &str = "J2K_BENCH_REQUIRE_LIBJPEG_TURBO";

/// Fails the build when the benchmark run requires libjpeg-turbo; otherwise
/// warns that the comparator is not linked.
#[cfg(not(test))]
fn report_missing_libjpeg_turbo(require: bool, reason: &str) {
    assert!(
        !require,
        "bench-libjpeg-turbo: {reason} ({REQUIRE_LIBJPEG_TURBO_ENV}=1)"
    );
    println!("cargo:warning=bench-libjpeg-turbo: {reason}; libjpeg-turbo is not linked");
}

fn is_v3(version: &[u8]) -> bool {
    std::str::from_utf8(version)
        .ok()
        .and_then(|version| version.trim().split('.').next())
        .and_then(|major| major.parse::<u32>().ok())
        .is_some_and(|major| major >= 3)
}

#[cfg(not(test))]
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
    println!("cargo:rerun-if-env-changed={REQUIRE_LIBJPEG_TURBO_ENV}");

    // `--all-features` builds (docs.rs, clippy, public-api) enable this
    // feature on machines without libjpeg-turbo, so a missing library is a
    // build warning by default. Benchmark runs set the require variable to
    // turn it into an error instead of building without the comparator.
    let require = env::var_os(REQUIRE_LIBJPEG_TURBO_ENV).is_some_and(|value| value == "1");
    let output = match Command::new("pkg-config")
        .args(LIBJPEG_TURBO_PKG_CONFIG_ARGS)
        .output()
    {
        Ok(output) if output.status.success() => output,
        Ok(_) => {
            return report_missing_libjpeg_turbo(
                require,
                "pkg-config cannot find libturbojpeg; set PKG_CONFIG_PATH",
            );
        }
        Err(err) => {
            return report_missing_libjpeg_turbo(require, &format!("cannot run pkg-config: {err}"));
        }
    };

    println!("cargo:rustc-cfg=has_libjpeg_turbo");
    let version = Command::new("pkg-config")
        .args(LIBJPEG_TURBO_VERSION_ARGS)
        .output()
        .ok()
        .filter(|version| version.status.success())
        .map(|version| version.stdout)
        .unwrap_or_default();
    if is_v3(&version) {
        println!("cargo:rustc-cfg=has_libjpeg_turbo_v3");
    }
    println!(
        "cargo:rustc-env=J2K_LIBJPEG_TURBO_VERSION={}",
        String::from_utf8_lossy(&version).trim()
    );
    let flags = String::from_utf8_lossy(&output.stdout);
    for token in flags.split_whitespace() {
        if let Some(path) = token.strip_prefix("-L") {
            println!("cargo:rustc-link-search=native={path}");
            // Benches load the library pkg-config named, not another copy on
            // the default search path. MSVC's linker has no rpath.
            if env::var("CARGO_CFG_TARGET_ENV").as_deref() != Ok("msvc") {
                println!("cargo:rustc-link-arg-benches=-Wl,-rpath,{path}");
            }
            println!("cargo:rustc-env=J2K_LIBJPEG_TURBO_LIBDIR={path}");
        } else if let Some(lib) = token.strip_prefix("-l") {
            println!("cargo:rustc-link-lib={lib}");
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn turbojpeg_major_version_selects_the_supported_ffi() {
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
