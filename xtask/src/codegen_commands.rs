use std::fmt::Write as _;
use std::fs;
use std::path::Path;

mod transaction;

use transaction::write_generated_pair_transactionally;

use crate::release_commands::STABLE_DOC_LIBRARY_PACKAGES;
use crate::stable_api::{
    collect_package_apis, verify_cargo_public_api_version, CARGO_PUBLIC_API_VERSION,
    HIDDEN_API_SNAPSHOT, ORDINARY_RUSTDOCFLAGS, PUBLIC_API_SNAPSHOT, PUBLIC_API_TARGET,
    PUBLIC_API_TOOLCHAIN,
};

const CODEC_MATH_DWT97_METAL_FRAGMENT: &str =
    "crates/j2k-codec-math/generated/dwt97_constants.metal";
const CODEC_MATH_DWT97_RUST_FRAGMENT: &str = "crates/j2k-codec-math/generated/dwt97_constants.rs";

pub(super) fn stable_api(args: impl Iterator<Item = String>) -> Result<(), String> {
    let mut write_snapshot = false;
    for arg in args {
        match arg.as_str() {
            "--write" => write_snapshot = true,
            "--help" | "-h" => {
                print_stable_api_help();
                return Ok(());
            }
            other => return Err(format!("unknown stable-api argument `{other}`")),
        }
    }

    let (public_api, implementation_api) = render_stable_api_snapshots()?;
    let snapshots = [
        (PUBLIC_API_SNAPSHOT, public_api),
        (HIDDEN_API_SNAPSHOT, implementation_api),
    ];
    if write_snapshot {
        return write_generated_pair_transactionally(&snapshots);
    }

    let mut stale = Vec::new();
    for (path, rendered) in &snapshots {
        let committed =
            fs::read_to_string(path).map_err(|err| format!("failed to read {path}: {err}"))?;
        if committed != *rendered {
            stale.push(*path);
        }
    }
    if stale.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "stable API snapshots are stale: {}; run `cargo xtask stable-api --write` and review both API inventories",
            stale.join(", ")
        ))
    }
}

pub(super) fn codec_math_codegen(args: impl Iterator<Item = String>) -> Result<(), String> {
    let mut write_fragments = false;
    for arg in args {
        match arg.as_str() {
            "--write" => write_fragments = true,
            "--help" | "-h" => {
                print_codec_math_codegen_help();
                return Ok(());
            }
            other => return Err(format!("unknown codec-math-codegen argument `{other}`")),
        }
    }

    let fragments = [
        (
            CODEC_MATH_DWT97_METAL_FRAGMENT,
            render_codec_math_dwt97_metal_fragment(),
        ),
        (
            CODEC_MATH_DWT97_RUST_FRAGMENT,
            render_codec_math_dwt97_rust_fragment(),
        ),
    ];

    if write_fragments {
        for (path, _) in &fragments {
            let path = Path::new(path);
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)
                    .map_err(|err| format!("failed to create {}: {err}", parent.display()))?;
            }
        }
        return write_generated_pair_transactionally(&fragments);
    }

    let mut stale = Vec::new();
    for (path, rendered) in fragments {
        let committed =
            fs::read_to_string(path).map_err(|err| format!("failed to read {path}: {err}"))?;
        if committed != rendered {
            stale.push(path);
        }
    }
    if stale.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "codec math generated fragments are stale: {}; run `cargo xtask codec-math-codegen --write` and review the diff",
            stale.join(", ")
        ))
    }
}

fn render_codec_math_dwt97_metal_fragment() -> String {
    use j2k_codec_math::dwt;

    [
        "// Generated from crates/j2k-codec-math/src/dwt.rs.".to_string(),
        format!(
            "constant float CODEC_MATH_DWT97_ALPHA = {}f;",
            compact_f32(dwt::DWT97_ALPHA_F32)
        ),
        format!(
            "constant float CODEC_MATH_DWT97_BETA = {}f;",
            compact_f32(dwt::DWT97_BETA_F32)
        ),
        format!(
            "constant float CODEC_MATH_DWT97_GAMMA = {}f;",
            compact_f32(dwt::DWT97_GAMMA_F32)
        ),
        format!(
            "constant float CODEC_MATH_DWT97_DELTA = {}f;",
            compact_f32(dwt::DWT97_DELTA_F32)
        ),
        format!(
            "constant float CODEC_MATH_DWT97_KAPPA = {}f;",
            compact_f32(dwt::DWT97_KAPPA_F32)
        ),
        "constant float CODEC_MATH_DWT97_INV_KAPPA = 1.0f / CODEC_MATH_DWT97_KAPPA;".to_string(),
        format!(
            "constant float CODEC_MATH_IDWT97_NEG_ALPHA = {}f;",
            compact_f32(dwt::IDWT97_NEG_ALPHA_F32)
        ),
        format!(
            "constant float CODEC_MATH_IDWT97_NEG_BETA = {}f;",
            compact_f32(dwt::IDWT97_NEG_BETA_F32)
        ),
        format!(
            "constant float CODEC_MATH_IDWT97_NEG_GAMMA = {}f;",
            compact_f32(dwt::IDWT97_NEG_GAMMA_F32)
        ),
        format!(
            "constant float CODEC_MATH_IDWT97_NEG_DELTA = {}f;",
            compact_f32(dwt::IDWT97_NEG_DELTA_F32)
        ),
    ]
    .join("\n")
        + "\n"
}

fn render_codec_math_dwt97_rust_fragment() -> String {
    use j2k_codec_math::dwt;

    [
        "// Generated from crates/j2k-codec-math/src/dwt.rs.".to_string(),
        format!(
            "pub const CODEC_MATH_DWT97_ALPHA: f32 = {};",
            compact_f32(dwt::DWT97_ALPHA_F32)
        ),
        format!(
            "pub const CODEC_MATH_DWT97_BETA: f32 = {};",
            compact_f32(dwt::DWT97_BETA_F32)
        ),
        format!(
            "pub const CODEC_MATH_DWT97_GAMMA: f32 = {};",
            compact_f32(dwt::DWT97_GAMMA_F32)
        ),
        format!(
            "pub const CODEC_MATH_DWT97_DELTA: f32 = {};",
            compact_f32(dwt::DWT97_DELTA_F32)
        ),
        format!(
            "pub const CODEC_MATH_DWT97_KAPPA: f32 = {};",
            compact_f32(dwt::DWT97_KAPPA_F32)
        ),
        "pub const CODEC_MATH_DWT97_INV_KAPPA: f32 = 1.0 / CODEC_MATH_DWT97_KAPPA;".to_string(),
        format!(
            "pub const CODEC_MATH_IDWT97_NEG_ALPHA: f32 = {};",
            compact_f32(dwt::IDWT97_NEG_ALPHA_F32)
        ),
        format!(
            "pub const CODEC_MATH_IDWT97_NEG_BETA: f32 = {};",
            compact_f32(dwt::IDWT97_NEG_BETA_F32)
        ),
        format!(
            "pub const CODEC_MATH_IDWT97_NEG_GAMMA: f32 = {};",
            compact_f32(dwt::IDWT97_NEG_GAMMA_F32)
        ),
        format!(
            "pub const CODEC_MATH_IDWT97_NEG_DELTA: f32 = {};",
            compact_f32(dwt::IDWT97_NEG_DELTA_F32)
        ),
    ]
    .join("\n")
        + "\n"
}

fn compact_f32(value: f32) -> String {
    format!("{value:?}")
}

fn render_stable_api_snapshots() -> Result<(String, String), String> {
    if !cfg!(target_os = "macos") {
        return Err(
            "stable-api snapshot must be generated on macOS so target-gated Metal APIs are included"
                .to_string(),
        );
    }
    verify_cargo_public_api_version()?;
    let inventories = collect_package_apis(STABLE_DOC_LIBRARY_PACKAGES)?;
    let tool_version = format!("cargo-public-api {CARGO_PUBLIC_API_VERSION}");

    let mut public_out = String::new();
    writeln!(
        &mut public_out,
        "# J2K 1.0 Public API Snapshot\n\n\
         This file is generated by `cargo xtask stable-api --write` from \
         `RUSTDOCFLAGS='{ORDINARY_RUSTDOCFLAGS}' rustup run \
         {PUBLIC_API_TOOLCHAIN} cargo \
         public-api -p <package> --all-features -sss --color never \
         --target {PUBLIC_API_TARGET}`.\n\
         It is generated on macOS for the pinned target so target-gated Metal \
         APIs are included.\n\n\
         Generator: `{tool_version}`.\n\n\
         Rustdoc toolchain: `{PUBLIC_API_TOOLCHAIN}`.\n\
         Target: `{PUBLIC_API_TARGET}`.\n\n\
         It is the item-level companion to `docs/stable-api-1.0.md`: every \
         public module, type, trait, function, method, constant, variant, and \
         field reported here is semver-visible unless moved private before 1.0. \
         Rustdoc-hidden items are tracked separately in `{HIDDEN_API_SNAPSHOT}`.\n"
    )
    .unwrap();

    let mut implementation_out = String::new();
    writeln!(
        &mut implementation_out,
        "# J2K 1.0 Rustdoc-Hidden Public API Snapshot\n\n\
         This file is generated by `cargo xtask stable-api --write`. For each \
         package it records the conservative additional reachable inventory \
         formed from the union of the ordinary and hidden-enabled passes. \
         Rustdoc may rewrite equivalent re-export paths when hidden modules \
         become visible, so rewritten variants remain reviewable here. The \
         hidden-enabled pass uses \
         `RUSTDOCFLAGS='-D warnings --document-hidden-items' rustup run \
         {PUBLIC_API_TOOLCHAIN} cargo public-api -p <package> --all-features -sss \
         --color never --target {PUBLIC_API_TARGET}`. The ordinary public \
         inventory remains in `{PUBLIC_API_SNAPSHOT}` so its \
         comparison with the 0.7.3 baseline keeps the same generator scope.\n\n\
         The published 0.7.3 artifact recorded a hidden-enabled pass. This \
         companion is staged-candidate inventory; the ordinary baseline remains \
         the compatibility input and this full hidden inventory remains exact \
         candidate-review evidence.\n\n\
         Rustdoc-hidden implementation adapters are still public Rust API. \
         They must be reviewed explicitly and must not become a compatibility \
         escape hatch.\n\n\
         Generator: `{tool_version}`.\n\n\
         Rustdoc toolchain: `{PUBLIC_API_TOOLCHAIN}`.\n\
         Target: `{PUBLIC_API_TARGET}`.\n"
    )
    .unwrap();

    for package in STABLE_DOC_LIBRARY_PACKAGES {
        let inventory = inventories.get(*package).ok_or_else(|| {
            format!("collected public API inventory is missing package `{package}`")
        })?;

        writeln!(&mut public_out, "## `{package}`\n\n```text").unwrap();
        for item in &inventory.ordinary {
            writeln!(&mut public_out, "{item}").unwrap();
        }
        writeln!(&mut public_out, "```\n").unwrap();
        writeln!(&mut implementation_out, "## `{package}`\n\n```text").unwrap();
        for item in &inventory.hidden {
            writeln!(&mut implementation_out, "{item}").unwrap();
        }
        writeln!(&mut implementation_out, "```\n").unwrap();
    }

    writeln!(
        &mut public_out,
        "## `j2k-cli`\n\n\
         `j2k-cli` is a binary package. Its stable command, stdout/stderr, \
         and exit-code contract is documented in `docs/stable-api-1.0.md`.\n"
    )
    .unwrap();

    Ok((
        finalize_text_snapshot(&public_out),
        finalize_text_snapshot(&implementation_out),
    ))
}

fn finalize_text_snapshot(snapshot: &str) -> String {
    let content = snapshot.trim_end_matches('\n');
    if content.is_empty() {
        String::new()
    } else {
        format!("{content}\n")
    }
}

fn print_stable_api_help() {
    println!(
        "usage: cargo xtask stable-api [--write]\n\n\
         Without --write, checks the ordinary and rustdoc-hidden API snapshots \
         against cargo-public-api output for all 1.0-stable library crates. \
         With --write, refreshes both snapshots. This task must run on macOS \
         so target-gated Metal APIs are included."
    );
}

fn print_codec_math_codegen_help() {
    println!(
        "usage: cargo xtask codec-math-codegen [--write]\n\n\
         Without --write, checks generated Rust and Metal codec-math fragments \
         against the Rust source of truth. With --write, refreshes the fragments."
    );
}

#[cfg(test)]
mod tests;
