use std::fmt::Write as _;
use std::fs;
use std::path::Path;

use crate::command_support::command_output_os_detailed;
use crate::process::cargo;
use crate::release_commands::STABLE_DOC_LIBRARY_PACKAGES;

const STABLE_API_SNAPSHOT: &str = "docs/stable-api-1.0.public-api.txt";
pub(super) const CARGO_PUBLIC_API_VERSION: &str = "0.52.0";
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

    let rendered = render_stable_api_snapshot()?;
    if write_snapshot {
        fs::write(STABLE_API_SNAPSHOT, rendered)
            .map_err(|err| format!("failed to write {STABLE_API_SNAPSHOT}: {err}"))?;
        return Ok(());
    }

    let committed = fs::read_to_string(STABLE_API_SNAPSHOT)
        .map_err(|err| format!("failed to read {STABLE_API_SNAPSHOT}: {err}"))?;
    if committed == rendered {
        Ok(())
    } else {
        Err(format!(
            "{STABLE_API_SNAPSHOT} is stale; run `cargo xtask stable-api --write` and review the public API diff"
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
        for (path, rendered) in fragments {
            let path = Path::new(path);
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)
                    .map_err(|err| format!("failed to create {}: {err}", parent.display()))?;
            }
            fs::write(path, rendered)
                .map_err(|err| format!("failed to write {}: {err}", path.display()))?;
        }
        return Ok(());
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

fn render_stable_api_snapshot() -> Result<String, String> {
    if !cfg!(target_os = "macos") {
        return Err(
            "stable-api snapshot must be generated on macOS so target-gated Metal APIs are included"
                .to_string(),
        );
    }

    let tool_version =
        command_output_os_detailed(cargo(), &["public-api", "--version"]).map_err(|err| {
            format!(
                "failed to detect cargo-public-api: {err}; \
                 install cargo-public-api with `cargo install cargo-public-api --version {CARGO_PUBLIC_API_VERSION} --locked`"
            )
        })?;
    if !tool_version.contains(CARGO_PUBLIC_API_VERSION) {
        return Err(format!(
            "cargo-public-api version must be {CARGO_PUBLIC_API_VERSION}; found `{tool_version}`"
        ));
    }

    let mut out = String::new();
    writeln!(
        &mut out,
        "# J2K 1.0 Public API Snapshot\n\n\
         This file is generated by `cargo xtask stable-api --write` from \
         `cargo public-api -p <package> --all-features -sss --color never`.\n\
         It is generated on macOS so target-gated Metal APIs are included; \
         non-macOS builds expose a smaller cfg-gated subset.\n\n\
         Generator: `{tool_version}`.\n\n\
         It is the item-level companion to `docs/stable-api-1.0.md`: every \
         public module, type, trait, function, method, constant, variant, and \
         field reported here is semver-visible unless moved private before 1.0.\n"
    )
    .unwrap();

    for package in STABLE_DOC_LIBRARY_PACKAGES {
        let api = command_output_os_detailed(
            cargo(),
            &[
                "public-api",
                "-p",
                package,
                "--all-features",
                "-sss",
                "--color",
                "never",
            ],
        )
        .map_err(|err| {
            format!(
                "failed to generate public API for {package}: {err}; \
                 install cargo-public-api with `cargo install cargo-public-api --version {CARGO_PUBLIC_API_VERSION} --locked`"
            )
        })?;
        writeln!(&mut out, "## `{package}`\n\n```text").unwrap();
        writeln!(&mut out, "{api}").unwrap();
        writeln!(&mut out, "```\n").unwrap();
    }

    writeln!(
        &mut out,
        "## `j2k-cli`\n\n\
         `j2k-cli` is a binary package. Its stable command, stdout/stderr, \
         and exit-code contract is documented in `docs/stable-api-1.0.md`.\n"
    )
    .unwrap();

    Ok(out)
}

fn print_stable_api_help() {
    println!(
        "usage: cargo xtask stable-api [--write]\n\n\
         Without --write, checks docs/stable-api-1.0.public-api.txt against \
         cargo-public-api output for all 1.0-stable library crates. With \
         --write, refreshes the snapshot. This task must run on macOS so \
         target-gated Metal APIs are included."
    );
}

fn print_codec_math_codegen_help() {
    println!(
        "usage: cargo xtask codec-math-codegen [--write]\n\n\
         Without --write, checks generated Rust and Metal codec-math fragments \
         against the Rust source of truth. With --write, refreshes the fragments."
    );
}
