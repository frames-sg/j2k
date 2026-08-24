// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    fmt::Write as _,
    fs,
    path::Path,
};

use serde::Deserialize;

const REQUIRED_OPERATIONS: [&str; 6] = [
    "batch_decode",
    "full_decode",
    "lossless_encode",
    "lossy_encode",
    "roi_decode",
    "scaled_decode",
];

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema_version: u32,
    sources: Vec<EvidenceSource>,
    cuda_decode: Vec<DecodeCell>,
    metal_decode: Vec<DecodeCell>,
    metal_host_output: Vec<HostOutputCell>,
    metal_lossy_rgb8: PixelThreshold,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct EvidenceSource {
    name: String,
    backend: String,
    artifact_sha256: String,
    operation_coverage: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DecodeCell {
    source: String,
    source_components: u16,
    format: String,
    transfer_syntax: String,
    payload_kind: String,
    operation: String,
    minimum_width: u32,
    minimum_height: u32,
    minimum_pixels: u64,
    minimum_count: usize,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct HostOutputCell {
    source: String,
    source_components: u16,
    width: u32,
    height: u32,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PixelThreshold {
    source: String,
    minimum_pixels: usize,
}

pub(crate) fn promotion_codegen(mut args: impl Iterator<Item = String>) -> Result<(), String> {
    let check = match args.next().as_deref() {
        None => false,
        Some("--check") => true,
        Some(other) => return Err(format!("unknown promotion-codegen argument `{other}`")),
    };
    if let Some(other) = args.next() {
        return Err(format!("unknown promotion-codegen argument `{other}`"));
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .ok_or_else(|| "xtask manifest has no repository parent".to_string())?;
    let manifest_path = root.join("docs/routing-promotion-evidence.json");
    let source = fs::read_to_string(&manifest_path)
        .map_err(|error| format!("read {}: {error}", manifest_path.display()))?;
    let manifest: Manifest = serde_json::from_str(&source)
        .map_err(|error| format!("parse {}: {error}", manifest_path.display()))?;
    validate(&manifest)?;
    let outputs = render(&manifest)?;
    for (relative, expected) in outputs {
        let path = root.join(relative);
        if check {
            let actual = fs::read_to_string(&path)
                .map_err(|error| format!("read generated output {}: {error}", path.display()))?;
            if actual != expected {
                return Err(format!(
                    "generated promotion table is stale: {}; run `cargo xtask promotion-codegen`",
                    path.display()
                ));
            }
        } else {
            fs::write(&path, expected)
                .map_err(|error| format!("write generated output {}: {error}", path.display()))?;
        }
    }
    Ok(())
}

fn validate(manifest: &Manifest) -> Result<(), String> {
    if manifest.schema_version != 1 {
        return Err("unsupported promotion evidence schema version".to_string());
    }
    let mut sources = BTreeMap::new();
    for source in &manifest.sources {
        if !matches!(source.backend.as_str(), "cuda" | "metal") {
            return Err(format!("source {} has an unsupported backend", source.name));
        }
        if !is_lower_hex_sha256(&source.artifact_sha256) {
            return Err(format!(
                "source {} has an invalid artifact SHA-256",
                source.name
            ));
        }
        let mut coverage = source
            .operation_coverage
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>();
        coverage.sort_unstable();
        coverage.dedup();
        if coverage != REQUIRED_OPERATIONS {
            return Err(format!(
                "source {} does not cover the required operation matrix",
                source.name
            ));
        }
        if sources.insert(source.name.as_str(), source).is_some() {
            return Err(format!("duplicate evidence source {}", source.name));
        }
    }
    validate_decode_cells("cuda", &manifest.cuda_decode, &sources)?;
    validate_decode_cells("metal", &manifest.metal_decode, &sources)?;
    let mut host_identities = BTreeSet::new();
    for cell in &manifest.metal_host_output {
        require_source(&sources, &cell.source, "metal")?;
        if cell.width == 0 || cell.height == 0 || !matches!(cell.source_components, 1 | 3) {
            return Err("invalid Metal host-output promotion cell".to_string());
        }
        let identity = (cell.source_components, cell.width, cell.height);
        if !host_identities.insert(identity) {
            return Err("duplicate Metal host-output workload identity".to_string());
        }
    }
    require_source(&sources, &manifest.metal_lossy_rgb8.source, "metal")?;
    if manifest.metal_lossy_rgb8.minimum_pixels == 0 {
        return Err("Metal lossy RGB8 threshold must be nonzero".to_string());
    }
    Ok(())
}

fn validate_decode_cells(
    backend: &str,
    cells: &[DecodeCell],
    sources: &BTreeMap<&str, &EvidenceSource>,
) -> Result<(), String> {
    let mut identities = BTreeSet::new();
    for cell in cells {
        require_source(sources, &cell.source, backend)?;
        pixel_format(&cell.format)?;
        transfer_syntax(&cell.transfer_syntax)?;
        payload_kind(&cell.payload_kind)?;
        operation(&cell.operation)?;
        if cell.source_components == 0
            || cell.minimum_count == 0
            || (cell.minimum_width == 0 && cell.minimum_height == 0 && cell.minimum_pixels == 0)
        {
            return Err(format!("invalid {backend} decode promotion boundary"));
        }
        let identity = (
            cell.source_components,
            cell.format.as_str(),
            cell.transfer_syntax.as_str(),
            cell.payload_kind.as_str(),
            cell.operation.as_str(),
        );
        if !identities.insert(identity) {
            return Err(format!("duplicate {backend} decode workload identity"));
        }
    }
    Ok(())
}

fn require_source<'a>(
    sources: &'a BTreeMap<&str, &EvidenceSource>,
    name: &str,
    backend: &str,
) -> Result<&'a EvidenceSource, String> {
    let source = sources
        .get(name)
        .ok_or_else(|| format!("unknown evidence source {name}"))?;
    if source.backend != backend {
        return Err(format!(
            "evidence source {name} does not belong to {backend}"
        ));
    }
    Ok(source)
}

fn is_lower_hex_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn render(manifest: &Manifest) -> Result<Vec<(&'static str, String)>, String> {
    Ok(vec![
        (
            "crates/j2k-cuda/src/generated/promotion.rs",
            render_cuda(manifest)?,
        ),
        (
            "crates/j2k-metal/src/generated/promotion.rs",
            render_metal(manifest)?,
        ),
    ])
}

fn generated_header() -> &'static str {
    "// SPDX-License-Identifier: MIT OR Apache-2.0\n// @generated by `cargo xtask promotion-codegen`; do not edit by hand.\n\n"
}

fn render_cuda(manifest: &Manifest) -> Result<String, String> {
    let mut out = String::from(generated_header());
    out.push_str("use j2k_core::{CompressedPayloadKind as Payload, CompressedTransferSyntax as Syntax, PixelFormat};\n\nuse crate::routing::promotion::{PromotionCell, PromotionOperation as Operation};\n\n");
    render_sources(&mut out, manifest, "cuda");
    out.push_str("\nconst fn cell(\n    surface: (u16, PixelFormat),\n    transfer_syntax: Syntax,\n    payload_kind: Payload,\n    operation: Operation,\n    minimum: (u32, u32),\n    minimum_count: usize,\n    source_evidence: &'static str,\n) -> PromotionCell {\n    PromotionCell {\n        source_components: surface.0,\n        format: surface.1,\n        transfer_syntax,\n        payload_kind,\n        operation,\n        minimum_width: minimum.0,\n        minimum_height: minimum.1,\n        minimum_count,\n        source_evidence,\n    }\n}\n\npub(crate) const PROMOTION_CELLS: &[PromotionCell] = &[\n");
    for cell in &manifest.cuda_decode {
        render_cuda_cell(&mut out, cell)?;
    }
    out.push_str("];\n");
    Ok(out)
}

fn render_metal(manifest: &Manifest) -> Result<String, String> {
    let mut out = String::from(generated_header());
    out.push_str("#[cfg(any(test, target_os = \"macos\"))]\nuse j2k_core::{CompressedPayloadKind as Payload, CompressedTransferSyntax as Syntax, PixelFormat};\n\n#[cfg(any(test, target_os = \"macos\"))]\nuse crate::routing::promotion::{PromotionCell, PromotionOperation as Operation};\n\n");
    render_sources(&mut out, manifest, "metal");
    out.push_str("\n#[cfg(any(test, target_os = \"macos\"))]\nconst fn decode_cell(\n    format: PixelFormat,\n    transfer_syntax: Syntax,\n    payload_kind: Payload,\n    operation: Operation,\n    boundary: (u32, u32, u64, usize),\n    source_evidence: &'static str,\n) -> PromotionCell {\n    PromotionCell {\n        format,\n        transfer_syntax,\n        payload_kind,\n        operation,\n        minimum_width: boundary.0,\n        minimum_height: boundary.1,\n        minimum_pixels: boundary.2,\n        minimum_count: boundary.3,\n        source_evidence,\n    }\n}\n\n#[cfg(any(test, target_os = \"macos\"))]\npub(crate) const PROMOTION_CELLS: &[PromotionCell] = &[\n");
    for cell in &manifest.metal_decode {
        render_metal_cell(&mut out, cell)?;
    }
    out.push_str("];\n\n#[cfg(target_os = \"macos\")]\nconst HOST_OUTPUT_CELLS: &[(u16, u32, u32, &str)] = &[\n");
    for cell in &manifest.metal_host_output {
        writeln!(
            out,
            "    ({}, {}, {}, {}),",
            cell.source_components,
            cell.width,
            cell.height,
            rust_source_name(&cell.source)
        )
        .expect("writing to a String cannot fail");
    }
    out.push_str("];\n\n#[cfg(target_os = \"macos\")]\npub(crate) fn auto_host_output_encode_qualifies(components: u16, width: u32, height: u32) -> bool {\n    HOST_OUTPUT_CELLS.iter().any(|cell| {\n        cell.0 == components\n            && cell.1 == width\n            && cell.2 == height\n            && SOURCE_EVIDENCE.contains(&cell.3)\n    })\n}\n\npub(crate) fn auto_lossy_rgb8_encode_qualifies(pixels: usize) -> bool {\n");
    write!(
        out,
        "    pixels >= {} && SOURCE_EVIDENCE.contains(&{})\n}}\n",
        rust_number(&manifest.metal_lossy_rgb8.minimum_pixels),
        rust_source_name(&manifest.metal_lossy_rgb8.source)
    )
    .expect("writing to a String cannot fail");
    Ok(out)
}

fn render_sources(out: &mut String, manifest: &Manifest, backend: &str) {
    let sources = manifest
        .sources
        .iter()
        .filter(|source| source.backend == backend)
        .collect::<Vec<_>>();
    for source in &sources {
        writeln!(
            out,
            "const {}: &str = \"{}\";",
            rust_source_name(&source.name),
            source.artifact_sha256
        )
        .expect("writing to a String cannot fail");
    }
    out.push_str("\npub(crate) const SOURCE_EVIDENCE: &[&str] = &[");
    for (index, source) in sources.iter().enumerate() {
        if index != 0 {
            out.push_str(", ");
        }
        out.push_str(&rust_source_name(&source.name));
    }
    out.push_str("];\n");
}

fn render_cuda_cell(out: &mut String, cell: &DecodeCell) -> Result<(), String> {
    write!(
        out,
        "    cell(\n        ({}, PixelFormat::{}),\n        Syntax::{},\n        Payload::{},\n        Operation::{},\n        ({}, {}),\n        {},\n        {},\n    ),\n",
        cell.source_components,
        pixel_format(&cell.format)?,
        transfer_syntax(&cell.transfer_syntax)?,
        payload_kind(&cell.payload_kind)?,
        operation(&cell.operation)?,
        cell.minimum_width,
        cell.minimum_height,
        cell.minimum_count,
        rust_source_name(&cell.source),
    )
    .expect("writing to a String cannot fail");
    Ok(())
}

fn render_metal_cell(out: &mut String, cell: &DecodeCell) -> Result<(), String> {
    write!(
        out,
        "    decode_cell(\n        PixelFormat::{},\n        Syntax::{},\n        Payload::{},\n        Operation::{},\n        ({}, {}, {}, {}),\n        {},\n    ),\n",
        pixel_format(&cell.format)?,
        transfer_syntax(&cell.transfer_syntax)?,
        payload_kind(&cell.payload_kind)?,
        operation(&cell.operation)?,
        cell.minimum_width,
        cell.minimum_height,
        rust_number(&cell.minimum_pixels),
        cell.minimum_count,
        rust_source_name(&cell.source),
    )
    .expect("writing to a String cannot fail");
    Ok(())
}

fn rust_source_name(name: &str) -> String {
    name.to_ascii_uppercase()
}

fn rust_number(value: &impl ToString) -> String {
    let digits = value.to_string();
    let mut rendered = String::with_capacity(digits.len() + digits.len() / 3);
    for (index, digit) in digits.chars().enumerate() {
        if index != 0 && (digits.len() - index).is_multiple_of(3) {
            rendered.push('_');
        }
        rendered.push(digit);
    }
    rendered
}

fn pixel_format(value: &str) -> Result<&str, String> {
    match value {
        "Gray8" | "Rgb8" => Ok(value),
        _ => Err(format!("unsupported promotion pixel format {value}")),
    }
}

fn transfer_syntax(value: &str) -> Result<&str, String> {
    match value {
        "Jpeg2000Lossless" | "Jpeg2000Lossy" | "HtJpeg2000Lossless" | "HtJpeg2000Lossy" => {
            Ok(value)
        }
        _ => Err(format!("unsupported promotion transfer syntax {value}")),
    }
}

fn payload_kind(value: &str) -> Result<&str, String> {
    match value {
        "Jpeg2000Codestream" | "JphFile" => Ok(value),
        _ => Err(format!("unsupported promotion payload kind {value}")),
    }
}

fn operation(value: &str) -> Result<&str, String> {
    match value {
        "Full" | "Region" | "ScaledHalf" | "Repeated" => Ok(value),
        _ => Err(format!("unsupported promotion operation {value}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest() -> Manifest {
        serde_json::from_str(include_str!("../../docs/routing-promotion-evidence.json"))
            .expect("checked-in promotion manifest")
    }

    #[test]
    fn checked_in_manifest_validates_and_renders_deterministically() {
        let manifest = manifest();
        validate(&manifest).expect("valid promotion manifest");
        assert_eq!(
            render(&manifest).expect("first render"),
            render(&manifest).expect("second render")
        );
    }

    #[test]
    fn invalid_hash_is_rejected() {
        let mut manifest = manifest();
        manifest.sources[0].artifact_sha256 = "not-a-hash".to_string();
        assert!(validate(&manifest)
            .unwrap_err()
            .contains("invalid artifact SHA-256"));
    }

    #[test]
    fn duplicate_workload_identity_is_rejected() {
        let mut manifest = manifest();
        manifest.cuda_decode.push(manifest.cuda_decode[0].clone());
        assert!(validate(&manifest)
            .unwrap_err()
            .contains("duplicate cuda decode workload identity"));
    }

    #[test]
    fn incomplete_operation_coverage_is_rejected() {
        let mut manifest = manifest();
        manifest.sources[0].operation_coverage.pop();
        assert!(validate(&manifest)
            .unwrap_err()
            .contains("required operation matrix"));
    }
}
