// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{fs, run_encoder_once, EncoderKind, EncoderTool, ImageCase, J2kDecoder, Path};

pub(super) fn validate_case_encoder(
    case: &ImageCase,
    tool: &EncoderTool,
    work_dir: &Path,
) -> Result<(), String> {
    let output = run_encoder_once(case, tool, work_dir, "validate")?;
    validate_encoded_profile(&output, case, tool.kind)?;
    let decoded = decode_encoded_output(&output, case)?;
    if decoded != case.pixels {
        return Err(format!(
            "{} {} output did not round-trip losslessly",
            tool.kind.label(),
            case.name
        ));
    }
    Ok(())
}

pub(super) fn validate_encoded_profile(
    path: &Path,
    case: &ImageCase,
    encoder: EncoderKind,
) -> Result<(), String> {
    let bytes = fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?;
    let payload = j2k::extract_j2k_codestream_payload(&bytes)
        .map_err(|error| format!("extract {} codestream: {error}", path.display()))?;
    if payload.payload_kind() == j2k::CompressedPayloadKind::Jpeg2000Codestream {
        return Err("encoded output is not a JP2 container".to_string());
    }
    let codestream = payload.codestream();
    let header = j2k_native::inspect_j2k_codestream_header(codestream)
        .map_err(|error| format!("inspect {} profile: {error}", path.display()))?;
    validate_header_profile(&header, case, encoder)?;

    let cod = cod_profile(codestream)?;
    validate_cod_profile(&cod, case, encoder)
}

fn validate_header_profile(
    header: &j2k_native::J2kCodestreamHeaderMetadata,
    case: &ImageCase,
    encoder: EncoderKind,
) -> Result<(), String> {
    if header.dimensions != (case.width, case.height) {
        return Err(format!(
            "{} {} profile dimensions {:?} != expected {:?}",
            encoder.label(),
            case.name,
            header.dimensions,
            (case.width, case.height)
        ));
    }
    if header.components != u16::from(case.components) {
        return Err(format!(
            "{} {} profile components {} != expected {}",
            encoder.label(),
            case.name,
            header.components,
            case.components
        ));
    }
    if header.tile_count != (1, 1) {
        return Err(format!(
            "{} {} profile tile count {:?} != expected single tile",
            encoder.label(),
            case.name,
            header.tile_count
        ));
    }
    if header.resolution_levels != 3 {
        return Err(format!(
            "{} {} profile resolution levels {} != expected 3",
            encoder.label(),
            case.name,
            header.resolution_levels
        ));
    }
    if !header.reversible {
        return Err(format!(
            "{} {} profile is not reversible 5/3",
            encoder.label(),
            case.name
        ));
    }
    if header.high_throughput {
        return Err(format!(
            "{} {} profile used HT block coding, expected classic",
            encoder.label(),
            case.name
        ));
    }
    if case.components == 3 && !header.has_mct {
        return Err(format!(
            "{} {} profile missing RGB reversible color transform",
            encoder.label(),
            case.name
        ));
    }
    if case.components == 1 && header.has_mct {
        return Err(format!(
            "{} {} grayscale profile unexpectedly enables MCT",
            encoder.label(),
            case.name
        ));
    }
    Ok(())
}

fn validate_cod_profile(
    cod: &CodProfile,
    case: &ImageCase,
    encoder: EncoderKind,
) -> Result<(), String> {
    if cod.progression_order != 0 {
        return Err(format!(
            "{} {} profile progression order {} != LRCP",
            encoder.label(),
            case.name,
            cod.progression_order
        ));
    }
    if cod.decomposition_levels != 2 {
        return Err(format!(
            "{} {} profile decomposition levels {} != expected 2",
            encoder.label(),
            case.name,
            cod.decomposition_levels
        ));
    }
    if cod.code_block_width_exp != 4 || cod.code_block_height_exp != 4 {
        return Err(format!(
            "{} {} profile code-block exponents {},{} != expected 4,4",
            encoder.label(),
            case.name,
            cod.code_block_width_exp,
            cod.code_block_height_exp
        ));
    }
    if cod.code_block_style & 0x40 != 0 {
        return Err(format!(
            "{} {} profile used HT code-block style",
            encoder.label(),
            case.name
        ));
    }
    if cod.transform != 1 {
        return Err(format!(
            "{} {} profile transform {} != reversible 5/3",
            encoder.label(),
            case.name,
            cod.transform
        ));
    }
    if cod.scod & 0x01 != 0 {
        return Err(format!(
            "{} {} profile overrides precincts",
            encoder.label(),
            case.name
        ));
    }
    if cod.scod & 0x02 != 0 {
        return Err(format!(
            "{} {} profile enables SOP markers",
            encoder.label(),
            case.name
        ));
    }
    if cod.scod & 0x04 != 0 {
        return Err(format!(
            "{} {} profile enables EPH markers",
            encoder.label(),
            case.name
        ));
    }
    Ok(())
}

pub(super) struct CodProfile {
    pub(super) scod: u8,
    pub(super) progression_order: u8,
    pub(super) decomposition_levels: u8,
    pub(super) code_block_width_exp: u8,
    pub(super) code_block_height_exp: u8,
    pub(super) code_block_style: u8,
    pub(super) transform: u8,
}

pub(super) fn cod_profile(codestream: &[u8]) -> Result<CodProfile, String> {
    if !j2k_native::looks_like_j2k_codestream(codestream) {
        return Err("codestream is missing SOC marker".to_string());
    }
    let mut offset = 2_usize;
    while offset
        .checked_add(2)
        .is_some_and(|end| end <= codestream.len())
    {
        if codestream[offset] != 0xFF {
            return Err(format!("invalid codestream marker at offset {offset}"));
        }
        let marker = codestream[offset + 1];
        offset += 2;
        match marker {
            0x52 => {
                let payload = codestream_segment_payload(codestream, &mut offset, "COD")?;
                return parse_cod_profile(payload);
            }
            0x90 | 0x93 | 0xD9 => break,
            _ => {
                let _ = codestream_segment_payload(codestream, &mut offset, "marker segment")?;
            }
        }
    }
    Err("codestream is missing COD marker".to_string())
}

pub(super) fn codestream_segment_payload<'a>(
    codestream: &'a [u8],
    offset: &mut usize,
    label: &str,
) -> Result<&'a [u8], String> {
    let length_end = offset
        .checked_add(2)
        .ok_or_else(|| format!("{label} length offset overflow"))?;
    if length_end > codestream.len() {
        return Err(format!("truncated {label} segment length"));
    }
    let length = u16::from_be_bytes([codestream[*offset], codestream[*offset + 1]]) as usize;
    if length < 2 {
        return Err(format!("invalid {label} segment length"));
    }
    let payload_start = *offset + 2;
    let segment_end = offset
        .checked_add(length)
        .ok_or_else(|| format!("{label} segment length overflow"))?;
    if segment_end > codestream.len() {
        return Err(format!("truncated {label} segment"));
    }
    *offset = segment_end;
    Ok(&codestream[payload_start..segment_end])
}

pub(super) fn parse_cod_profile(payload: &[u8]) -> Result<CodProfile, String> {
    if payload.len() < 10 {
        return Err("COD payload is shorter than the fixed profile fields".to_string());
    }
    Ok(CodProfile {
        scod: payload[0],
        progression_order: payload[1],
        decomposition_levels: payload[5],
        code_block_width_exp: payload[6],
        code_block_height_exp: payload[7],
        code_block_style: payload[8],
        transform: payload[9],
    })
}

pub(super) fn decode_encoded_output(path: &Path, case: &ImageCase) -> Result<Vec<u8>, String> {
    let bytes = fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?;
    let mut decoder = J2kDecoder::new(&bytes).map_err(|error| error.to_string())?;
    let format = case.pixel_format()?;
    let stride = case.width as usize * format.bytes_per_pixel();
    let mut out = vec![0_u8; stride * case.height as usize];
    decoder
        .decode_into(&mut out, stride, format)
        .map_err(|error| error.to_string())?;
    Ok(out)
}
