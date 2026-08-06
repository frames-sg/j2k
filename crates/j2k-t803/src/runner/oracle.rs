// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{collections::BTreeSet, fs, path::Path};

use j2k::{J2kDecodedNativeComponents, J2kDecoder, J2kNativeComponentPlane};
use j2k_compare::openjpeg;
use sha2::{Digest, Sha256};

use crate::{NativeComponentOracleEvidence, T803Manifest};

use super::cases;

const SELECTION: &str = "COD MCT enabled with more than four codestream components";

pub(super) fn run(
    manifest: &T803Manifest,
    corpus: &Path,
) -> Result<Vec<NativeComponentOracleEvidence>, String> {
    let paths = manifest
        .decoder_cases
        .iter()
        .map(|case| case.codestream.as_str())
        .collect::<BTreeSet<_>>();
    let mut evidence = Vec::new();
    for path in paths {
        let input_path = corpus.join(path);
        let input = fs::read(&input_path)
            .map_err(|error| format!("read {}: {error}", input_path.display()))?;
        let payload = j2k::extract_j2k_codestream_payload(&input).map_err(|error| {
            format!(
                "extract codestream payload from {}: {error}",
                input_path.display()
            )
        })?;
        let header = j2k_native::inspect_j2k_codestream_header(payload.codestream())
            .map_err(|error| format!("inspect {}: {error}", input_path.display()))?;
        if !header.has_mct || header.components <= 4 {
            continue;
        }
        let codestream_sha256 = manifest
            .files
            .iter()
            .find(|entry| entry.path == path)
            .map(|entry| entry.sha256.clone())
            .ok_or_else(|| format!("{path} is absent from the pinned corpus inventory"))?;
        evidence.push(compare(path, codestream_sha256, &input)?);
    }
    if evidence.is_empty() {
        return Err(format!(
            "the selected decoder matrix contains no codestream satisfying {SELECTION:?}"
        ));
    }
    Ok(evidence)
}

fn compare(
    codestream_path: &str,
    codestream_sha256: String,
    input: &[u8],
) -> Result<NativeComponentOracleEvidence, String> {
    let mut decoder = J2kDecoder::new(input)
        .map_err(|error| format!("production decoder open {codestream_path}: {error}"))?;
    let production = decoder
        .decode_native_components_at_reduction(0)
        .map_err(|error| format!("production decoder decode {codestream_path}: {error}"))?;
    let reference = openjpeg::decode_components(input)
        .map_err(|error| format!("OpenJPEG decode {codestream_path}: {error}"))?;
    validate_native_shapes(codestream_path, &production, &reference)?;

    let mut production_hash = Sha256::new();
    let mut openjpeg_hash = Sha256::new();
    update_image_header(
        &mut production_hash,
        production.dimensions(),
        production.planes().len(),
    )?;
    update_image_header(
        &mut openjpeg_hash,
        reference.dimensions,
        reference.components.len(),
    )?;
    let mut compared_sample_count = 0_u64;
    for (index, (actual, expected)) in production
        .planes()
        .iter()
        .zip(&reference.components)
        .enumerate()
    {
        let actual_samples = cases::unpack_native_plane(actual)?;
        let expected_metadata = (
            expected.dimensions,
            expected.sampling,
            expected.bit_depth,
            expected.signed,
        );
        let actual_metadata = (
            actual.dimensions(),
            (
                u32::from(actual.sampling().0),
                u32::from(actual.sampling().1),
            ),
            actual.bit_depth(),
            actual.signed(),
        );
        if actual_metadata != expected_metadata {
            return Err(format!(
                "{codestream_path} component {index} production metadata {actual_metadata:?} differs from OpenJPEG {expected_metadata:?}"
            ));
        }
        if actual_samples.len() != expected.samples.len() {
            return Err(format!(
                "{codestream_path} component {index} production returned {} samples, OpenJPEG returned {}",
                actual_samples.len(),
                expected.samples.len()
            ));
        }
        update_component_header(&mut production_hash, index, actual, actual_samples.len())?;
        update_openjpeg_component_header(
            &mut openjpeg_hash,
            index,
            expected,
            expected.samples.len(),
        )?;
        for (sample_index, (&actual_sample, &expected_sample)) in
            actual_samples.iter().zip(&expected.samples).enumerate()
        {
            let expected_sample = i64::from(expected_sample);
            if actual_sample != expected_sample {
                return Err(format!(
                    "{codestream_path} component {index} sample {sample_index} is {actual_sample}, OpenJPEG returned {expected_sample}"
                ));
            }
            production_hash.update(actual_sample.to_le_bytes());
            openjpeg_hash.update(expected_sample.to_le_bytes());
        }
        compared_sample_count = compared_sample_count
            .checked_add(u64::try_from(actual_samples.len()).map_err(|_| {
                format!("{codestream_path} component {index} sample count exceeds u64")
            })?)
            .ok_or_else(|| format!("{codestream_path} total sample count exceeds u64"))?;
    }
    let production_components_sha256 = format!("{:x}", production_hash.finalize());
    let openjpeg_components_sha256 = format!("{:x}", openjpeg_hash.finalize());
    if production_components_sha256 != openjpeg_components_sha256 {
        return Err(format!(
            "{codestream_path} canonical native component hashes differ after exact comparison"
        ));
    }
    Ok(NativeComponentOracleEvidence {
        codestream_path: codestream_path.to_string(),
        codestream_sha256,
        selection: SELECTION.to_string(),
        implementation: "OpenJPEG".to_string(),
        version: openjpeg::version(),
        library: openjpeg::library_path().to_string(),
        component_count: production.planes().len(),
        compared_sample_count,
        production_components_sha256: production_components_sha256.clone(),
        openjpeg_components_sha256,
        exact: true,
    })
}

fn validate_native_shapes(
    codestream_path: &str,
    production: &J2kDecodedNativeComponents,
    reference: &openjpeg::OpenJpegDecodedImage,
) -> Result<(), String> {
    if production.dimensions() != reference.dimensions {
        return Err(format!(
            "{codestream_path} production dimensions {:?} differ from OpenJPEG {:?}",
            production.dimensions(),
            reference.dimensions
        ));
    }
    if production.planes().len() != reference.components.len() {
        return Err(format!(
            "{codestream_path} production returned {} components, OpenJPEG returned {}",
            production.planes().len(),
            reference.components.len()
        ));
    }
    Ok(())
}

fn update_image_header(
    hasher: &mut Sha256,
    dimensions: (u32, u32),
    component_count: usize,
) -> Result<(), String> {
    hasher.update(dimensions.0.to_le_bytes());
    hasher.update(dimensions.1.to_le_bytes());
    hasher.update(
        u32::try_from(component_count)
            .map_err(|_| "native component count exceeds u32".to_string())?
            .to_le_bytes(),
    );
    Ok(())
}

fn update_component_header(
    hasher: &mut Sha256,
    index: usize,
    component: &J2kNativeComponentPlane,
    sample_count: usize,
) -> Result<(), String> {
    update_component_metadata(
        hasher,
        index,
        component.dimensions(),
        (
            u32::from(component.sampling().0),
            u32::from(component.sampling().1),
        ),
        component.bit_depth(),
        component.signed(),
        sample_count,
    )
}

fn update_openjpeg_component_header(
    hasher: &mut Sha256,
    index: usize,
    component: &openjpeg::OpenJpegDecodedComponent,
    sample_count: usize,
) -> Result<(), String> {
    update_component_metadata(
        hasher,
        index,
        component.dimensions,
        component.sampling,
        component.bit_depth,
        component.signed,
        sample_count,
    )
}

fn update_component_metadata(
    hasher: &mut Sha256,
    index: usize,
    dimensions: (u32, u32),
    sampling: (u32, u32),
    bit_depth: u8,
    signed: bool,
    sample_count: usize,
) -> Result<(), String> {
    hasher.update(
        u32::try_from(index)
            .map_err(|_| "native component index exceeds u32".to_string())?
            .to_le_bytes(),
    );
    hasher.update(dimensions.0.to_le_bytes());
    hasher.update(dimensions.1.to_le_bytes());
    hasher.update(sampling.0.to_le_bytes());
    hasher.update(sampling.1.to_le_bytes());
    hasher.update([bit_depth, u8::from(signed)]);
    hasher.update(
        u64::try_from(sample_count)
            .map_err(|_| "native component sample count exceeds u64".to_string())?
            .to_le_bytes(),
    );
    Ok(())
}
