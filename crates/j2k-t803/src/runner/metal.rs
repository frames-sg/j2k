// SPDX-License-Identifier: MIT OR Apache-2.0

use std::path::{Path, PathBuf};

#[cfg(target_os = "macos")]
use std::{process::Command, sync::Arc};

#[cfg(target_os = "macos")]
use j2k::{BatchDecodeOptions, BatchLayout, EncodedImage};
#[cfg(target_os = "macos")]
use j2k_core::SurfaceResidency;
#[cfg(target_os = "macos")]
use j2k_metal::MetalBatchDecoder;

#[cfg(target_os = "macos")]
use crate::{ExecutionLocation, PlatformIdentity};

#[cfg(target_os = "macos")]
use super::{cases, encoder, execute};

#[cfg(target_os = "macos")]
const METAL_CLAIM: &str = "Profile-1 Cclass-1 adapter IUT; Profile-1 Cclass-1HF adapter IUT; Annex G JP2 reader via j2k CPU stages (candidate evidence)";

#[cfg(target_os = "macos")]
pub(super) fn run(
    cache_dir: &Path,
    output_dir: Option<PathBuf>,
    development: bool,
) -> Result<(), String> {
    let mut iut = MetalIut::new()?;
    let platform = iut.platform()?;
    let encoder = encoder::run_metal()?;
    execute::run(
        cache_dir,
        output_dir,
        development,
        execute::IutConfig {
            name: "j2k-metal",
            claim: METAL_CLAIM,
            report_stem: "metal",
            features: Vec::from([
                "adapter-iut".to_string(),
                "metal".to_string(),
                "production-batch-decode".to_string(),
            ]),
            platform,
        },
        encoder,
        move |input, reduction_levels| iut.decode(&input, reduction_levels),
    )
}

#[cfg(not(target_os = "macos"))]
pub(super) fn run(
    _cache_dir: &Path,
    _output_dir: Option<PathBuf>,
    _development: bool,
) -> Result<(), String> {
    Err("the Metal T.803 adapter IUT requires macOS and a real Metal device".to_string())
}

#[cfg(target_os = "macos")]
struct MetalIut {
    decoder: MetalBatchDecoder,
}

#[cfg(target_os = "macos")]
impl MetalIut {
    fn new() -> Result<Self, String> {
        let options = BatchDecodeOptions {
            layout: BatchLayout::Nhwc,
            ..BatchDecodeOptions::default()
        };
        let decoder = MetalBatchDecoder::system_default_with_options(options)
            .map_err(|error| error.to_string())?;
        Ok(Self { decoder })
    }

    fn platform(&self) -> Result<PlatformIdentity, String> {
        let device = self.decoder.backend_session().device();
        Ok(PlatformIdentity {
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            hardware: format!("{} (registry {})", device.name(), device.registry_id()),
            driver: macos_driver_identity()?,
        })
    }

    fn decode(
        &mut self,
        input: &Arc<[u8]>,
        reduction_levels: u8,
    ) -> Result<cases::DecodedImage, cases::DecodeFailure> {
        let Some(request) = cases::reduction_request(reduction_levels) else {
            return cases::decode_cpu(input, reduction_levels);
        };
        let component_transform = cases::codestream_component_transform(input)
            .map_err(|error| cases::DecodeFailure::new(error, cases::cpu_route(false)))?;
        let prepared = self
            .decoder
            .prepare(Vec::from([EncodedImage::new(Arc::clone(input), request)]))
            .map_err(|error| {
                cases::DecodeFailure::new(error.to_string(), cases::cpu_route(false))
            })?;
        match cases::prepared_requires_cpu(&prepared) {
            Ok(true) => return cases::decode_cpu(input, reduction_levels),
            Ok(false) => {}
            Err(error) => {
                return Err(cases::DecodeFailure::new(error, cases::cpu_route(false)));
            }
        }
        let info = prepared.groups()[0].info().clone();
        let route = cases::device_route(ExecutionLocation::Metal, component_transform.is_some());
        let decoded = self
            .decoder
            .decode_prepared(&prepared)
            .map_err(|error| cases::DecodeFailure::new(error.to_string(), route.clone()))?;
        if !decoded.errors().is_empty() || !decoded.group_errors().is_empty() {
            let errors = decoded
                .errors()
                .iter()
                .map(ToString::to_string)
                .chain(decoded.group_errors().iter().map(ToString::to_string))
                .collect::<Vec<_>>()
                .join("; ");
            return Err(cases::DecodeFailure::new(errors, route));
        }
        let [group] = decoded.groups() else {
            return Err(cases::DecodeFailure::new(
                format!(
                    "Metal T.803 adapter produced {} groups for one input",
                    decoded.groups().len()
                ),
                route,
            ));
        };
        let [surface] = group.surfaces() else {
            return Err(cases::DecodeFailure::new(
                format!(
                    "Metal T.803 adapter produced {} NHWC surfaces for one input",
                    group.surfaces().len()
                ),
                route,
            ));
        };
        if surface.residency() != SurfaceResidency::MetalResidentDecode {
            return Err(cases::DecodeFailure::new(
                format!(
                    "Metal T.803 adapter returned unexpected {:?} residency",
                    surface.residency()
                ),
                route,
            ));
        }
        let bytes = surface
            .as_bytes()
            .map_err(|error| cases::DecodeFailure::new(error.to_string(), route.clone()))?;
        cases::decoded_interleaved(&info, bytes.as_ref(), component_transform, route.clone())
            .map_err(|error| cases::DecodeFailure::new(error, route))
    }
}

#[cfg(target_os = "macos")]
fn macos_driver_identity() -> Result<String, String> {
    let version = command_output("sw_vers", &["-productVersion"])?;
    let build = command_output("sw_vers", &["-buildVersion"])?;
    Ok(format!("macOS {version} build {build} Metal driver"))
}

#[cfg(target_os = "macos")]
fn command_output(command: &str, args: &[&str]) -> Result<String, String> {
    let output = Command::new(command)
        .args(args)
        .output()
        .map_err(|error| format!("start {command}: {error}"))?;
    if !output.status.success() {
        return Err(format!("{command} exited with {}", output.status));
    }
    let value = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if value.is_empty() {
        return Err(format!("{command} returned an empty value"));
    }
    Ok(value)
}
