// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{path::Path, path::PathBuf, process::Command, sync::Arc};

use j2k::{BatchDecodeOptions, BatchLayout, EncodedImage};
use j2k_core::SurfaceResidency;
use j2k_cuda::{CudaBatchDecoder, CudaSession, Surface};
use j2k_cuda_runtime::CudaContext;

use crate::{ExecutionLocation, PlatformIdentity};

use super::{cases, encoder, execute};

const CUDA_CLAIM: &str = "Profile-1 Cclass-1 adapter IUT; Profile-1 Cclass-1HF adapter IUT; Annex G JP2 reader via j2k CPU stages (candidate evidence)";

pub(super) fn run(
    cache_dir: &Path,
    output_dir: Option<PathBuf>,
    development: bool,
) -> Result<(), String> {
    let mut iut = CudaIut::new()?;
    let platform = iut.platform()?;
    let encoder = encoder::run_cuda()?;
    execute::run(
        cache_dir,
        output_dir,
        development,
        execute::IutConfig {
            name: "j2k-cuda",
            claim: CUDA_CLAIM,
            report_stem: "cuda",
            features: Vec::from([
                "adapter-iut".to_string(),
                "cuda".to_string(),
                "production-batch-decode".to_string(),
            ]),
            platform,
        },
        encoder,
        move |input, reduction_levels| iut.decode(&input, reduction_levels),
    )
}

struct CudaIut {
    decoder: CudaBatchDecoder,
    device_ordinal: usize,
}

impl CudaIut {
    fn new() -> Result<Self, String> {
        let context = CudaContext::system_default().map_err(|error| error.to_string())?;
        let device_ordinal = context.device_ordinal();
        let options = BatchDecodeOptions {
            layout: BatchLayout::Nhwc,
            ..BatchDecodeOptions::default()
        };
        let decoder =
            CudaBatchDecoder::with_session_and_options(CudaSession::with_context(context), options);
        Ok(Self {
            decoder,
            device_ordinal,
        })
    }

    fn platform(&self) -> Result<PlatformIdentity, String> {
        let (name, driver) = nvidia_device_identity(self.device_ordinal)?;
        Ok(PlatformIdentity {
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            hardware: format!("{name} (CUDA device {})", self.device_ordinal),
            driver: format!("NVIDIA {driver}"),
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
        let route = cases::device_route(ExecutionLocation::Cuda, component_transform.is_some());
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
                    "CUDA T.803 adapter produced {} groups for one input",
                    decoded.groups().len()
                ),
                route,
            ));
        };
        let [surface] = group.surfaces() else {
            return Err(cases::DecodeFailure::new(
                format!(
                    "CUDA T.803 adapter produced {} NHWC surfaces for one input",
                    group.surfaces().len()
                ),
                route,
            ));
        };
        if surface.residency() != SurfaceResidency::CudaResidentDecode {
            return Err(cases::DecodeFailure::new(
                format!(
                    "CUDA T.803 adapter returned unexpected {:?} residency",
                    surface.residency()
                ),
                route,
            ));
        }
        let bytes = Surface::download_batch_tight(group.surfaces())
            .map_err(|error| cases::DecodeFailure::new(error.to_string(), route.clone()))?;
        cases::decoded_interleaved(&info, &bytes, component_transform, route.clone())
            .map_err(|error| cases::DecodeFailure::new(error, route))
    }
}

fn nvidia_device_identity(device_ordinal: usize) -> Result<(String, String), String> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,driver_version",
            "--format=csv,noheader,nounits",
            "-i",
            &device_ordinal.to_string(),
        ])
        .output()
        .map_err(|error| format!("start nvidia-smi: {error}"))?;
    if !output.status.success() {
        return Err(format!("nvidia-smi exited with {}", output.status));
    }
    parse_nvidia_identity(&String::from_utf8_lossy(&output.stdout))
}

fn parse_nvidia_identity(output: &str) -> Result<(String, String), String> {
    let line = output
        .lines()
        .find(|line| !line.trim().is_empty())
        .ok_or_else(|| "nvidia-smi returned an empty value".to_string())?;
    let (name, driver) = line
        .split_once(',')
        .ok_or_else(|| "nvidia-smi returned malformed device identity".to_string())?;
    let name = name.trim();
    let driver = driver.trim();
    if name.is_empty() || driver.is_empty() {
        return Err("nvidia-smi returned malformed device identity".to_string());
    }
    Ok((name.to_string(), driver.to_string()))
}

#[cfg(test)]
mod tests {
    use super::parse_nvidia_identity;

    #[test]
    fn parses_selected_nvidia_device_identity() {
        assert_eq!(
            parse_nvidia_identity("NVIDIA A100-SXM4-80GB, 580.65.06\n").expect("NVIDIA identity"),
            ("NVIDIA A100-SXM4-80GB".to_string(), "580.65.06".to_string())
        );
        assert!(parse_nvidia_identity("malformed").is_err());
        assert!(parse_nvidia_identity(" , 580.65.06").is_err());
    }
}
