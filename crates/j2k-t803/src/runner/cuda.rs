// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{path::Path, path::PathBuf, process::Command, sync::Arc};

use j2k::{BatchDecodeOptions, BatchLayout, EncodedImage};
use j2k_core::SurfaceResidency;
use j2k_cuda::{CudaBatchDecoder, CudaHtj2kProfileReport, CudaSession, Surface};
use j2k_cuda_runtime::CudaContext;

use crate::{
    AcceleratorExecutionEvidence, ExecutionLocation, PlatformIdentity, RouteKind, RouteStage,
    RouteStageName, T803Suite,
};

use super::{adapter_claim, cases, encoder, execute};

pub(super) fn run(
    cache_dir: &Path,
    output_dir: Option<PathBuf>,
    development: bool,
    suite: T803Suite,
) -> Result<(), String> {
    let mut iut = CudaIut::new()?;
    let platform = iut.platform()?;
    execute::run(
        cache_dir,
        output_dir,
        development,
        suite,
        execute::IutConfig {
            name: "j2k-cuda",
            claim: adapter_claim(suite),
            report_stem: "cuda",
            features: Vec::from([
                "adapter-iut".to_string(),
                "cuda".to_string(),
                "production-batch-decode".to_string(),
            ]),
            platform,
        },
        encoder::run_cuda,
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
        let requirements = cases::codestream_requirements(input)
            .map_err(|error| cases::DecodeFailure::new(error, cases::parse_only_route()))?;
        let component_transform = requirements.component_transform;
        let prepared = self
            .decoder
            .prepare(Vec::from([EncodedImage::new(Arc::clone(input), request)]))
            .map_err(|error| {
                cases::DecodeFailure::new(error.to_string(), cases::parse_only_route())
            })?;
        match cases::prepared_requires_cpu(&prepared) {
            Ok(true) => return cases::decode_cpu(input, reduction_levels),
            Ok(false) => {}
            Err(error) => {
                return Err(cases::DecodeFailure::new(error, cases::parse_only_route()));
            }
        }
        let info = prepared.groups()[0].info().clone();
        let decoded = self.decoder.decode_prepared(&prepared).map_err(|error| {
            cases::DecodeFailure::new(error.to_string(), cases::parse_only_route())
        })?;
        if !decoded.errors().is_empty() || !decoded.group_errors().is_empty() {
            let errors = decoded
                .errors()
                .iter()
                .map(ToString::to_string)
                .chain(decoded.group_errors().iter().map(ToString::to_string))
                .collect::<Vec<_>>()
                .join("; ");
            return Err(cases::DecodeFailure::new(errors, cases::parse_only_route()));
        }
        let [group] = decoded.groups() else {
            return Err(cases::DecodeFailure::new(
                format!(
                    "CUDA T.803 adapter produced {} groups for one input",
                    decoded.groups().len()
                ),
                cases::parse_only_route(),
            ));
        };
        let route = cuda_route_from_profile(
            group.profile_report(),
            component_transform.is_some(),
            requirements.high_throughput,
            false,
        )
        .map_err(|error| cases::DecodeFailure::new(error, cases::parse_only_route()))?;
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
        let route = cuda_route_from_profile(
            group.profile_report(),
            component_transform.is_some(),
            requirements.high_throughput,
            true,
        )
        .map_err(|error| cases::DecodeFailure::new(error, route))?;
        cases::decoded_interleaved(&info, &bytes, requirements, route.clone())
            .map_err(|error| cases::DecodeFailure::new(error, route))
    }
}

fn cuda_route_from_profile(
    profile: &CudaHtj2kProfileReport,
    mct: bool,
    high_throughput: bool,
    download_completed: bool,
) -> Result<cases::RouteEvidence, String> {
    let tier1_dispatches = profile
        .detail
        .ht_dispatch_count
        .saturating_add(profile.detail.classic_dispatch_count);
    if profile.dispatch_count == 0 || tier1_dispatches == 0 {
        return Err("production batch result reported no completed CUDA dispatch".to_string());
    }
    if high_throughput && profile.detail.ht_dispatch_count == 0 {
        return Err(
            "production batch result reported no completed CUDA HT Tier-1 dispatch".to_string(),
        );
    }
    if !high_throughput && profile.detail.classic_dispatch_count == 0 {
        return Err(
            "production batch result reported no completed CUDA classic Tier-1 dispatch"
                .to_string(),
        );
    }
    if profile.payload_bytes == 0 {
        return Err("production batch result reported no CUDA payload upload".to_string());
    }
    if mct && profile.detail.mct_dispatch_count == 0 {
        return Err("production batch result reported no CUDA MCT dispatch".to_string());
    }
    if !mct && profile.detail.mct_dispatch_count != 0 {
        return Err("production batch result reported an unexpected CUDA MCT dispatch".to_string());
    }
    let dequant_dispatches = profile
        .detail
        .dequant_dispatch_count
        .saturating_add(profile.detail.fused_dequant_dispatch_count);
    if dequant_dispatches == 0 {
        return Err("production batch result reported no CUDA dequantization dispatch".to_string());
    }
    if profile.detail.store_dispatch_count == 0 {
        return Err("production batch result reported no CUDA output dispatch".to_string());
    }

    let idwt = if profile.detail.idwt_dispatch_count == 0 {
        ExecutionLocation::NotUsed
    } else {
        ExecutionLocation::Cuda
    };
    Ok(cases::RouteEvidence {
        kind: RouteKind::Hybrid,
        stages: Vec::from([
            RouteStage {
                stage: RouteStageName::Parsing,
                location: ExecutionLocation::Cpu,
            },
            RouteStage {
                stage: RouteStageName::Tier1,
                location: ExecutionLocation::Cuda,
            },
            RouteStage {
                stage: RouteStageName::Dequantization,
                location: ExecutionLocation::Cuda,
            },
            RouteStage {
                stage: RouteStageName::Idwt,
                location: idwt,
            },
            RouteStage {
                stage: RouteStageName::Mct,
                location: if mct {
                    ExecutionLocation::Cuda
                } else {
                    ExecutionLocation::NotUsed
                },
            },
            RouteStage {
                stage: RouteStageName::ColorOutput,
                location: ExecutionLocation::Cuda,
            },
            RouteStage {
                stage: RouteStageName::HostToDevice,
                location: ExecutionLocation::Cuda,
            },
            RouteStage {
                stage: RouteStageName::DeviceToHost,
                location: if download_completed {
                    ExecutionLocation::Cuda
                } else {
                    ExecutionLocation::NotUsed
                },
            },
        ]),
        accelerator_execution: Some(AcceleratorExecutionEvidence {
            backend: ExecutionLocation::Cuda,
            ht_tier1_dispatches: profile.detail.ht_dispatch_count,
            ht_refinement_dispatches: profile.detail.ht_refinement_dispatch_count,
            classic_tier1_dispatches: profile.detail.classic_dispatch_count,
            dequantization_dispatches: dequant_dispatches,
            idwt_dispatches: profile.detail.idwt_dispatch_count,
            mct_dispatches: profile.detail.mct_dispatch_count,
            color_output_dispatches: profile.detail.store_dispatch_count,
            uploaded_payload_bytes: Some(profile.payload_bytes),
            metal_host_inputs: None,
            device_to_host_completed: download_completed,
        }),
    })
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
    use super::{cuda_route_from_profile, parse_nvidia_identity};
    use crate::{ExecutionLocation, RouteKind, RouteStageName};
    use j2k_cuda::CudaHtj2kProfileReport;

    #[test]
    fn parses_selected_nvidia_device_identity() {
        assert_eq!(
            parse_nvidia_identity("NVIDIA A100-SXM4-80GB, 580.65.06\n").expect("NVIDIA identity"),
            ("NVIDIA A100-SXM4-80GB".to_string(), "580.65.06".to_string())
        );
        assert!(parse_nvidia_identity("malformed").is_err());
        assert!(parse_nvidia_identity(" , 580.65.06").is_err());
    }

    #[test]
    fn route_uses_completed_cuda_dispatch_counters() {
        let mut profile = CudaHtj2kProfileReport::new();
        profile.payload_bytes = 64;
        profile.dispatch_count = 3;
        profile.detail.ht_dispatch_count = 1;
        profile.detail.dequant_dispatch_count = 1;
        profile.detail.idwt_dispatch_count = 1;
        profile.detail.mct_dispatch_count = 1;
        profile.detail.store_dispatch_count = 1;

        let route =
            cuda_route_from_profile(&profile, true, true, true).expect("observed CUDA route");
        assert_eq!(route.kind, RouteKind::Hybrid);
        assert_eq!(
            route
                .stages
                .iter()
                .map(|stage| (stage.stage, stage.location))
                .collect::<Vec<_>>(),
            vec![
                (RouteStageName::Parsing, ExecutionLocation::Cpu),
                (RouteStageName::Tier1, ExecutionLocation::Cuda),
                (RouteStageName::Dequantization, ExecutionLocation::Cuda),
                (RouteStageName::Idwt, ExecutionLocation::Cuda),
                (RouteStageName::Mct, ExecutionLocation::Cuda),
                (RouteStageName::ColorOutput, ExecutionLocation::Cuda),
                (RouteStageName::HostToDevice, ExecutionLocation::Cuda),
                (RouteStageName::DeviceToHost, ExecutionLocation::Cuda),
            ]
        );
        let execution = route
            .accelerator_execution
            .as_ref()
            .expect("raw CUDA observations");
        assert_eq!(execution.backend, ExecutionLocation::Cuda);
        assert_eq!(execution.ht_tier1_dispatches, 1);
        assert_eq!(execution.mct_dispatches, 1);
        assert_eq!(execution.uploaded_payload_bytes, Some(64));
        assert!(execution.device_to_host_completed);
    }

    #[test]
    fn route_accepts_observed_fused_cuda_dequantization() {
        let mut profile = CudaHtj2kProfileReport::new();
        profile.payload_bytes = 64;
        profile.dispatch_count = 2;
        profile.detail.ht_dispatch_count = 1;
        profile.detail.fused_dequant_dispatch_count = 1;
        profile.detail.store_dispatch_count = 1;

        let route = cuda_route_from_profile(&profile, false, true, true)
            .expect("fused cleanup/dequantization is observed CUDA work");
        assert_eq!(route.kind, RouteKind::Hybrid);
        assert_eq!(
            route
                .stages
                .iter()
                .find(|stage| stage.stage == RouteStageName::Dequantization)
                .map(|stage| stage.location),
            Some(ExecutionLocation::Cuda)
        );
    }

    #[test]
    fn route_rejects_missing_cuda_execution_observations() {
        let error = cuda_route_from_profile(&CudaHtj2kProfileReport::default(), false, true, false)
            .expect_err("empty CUDA profile must not become inferred evidence");
        assert!(error.contains("no completed CUDA dispatch"), "{error}");

        let mut profile = CudaHtj2kProfileReport::new();
        profile.payload_bytes = 64;
        profile.dispatch_count = 2;
        profile.detail.ht_dispatch_count = 1;
        profile.detail.store_dispatch_count = 1;
        let error = cuda_route_from_profile(&profile, true, true, false)
            .expect_err("declared MCT must have an observed CUDA dispatch");
        assert!(error.contains("no CUDA MCT dispatch"), "{error}");

        let mut profile = CudaHtj2kProfileReport::new();
        profile.payload_bytes = 64;
        profile.dispatch_count = 2;
        profile.detail.ht_dispatch_count = 1;
        profile.detail.store_dispatch_count = 1;
        let error = cuda_route_from_profile(&profile, false, true, false)
            .expect_err("dequantization must have an observed CUDA dispatch");
        assert!(error.contains("no CUDA dequantization"), "{error}");

        let mut profile = CudaHtj2kProfileReport::new();
        profile.payload_bytes = 64;
        profile.dispatch_count = 2;
        profile.detail.classic_dispatch_count = 1;
        profile.detail.dequant_dispatch_count = 1;
        profile.detail.store_dispatch_count = 1;
        let error = cuda_route_from_profile(&profile, false, true, false)
            .expect_err("classic dispatches must not prove native HT Tier-1 execution");
        assert!(error.contains("no completed CUDA HT Tier-1"), "{error}");
    }
}
