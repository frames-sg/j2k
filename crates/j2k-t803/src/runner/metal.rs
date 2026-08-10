// SPDX-License-Identifier: MIT OR Apache-2.0

use std::path::{Path, PathBuf};

#[cfg(target_os = "macos")]
use std::{process::Command, sync::Arc};

#[cfg(target_os = "macos")]
use j2k::{BatchDecodeOptions, BatchLayout, EncodedImage};
#[cfg(target_os = "macos")]
use j2k_core::SurfaceResidency;
#[cfg(target_os = "macos")]
use j2k_metal::{MetalBatchDecoder, MetalDecodeDispatchReport};
use objc2_metal::MTLDevice as _;

#[cfg(target_os = "macos")]
use crate::{
    AcceleratorExecutionEvidence, ExecutionLocation, PlatformIdentity, RouteKind, RouteStage,
    RouteStageName, T803Suite,
};

#[cfg(target_os = "macos")]
use super::{adapter_claim, cases, encoder, execute};

#[cfg(target_os = "macos")]
pub(super) fn run(
    cache_dir: &Path,
    output_dir: Option<PathBuf>,
    development: bool,
    suite: T803Suite,
) -> Result<(), String> {
    let mut iut = MetalIut::new()?;
    let platform = iut.platform()?;
    execute::run(
        cache_dir,
        output_dir,
        development,
        suite,
        execute::IutConfig {
            name: "j2k-metal",
            claim: adapter_claim(suite),
            report_stem: "metal",
            features: Vec::from([
                "adapter-iut".to_string(),
                "metal".to_string(),
                "production-batch-decode".to_string(),
            ]),
            platform,
        },
        encoder::run_metal,
        move |input, reduction_levels| iut.decode(&input, reduction_levels),
    )
}

#[cfg(not(target_os = "macos"))]
pub(super) fn run(
    _cache_dir: &Path,
    _output_dir: Option<PathBuf>,
    _development: bool,
    _suite: crate::T803Suite,
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
            hardware: format!("{} (registry {})", device.name(), device.registryID()),
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
                    "Metal T.803 adapter produced {} groups for one input",
                    decoded.groups().len()
                ),
                cases::parse_only_route(),
            ));
        };
        let route = metal_route_from_dispatch(
            group.dispatch_report(),
            component_transform.is_some(),
            requirements.high_throughput,
            false,
        )
        .map_err(|error| cases::DecodeFailure::new(error, cases::parse_only_route()))?;
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
        let route = metal_route_from_dispatch(
            group.dispatch_report(),
            component_transform.is_some(),
            requirements.high_throughput,
            true,
        )
        .map_err(|error| cases::DecodeFailure::new(error, route))?;
        cases::decoded_interleaved(&info, bytes.as_ref(), requirements, route.clone())
            .map_err(|error| cases::DecodeFailure::new(error, route))
    }
}

#[cfg(target_os = "macos")]
fn metal_route_from_dispatch(
    report: &MetalDecodeDispatchReport,
    mct: bool,
    high_throughput: bool,
    download_completed: bool,
) -> Result<cases::RouteEvidence, String> {
    if report.tier1 == 0 {
        return Err(
            "production batch result reported no completed Metal Tier-1 dispatch".to_string(),
        );
    }
    if high_throughput && report.ht_tier1 == 0 {
        return Err(
            "production batch result reported no completed Metal HT Tier-1 dispatch".to_string(),
        );
    }
    if !high_throughput && report.classic_tier1 == 0 {
        return Err(
            "production batch result reported no completed Metal classic Tier-1 dispatch"
                .to_string(),
        );
    }
    if report.dequantization == 0 {
        return Err(
            "production batch result reported no completed Metal dequantization".to_string(),
        );
    }
    if report.color_output == 0 {
        return Err(
            "production batch result reported no completed Metal output dispatch".to_string(),
        );
    }
    if report.host_to_device == 0 {
        return Err("production batch result reported no Metal input transfer".to_string());
    }
    if mct && report.mct == 0 {
        return Err("production batch result reported no completed Metal MCT".to_string());
    }
    if !mct && report.mct != 0 {
        return Err(
            "production batch result reported an unexpected completed Metal MCT".to_string(),
        );
    }

    Ok(cases::RouteEvidence {
        kind: RouteKind::Hybrid,
        stages: Vec::from([
            RouteStage {
                stage: RouteStageName::Parsing,
                location: ExecutionLocation::Cpu,
            },
            RouteStage {
                stage: RouteStageName::Tier1,
                location: ExecutionLocation::Metal,
            },
            RouteStage {
                stage: RouteStageName::Dequantization,
                location: ExecutionLocation::Metal,
            },
            RouteStage {
                stage: RouteStageName::Idwt,
                location: if report.idwt == 0 {
                    ExecutionLocation::NotUsed
                } else {
                    ExecutionLocation::Metal
                },
            },
            RouteStage {
                stage: RouteStageName::Mct,
                location: if mct {
                    ExecutionLocation::Metal
                } else {
                    ExecutionLocation::NotUsed
                },
            },
            RouteStage {
                stage: RouteStageName::ColorOutput,
                location: ExecutionLocation::Metal,
            },
            RouteStage {
                stage: RouteStageName::HostToDevice,
                location: ExecutionLocation::Metal,
            },
            RouteStage {
                stage: RouteStageName::DeviceToHost,
                location: if download_completed {
                    ExecutionLocation::Metal
                } else {
                    ExecutionLocation::NotUsed
                },
            },
        ]),
        accelerator_execution: Some(AcceleratorExecutionEvidence {
            backend: ExecutionLocation::Metal,
            ht_tier1_dispatches: report.ht_tier1,
            ht_refinement_dispatches: report.ht_refinement,
            classic_tier1_dispatches: report.classic_tier1,
            dequantization_dispatches: report.dequantization,
            idwt_dispatches: report.idwt,
            mct_dispatches: report.mct,
            color_output_dispatches: report.color_output,
            uploaded_payload_bytes: None,
            metal_host_inputs: Some(report.host_to_device),
            device_to_host_completed: download_completed,
        }),
    })
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

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::metal_route_from_dispatch;
    use crate::{ExecutionLocation, RouteKind, RouteStageName};
    use j2k_metal::MetalDecodeDispatchReport;

    #[test]
    fn route_uses_completed_metal_stage_observations() {
        let mut report = MetalDecodeDispatchReport::new();
        report.tier1 = 1;
        report.ht_tier1 = 1;
        report.dequantization = 1;
        report.idwt = 1;
        report.mct = 1;
        report.color_output = 1;
        report.host_to_device = 1;

        let route =
            metal_route_from_dispatch(&report, true, true, true).expect("observed Metal route");
        assert_eq!(route.kind, RouteKind::Hybrid);
        assert_eq!(
            route
                .stages
                .iter()
                .map(|stage| (stage.stage, stage.location))
                .collect::<Vec<_>>(),
            vec![
                (RouteStageName::Parsing, ExecutionLocation::Cpu),
                (RouteStageName::Tier1, ExecutionLocation::Metal),
                (RouteStageName::Dequantization, ExecutionLocation::Metal),
                (RouteStageName::Idwt, ExecutionLocation::Metal),
                (RouteStageName::Mct, ExecutionLocation::Metal),
                (RouteStageName::ColorOutput, ExecutionLocation::Metal),
                (RouteStageName::HostToDevice, ExecutionLocation::Metal),
                (RouteStageName::DeviceToHost, ExecutionLocation::Metal),
            ]
        );
        let execution = route
            .accelerator_execution
            .as_ref()
            .expect("raw Metal observations");
        assert_eq!(execution.backend, ExecutionLocation::Metal);
        assert_eq!(execution.ht_tier1_dispatches, 1);
        assert_eq!(execution.mct_dispatches, 1);
        assert_eq!(execution.metal_host_inputs, Some(1));
        assert!(execution.device_to_host_completed);
    }

    #[test]
    fn route_rejects_missing_metal_execution_observations() {
        let error =
            metal_route_from_dispatch(&MetalDecodeDispatchReport::new(), false, true, false)
                .expect_err("empty Metal report must not become inferred evidence");
        assert!(error.contains("no completed Metal Tier-1"), "{error}");

        let mut report = MetalDecodeDispatchReport::new();
        report.tier1 = 1;
        report.ht_tier1 = 1;
        report.dequantization = 1;
        report.mct = 1;
        report.color_output = 1;
        report.host_to_device = 1;
        let error = metal_route_from_dispatch(&report, false, true, false)
            .expect_err("an observed MCT dispatch cannot be reported as unused");
        assert!(error.contains("unexpected completed Metal MCT"), "{error}");

        let mut report = MetalDecodeDispatchReport::new();
        report.tier1 = 1;
        report.classic_tier1 = 1;
        report.dequantization = 1;
        report.color_output = 1;
        report.host_to_device = 1;
        let error = metal_route_from_dispatch(&report, false, true, false)
            .expect_err("classic dispatches must not prove native HT Tier-1 execution");
        assert!(error.contains("no completed Metal HT Tier-1"), "{error}");
    }
}
