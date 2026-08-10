// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{fs, path::Path, sync::Arc};

#[cfg(any(
    feature = "cuda-runner",
    all(feature = "metal-runner", target_os = "macos"),
    test
))]
use j2k::{BatchGroupInfo, BatchLayout, DecodeRequest, NativeSampleType};
#[cfg(any(
    feature = "cuda-runner",
    all(feature = "metal-runner", target_os = "macos")
))]
use j2k::{BatchItemError, PreparationDepth, PreparedBatch};
use j2k::{J2kDecodedNativeComponents, J2kDecoder, J2kNativeComponentPlane};
use j2k_codec_math::mct;
use j2k_core::Colorspace;
#[cfg(any(
    feature = "cuda-runner",
    all(feature = "metal-runner", target_os = "macos"),
    test
))]
use j2k_core::Downscale;

use crate::{
    compare_samples, normalize_component, parse_pgx, CaseReport, CaseStatus, Component,
    DecoderCase, ErrorBounds, NormalizationTarget, Part15CaseEvidence,
    Part15EvidenceClassification,
};

mod codestream;
mod file_format;
mod route;

pub(super) use codestream::{codestream_requirements, CodestreamRequirements};
pub(super) use file_format::{run_jp2_cases, run_jph_cases};
#[cfg(any(feature = "cuda-runner", feature = "metal-runner"))]
pub(super) use route::parse_only_route;
pub(super) use route::{cpu_route, RouteEvidence};

#[derive(Debug)]
pub(super) struct DecodedPlane {
    pub(super) dimensions: (u32, u32),
    pub(super) bit_depth: u8,
    pub(super) signed: bool,
    pub(super) sampling: (u8, u8),
    pub(super) samples: Vec<i64>,
}

#[derive(Debug)]
pub(super) struct DecodedImage {
    pub(super) dimensions: (u32, u32),
    pub(super) requirements: CodestreamRequirements,
    pub(super) planes: Vec<DecodedPlane>,
    pub(super) route: RouteEvidence,
}

#[cfg(any(
    feature = "cuda-runner",
    all(feature = "metal-runner", target_os = "macos"),
    test
))]
pub(super) fn decoded_interleaved(
    info: &BatchGroupInfo,
    bytes: &[u8],
    requirements: CodestreamRequirements,
    route: RouteEvidence,
) -> Result<DecodedImage, String> {
    if info.layout != BatchLayout::Nhwc {
        return Err("T.803 adapter output must use NHWC layout".to_string());
    }
    let pixel_count = (info.dimensions.0 as usize)
        .checked_mul(info.dimensions.1 as usize)
        .ok_or_else(|| "T.803 adapter output dimensions overflow".to_string())?;
    let channels = info.color.channels();
    let bytes_per_sample = match info.sample_type {
        NativeSampleType::U8 => 1,
        NativeSampleType::U16 | NativeSampleType::I16 => 2,
        _ => return Err("T.803 adapter output uses an unsupported sample type".to_string()),
    };
    let bytes_per_pixel = channels
        .checked_mul(bytes_per_sample)
        .ok_or_else(|| "T.803 adapter pixel size overflows".to_string())?;
    let expected_len = pixel_count
        .checked_mul(bytes_per_pixel)
        .ok_or_else(|| "T.803 adapter output length overflows".to_string())?;
    if bytes.len() != expected_len {
        return Err(format!(
            "T.803 adapter output length is {}, expected {expected_len}",
            bytes.len()
        ));
    }

    let mut component_samples = Vec::new();
    component_samples
        .try_reserve_exact(channels)
        .map_err(|_| "cannot allocate T.803 adapter component owners".to_string())?;
    for _ in 0..channels {
        let mut samples = Vec::new();
        samples
            .try_reserve_exact(pixel_count)
            .map_err(|_| "cannot allocate T.803 adapter component samples".to_string())?;
        component_samples.push(samples);
    }
    for pixel in bytes.chunks_exact(bytes_per_pixel) {
        for (channel, samples) in component_samples.iter_mut().enumerate() {
            let start = channel * bytes_per_sample;
            let sample = match info.sample_type {
                NativeSampleType::U8 => i64::from(pixel[start]),
                NativeSampleType::U16 => {
                    i64::from(u16::from_ne_bytes([pixel[start], pixel[start + 1]]))
                }
                NativeSampleType::I16 => {
                    i64::from(i16::from_ne_bytes([pixel[start], pixel[start + 1]]))
                }
                _ => unreachable!("sample type was validated above"),
            };
            samples.push(sample);
        }
    }
    let planes = component_samples
        .into_iter()
        .map(|samples| DecodedPlane {
            dimensions: info.dimensions,
            bit_depth: info.precision,
            signed: info.signed,
            sampling: (1, 1),
            samples,
        })
        .collect();
    Ok(DecodedImage {
        dimensions: info.dimensions,
        requirements,
        planes,
        route,
    })
}

#[cfg(any(
    feature = "cuda-runner",
    all(feature = "metal-runner", target_os = "macos"),
    test
))]
pub(super) const fn reduction_request(reduction_levels: u8) -> Option<DecodeRequest> {
    let scale = match reduction_levels {
        0 => return Some(DecodeRequest::Full),
        1 => Downscale::Half,
        2 => Downscale::Quarter,
        3 => Downscale::Eighth,
        _ => return None,
    };
    Some(DecodeRequest::Reduced { scale })
}

#[cfg(any(
    feature = "cuda-runner",
    all(feature = "metal-runner", target_os = "macos")
))]
pub(super) fn prepared_requires_cpu(prepared: &PreparedBatch) -> Result<bool, String> {
    if !prepared.errors().is_empty() {
        if prepared.groups().is_empty()
            && prepared.errors().iter().all(|error| {
                matches!(
                    error.source,
                    BatchItemError::NonRepresentableBatchOutput { .. }
                )
            })
        {
            return Ok(true);
        }
        return Err(prepared
            .errors()
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("; "));
    }
    let [group] = prepared.groups() else {
        return Err(format!(
            "T.803 adapter preparation produced {} groups for one input",
            prepared.groups().len()
        ));
    };
    let [image] = group.images() else {
        return Err(format!(
            "T.803 adapter preparation retained {} images for one input",
            group.images().len()
        ));
    };
    Ok(image.preparation_depth() == PreparationDepth::MetadataOnly)
}

#[derive(Debug)]
pub(super) struct DecodeFailure {
    message: String,
    route: Box<RouteEvidence>,
}

impl DecodeFailure {
    pub(super) fn new(message: impl Into<String>, route: RouteEvidence) -> Self {
        Self {
            message: message.into(),
            route: Box::new(route),
        }
    }
}

pub(super) fn run_decoder_cases(
    cases: &[DecoderCase],
    corpus: &Path,
    mut decode: impl FnMut(Arc<[u8]>, u8) -> Result<DecodedImage, DecodeFailure>,
) -> Vec<CaseReport> {
    let mut reports = Vec::new();
    let mut start = 0;
    while start < cases.len() {
        let first = &cases[start];
        let mut end = start + 1;
        while end < cases.len()
            && cases[end].codestream == first.codestream
            && cases[end].reduction_levels == first.reduction_levels
        {
            end += 1;
        }
        let input_path = corpus.join(&first.codestream);
        let decoded = fs::read(&input_path)
            .map_err(|error| {
                DecodeFailure::new(
                    format!("read {}: {error}", input_path.display()),
                    cpu_route(false),
                )
            })
            .and_then(|input| decode(Arc::from(input), first.reduction_levels));
        for case in &cases[start..end] {
            reports.push(match &decoded {
                Ok(decoded) => {
                    compare_decoder_case(case, decoded, corpus).unwrap_or_else(|error| {
                        error_report(case, Some(case.mse), error, decoded.route.clone())
                    })
                }
                Err(error) => error_report(
                    case,
                    Some(case.mse),
                    error.message.clone(),
                    error.route.as_ref().clone(),
                ),
            });
        }
        start = end;
    }
    reports
}

pub(super) fn decode_cpu(
    input: &[u8],
    reduction_levels: u8,
) -> Result<DecodedImage, DecodeFailure> {
    let requirements = codestream_requirements(input)
        .map_err(|error| DecodeFailure::new(error, cpu_route(false)))?;
    let component_transform = requirements.component_transform;
    let mut iut = J2kDecoder::new(input)
        .map_err(|error| DecodeFailure::new(error.to_string(), cpu_route(false)))?;
    let native = iut
        .decode_native_components_at_reduction(reduction_levels)
        .map_err(|error| {
            DecodeFailure::new(error.to_string(), cpu_route(component_transform.is_some()))
        })?;
    decoded_native_components(&native, requirements)
        .map_err(|error| DecodeFailure::new(error, cpu_route(component_transform.is_some())))
}

fn decoded_native_components(
    decoded: &J2kDecodedNativeComponents,
    requirements: CodestreamRequirements,
) -> Result<DecodedImage, String> {
    let component_transform = requirements.component_transform;
    let dimensions = decoded.dimensions();
    let mut planes = Vec::new();
    planes
        .try_reserve_exact(decoded.planes().len())
        .map_err(|_| "cannot allocate T.803 decoded component owners".to_string())?;
    for plane in decoded.planes() {
        planes.push(DecodedPlane {
            dimensions: plane.dimensions(),
            bit_depth: plane.bit_depth(),
            signed: plane.signed(),
            sampling: plane.sampling(),
            samples: unpack_native_plane(plane)?,
        });
    }
    Ok(DecodedImage {
        dimensions,
        requirements,
        planes,
        route: cpu_route(component_transform.is_some()),
    })
}

fn compare_decoder_case(
    case: &DecoderCase,
    decoded: &DecodedImage,
    corpus: &Path,
) -> Result<CaseReport, String> {
    let plane = decoded
        .planes
        .get(case.component)
        .ok_or_else(|| format!("decoded output has no component {}", case.component))?;
    let decoded_samples = if let (true, Some(component_transform)) = (
        matches!(case.table.as_str(), "C.1" | "C.4"),
        decoded.requirements.component_transform,
    ) {
        let planes = decoded.planes.get(..3).ok_or_else(|| {
            "multi-component transform output has fewer than three planes".to_string()
        })?;
        if planes
            .iter()
            .any(|candidate| candidate.dimensions != plane.dimensions)
        {
            return Err("multi-component transform plane dimensions differ".to_string());
        }
        forward_first_component(
            [&planes[0].samples, &planes[1].samples, &planes[2].samples],
            component_transform,
        )?
    } else {
        plane.samples.clone()
    };
    let normalized = normalize_component(
        Component {
            width: plane.dimensions.0,
            height: plane.dimensions.1,
            bit_depth: plane.bit_depth,
            signed: plane.signed,
            post_decode_subsampling: post_decode_subsampling(plane, decoded),
            samples: &decoded_samples,
        },
        NormalizationTarget {
            width: case.width,
            height: case.height,
            bit_depth: case.bit_depth,
            signed: case.signed,
        },
    )
    .map_err(|error| error.to_string())?;
    let reference_path = corpus.join(&case.reference);
    let reference_bytes = fs::read(&reference_path)
        .map_err(|error| format!("read {}: {error}", reference_path.display()))?;
    let reference = parse_pgx(&reference_bytes).map_err(|error| error.to_string())?;
    if (
        reference.width,
        reference.height,
        reference.bit_depth,
        reference.signed,
    ) != (case.width, case.height, case.bit_depth, case.signed)
    {
        return Err("PGX metadata does not match the pinned case".to_string());
    }
    let comparison = compare_samples(
        &reference.samples,
        &normalized,
        ErrorBounds {
            peak: case.peak,
            mse: case.mse,
        },
    )
    .map_err(|error| error.to_string())?;
    let part15 = match (&case.part15, &decoded.requirements.part15) {
        (Some(selection), Some(codestream)) => Some(Part15CaseEvidence {
            classification: Part15EvidenceClassification::Formal,
            selection: selection.clone(),
            codestream: codestream.clone(),
        }),
        (Some(_), None) => {
            return Err("selected Part 15 case has no parsed CAP/CPF evidence".to_string());
        }
        (None, _) => None,
    };
    Ok(CaseReport {
        id: case.id.clone(),
        table: case.table.clone(),
        status: if comparison.passed {
            CaseStatus::Pass
        } else {
            CaseStatus::Fail
        },
        route: decoded.route.kind,
        peak: Some(comparison.peak),
        mse: Some(comparison.mse),
        allowed_peak: case.peak,
        allowed_mse: Some(case.mse),
        error: None,
        stages: decoded.route.stages.clone(),
        accelerator_execution: decoded.route.accelerator_execution.clone(),
        part15,
    })
}

fn post_decode_subsampling(plane: &DecodedPlane, decoded: &DecodedImage) -> (u8, u8) {
    let common_sampling = decoded
        .planes
        .first()
        .map(|plane| plane.sampling)
        .filter(|first| {
            decoded
                .planes
                .iter()
                .all(|candidate| candidate.sampling == *first)
        })
        .unwrap_or((1, 1));
    let output_sampling = (
        plane.sampling.0 / common_sampling.0,
        plane.sampling.1 / common_sampling.1,
    );
    let native_dimensions = (
        decoded.dimensions.0.div_ceil(u32::from(output_sampling.0)),
        decoded.dimensions.1.div_ceil(u32::from(output_sampling.1)),
    );
    (
        if plane.dimensions.0 == decoded.dimensions.0 && plane.dimensions.0 != native_dimensions.0 {
            output_sampling.0
        } else {
            1
        },
        if plane.dimensions.1 == decoded.dimensions.1 && plane.dimensions.1 != native_dimensions.1 {
            output_sampling.1
        } else {
            1
        },
    )
}

pub(super) fn unpack_native_plane(plane: &J2kNativeComponentPlane) -> Result<Vec<i64>, String> {
    let bytes_per_sample = usize::from(plane.bytes_per_sample());
    let mut samples = Vec::new();
    samples
        .try_reserve_exact(plane.data().len() / bytes_per_sample)
        .map_err(|_| "cannot allocate T.803 native component samples".to_string())?;
    for bytes in plane.data().chunks_exact(bytes_per_sample) {
        let value = match (plane.signed(), bytes) {
            (false, [value]) => i64::from(*value),
            (true, [value]) => i64::from(i8::from_le_bytes([*value])),
            (false, [a, b]) => i64::from(u16::from_le_bytes([*a, *b])),
            (true, [a, b]) => i64::from(i16::from_le_bytes([*a, *b])),
            (false, [a, b, c, d]) => i64::from(u32::from_le_bytes([*a, *b, *c, *d])),
            (true, [a, b, c, d]) => i64::from(i32::from_le_bytes([*a, *b, *c, *d])),
            _ => return Err("unsupported native component storage width".to_string()),
        };
        samples.push(value);
    }
    if samples.len()
        != (plane.dimensions().0 as usize)
            .checked_mul(plane.dimensions().1 as usize)
            .ok_or_else(|| "component dimensions overflow".to_string())?
    {
        return Err("native component storage length does not match dimensions".to_string());
    }
    Ok(samples)
}

fn forward_first_component(
    planes: [&[i64]; 3],
    colorspace: Colorspace,
) -> Result<Vec<i64>, String> {
    let [red, green, blue] = planes;
    if red.len() != green.len() || green.len() != blue.len() {
        return Err("multi-component transform plane dimensions differ".to_string());
    }
    let mut output = Vec::new();
    output
        .try_reserve_exact(red.len())
        .map_err(|_| "cannot allocate T.803 component transform output".to_string())?;
    match colorspace {
        Colorspace::Rct => {
            for ((&red, &green), &blue) in red.iter().zip(green).zip(blue) {
                let numerator = red
                    .checked_add(green.checked_mul(2).ok_or_else(mct_overflow)?)
                    .and_then(|value| value.checked_add(blue))
                    .ok_or_else(mct_overflow)?;
                output.push(numerator.div_euclid(4));
            }
        }
        Colorspace::Ict => {
            for ((&red, &green), &blue) in red.iter().zip(green).zip(blue) {
                #[expect(
                    clippy::cast_precision_loss,
                    clippy::cast_possible_truncation,
                    reason = "T.803 forward ICT intentionally converts bounded integer components into the JPEG 2000 float domain and rounds the finite result back to an integer reference sample"
                )]
                let rounded = (mct::ICT_FWD_Y_R * red as f32
                    + mct::ICT_FWD_Y_G * green as f32
                    + mct::ICT_FWD_Y_B * blue as f32)
                    .round() as i64;
                output.push(rounded);
            }
        }
        _ => return Err("component transform metadata is not RCT or ICT".to_string()),
    }
    Ok(output)
}

fn mct_overflow() -> String {
    "multi-component transform arithmetic overflow".to_string()
}

fn error_report(
    case: &DecoderCase,
    allowed_mse: Option<f64>,
    error: String,
    route: RouteEvidence,
) -> CaseReport {
    CaseReport {
        id: case.id.clone(),
        table: case.table.clone(),
        status: CaseStatus::Error,
        route: route.kind,
        peak: None,
        mse: None,
        allowed_peak: case.peak,
        allowed_mse,
        error: Some(error),
        stages: route.stages,
        accelerator_execution: route.accelerator_execution,
        part15: None,
    }
}

#[cfg(test)]
mod tests {
    use j2k::{
        BatchAlpha, BatchCodecRoute, BatchColor, BatchGroupInfo, BatchLayout,
        BatchWaveletTransform, DecodeRequest, NativeSampleType,
    };
    use j2k_core::{Colorspace, CompressedPayloadKind, CompressedTransferSyntax, Downscale};

    use super::{
        codestream_requirements, decoded_interleaved, forward_first_component, reduction_request,
        CodestreamRequirements, RouteEvidence,
    };
    use crate::{ExecutionLocation, RouteKind, RouteStage, RouteStageName};
    use j2k::J2kDecoder;
    use j2k_test_support::{minimal_j2k_codestream, wrap_jp2_codestream};

    fn route() -> RouteEvidence {
        let stages = Vec::from([RouteStage {
            stage: RouteStageName::Parsing,
            location: ExecutionLocation::Cpu,
        }]);
        RouteEvidence {
            kind: RouteKind::Hybrid,
            stages,
            accelerator_execution: None,
        }
    }

    fn requirements(component_transform: Option<Colorspace>) -> CodestreamRequirements {
        CodestreamRequirements {
            component_transform,
            #[cfg(any(feature = "cuda-runner", feature = "metal-runner"))]
            high_throughput: false,
            part15: None,
        }
    }

    fn info(
        color: BatchColor,
        sample_type: NativeSampleType,
        precision: u8,
        signed: bool,
    ) -> BatchGroupInfo {
        BatchGroupInfo {
            dimensions: (2, 1),
            color,
            alpha: if color == BatchColor::Rgba {
                BatchAlpha::Straight
            } else {
                BatchAlpha::None
            },
            precision,
            signed,
            sample_type,
            layout: BatchLayout::Nhwc,
            colorspace: Colorspace::Rgb,
            route: BatchCodecRoute::Classic,
            transform: BatchWaveletTransform::Reversible53,
            transfer_syntax: CompressedTransferSyntax::Jpeg2000Lossless,
            payload_kind: CompressedPayloadKind::Jpeg2000Codestream,
        }
    }

    #[test]
    fn interleaved_native_bytes_are_split_into_component_planes() {
        let decoded = decoded_interleaved(
            &info(BatchColor::Rgb, NativeSampleType::U8, 8, false),
            &[1, 2, 3, 4, 5, 6],
            requirements(None),
            route(),
        )
        .expect("decode interleaved RGB bytes");

        assert_eq!(decoded.planes[0].samples, [1, 4]);
        assert_eq!(decoded.planes[1].samples, [2, 5]);
        assert_eq!(decoded.planes[2].samples, [3, 6]);
    }

    #[test]
    fn interleaved_signed_samples_preserve_native_endianness() {
        let bytes = [-2048_i16, 2047]
            .into_iter()
            .flat_map(i16::to_ne_bytes)
            .collect::<Vec<_>>();
        let decoded = decoded_interleaved(
            &info(BatchColor::Gray, NativeSampleType::I16, 12, true),
            &bytes,
            requirements(None),
            route(),
        )
        .expect("decode interleaved signed bytes");

        assert_eq!(decoded.planes[0].samples, [-2048, 2047]);
    }

    #[test]
    fn interleaved_decode_rejects_layout_or_length_mismatch() {
        let mut planar = info(BatchColor::Rgb, NativeSampleType::U8, 8, false);
        planar.layout = BatchLayout::Nchw;
        assert!(
            decoded_interleaved(&planar, &[0; 6], requirements(None), route())
                .expect_err("planar input must be rejected")
                .contains("NHWC")
        );
        assert!(decoded_interleaved(
            &info(BatchColor::Rgb, NativeSampleType::U8, 8, false),
            &[0; 5],
            requirements(None),
            route(),
        )
        .expect_err("short input must be rejected")
        .contains("length"));
    }

    #[test]
    fn adapter_reduction_uses_only_the_public_downscale_range() {
        assert_eq!(reduction_request(0), Some(DecodeRequest::Full));
        assert_eq!(
            reduction_request(1),
            Some(DecodeRequest::Reduced {
                scale: Downscale::Half,
            })
        );
        assert_eq!(
            reduction_request(3),
            Some(DecodeRequest::Reduced {
                scale: Downscale::Eighth,
            })
        );
        assert_eq!(reduction_request(4), None);
    }

    #[test]
    fn forward_mct_recovers_the_first_codestream_component() {
        let red = [100, 3];
        let green = [150, 4];
        let blue = [200, 8];

        assert_eq!(
            forward_first_component([&red, &green, &blue], Colorspace::Rct).expect("forward RCT"),
            [150, 4]
        );
        assert_eq!(
            forward_first_component([&red, &green, &blue], Colorspace::Ict).expect("forward ICT"),
            [141, 4]
        );
    }

    #[test]
    fn codestream_transform_detection_is_independent_of_color_space_inference() {
        let mut codestream = minimal_j2k_codestream();
        let siz = codestream
            .windows(2)
            .position(|marker| marker == [0xff, 0x51])
            .expect("SIZ marker");
        let cod = codestream
            .windows(2)
            .position(|marker| marker == [0xff, 0x52])
            .expect("COD marker");
        let extra_components = (3..257).flat_map(|_| [0x07, 0x01, 0x01]);
        codestream.splice(cod..cod, extra_components);
        codestream[siz + 38..siz + 40].copy_from_slice(&257_u16.to_be_bytes());
        let siz_length = u16::from_be_bytes([codestream[siz + 2], codestream[siz + 3]])
            .checked_add(254 * 3)
            .expect("expanded SIZ length");
        codestream[siz + 2..siz + 4].copy_from_slice(&siz_length.to_be_bytes());

        assert_eq!(
            J2kDecoder::inspect(&codestream)
                .expect("wide-component inspect")
                .colorspace,
            Colorspace::IccTagged
        );
        assert_eq!(
            codestream_requirements(&codestream)
                .expect("COD requirements")
                .component_transform,
            Some(Colorspace::Rct)
        );
        let jp2 = wrap_jp2_codestream(&codestream, 128, 64, 257, 8, 16);
        assert_eq!(
            codestream_requirements(&jp2)
                .expect("wrapped COD requirements")
                .component_transform,
            Some(Colorspace::Rct)
        );
    }

    #[test]
    fn forward_mct_rejects_mismatched_plane_lengths() {
        let one = [1];
        let two = [2, 3];
        let three = [4];
        let error = forward_first_component([&one, &two, &three], Colorspace::Rct)
            .expect_err("mismatched planes must fail");
        assert!(error.contains("dimensions"));
    }
}
