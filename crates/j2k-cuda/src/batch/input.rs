// SPDX-License-Identifier: MIT OR Apache-2.0

//! Prepared-input validation and CUDA adapter mapping.

use j2k::{PreparedDeviceOperation, PreparedDeviceOperationGroup};

use super::{
    cuda_range_storage, Arc, BackendKind, BatchColor, BatchGroupInfo, BatchLayout,
    CudaResidentBatchBuffer, CudaSurfaceStats, Error, J2kDecodeWarning, PixelFormat,
    PreparedBatchGroup, PreparedImage, Rect, Surface, SurfaceResidency,
};
use crate::decoder::PreparedInputGeometry;

/// One homogeneous output group as CUDA submission sees it: the images of a
/// prepared batch group, or exact operations prepared for the device.
#[derive(Clone, Copy)]
pub(super) struct CudaGroupView<'g> {
    pub(super) info: &'g BatchGroupInfo,
    pub(super) source_indices: &'g [usize],
    images: GroupImages<'g>,
}

#[derive(Clone, Copy)]
enum GroupImages<'g> {
    /// A prepared batch group's images and the settings it was prepared with.
    Prepared(&'g [PreparedImage], j2k::DecodeSettings),
    /// Exact operations sharing one cancellation token.
    Operations(&'g PreparedDeviceOperationGroup),
}

/// One image of a [`CudaGroupView`].
#[derive(Clone, Copy)]
pub(super) enum GroupImage<'g> {
    Prepared(&'g PreparedImage, j2k::DecodeSettings),
    Operation(&'g PreparedDeviceOperation),
}

impl<'g> From<&'g PreparedBatchGroup> for CudaGroupView<'g> {
    fn from(group: &'g PreparedBatchGroup) -> Self {
        Self {
            info: group.info(),
            source_indices: group.source_indices(),
            images: GroupImages::Prepared(group.images(), group.options().settings),
        }
    }
}

impl<'g> From<&'g PreparedDeviceOperationGroup> for CudaGroupView<'g> {
    fn from(group: &'g PreparedDeviceOperationGroup) -> Self {
        Self {
            info: group.info(),
            source_indices: group.source_indices(),
            images: GroupImages::Operations(group),
        }
    }
}

impl<'g> CudaGroupView<'g> {
    /// Stop before planning or submitting a cancelled operation group.
    /// Prepared batches carry no cancellation.
    pub(super) fn check_cancelled(self) -> Result<(), Error> {
        match self.images {
            GroupImages::Prepared(..) => Ok(()),
            GroupImages::Operations(group) => group.check_cancelled().map_err(Error::Decode),
        }
    }

    /// Images in dense group order.
    pub(super) fn images(self) -> impl ExactSizeIterator<Item = GroupImage<'g>> {
        let len = match self.images {
            GroupImages::Prepared(images, _) => images.len(),
            GroupImages::Operations(group) => group.operations().len(),
        };
        (0..len).map(move |index| match self.images {
            GroupImages::Prepared(images, settings) => {
                GroupImage::Prepared(&images[index], settings)
            }
            GroupImages::Operations(group) => GroupImage::Operation(&group.operations()[index]),
        })
    }

    /// Settings of the legacy grayscale parse-from-bytes route, which exact
    /// operations never take.
    pub(super) fn fallback_settings(self) -> j2k_native::DecodeSettings {
        native_decode_settings(match self.images {
            GroupImages::Prepared(_, settings) => settings,
            GroupImages::Operations(group) => group.operations().first().map_or_else(
                j2k::DecodeSettings::default,
                PreparedDeviceOperation::decode_settings,
            ),
        })
    }
}

impl<'g> GroupImage<'g> {
    fn bytes(self) -> &'g [u8] {
        match self {
            Self::Prepared(image, _) => image.bytes(),
            Self::Operation(operation) => operation.bytes(),
        }
    }

    fn htj2k_plan(self) -> Option<&'g j2k::PreparedHtj2kPlan> {
        match self {
            Self::Prepared(image, _) => image.htj2k_plan(),
            Self::Operation(operation) => operation.htj2k_plan(),
        }
    }

    fn classic_plan(self) -> Option<&'g j2k::PreparedClassicPlan> {
        match self {
            Self::Prepared(image, _) => image.classic_plan(),
            Self::Operation(operation) => operation.classic_plan(),
        }
    }

    /// Exact operations describe only their decoded rectangle, so they can
    /// never take the legacy parse-from-bytes route.
    fn geometry(self) -> PreparedInputGeometry {
        match self {
            Self::Prepared(image, _) => PreparedInputGeometry::Request(image.plan()),
            Self::Operation(operation) => PreparedInputGeometry::Exact(operation.output_rect()),
        }
    }

    /// Decoded rectangle in the coordinates of the decoded level.
    pub(super) fn output_rect(self) -> Rect {
        self.geometry().output_rect()
    }

    fn settings(self) -> j2k_native::DecodeSettings {
        native_decode_settings(match self {
            Self::Prepared(_, settings) => settings,
            Self::Operation(operation) => operation.decode_settings(),
        })
    }

    fn used_lenient_metadata_recovery(self) -> bool {
        match self {
            Self::Prepared(image, _) => image.used_lenient_metadata_recovery(),
            Self::Operation(operation) => operation.used_lenient_metadata_recovery(),
        }
    }
}

pub(super) fn group_pixel_format(info: &BatchGroupInfo) -> Result<PixelFormat, Error> {
    info.native_pixel_format().ok_or(Error::capability_rejected(
        j2k_core::CapabilityRejection::unsupported_format(
            "CUDA batch output color/sample type is unsupported",
        ),
    ))
}

pub(super) fn validate_layout(info: &BatchGroupInfo) -> Result<(), Error> {
    if !matches!(info.layout, BatchLayout::Nchw | BatchLayout::Nhwc) {
        return Err(Error::capability_rejected(
            j2k_core::CapabilityRejection::unsupported_format(
                "CUDA batch output layout is unsupported",
            ),
        ));
    }
    Ok(())
}

pub(super) fn native_decode_settings(settings: j2k::DecodeSettings) -> j2k_native::DecodeSettings {
    j2k_native::DecodeSettings {
        resolve_palette_indices: true,
        strict: settings.is_strict(),
        target_resolution: None,
    }
}

fn native_referenced_htj2k_plan(
    plan: &j2k::PreparedHtj2kPlan,
) -> &j2k_native::J2kReferencedHtj2kPlan {
    plan.geometry()
}

fn native_referenced_classic_plan(
    plan: &j2k::PreparedClassicPlan,
) -> &j2k_native::J2kReferencedClassicPlan {
    plan.geometry()
}

pub(super) fn native_color_inputs(
    group: CudaGroupView<'_>,
) -> Result<Vec<crate::decoder::NativeColorBatchInput<'_>>, Error> {
    group
        .images()
        .zip(group.source_indices.iter().copied())
        .map(|(image, source_index)| {
            let referenced_plan = match image.htj2k_plan() {
                Some(prepared_plan)
                    if group.info.color == BatchColor::Rgb && prepared_plan.is_color() =>
                {
                    Some(native_referenced_htj2k_plan(prepared_plan))
                }
                Some(prepared_plan)
                    if group.info.color == BatchColor::Rgba && prepared_plan.is_rgba() =>
                {
                    Some(native_referenced_htj2k_plan(prepared_plan))
                }
                Some(_) => {
                    return Err(Error::capability_rejected(
                        j2k_core::CapabilityRejection::geometry_mismatch(
                            "exact CUDA color batch received incompatible prepared geometry",
                        ),
                    ))
                }
                None => None,
            };
            let referenced_classic_plan = match image.classic_plan() {
                Some(prepared_plan)
                    if group.info.color == BatchColor::Rgb && prepared_plan.is_color() =>
                {
                    Some(native_referenced_classic_plan(prepared_plan))
                }
                Some(prepared_plan)
                    if group.info.color == BatchColor::Rgba && prepared_plan.is_rgba() =>
                {
                    Some(native_referenced_classic_plan(prepared_plan))
                }
                Some(_) => {
                    return Err(Error::capability_rejected(
                        j2k_core::CapabilityRejection::geometry_mismatch(
                            "exact CUDA color batch received incompatible classic geometry",
                        ),
                    ));
                }
                None => None,
            };
            if referenced_plan.is_none() && referenced_classic_plan.is_none() {
                return Err(Error::capability_rejected(
                    j2k_core::CapabilityRejection::missing_prepared_plan(
                        "exact CUDA color batch requires a supported prepared device plan",
                    ),
                ));
            }
            Ok(crate::decoder::NativeColorBatchInput {
                source_index,
                bytes: image.bytes(),
                geometry: image.geometry(),
                referenced_plan,
                referenced_classic_plan,
                settings: image.settings(),
            })
        })
        .collect()
}

/// Grayscale inputs of a group. A prepared image without an offset plan is
/// planned from its bytes; an exact operation without one fails its group
/// during planning.
pub(super) fn grayscale_inputs(
    group: CudaGroupView<'_>,
) -> Vec<crate::decoder::grayscale_batch::GrayscaleBatchInput<'_>> {
    group
        .images()
        .zip(group.source_indices.iter().copied())
        .map(
            |(image, source_index)| crate::decoder::grayscale_batch::GrayscaleBatchInput {
                source_index,
                bytes: image.bytes(),
                geometry: Some(image.geometry()),
                referenced_plan: image.htj2k_plan().map(native_referenced_htj2k_plan),
                referenced_classic_plan: image.classic_plan().map(native_referenced_classic_plan),
            },
        )
        .collect()
}

pub(super) fn native_color_group_storage(
    info: &BatchGroupInfo,
    fmt: PixelFormat,
    output: crate::decoder::NativeColorOwnedBatch,
) -> (Vec<Surface>, CudaResidentBatchBuffer) {
    let crate::decoder::NativeColorOwnedBatch {
        buffer,
        ranges,
        execution,
    } = output;
    let shared = Arc::new(buffer);
    let surfaces = if info.layout == BatchLayout::Nhwc {
        ranges
            .iter()
            .map(|range| Surface {
                backend: BackendKind::Cuda,
                residency: SurfaceResidency::CudaResidentDecode,
                dimensions: info.dimensions,
                fmt,
                pitch_bytes: info.dimensions.0 as usize * fmt.bytes_per_pixel(),
                stats: CudaSurfaceStats {
                    total: execution.kernel_dispatches(),
                    copy: execution.copy_kernel_dispatches(),
                    decode: execution.decode_kernel_dispatches(),
                },
                storage: cuda_range_storage(shared.clone(), range.offset, range.len),
            })
            .collect()
    } else {
        Vec::new()
    };
    (
        surfaces,
        CudaResidentBatchBuffer {
            buffer: shared,
            ranges,
        },
    )
}

pub(super) fn decode_warnings(
    group: CudaGroupView<'_>,
) -> Result<Vec<Vec<J2kDecodeWarning>>, Error> {
    let mut budget = crate::allocation::HostPhaseBudget::new("CUDA batch decode warnings");
    let mut warnings = budget.try_vec_with_capacity(group.images().len())?;
    for image in group.images() {
        let mut image_warnings = Vec::new();
        if image.used_lenient_metadata_recovery() {
            // `J2kDecodeWarning` is a ZST, so this push does not allocate.
            image_warnings.push(J2kDecodeWarning::LenientMetadataRecovery);
        }
        warnings.push(image_warnings);
    }
    Ok(warnings)
}
