// SPDX-License-Identifier: Apache-2.0

//! Facade crate for the `signinum` pathology image codecs.
//!
//! Runtime backend requests default to [`BackendRequest::Auto`]. The facade
//! compiles portable CPU codecs plus the Metal adapter by default, then uses
//! device backends for supported workloads when they are compiled and available.
//! CPU is the fallback for `Auto`, not the policy default.
//!
//! # Examples
//!
//! JPEG decode imports:
//!
//! ```no_run
//! use signinum::jpeg::{Decoder, PixelFormat};
//!
//! let bytes = std::fs::read("tile.jpg").unwrap();
//! let mut decoder = Decoder::new(&bytes).unwrap();
//! let info = decoder.info();
//! let stride = info.dimensions.0 as usize * PixelFormat::Rgb8.bytes_per_pixel();
//! let mut out = vec![0; stride * info.dimensions.1 as usize];
//! decoder.decode_into(&mut out, stride, PixelFormat::Rgb8).unwrap();
//! ```
//!
//! JPEG 2000 lossless encode with the runtime default:
//!
//! ```
//! use signinum::j2k::{encode_j2k_lossless, J2kLosslessEncodeOptions, J2kLosslessSamples};
//! use signinum::BackendRequest;
//!
//! assert_eq!(BackendRequest::default(), BackendRequest::Auto);
//! let pixels = [0u8; 4 * 4];
//! let samples = J2kLosslessSamples::new(&pixels, 4, 4, 1, 8, false).unwrap();
//! let encoded = encode_j2k_lossless(samples, &J2kLosslessEncodeOptions::default()).unwrap();
//! assert!(encoded.codestream.starts_with(&[0xFF, 0x4F]));
//! ```
//!
//! Tile decompression imports:
//!
//! ```
//! use signinum::tilecodec::UncompressedCodec;
//! use signinum::TileDecompress;
//!
//! let mut pool = <UncompressedCodec as TileDecompress>::Pool::default();
//! let mut out = [0u8; 3];
//! let written = UncompressedCodec::decompress_into(&mut pool, &[1, 2, 3], &mut out).unwrap();
//! assert_eq!(written, 3);
//! ```

#![warn(unreachable_pub)]

pub mod core {
    //! Shared codec contracts and backend selection types.

    pub use signinum_core::*;
}

pub mod jpeg {
    //! Baseline JPEG decode APIs.

    pub use signinum_jpeg::*;

    #[cfg(feature = "cuda")]
    pub mod cuda {
        //! CUDA JPEG adapter APIs.

        pub use signinum_jpeg_cuda::*;
    }

    #[cfg(feature = "metal")]
    pub mod metal {
        //! Metal JPEG adapter APIs.

        pub use signinum_jpeg_metal::*;
    }
}

pub mod j2k {
    //! JPEG 2000 inspect, decode, and encode APIs.

    pub use signinum_j2k::{
        adapter, context, encode_j2k_lossless as encode_j2k_lossless_cpu,
        encode_j2k_lossless_with_accelerator, error, j2k_lossless_decomposition_levels,
        recode_j2k_to_htj2k_lossless, scratch, view, BackendKind, BackendRequest, BufferError,
        CodecError, CompressedPayloadKind, CompressedTransferSyntax, DecodeOutcome,
        DecodeRowsError, DecoderContext, Downscale, EncodeBackendPreference, EncodedJ2k,
        ImageCodec, ImageDecode, ImageDecodeRows, J2kAdaptiveBackendRequest,
        J2kAdaptiveBenchmarkEvidence, J2kAdaptiveBenchmarkScope, J2kAdaptiveBenchmarks,
        J2kAdaptiveCodecMode, J2kAdaptiveGatePolicy, J2kAdaptiveOperation,
        J2kAdaptiveOutputResidency, J2kAdaptiveQualityMode, J2kAdaptiveRcaFinding,
        J2kAdaptiveRcaReason, J2kAdaptiveRouteKind, J2kAdaptiveRoutePlanner,
        J2kAdaptiveRouteReport, J2kAdaptiveStage, J2kAdaptiveStageDecision,
        J2kAdaptiveStageGateStatus, J2kAdaptiveStageOwner, J2kAdaptiveWorkload, J2kBlockCodingMode,
        J2kCodec, J2kContext, J2kDecoder, J2kEncodeDispatchReport, J2kEncodeStageAccelerator,
        J2kEncodeValidation, J2kError, J2kLosslessEncodeOptions, J2kLosslessSamples,
        J2kProgressionOrder, J2kScratchPool, J2kToHtj2kMode, J2kToHtj2kOptions, J2kToHtj2kReport,
        J2kView, PassthroughCandidate, PassthroughDecision, PassthroughRejectReason,
        PassthroughRequirements, PixelFormat, Rect, ReencodedHtj2k, ReversibleTransform, RowSink,
        TileBatchDecode,
    };

    #[cfg(feature = "cuda")]
    pub mod cuda {
        //! CUDA JPEG 2000 adapter APIs.

        pub use signinum_j2k_cuda::*;
    }

    #[cfg(feature = "metal")]
    pub mod metal {
        //! Metal JPEG 2000 adapter APIs.

        pub use signinum_j2k_metal::*;
    }

    /// Encode interleaved samples into a raw JPEG 2000 lossless codestream.
    ///
    /// With [`EncodeBackendPreference::Auto`], the facade uses adaptive
    /// accelerated routing: CPU-shaped stages stay on CPU and device-shaped
    /// stages run on Metal/CUDA only for benchmark-approved workload shapes.
    /// [`EncodeBackendPreference::RequireDevice`] keeps the strict diagnostic
    /// path and fails instead of silently falling back when required device
    /// stages do not dispatch.
    pub fn encode_j2k_lossless(
        samples: J2kLosslessSamples<'_>,
        options: &J2kLosslessEncodeOptions,
    ) -> Result<EncodedJ2k, J2kError> {
        if options.backend == EncodeBackendPreference::CpuOnly {
            return signinum_j2k::encode_j2k_lossless(samples, options);
        }
        if matches!(
            options.backend,
            EncodeBackendPreference::Auto | EncodeBackendPreference::PreferDevice
        ) {
            let route = adaptive_lossless_encode_route(samples, *options)?;
            if route.route_kind == J2kAdaptiveRouteKind::CpuOnly {
                return signinum_j2k::encode_j2k_lossless(samples, options);
            }
        }

        if let Some(encoded) = try_metal_encode(samples, *options)? {
            return Ok(encoded);
        }
        if let Some(encoded) = try_cuda_encode(samples, *options)? {
            return Ok(encoded);
        }

        signinum_j2k::encode_j2k_lossless(samples, options)
    }

    fn adaptive_lossless_encode_route(
        samples: J2kLosslessSamples<'_>,
        options: J2kLosslessEncodeOptions,
    ) -> Result<J2kAdaptiveRouteReport, J2kError> {
        let workload = lossless_encode_workload(samples, options);
        let benchmarks = facade_lossless_encode_benchmarks(workload);
        facade_adaptive_route_planner(facade_adaptive_capabilities()).plan(
            workload,
            J2kAdaptiveBackendRequest::Accelerated,
            &benchmarks,
        )
    }

    fn lossless_encode_workload(
        samples: J2kLosslessSamples<'_>,
        options: J2kLosslessEncodeOptions,
    ) -> J2kAdaptiveWorkload {
        let codec_mode = match options.block_coding_mode {
            J2kBlockCodingMode::Classic => J2kAdaptiveCodecMode::ClassicJ2k,
            J2kBlockCodingMode::HighThroughput => J2kAdaptiveCodecMode::Htj2k,
        };
        J2kAdaptiveWorkload::new(
            J2kAdaptiveOperation::Encode,
            codec_mode,
            J2kAdaptiveQualityMode::Lossless,
            samples.components,
            samples.bit_depth,
            (samples.width, samples.height),
            1,
        )
    }

    fn facade_adaptive_capabilities() -> signinum_core::BackendCapabilities {
        let mut capabilities = signinum_core::BackendCapabilities::detect();
        capabilities.metal = capabilities.metal && cfg!(feature = "metal");
        capabilities.cuda = cfg!(feature = "cuda-runtime");
        capabilities
    }

    fn facade_adaptive_route_planner(
        capabilities: signinum_core::BackendCapabilities,
    ) -> J2kAdaptiveRoutePlanner {
        J2kAdaptiveRoutePlanner::new(capabilities).with_rca_finding(
            J2kAdaptiveRcaFinding::reclassify_cpu(
                J2kAdaptiveStage::Mct,
                BackendKind::Cuda,
                J2kAdaptiveRcaReason::CpuGenuinelyBetter,
            ),
        )
    }

    fn facade_lossless_encode_benchmarks(workload: J2kAdaptiveWorkload) -> J2kAdaptiveBenchmarks {
        let mut benchmarks = J2kAdaptiveBenchmarks::default();
        if should_use_measured_cuda_htj2k_host_encode(workload) {
            // Lossless host-pixel RGB/RGBA8 HTJ2K facade measurements. This is
            // intentionally separate from the JPEG DCT-grid 9/7 transcode
            // resident-HT measurements documented in
            // docs/cuda-htj2k-resident-encode.md.
            benchmarks.push_stage(J2kAdaptiveBenchmarkEvidence::stage(
                J2kAdaptiveStage::Dwt,
                BackendKind::Cuda,
                19_506_000,
                2_616_000,
                2.0,
            ));
            benchmarks.push_stage(J2kAdaptiveBenchmarkEvidence::stage(
                J2kAdaptiveStage::HtBlockCoding,
                BackendKind::Cuda,
                4_566_000,
                2_002_000,
                2.0,
            ));
            let (cpu_ns, accelerated_ns) = if workload.components == 4 {
                (108_350_000, 53_360_000)
            } else {
                (81_419_000, 41_307_000)
            };
            benchmarks.push_end_to_end(J2kAdaptiveBenchmarkEvidence::end_to_end(
                BackendKind::Cuda,
                cpu_ns,
                accelerated_ns,
                2.0,
            ));
        }
        benchmarks
    }

    fn should_use_measured_cuda_htj2k_host_encode(workload: J2kAdaptiveWorkload) -> bool {
        const MIN_CUDA_HTJ2K_AUTO_PIXELS: u64 = 1024 * 1024;
        let pixels =
            u64::from(workload.tile_size.0).saturating_mul(u64::from(workload.tile_size.1));
        workload.operation == J2kAdaptiveOperation::Encode
            && workload.codec_mode == J2kAdaptiveCodecMode::Htj2k
            && workload.quality_mode == J2kAdaptiveQualityMode::Lossless
            && matches!(workload.components, 3 | 4)
            && workload.bit_depth == 8
            && workload.batch_size == 1
            && !workload.roi
            && !workload.scaled
            && workload.quality_layers == 1
            && workload.output_residency == J2kAdaptiveOutputResidency::Host
            && pixels >= MIN_CUDA_HTJ2K_AUTO_PIXELS
    }

    #[cfg(feature = "metal")]
    fn try_metal_encode(
        samples: J2kLosslessSamples<'_>,
        options: J2kLosslessEncodeOptions,
    ) -> Result<Option<EncodedJ2k>, J2kError> {
        let mut accelerator = if options.backend == EncodeBackendPreference::Auto {
            signinum_j2k_metal::MetalEncodeStageAccelerator::for_auto_host_output()
        } else {
            signinum_j2k_metal::MetalEncodeStageAccelerator::with_cpu_forward_rct()
        };
        encode_with_device_accelerator(samples, options, BackendKind::Metal, &mut accelerator)
    }

    #[cfg(not(feature = "metal"))]
    #[allow(clippy::unnecessary_wraps)]
    fn try_metal_encode(
        _samples: J2kLosslessSamples<'_>,
        _options: J2kLosslessEncodeOptions,
    ) -> Result<Option<EncodedJ2k>, J2kError> {
        Ok(None)
    }

    #[cfg(feature = "cuda")]
    fn try_cuda_encode(
        samples: J2kLosslessSamples<'_>,
        options: J2kLosslessEncodeOptions,
    ) -> Result<Option<EncodedJ2k>, J2kError> {
        let mut accelerator = if options.backend == EncodeBackendPreference::Auto {
            signinum_j2k_cuda::CudaEncodeStageAccelerator::for_auto_host_output()
        } else {
            signinum_j2k_cuda::CudaEncodeStageAccelerator::default()
        };
        encode_with_device_accelerator(samples, options, BackendKind::Cuda, &mut accelerator)
    }

    #[cfg(not(feature = "cuda"))]
    #[allow(clippy::unnecessary_wraps)]
    fn try_cuda_encode(
        _samples: J2kLosslessSamples<'_>,
        _options: J2kLosslessEncodeOptions,
    ) -> Result<Option<EncodedJ2k>, J2kError> {
        Ok(None)
    }

    #[cfg_attr(not(any(feature = "metal", feature = "cuda")), allow(dead_code))]
    fn encode_with_device_accelerator(
        samples: J2kLosslessSamples<'_>,
        options: J2kLosslessEncodeOptions,
        backend: BackendKind,
        accelerator: &mut impl J2kEncodeStageAccelerator,
    ) -> Result<Option<EncodedJ2k>, J2kError> {
        let requested_backend = options.backend;
        let device_options = options.with_backend(EncodeBackendPreference::ACCELERATED);
        let before = accelerator.dispatch_report();
        let encoded = signinum_j2k::encode_j2k_lossless_with_accelerator(
            samples,
            &device_options,
            backend,
            accelerator,
        )?;
        let dispatch = accelerator.dispatch_report().saturating_delta(before);

        let keep_cpu_backed_partial_auto_result =
            requested_backend == EncodeBackendPreference::Auto && dispatch.any();
        Ok((encoded.backend == backend || keep_cpu_backed_partial_auto_result).then_some(encoded))
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[derive(Default)]
        struct PacketizationOnlyAccelerator {
            packetization_dispatches: usize,
        }

        impl J2kEncodeStageAccelerator for PacketizationOnlyAccelerator {
            fn dispatch_report(&self) -> J2kEncodeDispatchReport {
                J2kEncodeDispatchReport {
                    packetization: self.packetization_dispatches,
                    ..J2kEncodeDispatchReport::default()
                }
            }

            fn encode_packetization(
                &mut self,
                job: signinum_j2k::J2kPacketizationEncodeJob<'_>,
            ) -> core::result::Result<Option<Vec<u8>>, &'static str> {
                self.packetization_dispatches = self.packetization_dispatches.saturating_add(1);
                encode_packetization_scalar(job).map(Some)
            }
        }

        fn encode_packetization_scalar(
            job: signinum_j2k::J2kPacketizationEncodeJob<'_>,
        ) -> core::result::Result<Vec<u8>, &'static str> {
            let packet_descriptors = job
                .packet_descriptors
                .iter()
                .copied()
                .map(native_packet_descriptor)
                .collect::<Vec<_>>();
            let resolutions = job
                .resolutions
                .iter()
                .map(native_packet_resolution)
                .collect::<Vec<_>>();
            let native_job = signinum_j2k_native::J2kPacketizationEncodeJob {
                resolution_count: job.resolution_count,
                num_layers: job.num_layers,
                num_components: job.num_components,
                code_block_count: job.code_block_count,
                progression_order: native_packet_progression(job.progression_order),
                packet_descriptors: &packet_descriptors,
                resolutions: &resolutions,
            };
            signinum_j2k_native::encode_j2k_packetization_scalar(native_job)
        }

        fn native_packet_descriptor(
            descriptor: signinum_j2k::J2kPacketizationPacketDescriptor,
        ) -> signinum_j2k_native::J2kPacketizationPacketDescriptor {
            signinum_j2k_native::J2kPacketizationPacketDescriptor {
                packet_index: descriptor.packet_index,
                state_index: descriptor.state_index,
                layer: descriptor.layer,
                resolution: descriptor.resolution,
                component: descriptor.component,
                precinct: descriptor.precinct,
            }
        }

        fn native_packet_resolution<'a>(
            resolution: &signinum_j2k::J2kPacketizationResolution<'a>,
        ) -> signinum_j2k_native::J2kPacketizationResolution<'a> {
            signinum_j2k_native::J2kPacketizationResolution {
                subbands: resolution
                    .subbands
                    .iter()
                    .map(native_packet_subband)
                    .collect(),
            }
        }

        fn native_packet_subband<'a>(
            subband: &signinum_j2k::J2kPacketizationSubband<'a>,
        ) -> signinum_j2k_native::J2kPacketizationSubband<'a> {
            signinum_j2k_native::J2kPacketizationSubband {
                code_blocks: subband
                    .code_blocks
                    .iter()
                    .copied()
                    .map(native_packet_code_block)
                    .collect(),
                num_cbs_x: subband.num_cbs_x,
                num_cbs_y: subband.num_cbs_y,
            }
        }

        fn native_packet_code_block(
            code_block: signinum_j2k::J2kPacketizationCodeBlock<'_>,
        ) -> signinum_j2k_native::J2kPacketizationCodeBlock<'_> {
            signinum_j2k_native::J2kPacketizationCodeBlock {
                data: code_block.data,
                ht_cleanup_length: code_block.ht_cleanup_length,
                ht_refinement_length: code_block.ht_refinement_length,
                num_coding_passes: code_block.num_coding_passes,
                num_zero_bitplanes: code_block.num_zero_bitplanes,
                previously_included: code_block.previously_included,
                l_block: code_block.l_block,
                block_coding_mode: native_packet_block_coding_mode(code_block.block_coding_mode),
            }
        }

        fn native_packet_block_coding_mode(
            mode: signinum_j2k::J2kPacketizationBlockCodingMode,
        ) -> signinum_j2k_native::J2kPacketizationBlockCodingMode {
            match mode {
                signinum_j2k::J2kPacketizationBlockCodingMode::Classic => {
                    signinum_j2k_native::J2kPacketizationBlockCodingMode::Classic
                }
                signinum_j2k::J2kPacketizationBlockCodingMode::HighThroughput => {
                    signinum_j2k_native::J2kPacketizationBlockCodingMode::HighThroughput
                }
            }
        }

        fn native_packet_progression(
            progression: signinum_j2k::J2kPacketizationProgressionOrder,
        ) -> signinum_j2k_native::J2kPacketizationProgressionOrder {
            match progression {
                signinum_j2k::J2kPacketizationProgressionOrder::Lrcp => {
                    signinum_j2k_native::J2kPacketizationProgressionOrder::Lrcp
                }
                signinum_j2k::J2kPacketizationProgressionOrder::Rlcp => {
                    signinum_j2k_native::J2kPacketizationProgressionOrder::Rlcp
                }
                signinum_j2k::J2kPacketizationProgressionOrder::Rpcl => {
                    signinum_j2k_native::J2kPacketizationProgressionOrder::Rpcl
                }
                signinum_j2k::J2kPacketizationProgressionOrder::Pcrl => {
                    signinum_j2k_native::J2kPacketizationProgressionOrder::Pcrl
                }
                signinum_j2k::J2kPacketizationProgressionOrder::Cprl => {
                    signinum_j2k_native::J2kPacketizationProgressionOrder::Cprl
                }
            }
        }

        #[test]
        fn auto_keeps_cpu_backed_encode_after_partial_device_dispatch() {
            let pixels: Vec<u8> = (0..64 * 64 * 3)
                .map(|value| u8::try_from((value * 13) & 0xFF).expect("masked sample fits"))
                .collect();
            let samples =
                J2kLosslessSamples::new(&pixels, 64, 64, 3, 8, false).expect("valid samples");
            let mut accelerator = PacketizationOnlyAccelerator::default();

            let encoded = encode_with_device_accelerator(
                samples,
                J2kLosslessEncodeOptions::default()
                    .with_backend(EncodeBackendPreference::Auto)
                    .with_validation(J2kEncodeValidation::External),
                BackendKind::Metal,
                &mut accelerator,
            )
            .expect("partial accelerator encode succeeds")
            .expect("Auto should reuse the CPU-backed codestream after partial device dispatch");

            assert_eq!(encoded.backend, BackendKind::Cpu);
            assert_eq!(encoded.dispatch_report.packetization, 1);
            assert_eq!(accelerator.packetization_dispatches, 1);
        }

        #[test]
        fn facade_cuda_htj2k_1024_policy_uses_measured_hybrid_route() {
            let workload = J2kAdaptiveWorkload::new(
                J2kAdaptiveOperation::Encode,
                J2kAdaptiveCodecMode::Htj2k,
                J2kAdaptiveQualityMode::Lossless,
                3,
                8,
                (1024, 1024),
                1,
            );
            let benchmarks = facade_lossless_encode_benchmarks(workload);
            let report = facade_adaptive_route_planner(signinum_core::BackendCapabilities {
                cpu: signinum_core::CpuFeatures::default(),
                metal: false,
                cuda: true,
            })
            .plan(
                workload,
                J2kAdaptiveBackendRequest::Accelerated,
                &benchmarks,
            )
            .expect("facade CUDA route should plan");

            assert_eq!(report.route_kind, J2kAdaptiveRouteKind::Hybrid);
            assert_eq!(report.selected_device, Some(BackendKind::Cuda));
            assert_eq!(
                report
                    .stage(J2kAdaptiveStage::Mct)
                    .expect("MCT decision")
                    .selected_backend,
                BackendKind::Cpu
            );
            assert_eq!(
                report
                    .stage(J2kAdaptiveStage::Dwt)
                    .expect("DWT decision")
                    .selected_backend,
                BackendKind::Cuda
            );
            assert_eq!(
                report
                    .stage(J2kAdaptiveStage::HtBlockCoding)
                    .expect("HT decision")
                    .selected_backend,
                BackendKind::Cuda
            );
            assert_eq!(
                report
                    .stage(J2kAdaptiveStage::Packetization)
                    .expect("packetization decision")
                    .selected_backend,
                BackendKind::Cpu
            );
        }

        #[test]
        fn facade_cuda_htj2k_512_policy_stays_cpu_below_measured_win_gate() {
            let workload = J2kAdaptiveWorkload::new(
                J2kAdaptiveOperation::Encode,
                J2kAdaptiveCodecMode::Htj2k,
                J2kAdaptiveQualityMode::Lossless,
                3,
                8,
                (512, 512),
                1,
            );
            let benchmarks = facade_lossless_encode_benchmarks(workload);
            let report = facade_adaptive_route_planner(signinum_core::BackendCapabilities {
                cpu: signinum_core::CpuFeatures::default(),
                metal: false,
                cuda: true,
            })
            .plan(
                workload,
                J2kAdaptiveBackendRequest::Accelerated,
                &benchmarks,
            )
            .expect("facade CUDA route should plan");

            assert_eq!(report.route_kind, J2kAdaptiveRouteKind::CpuOnly);
            assert_eq!(report.selected_device, None);
            assert!(
                report
                    .stages
                    .iter()
                    .all(|stage| stage.selected_backend == BackendKind::Cpu),
                "512 host-pixel CUDA HTJ2K Auto must stay CPU until measured otherwise"
            );
        }
    }
}

pub mod tilecodec {
    //! Tile decompression codecs for container integrations.

    pub use signinum_tilecodec::*;
}

pub use core::{
    BackendCapabilities, BackendKind, BackendRequest, BufferError, CodecError, DecodeOutcome,
    DecodeRowsError, DecoderContext, DeviceSurface, Downscale, ImageCodec, ImageDecode,
    ImageDecodeDevice, ImageDecodeRows, PixelFormat, Rect, RowSink, TileBatchDecode,
    TileBatchDecodeManyDevice, TileDecompress,
};
pub use core::{
    CompressedPayloadKind, CompressedTransferSyntax, PassthroughCandidate, PassthroughDecision,
    PassthroughRejectReason, PassthroughRequirements,
};
pub use j2k::{
    encode_j2k_lossless, encode_j2k_lossless_with_accelerator, j2k_lossless_decomposition_levels,
    EncodeBackendPreference, EncodedJ2k, J2kBlockCodingMode, J2kCodec, J2kContext, J2kDecoder,
    J2kEncodeDispatchReport, J2kEncodeStageAccelerator, J2kEncodeValidation, J2kError,
    J2kLosslessEncodeOptions, J2kLosslessSamples, J2kProgressionOrder, ReversibleTransform,
};
pub use jpeg::{
    ColorSpace, ColorTransform, DecodeOptions, Decoder as JpegDecoder, JpegCodec, JpegError,
    JpegView,
};
pub use tilecodec::{DeflateCodec, LzwCodec, TileCodecError, UncompressedCodec, ZstdCodec};

pub mod prelude {
    //! Common imports for applications using the `signinum` facade.

    pub use crate::{
        BackendRequest, DeflateCodec, Downscale, EncodeBackendPreference, J2kDecoder,
        J2kLosslessEncodeOptions, J2kLosslessSamples, JpegDecoder, JpegView, LzwCodec, PixelFormat,
        TileDecompress, UncompressedCodec, ZstdCodec,
    };
}
