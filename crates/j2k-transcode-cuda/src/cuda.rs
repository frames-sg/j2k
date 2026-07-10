// SPDX-License-Identifier: MIT OR Apache-2.0

//! CUDA dispatch boundary for the transcode accelerator.
//!
//! Each function uploads a DCT-grid job to the device, runs the ported kernel
//! in `j2k-cuda-runtime`, and returns wavelet bands / prequantized
//! components matching the `j2k-transcode` scalar oracle. Kernels are
//! wired incrementally; until a path is wired its dispatch returns a typed
//! [`CudaTranscodeError::UnsupportedJob`], which Auto mode treats as a scalar
//! fallback and Explicit mode surfaces as an error.

use j2k_transcode::{
    htj2k97_subband_delta, htj2k97_subband_total_bitplanes, DctGridI16ToHtj2k97CodeBlockBatch,
    DctGridI16ToHtj2k97CodeBlockJob, DctGridToDwt53Job, DctGridToDwt97Job,
    DctGridToHtj2k97CodeBlockJob, DctGridToReversibleDwt53Job, Dwt53TwoDimensional,
    Dwt97BatchStageTimings, Dwt97TwoDimensional, EncodedHtJ2kCodeBlock, Htj2k97CodeBlockOptions,
    J2kSubBandType, PreencodedHtj2k97CodeBlock, PreencodedHtj2k97CompactBatch,
    PreencodedHtj2k97CompactBatchGroups, PreencodedHtj2k97CompactCodeBlock,
    PreencodedHtj2k97CompactComponent, PreencodedHtj2k97CompactResolution,
    PreencodedHtj2k97CompactSubband, PreencodedHtj2k97Component, PreencodedHtj2k97Resolution,
    PreencodedHtj2k97Subband, PrequantizedHtj2k97CodeBlock, PrequantizedHtj2k97Component,
    PrequantizedHtj2k97Resolution, PrequantizedHtj2k97Subband, ReversibleDwt53FirstLevel,
};

use std::sync::Arc;

use j2k_cuda_runtime::{
    transcode_kernels_built, CudaBufferPool, CudaContext, CudaDwt97BatchGeometry,
    CudaDwt97BatchStageTimings, CudaDwt97BatchWithPoolRequest, CudaHtj2k97CodeblockBands,
    CudaHtj2k97CodeblockBatchWithPoolRequest, CudaHtj2k97DeviceCodeblockBands,
    CudaHtj2k97I16CodeblockBatchWithPoolRequest, CudaHtj2k97QuantizeParams,
    CudaHtj2kCompactEncodedCodeBlock, CudaHtj2kEncodeCodeBlockJob, CudaHtj2kEncodeResidentTarget,
    CudaHtj2kEncodeResources, CudaHtj2kEncodeStageTimings, CudaHtj2kEncodeTables,
    CudaHtj2kEncodedCodeBlock, CudaPooledDeviceBuffer, CudaTranscodeDwt97Bands,
    CudaTranscodeReversible53Bands,
};

use crate::CudaTranscodeError;
mod transform;
use self::transform::{
    accumulate_batch_timings, add_ht_encode_timings, append_i16_blocks, map_batch_timings,
    set_ht_encode_timings,
};
pub(crate) use self::transform::{
    dispatch_dwt53, dispatch_dwt97, dispatch_dwt97_batch, dispatch_htj2k97_codeblock_batch,
    dispatch_htj2k97_preencoded_batch, dispatch_reversible_dwt53, dispatch_reversible_dwt53_batch,
};
mod resident_dispatch;
use self::resident_dispatch::{device_bands_to_preencoded_components, htj2k97_quantize_params};
pub(crate) use self::resident_dispatch::{
    dispatch_htj2k97_compact_preencoded_i16_batch,
    dispatch_htj2k97_compact_preencoded_i16_batch_groups, dispatch_htj2k97_preencoded_i16_batch,
    dispatch_htj2k97_preencoded_i16_batch_groups,
};
mod resident_encode;
use self::resident_encode::{
    assemble_compact_preencoded_components, assemble_preencoded_components,
    cuda_htj2k_encode_tables, device_band_groups_to_compact_preencoded_components,
    device_band_groups_to_preencoded_components, encode_resident_compact_subbands,
    encode_resident_subbands, htj2k97_code_block_dim, to_u32, validate_band_len,
    validate_htj2k97_codeblock_options, Htj2k97ComponentJob, ResidentDeviceGroup,
};

/// Returned until a given kernel path is wired to `j2k-cuda-runtime`.
const NOT_WIRED: CudaTranscodeError =
    CudaTranscodeError::UnsupportedJob("j2k-transcode-cuda kernel not yet wired");

type GroupedPreencodedComponents = Vec<(usize, Vec<PreencodedHtj2k97Component>)>;
type GroupedCompactPreencodedComponents = Vec<(usize, Vec<PreencodedHtj2k97CompactComponent>)>;
type ResidentPreencodedGroups = (
    GroupedPreencodedComponents,
    CudaHtj2kEncodeStageTimings,
    usize,
);
type ResidentCompactPreencodedGroups = (
    Vec<u8>,
    GroupedCompactPreencodedComponents,
    CudaHtj2kEncodeStageTimings,
    usize,
);

/// Caller-owned CUDA runtime state reused across transcode dispatches.
#[derive(Clone, Debug, Default)]
pub(crate) struct CudaTranscodeSession {
    context: Option<CudaContext>,
    buffer_pool: Option<CudaBufferPool>,
    encode_resources: Option<Arc<CudaHtj2kEncodeResources>>,
}

impl CudaTranscodeSession {
    fn context(&mut self) -> Result<CudaContext, CudaTranscodeError> {
        if self.context.is_none() {
            self.context = Some(
                CudaContext::system_default().map_err(|_| CudaTranscodeError::CudaUnavailable)?,
            );
        }
        self.context
            .clone()
            .ok_or(CudaTranscodeError::CudaUnavailable)
    }

    fn buffer_pool(&mut self, context: &CudaContext) -> CudaBufferPool {
        if let Some(pool) = &self.buffer_pool {
            return pool.clone();
        }
        let pool = context.buffer_pool();
        self.buffer_pool = Some(pool.clone());
        pool
    }

    fn encode_resources(
        &mut self,
        context: &CudaContext,
    ) -> Result<Arc<CudaHtj2kEncodeResources>, CudaTranscodeError> {
        if let Some(resources) = &self.encode_resources {
            return Ok(Arc::clone(resources));
        }
        let resources = Arc::new(
            context
                .upload_htj2k_encode_resources(cuda_htj2k_encode_tables())
                .map_err(|_| {
                    CudaTranscodeError::Kernel("CUDA HTJ2K encode resource upload failed")
                })?,
        );
        self.encode_resources = Some(Arc::clone(&resources));
        Ok(resources)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestComponentJob {
        x_rsiz: u8,
        y_rsiz: u8,
    }

    impl Htj2k97ComponentJob for TestComponentJob {
        fn x_rsiz(&self) -> u8 {
            self.x_rsiz
        }

        fn y_rsiz(&self) -> u8 {
            self.y_rsiz
        }
    }

    fn test_subband(sub_band_type: J2kSubBandType, marker: u8) -> PreencodedHtj2k97Subband {
        PreencodedHtj2k97Subband {
            sub_band_type,
            num_cbs_x: 1,
            num_cbs_y: 1,
            total_bitplanes: 8,
            code_blocks: vec![PreencodedHtj2k97CodeBlock {
                width: 1,
                height: 1,
                encoded: EncodedHtJ2kCodeBlock {
                    data: vec![marker; 8],
                    cleanup_length: 8,
                    refinement_length: 0,
                    num_coding_passes: 1,
                    num_zero_bitplanes: 0,
                },
            }],
        }
    }

    fn payload_ptr(subband: &PreencodedHtj2k97Subband) -> usize {
        subband.code_blocks[0].encoded.data.as_ptr() as usize
    }

    #[test]
    #[expect(
        clippy::similar_names,
        reason = "LL, HL, LH, and HH are standard wavelet subband names"
    )]
    fn assemble_preencoded_components_moves_subband_payloads_without_clone() {
        let jobs = [TestComponentJob {
            x_rsiz: 1,
            y_rsiz: 2,
        }];
        let ll = vec![test_subband(J2kSubBandType::LowLow, 1)];
        let hl = vec![test_subband(J2kSubBandType::HighLow, 2)];
        let lh = vec![test_subband(J2kSubBandType::LowHigh, 3)];
        let hh = vec![test_subband(J2kSubBandType::HighHigh, 4)];
        let ll_ptr = payload_ptr(&ll[0]);
        let hl_ptr = payload_ptr(&hl[0]);
        let lh_ptr = payload_ptr(&lh[0]);
        let hh_ptr = payload_ptr(&hh[0]);

        let components = assemble_preencoded_components(&jobs, ll, hl, lh, hh).expect("components");

        assert_eq!(components.len(), 1);
        assert_eq!(components[0].x_rsiz, 1);
        assert_eq!(components[0].y_rsiz, 2);
        assert_eq!(
            payload_ptr(&components[0].resolutions[0].subbands[0]),
            ll_ptr
        );
        assert_eq!(
            payload_ptr(&components[0].resolutions[1].subbands[0]),
            hl_ptr
        );
        assert_eq!(
            payload_ptr(&components[0].resolutions[1].subbands[1]),
            lh_ptr
        );
        assert_eq!(
            payload_ptr(&components[0].resolutions[1].subbands[2]),
            hh_ptr
        );
    }

    #[test]
    fn append_i16_blocks_preserves_prefix_and_flattens_blocks() {
        let mut first = [0i16; 64];
        first[0] = -7;
        first[63] = 42;
        let mut second = [0i16; 64];
        second[1] = 9;
        second[62] = -11;
        let mut out = vec![123];

        append_i16_blocks(&[first, second], &mut out);

        assert_eq!(out[0], 123);
        assert_eq!(out.len(), 1 + 128);
        assert_eq!(out[1], -7);
        assert_eq!(out[64], 42);
        assert_eq!(out[66], 9);
        assert_eq!(out[127], -11);
    }
}
