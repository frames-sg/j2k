// SPDX-License-Identifier: MIT OR Apache-2.0

use super::CudaKernel;

impl CudaKernel {
    pub(crate) fn is_transcode_reversible53_stage(self) -> bool {
        matches!(
            self,
            Self::TranscodeReversible53Idct
                | Self::TranscodeReversible53VerticalLow
                | Self::TranscodeReversible53VerticalHigh
                | Self::TranscodeReversible53HorizontalLow
                | Self::TranscodeReversible53HorizontalHigh
        )
    }

    pub(crate) fn is_transcode_dwt97_single_stage(self) -> bool {
        matches!(
            self,
            Self::TranscodeDwt97Idct | Self::TranscodeDwt97RowLift | Self::TranscodeDwt97ColumnLift
        )
    }

    pub(crate) fn is_transcode_dwt97_batch_stage(self) -> bool {
        matches!(
            self,
            Self::TranscodeDwt97IdctBatch
                | Self::TranscodeDwt97IdctI16Batch
                | Self::TranscodeDwt97RowLiftBatch
                | Self::TranscodeDwt97RowLiftBatchCoop
                | Self::TranscodeDwt97ColumnLiftBatch
                | Self::TranscodeDwt97QuantizeCodeblocks
                | Self::TranscodeDwt97ColumnLiftQuantizeCodeblocksBatch
        )
    }

    #[cfg_attr(
        all(not(feature = "cuda-oxide-transcode"), not(test)),
        expect(
            dead_code,
            reason = "classifier is used only by the transcode kernel feature"
        )
    )]
    pub(crate) fn is_cuda_oxide_transcode_stage(self) -> bool {
        self.is_transcode_reversible53_stage()
            || self.is_transcode_dwt97_single_stage()
            || self.is_transcode_dwt97_batch_stage()
    }
}

#[cfg(feature = "cuda-oxide-transcode")]
const CUDA_OXIDE_TRANSCODE_PTX: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/cuda_oxide_transcode.ptx"));

#[cfg(feature = "cuda-oxide-transcode")]
pub(crate) fn cuda_oxide_transcode_ptx() -> &'static [u8] {
    CUDA_OXIDE_TRANSCODE_PTX
}
