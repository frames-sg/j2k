// SPDX-License-Identifier: Apache-2.0

use j2k_native::{
    HtCodeBlockDecodeJob, HtCodeBlockDecoder, HtSubBandDecodeJob, J2kCodeBlockDecodeJob,
    J2kInverseMctJob, J2kSingleDecompositionIdwtJob, J2kStoreComponentJob, J2kSubBandDecodeJob,
};

use crate::{
    classic::MetalClassicBlockDecoder, ht::MetalHtBlockDecoder, idwt::MetalIdwtDecoder,
    mct::MetalMctDecoder, store::MetalStoreDecoder,
};

#[derive(Default)]
pub(super) struct MetalCodeBlockDecoder {
    classic: MetalClassicBlockDecoder,
    ht: MetalHtBlockDecoder,
    idwt: MetalIdwtDecoder,
    pub(super) mct: MetalMctDecoder,
    pub(super) store: MetalStoreDecoder,
}

impl HtCodeBlockDecoder for MetalCodeBlockDecoder {
    fn decode_j2k_sub_band(
        &mut self,
        job: J2kSubBandDecodeJob<'_>,
        output: &mut [f32],
    ) -> j2k_native::Result<bool> {
        self.classic.decode_j2k_sub_band(job, output)
    }

    fn decode_j2k_code_block(
        &mut self,
        job: J2kCodeBlockDecodeJob<'_>,
        output: &mut [f32],
    ) -> j2k_native::Result<bool> {
        self.classic.decode_j2k_code_block(job, output)
    }

    fn decode_sub_band(
        &mut self,
        job: HtSubBandDecodeJob<'_>,
        output: &mut [f32],
    ) -> j2k_native::Result<bool> {
        self.ht.decode_sub_band(job, output)
    }

    fn decode_code_block(
        &mut self,
        job: HtCodeBlockDecodeJob<'_>,
        output: &mut [f32],
    ) -> j2k_native::Result<()> {
        self.ht.decode_code_block(job, output)
    }

    fn decode_single_decomposition_idwt(
        &mut self,
        job: J2kSingleDecompositionIdwtJob<'_>,
        output: &mut [f32],
    ) -> j2k_native::Result<bool> {
        self.idwt.decode_single_decomposition_idwt(job, output)
    }

    fn decode_inverse_mct(&mut self, job: J2kInverseMctJob<'_>) -> j2k_native::Result<bool> {
        self.mct.decode_inverse_mct(job)
    }

    fn decode_store_component(
        &mut self,
        job: J2kStoreComponentJob<'_>,
    ) -> j2k_native::Result<bool> {
        self.store.decode_store_component(job)
    }
}
