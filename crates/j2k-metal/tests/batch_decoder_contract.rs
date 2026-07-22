// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k::{BatchDecodeOptions, BatchDecoder, EncodedImage, PreparedBatch};
#[cfg(target_os = "macos")]
use j2k_metal::SubmittedMetalPreparedBatch;
use j2k_metal::{Error, MetalBatchDecodeResult, MetalBatchDecoder};

const _: fn(&MetalBatchDecoder) -> BatchDecodeOptions =
    <MetalBatchDecoder as BatchDecoder>::options;
const _: fn(&MetalBatchDecoder, Vec<EncodedImage>) -> Result<PreparedBatch, Error> =
    MetalBatchDecoder::prepare;
const _: fn(&mut MetalBatchDecoder, Vec<EncodedImage>) -> Result<MetalBatchDecodeResult, Error> =
    MetalBatchDecoder::decode_batch;

#[cfg(target_os = "macos")]
const _: fn(
    &mut MetalBatchDecoder,
    Vec<EncodedImage>,
) -> Result<SubmittedMetalPreparedBatch, Error> = MetalBatchDecoder::submit_batch;

#[test]
fn metal_batch_decoder_implements_shared_codec_contract() {
    fn assert_contract<D>()
    where
        D: BatchDecoder<Output = MetalBatchDecodeResult, Error = Error>,
    {
    }

    assert_contract::<MetalBatchDecoder>();
}
