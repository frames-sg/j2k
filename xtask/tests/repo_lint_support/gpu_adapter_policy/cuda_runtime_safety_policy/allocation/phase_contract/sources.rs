// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{fs, path::Path};

pub(super) struct PhaseSources {
    pub(super) decode_planning: String,
    pub(super) decode_completion: String,
    pub(super) decode_queued: String,
    pub(super) idwt_sequence: String,
    pub(super) encode_planning: String,
    pub(super) encode_api: String,
    pub(super) encode_completion: String,
    pub(super) compact_expansion: String,
    pub(super) packetize: String,
    pub(super) store_batch: String,
    pub(super) band_transfer: String,
    pub(super) pool_readback: String,
    pub(super) transcode: String,
    pub(super) dwt: String,
}

impl PhaseSources {
    pub(super) fn read(root: &Path) -> Self {
        let read = |relative: &str| {
            fs::read_to_string(root.join(relative))
                .unwrap_or_else(|error| panic!("read {relative}: {error}"))
        };
        Self {
            decode_planning: read("crates/j2k-cuda-runtime/src/htj2k_decode/planning.rs"),
            decode_completion: [
                read("crates/j2k-cuda-runtime/src/htj2k_decode/completion.rs"),
                read("crates/j2k-cuda-runtime/src/htj2k_decode/completion/dequant.rs"),
                read(
                    "crates/j2k-cuda-runtime/src/htj2k_decode/completion/cleanup_enqueue.rs",
                ),
                read(
                    "crates/j2k-cuda-runtime/src/htj2k_decode/completion/cleanup_dequant_enqueue.rs",
                ),
            ]
            .concat(),
            decode_queued: read("crates/j2k-cuda-runtime/src/htj2k_decode/queued.rs"),
            idwt_sequence: read("crates/j2k-cuda-runtime/src/j2k_decode/idwt/sequence.rs"),
            encode_planning: [
                read("crates/j2k-cuda-runtime/src/htj2k_encode/planning.rs"),
                read("crates/j2k-cuda-runtime/src/htj2k_encode/planning/compact.rs"),
            ]
            .concat(),
            encode_api: read("crates/j2k-cuda-runtime/src/htj2k_encode/api.rs"),
            encode_completion: read("crates/j2k-cuda-runtime/src/htj2k_encode/completion.rs"),
            compact_expansion: read("crates/j2k-cuda-runtime/src/context/compact.rs"),
            packetize: read("crates/j2k-cuda-runtime/src/htj2k_packetize.rs"),
            store_batch: read("crates/j2k-cuda-runtime/src/j2k_decode/store/batch.rs"),
            band_transfer: read("crates/j2k-cuda-runtime/src/context/band_transfer.rs"),
            pool_readback: read("crates/j2k-cuda-runtime/src/memory/pool/readback.rs"),
            transcode: [
                read("crates/j2k-cuda-runtime/src/transcode/reversible53.rs"),
                read("crates/j2k-cuda-runtime/src/transcode/dwt97.rs"),
                read("crates/j2k-cuda-runtime/src/transcode/dwt97/single.rs"),
                read("crates/j2k-cuda-runtime/src/transcode/readback.rs"),
                read("crates/j2k-cuda-runtime/src/transcode/htj2k97.rs"),
            ]
            .concat(),
            dwt: [
                read("crates/j2k-cuda-runtime/src/j2k_encode/dwt.rs"),
                read("crates/j2k-cuda-runtime/src/j2k_encode/readback.rs"),
            ]
            .concat(),
        }
    }
}
