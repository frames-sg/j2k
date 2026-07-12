// SPDX-License-Identifier: MIT OR Apache-2.0

use std::hash::{BuildHasher, Hasher};

use super::{JpegCachedPlan, JpegFastPacketState, SharedJpegInput};
use crate::adapter::DeviceBatchSummary;
use crate::ColorSpace;

mod build;
mod ownership;
mod resolve;
mod store;

fn unsupported_plan(input: &[u8]) -> JpegCachedPlan {
    JpegCachedPlan::try_new(
        SharedJpegInput::try_copy_from_slice(input).expect("copy test JPEG input"),
        unsupported_summary(),
        ColorSpace::Grayscale,
        JpegFastPacketState::Unsupported,
    )
    .expect("construct unsupported test plan")
}

const fn unsupported_summary() -> DeviceBatchSummary {
    DeviceBatchSummary {
        restart_interval: None,
        checkpoint_count: 0,
        matches_fast_420: false,
        matches_fast_422: false,
        matches_fast_444: false,
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct ConstantDigestBuilder;

impl BuildHasher for ConstantDigestBuilder {
    type Hasher = ConstantDigest;

    fn build_hasher(&self) -> Self::Hasher {
        ConstantDigest
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct ConstantDigest;

impl Hasher for ConstantDigest {
    fn finish(&self) -> u64 {
        11
    }

    fn write(&mut self, _bytes: &[u8]) {}
}

fn rewrite_first_sof_quant_table_selector(mut bytes: Vec<u8>, selector: u8) -> Vec<u8> {
    let mut position = 2_usize;
    while position + 4 <= bytes.len() {
        assert_eq!(bytes[position], 0xff, "fixture marker alignment");
        let marker = bytes[position + 1];
        position += 2;
        let length = usize::from(u16::from_be_bytes([bytes[position], bytes[position + 1]]));
        let payload_start = position + 2;
        if matches!(marker, 0xc0..=0xc3) {
            bytes[payload_start + 8] = selector;
            return bytes;
        }
        position += length;
    }
    panic!("fixture must contain a supported SOF marker");
}
