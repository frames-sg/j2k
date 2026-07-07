// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_jpeg::adapter::{JpegFast420PacketV1, JpegFast422PacketV1, JpegFast444PacketV1};

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct JpegFastPackets<'a> {
    pub(crate) fast444: Option<&'a JpegFast444PacketV1>,
    pub(crate) fast422: Option<&'a JpegFast422PacketV1>,
    pub(crate) fast420: Option<&'a JpegFast420PacketV1>,
}

impl<'a> JpegFastPackets<'a> {
    pub(crate) fn new(
        fast444: Option<&'a JpegFast444PacketV1>,
        fast422: Option<&'a JpegFast422PacketV1>,
        fast420: Option<&'a JpegFast420PacketV1>,
    ) -> Self {
        Self {
            fast444,
            fast422,
            fast420,
        }
    }
}
