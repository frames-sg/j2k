// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(all(test, target_os = "macos"))]
use std::sync::Arc;

use j2k_jpeg::adapter::{
    JpegFast420PacketV1, JpegFast422PacketV1, JpegFast444PacketV1, SharedJpegFastPacket,
};
#[cfg(test)]
use j2k_jpeg::{
    adapter::{
        build_fast420_packet, build_fast422_packet, build_fast444_packet,
        classify_color_fast_packet_family, decoder_bytes, FastPacketError, JpegFastPacket,
        JpegFastPacketFamily,
    },
    Decoder as CpuDecoder,
};

#[cfg(test)]
use crate::Error;

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct JpegFastPackets<'a> {
    pub(crate) fast444: Option<&'a JpegFast444PacketV1>,
    pub(crate) fast422: Option<&'a JpegFast422PacketV1>,
    pub(crate) fast420: Option<&'a JpegFast420PacketV1>,
}

impl<'a> JpegFastPackets<'a> {
    #[cfg(test)]
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

    pub(crate) fn from_shared(packet: Option<&'a SharedJpegFastPacket>) -> Self {
        let Some(packet) = packet else {
            return Self::default();
        };
        Self {
            fast444: packet.fast444(),
            fast422: packet.fast422(),
            fast420: packet.fast420(),
        }
    }
}

/// Inspect sampling once and build only the matching fast-packet family.
#[cfg(test)]
pub(crate) fn build_shared_fast_packet(
    decoder: &CpuDecoder<'_>,
) -> Result<Option<SharedJpegFastPacket>, Error> {
    let bytes = decoder_bytes(decoder);
    match classify_color_fast_packet_family(decoder) {
        Some(JpegFastPacketFamily::Fast420) => {
            retain_fast_packet_failure(build_fast420_packet(bytes)).and_then(|packet| {
                packet
                    .map(|packet| SharedJpegFastPacket::try_new(JpegFastPacket::Fast420(packet)))
                    .transpose()
                    .map_err(Error::from)
            })
        }
        Some(JpegFastPacketFamily::Fast422) => {
            retain_fast_packet_failure(build_fast422_packet(bytes)).and_then(|packet| {
                packet
                    .map(|packet| SharedJpegFastPacket::try_new(JpegFastPacket::Fast422(packet)))
                    .transpose()
                    .map_err(Error::from)
            })
        }
        Some(JpegFastPacketFamily::Fast444) => {
            retain_fast_packet_failure(build_fast444_packet(bytes)).and_then(|packet| {
                packet
                    .map(|packet| SharedJpegFastPacket::try_new(JpegFastPacket::Fast444(packet)))
                    .transpose()
                    .map_err(Error::from)
            })
        }
        None => Ok(None),
    }
}

#[cfg(test)]
fn retain_fast_packet_failure<T>(result: Result<T, FastPacketError>) -> Result<Option<T>, Error> {
    match result {
        Ok(packet) => Ok(Some(packet)),
        Err(source) if source.is_capability_mismatch() => Ok(None),
        Err(source) => Err(Error::FastPacket { source }),
    }
}

#[cfg(all(test, target_os = "macos"))]
pub(crate) fn build_test_shared_fast_packet(
    input: &[u8],
    fast444: Option<Arc<JpegFast444PacketV1>>,
    fast422: Option<Arc<JpegFast422PacketV1>>,
    fast420: Option<Arc<JpegFast420PacketV1>>,
) -> Option<SharedJpegFastPacket> {
    match (fast444, fast422, fast420) {
        (Some(_), None, None) => Some(
            SharedJpegFastPacket::try_new(JpegFastPacket::Fast444(
                build_fast444_packet(input).expect("queued test fast444 packet"),
            ))
            .expect("queued test fast444 owner"),
        ),
        (None, Some(_), None) => Some(
            SharedJpegFastPacket::try_new(JpegFastPacket::Fast422(
                build_fast422_packet(input).expect("queued test fast422 packet"),
            ))
            .expect("queued test fast422 owner"),
        ),
        (None, None, Some(_)) => Some(
            SharedJpegFastPacket::try_new(JpegFastPacket::Fast420(
                build_fast420_packet(input).expect("queued test fast420 packet"),
            ))
            .expect("queued test fast420 owner"),
        ),
        (None, None, None) => None,
        _ => panic!("a queued test request must carry at most one fast-packet family"),
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;

    use j2k_jpeg::{
        adapter::{FastPacketError, TableKind},
        ColorSpace, Decoder as CpuDecoder, JpegError, SofKind,
    };

    use super::{build_shared_fast_packet, retain_fast_packet_failure};
    use crate::Error;

    const BASELINE_420: &[u8] = include_bytes!("../fixtures/jpeg/baseline_420_16x16.jpg");
    const BASELINE_422: &[u8] = include_bytes!("../fixtures/jpeg/baseline_422_16x8.jpg");
    const BASELINE_444: &[u8] = include_bytes!("../fixtures/jpeg/baseline_444_8x8.jpg");
    const BASELINE_420_RESTART: &[u8] =
        include_bytes!("../fixtures/jpeg/baseline_420_restart_32x16.jpg");
    const GRAYSCALE: &[u8] = include_bytes!("../fixtures/jpeg/grayscale_8x8.jpg");

    #[test]
    fn supported_sampling_selects_exactly_one_matching_family() {
        for (bytes, expected) in [
            (BASELINE_420, "420"),
            (BASELINE_420_RESTART, "420"),
            (BASELINE_422, "422"),
            (BASELINE_444, "444"),
        ] {
            let decoder = CpuDecoder::new(bytes).expect("fixture decoder");
            let packet = build_shared_fast_packet(&decoder)
                .expect("fast-packet selection")
                .expect("supported packet family");

            let actual = match packet.as_packet().family() {
                j2k_jpeg::adapter::JpegFastPacketFamily::Fast420 => "420",
                j2k_jpeg::adapter::JpegFastPacketFamily::Fast422 => "422",
                j2k_jpeg::adapter::JpegFastPacketFamily::Fast444 => "444",
            };
            assert_eq!(actual, expected, "selected the wrong fast-packet family");
        }
    }

    #[test]
    fn unsupported_sampling_is_an_explicit_no_packet_decision() {
        let decoder = CpuDecoder::new(GRAYSCALE).expect("grayscale decoder");

        assert!(build_shared_fast_packet(&decoder)
            .expect("capability mismatch is not a hard error")
            .is_none());
    }

    #[test]
    fn every_capability_mismatch_routes_to_no_packet() {
        for source in [
            FastPacketError::UnsupportedSof(SofKind::Progressive8),
            FastPacketError::UnsupportedColorSpace(ColorSpace::Rgb),
            FastPacketError::UnsupportedSampling,
            FastPacketError::UnsupportedComponentOrder,
            FastPacketError::EntropyMarkerUnsupported { marker: 0xdc },
        ] {
            assert!(retain_fast_packet_failure::<()>(Err(source))
                .expect("capability mismatch")
                .is_none());
        }
    }

    #[test]
    fn malformed_packet_failures_remain_typed_hard_errors() {
        for source in [
            FastPacketError::MissingScan,
            FastPacketError::MissingQuantTable { slot: 2 },
            FastPacketError::MissingHuffmanTable {
                kind: TableKind::Ac,
                slot: 1,
            },
            FastPacketError::TruncatedEntropy,
        ] {
            let error = retain_fast_packet_failure::<()>(Err(source.clone()))
                .expect_err("malformed packet input must not fall back silently");

            assert!(matches!(
                &error,
                Error::FastPacket { source: stored } if stored == &source
            ));
            assert!(error.source().is_some_and(|chained| {
                chained.downcast_ref::<FastPacketError>() == Some(&source)
            }));
        }
    }

    #[test]
    fn resource_and_invariant_failures_keep_the_nested_jpeg_source() {
        for nested in [
            JpegError::MemoryCapExceeded {
                requested: 65,
                cap: 64,
            },
            JpegError::HostAllocationFailed { bytes: 4096 },
            JpegError::InternalInvariant {
                reason: "test fast-packet invariant",
            },
        ] {
            let error =
                retain_fast_packet_failure::<()>(Err(FastPacketError::Decode(nested.clone())))
                    .expect_err("resource and invariant failures must not fall back silently");
            let packet_source = error
                .source()
                .and_then(|source| source.downcast_ref::<FastPacketError>())
                .expect("typed fast-packet source");
            let jpeg_source = packet_source
                .source()
                .and_then(|source| source.downcast_ref::<JpegError>())
                .expect("nested JPEG source");

            assert_eq!(jpeg_source, &nested);
        }
    }
}
