// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(test)]
use super::allocation::checked_entropy_live_bytes;
use super::allocation::try_vec_with_exact_capacity;
use super::error::FastPacketError;
use crate::error::{JpegError, MarkerKind};
use alloc::vec::Vec;
use j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES;

#[derive(Debug, PartialEq, Eq)]
pub(super) struct EntropySegments {
    pub(super) entropy_bytes: Vec<u8>,
    pub(super) restart_offsets: Vec<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct EntropySegmentLayout {
    pub(super) entropy_len: usize,
    pub(super) restart_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EntropyEvent {
    Byte(u8),
    Restart,
    Eoi,
}

struct EntropyEventCursor<'a> {
    bytes: &'a [u8],
    pos: usize,
    restart_markers_enabled: bool,
    expected_rst: u8,
    allow_missing_eoi: bool,
    finished: bool,
}

impl<'a> EntropyEventCursor<'a> {
    fn new(bytes: &'a [u8], restart_interval: Option<u16>) -> Self {
        Self::new_with_missing_eoi(bytes, restart_interval, false)
    }

    fn new_with_missing_eoi(
        bytes: &'a [u8],
        restart_interval: Option<u16>,
        allow_missing_eoi: bool,
    ) -> Self {
        Self {
            bytes,
            pos: 0,
            restart_markers_enabled: restart_interval.unwrap_or(0) != 0,
            expected_rst: 0xd0,
            allow_missing_eoi,
            finished: false,
        }
    }

    fn next_event(&mut self) -> Result<Option<EntropyEvent>, FastPacketError> {
        if self.finished {
            return Ok(None);
        }
        let Some(&byte) = self.bytes.get(self.pos) else {
            if self.allow_missing_eoi {
                self.finished = true;
                return Ok(Some(EntropyEvent::Eoi));
            }
            return Err(FastPacketError::Decode(JpegError::MissingMarker {
                marker: MarkerKind::Eoi,
            }));
        };
        if byte != 0xff {
            self.pos += 1;
            return Ok(Some(EntropyEvent::Byte(byte)));
        }

        let Some(&marker) = self.bytes.get(self.pos + 1) else {
            if self.allow_missing_eoi {
                self.finished = true;
                return Ok(Some(EntropyEvent::Eoi));
            }
            return Err(FastPacketError::TruncatedEntropy);
        };
        self.pos += 2;
        match marker {
            0x00 => Ok(Some(EntropyEvent::Byte(0xff))),
            0xd9 => {
                self.finished = true;
                Ok(Some(EntropyEvent::Eoi))
            }
            0xd0..=0xd7 if self.restart_markers_enabled => {
                if marker != self.expected_rst {
                    return Err(FastPacketError::EntropyMarkerUnsupported { marker });
                }
                self.expected_rst = if self.expected_rst == 0xd7 {
                    0xd0
                } else {
                    self.expected_rst + 1
                };
                Ok(Some(EntropyEvent::Restart))
            }
            marker => Err(FastPacketError::EntropyMarkerUnsupported { marker }),
        }
    }
}

#[cfg(test)]
pub(super) fn inspect_entropy_segments(
    bytes: &[u8],
    restart_interval: Option<u16>,
) -> Result<EntropySegmentLayout, FastPacketError> {
    inspect_entropy_segments_with_missing_eoi(bytes, restart_interval, false)
}

pub(super) fn inspect_entropy_segments_allow_missing_eoi(
    bytes: &[u8],
    restart_interval: Option<u16>,
) -> Result<EntropySegmentLayout, FastPacketError> {
    inspect_entropy_segments_with_missing_eoi(bytes, restart_interval, true)
}

fn inspect_entropy_segments_with_missing_eoi(
    bytes: &[u8],
    restart_interval: Option<u16>,
    allow_missing_eoi: bool,
) -> Result<EntropySegmentLayout, FastPacketError> {
    let mut entropy_len = 0usize;
    let mut restart_count = 1usize;
    let mut events =
        EntropyEventCursor::new_with_missing_eoi(bytes, restart_interval, allow_missing_eoi);
    while let Some(event) = events.next_event()? {
        match event {
            EntropyEvent::Byte(_) => {
                entropy_len = entropy_len.checked_add(1).ok_or_else(cap_overflow)?;
            }
            EntropyEvent::Restart => {
                u32::try_from(entropy_len).map_err(|_| FastPacketError::TruncatedEntropy)?;
                restart_count = restart_count.checked_add(1).ok_or_else(cap_overflow)?;
            }
            EntropyEvent::Eoi => {
                return Ok(EntropySegmentLayout {
                    entropy_len,
                    restart_count,
                });
            }
        }
    }
    Err(FastPacketError::Decode(JpegError::InternalInvariant {
        reason: "entropy event stream ended without EOI",
    }))
}

#[cfg(test)]
pub(super) fn extract_entropy_segments_with_cap(
    bytes: &[u8],
    restart_interval: Option<u16>,
    allocation_cap: usize,
) -> Result<EntropySegments, FastPacketError> {
    let layout = inspect_entropy_segments(bytes, restart_interval)?;
    checked_entropy_live_bytes(layout.entropy_len, layout.restart_count, allocation_cap)?;
    let mut live_bytes = 0;
    extract_entropy_segments_from_layout(
        bytes,
        restart_interval,
        layout,
        &mut live_bytes,
        allocation_cap,
    )
}

pub(super) fn extract_entropy_segments_from_layout(
    bytes: &[u8],
    restart_interval: Option<u16>,
    layout: EntropySegmentLayout,
    live_bytes: &mut usize,
    allocation_cap: usize,
) -> Result<EntropySegments, FastPacketError> {
    let mut entropy_bytes =
        try_vec_with_exact_capacity(layout.entropy_len, live_bytes, allocation_cap)?;
    let mut restart_offsets =
        try_vec_with_exact_capacity(layout.restart_count, live_bytes, allocation_cap)?;
    if layout.restart_count == 0 {
        return Err(materialization_exceeded_plan());
    }
    restart_offsets.push(0);

    let mut events = EntropyEventCursor::new(bytes, restart_interval);
    while let Some(event) = events.next_event()? {
        match event {
            EntropyEvent::Byte(byte) => {
                if entropy_bytes.len() >= layout.entropy_len {
                    return Err(materialization_exceeded_plan());
                }
                entropy_bytes.push(byte);
            }
            EntropyEvent::Restart => {
                if restart_offsets.len() >= layout.restart_count {
                    return Err(materialization_exceeded_plan());
                }
                let restart_offset = u32::try_from(entropy_bytes.len())
                    .map_err(|_| FastPacketError::TruncatedEntropy)?;
                restart_offsets.push(restart_offset);
            }
            EntropyEvent::Eoi => {
                if entropy_bytes.len() != layout.entropy_len
                    || restart_offsets.len() != layout.restart_count
                {
                    return Err(FastPacketError::Decode(JpegError::InternalInvariant {
                        reason: "entropy materialization disagrees with its allocation plan",
                    }));
                }
                return Ok(EntropySegments {
                    entropy_bytes,
                    restart_offsets,
                });
            }
        }
    }

    Err(FastPacketError::Decode(JpegError::InternalInvariant {
        reason: "entropy event stream ended without EOI",
    }))
}

fn cap_overflow() -> FastPacketError {
    FastPacketError::Decode(JpegError::MemoryCapExceeded {
        requested: usize::MAX,
        cap: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
    })
}

fn materialization_exceeded_plan() -> FastPacketError {
    FastPacketError::Decode(JpegError::InternalInvariant {
        reason: "entropy materialization exceeded its allocation plan",
    })
}
