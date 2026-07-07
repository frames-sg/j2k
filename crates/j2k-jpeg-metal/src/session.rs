// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::VecDeque;
use std::sync::{Arc, Mutex, MutexGuard};

use j2k_core::{BackendKind, BackendRequest};
use j2k_jpeg::adapter::{
    build_fast420_packet, build_fast422_packet, build_fast444_packet, JpegFast420PacketV1,
    JpegFast422PacketV1, JpegFast444PacketV1,
};
#[cfg(target_os = "macos")]
use j2k_metal_support::{MetalRuntimeSession, MetalSupportError};
#[cfg(target_os = "macos")]
use metal::Device;

#[cfg(target_os = "macos")]
use crate::compute;
use crate::{batch, Error};

const BATCH_SHAPE_CACHE_SLOTS: usize = 8;
const FAST_PACKET_CACHE_SLOTS: usize = 8;
const INPUT_ALIAS_CACHE_SLOTS: usize = 8;

pub(crate) type SharedFastPackets = (
    Option<Arc<JpegFast444PacketV1>>,
    Option<Arc<JpegFast422PacketV1>>,
    Option<Arc<JpegFast420PacketV1>>,
);

#[cfg(target_os = "macos")]
#[derive(Clone)]
/// Reusable Metal device session for decode and encode submissions.
pub struct MetalBackendSession {
    runtime_session: MetalRuntimeSession<compute::MetalRuntime, MetalSupportError>,
}

#[cfg(target_os = "macos")]
impl MetalBackendSession {
    /// Create a session bound to an existing Metal device.
    pub fn new(device: Device) -> Self {
        Self {
            runtime_session: MetalRuntimeSession::new(device),
        }
    }

    /// Create a session from the system default Metal device.
    pub fn system_default() -> Result<Self, Error> {
        MetalRuntimeSession::system_default()
            .map(|runtime_session| Self { runtime_session })
            .map_err(|error| compute::runtime_initialization_error(&error))
    }

    /// Metal device used by this session.
    pub fn device(&self) -> &metal::DeviceRef {
        self.runtime_session.device()
    }

    pub(crate) fn runtime_result(&self) -> &Result<compute::MetalRuntime, MetalSupportError> {
        self.runtime_session
            .get_or_init_runtime(|device| compute::MetalRuntime::new_with_device(device.clone()))
    }

    #[cfg(test)]
    pub(crate) fn runtime_initialized_for_test(&self) -> bool {
        self.runtime_session.runtime_initialized()
    }

    #[cfg(test)]
    pub(crate) fn runtime_ptr_for_test(&self) -> Option<*const compute::MetalRuntime> {
        self.runtime_session
            .runtime_result()
            .and_then(|runtime| runtime.as_ref().ok())
            .map(std::ptr::from_ref::<compute::MetalRuntime>)
    }
}

#[cfg(target_os = "macos")]
#[doc(hidden)]
impl j2k_core::AcceleratorSession for MetalBackendSession {
    fn backend_kind(&self) -> BackendKind {
        BackendKind::Metal
    }
}

#[cfg(target_os = "macos")]
impl core::fmt::Debug for MetalBackendSession {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MetalBackendSession")
            .field("device", &self.runtime_session.device_handle().name())
            .field(
                "runtime_initialized",
                &self.runtime_session.runtime_initialized(),
            )
            .finish()
    }
}

#[cfg(not(target_os = "macos"))]
#[derive(Clone, Copy, Debug, Default)]
/// Placeholder Metal session for non-macOS builds.
pub struct MetalBackendSession {
    _private: (),
}

#[cfg(not(target_os = "macos"))]
impl MetalBackendSession {
    /// Return `Error::MetalUnavailable` on hosts without Metal support.
    pub fn system_default() -> Result<Self, Error> {
        Err(Error::MetalUnavailable)
    }
}

#[derive(Clone)]
pub(crate) struct CachedBatchShape {
    digest: u64,
    input: Arc<[u8]>,
    shape: batch::BatchShape,
}

#[derive(Clone)]
pub(crate) struct CachedFastPackets {
    digest: u64,
    input: Arc<[u8]>,
    fast444_packet: Option<Arc<JpegFast444PacketV1>>,
    fast422_packet: Option<Arc<JpegFast422PacketV1>>,
    fast420_packet: Option<Arc<JpegFast420PacketV1>>,
}

#[derive(Clone)]
struct CachedInputAlias {
    source_ptr: usize,
    source_len: usize,
    digest: u64,
    input: Arc<[u8]>,
}

#[derive(Default)]
pub(crate) struct SessionState {
    pub(crate) submissions: u64,
    pub(crate) queued: Vec<crate::batch::QueuedRequest>,
    pub(crate) completed: Vec<Option<Result<crate::Surface, crate::Error>>>,
    #[cfg(target_os = "macos")]
    pub(crate) backend_session: Option<MetalBackendSession>,
    batch_shapes: VecDeque<CachedBatchShape>,
    fast_packets: VecDeque<CachedFastPackets>,
    input_aliases: VecDeque<CachedInputAlias>,
}

impl SessionState {
    #[cfg(target_os = "macos")]
    pub(crate) fn with_backend_session(backend_session: MetalBackendSession) -> Self {
        Self {
            backend_session: Some(backend_session),
            ..Self::default()
        }
    }

    pub(crate) fn queue_request(&mut self, request: crate::batch::QueuedRequest) -> usize {
        let slot = self.completed.len();
        self.completed.push(None);
        self.queued.push(request.with_output_slot(slot));
        slot
    }

    pub(crate) fn intern_input_slice(&mut self, input: &[u8]) -> Arc<[u8]> {
        let source_ptr = input.as_ptr() as usize;
        let source_len = input.len();
        // Pointer identity alone is unsound: a caller may reuse one allocation
        // for different payloads, so every hit is verified by digest plus byte
        // equality, mirroring resolve_batch_shape/resolve_fast_packets.
        let digest = digest_bytes(input);
        if let Some(entry) = self
            .input_aliases
            .iter_mut()
            .find(|entry| entry.source_ptr == source_ptr && entry.source_len == source_len)
        {
            if entry.digest == digest && entry.input.as_ref() == input {
                return Arc::clone(&entry.input);
            }
            let refreshed = Arc::<[u8]>::from(input);
            entry.digest = digest;
            entry.input = Arc::clone(&refreshed);
            return refreshed;
        }

        let input = Arc::<[u8]>::from(input);
        if self.input_aliases.len() == INPUT_ALIAS_CACHE_SLOTS {
            self.input_aliases.pop_front();
        }
        self.input_aliases.push_back(CachedInputAlias {
            source_ptr,
            source_len,
            digest,
            input: Arc::clone(&input),
        });
        input
    }

    pub(crate) fn resolve_batch_shape(
        &mut self,
        input: &Arc<[u8]>,
        backend: BackendRequest,
    ) -> Result<batch::BatchShape, Error> {
        #[cfg(not(target_os = "macos"))]
        {
            if matches!(backend, BackendRequest::Auto | BackendRequest::Metal) {
                return Ok(batch::BatchShape {
                    restart_interval: None,
                    checkpoint_count: 0,
                    sampling_family: batch::SamplingFamily::Unknown,
                });
            }
        }

        match backend {
            BackendRequest::Auto | BackendRequest::Metal => {}
            BackendRequest::Cpu | BackendRequest::Cuda => {
                return Ok(batch::BatchShape {
                    restart_interval: None,
                    checkpoint_count: 0,
                    sampling_family: batch::SamplingFamily::Unknown,
                });
            }
        }

        if let Some(entry) = self
            .batch_shapes
            .iter()
            .find(|entry| Arc::ptr_eq(&entry.input, input))
        {
            return Ok(entry.shape);
        }

        let digest = digest_bytes(input.as_ref());
        if let Some(entry) = self
            .batch_shapes
            .iter()
            .find(|entry| entry.digest == digest && entry.input.as_ref() == input.as_ref())
        {
            return Ok(entry.shape);
        }

        let decoder = j2k_jpeg::Decoder::new(input.as_ref())?;
        let summary = j2k_jpeg::adapter::summarize_device_batch(&decoder, 4);
        let shape = batch::BatchShape {
            restart_interval: summary.restart_interval,
            checkpoint_count: summary.checkpoint_count,
            sampling_family: if summary.matches_fast_420 {
                batch::SamplingFamily::Fast420
            } else if summary.matches_fast_422 {
                batch::SamplingFamily::Fast422
            } else if summary.matches_fast_444 {
                batch::SamplingFamily::Fast444
            } else {
                batch::SamplingFamily::Other
            },
        };

        if self.batch_shapes.len() == BATCH_SHAPE_CACHE_SLOTS {
            self.batch_shapes.pop_front();
        }
        self.batch_shapes.push_back(CachedBatchShape {
            digest,
            input: Arc::clone(input),
            shape,
        });

        Ok(shape)
    }

    pub(crate) fn resolve_fast_packets(
        &mut self,
        input: &Arc<[u8]>,
        backend: BackendRequest,
    ) -> SharedFastPackets {
        if !matches!(backend, BackendRequest::Auto | BackendRequest::Metal) {
            return (None, None, None);
        }

        if let Some(entry) = self
            .fast_packets
            .iter()
            .find(|entry| Arc::ptr_eq(&entry.input, input))
        {
            return (
                entry.fast444_packet.clone(),
                entry.fast422_packet.clone(),
                entry.fast420_packet.clone(),
            );
        }

        let digest = digest_bytes(input.as_ref());
        if let Some(entry) = self
            .fast_packets
            .iter()
            .find(|entry| entry.digest == digest && entry.input.as_ref() == input.as_ref())
        {
            return (
                entry.fast444_packet.clone(),
                entry.fast422_packet.clone(),
                entry.fast420_packet.clone(),
            );
        }

        let fast444_packet = build_fast444_packet(input.as_ref()).ok().map(Arc::new);
        let fast422_packet = build_fast422_packet(input.as_ref()).ok().map(Arc::new);
        let fast420_packet = build_fast420_packet(input.as_ref()).ok().map(Arc::new);
        if self.fast_packets.len() == FAST_PACKET_CACHE_SLOTS {
            self.fast_packets.pop_front();
        }
        self.fast_packets.push_back(CachedFastPackets {
            digest,
            input: Arc::clone(input),
            fast444_packet: fast444_packet.clone(),
            fast422_packet: fast422_packet.clone(),
            fast420_packet: fast420_packet.clone(),
        });

        (fast444_packet, fast422_packet, fast420_packet)
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn backend_session(&mut self) -> Result<&MetalBackendSession, Error> {
        if self.backend_session.is_none() {
            self.backend_session = Some(MetalBackendSession::system_default()?);
        }
        Ok(self
            .backend_session
            .as_ref()
            .expect("backend session initialized"))
    }
}

#[derive(Clone, Default)]
pub(crate) struct SharedSession(pub(crate) Arc<Mutex<SessionState>>);

impl SharedSession {
    pub(crate) fn lock(&self) -> Result<MutexGuard<'_, SessionState>, Error> {
        self.0.lock().map_err(|_| Error::MetalStatePoisoned {
            state: "JPEG Metal session",
        })
    }
}

#[derive(Default)]
/// Shared batching session used by `JpegTileBatch` and submit APIs.
pub struct MetalSession {
    pub(crate) shared: SharedSession,
}

impl MetalSession {
    /// Create a tile batching session that reuses an existing Metal backend session.
    #[cfg(target_os = "macos")]
    pub fn with_backend_session(backend_session: MetalBackendSession) -> Self {
        Self {
            shared: SharedSession(Arc::new(Mutex::new(SessionState::with_backend_session(
                backend_session,
            )))),
        }
    }

    /// Number of Metal or emulated submissions flushed through this session.
    pub fn submissions(&self) -> Result<u64, Error> {
        Ok(self.shared.lock()?.submissions)
    }
}

impl core::fmt::Debug for MetalSession {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MetalSession")
            .field("submissions", &self.submissions())
            .finish()
    }
}

fn digest_bytes(bytes: &[u8]) -> u64 {
    j2k_core::__j2k_fnv1a64_bytes!(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_os = "macos")]
    #[test]
    fn batch_shape_cache_hits_for_repeated_input() {
        let mut session = SessionState::default();
        let input =
            Arc::<[u8]>::from(include_bytes!("../fixtures/jpeg/baseline_420_16x16.jpg").as_slice());

        let first = session
            .resolve_batch_shape(&input, BackendRequest::Metal)
            .expect("first shape");
        let second = session
            .resolve_batch_shape(&input, BackendRequest::Metal)
            .expect("second shape");

        assert_eq!(first, second);
        assert_eq!(session.batch_shapes.len(), 1);
    }

    #[test]
    fn fast_packet_cache_hits_for_repeated_input() {
        let mut session = SessionState::default();
        let input =
            Arc::<[u8]>::from(include_bytes!("../fixtures/jpeg/baseline_420_16x16.jpg").as_slice());

        let first = session.resolve_fast_packets(&input, BackendRequest::Metal);
        let second = session.resolve_fast_packets(&input, BackendRequest::Metal);

        assert!(first.2.is_some());
        assert_eq!(first, second);
        assert_eq!(session.fast_packets.len(), 1);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn batch_shape_tracks_fast422_sampling_family() {
        let mut session = SessionState::default();
        let input =
            Arc::<[u8]>::from(include_bytes!("../fixtures/jpeg/baseline_422_16x8.jpg").as_slice());

        let shape = session
            .resolve_batch_shape(&input, BackendRequest::Metal)
            .expect("fast422 shape");

        assert_eq!(shape.sampling_family, batch::SamplingFamily::Fast422);
    }

    #[test]
    fn intern_input_slice_refreshes_when_buffer_is_overwritten() {
        let mut session = SessionState::default();
        let mut buffer = vec![0xAAu8; 64];

        let first = session.intern_input_slice(&buffer);
        assert_eq!(first.as_ref(), [0xAAu8; 64].as_slice());

        // Reuse the same allocation (same pointer, same length) with new contents.
        buffer.fill(0xBB);
        let second = session.intern_input_slice(&buffer);
        assert_eq!(
            second.as_ref(),
            [0xBBu8; 64].as_slice(),
            "input-alias cache returned stale bytes for a reused buffer"
        );
        assert_eq!(session.input_aliases.len(), 1);
    }

    #[test]
    fn intern_input_slice_reuses_interned_arc_for_unchanged_buffer() {
        let mut session = SessionState::default();
        let buffer = vec![0x42u8; 64];

        let first = session.intern_input_slice(&buffer);
        let second = session.intern_input_slice(&buffer);

        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(session.input_aliases.len(), 1);
    }

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn non_macos_auto_and_metal_shape_resolution_stays_unparsed() {
        let mut session = SessionState::default();
        let invalid = Arc::<[u8]>::from(&b"not a jpeg"[..]);

        let auto = session
            .resolve_batch_shape(&invalid, BackendRequest::Auto)
            .expect("auto shape");
        let metal = session
            .resolve_batch_shape(&invalid, BackendRequest::Metal)
            .expect("metal shape");

        assert_eq!(auto.sampling_family, batch::SamplingFamily::Unknown);
        assert_eq!(metal.sampling_family, batch::SamplingFamily::Unknown);
        assert!(session.batch_shapes.is_empty());
    }
}
