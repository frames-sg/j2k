// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(feature = "cuda-runtime")]
use std::collections::VecDeque;
#[cfg(feature = "cuda-runtime")]
use std::sync::Arc;

#[cfg(feature = "cuda-runtime")]
use j2k_cuda_runtime::{CudaBufferPool, CudaContext, CudaDeviceBuffer};
#[cfg(feature = "cuda-runtime")]
use j2k_jpeg::adapter::{
    build_fast420_packet, build_fast422_packet, build_fast444_packet, FastPacketError,
    JpegFast420PacketV1, JpegFast422PacketV1, JpegFast444PacketV1,
};

#[cfg(feature = "cuda-runtime")]
use crate::runtime::cuda_error;
#[cfg(feature = "cuda-runtime")]
use crate::Error;

#[cfg(feature = "cuda-runtime")]
const OWNED_PACKET_CACHE_SLOTS: usize = 8;
#[cfg(feature = "cuda-runtime")]
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
#[cfg(feature = "cuda-runtime")]
const FNV_PRIME: u64 = 0x0000_0100_0000_01B3;

#[cfg(feature = "cuda-runtime")]
#[derive(Clone)]
struct CachedOwnedFastPacket<T> {
    digest: u64,
    input: Arc<[u8]>,
    packet: Arc<T>,
}

#[derive(Clone, Default)]
/// Reusable CUDA JPEG decode session.
pub struct CudaSession {
    submissions: u64,
    #[cfg(feature = "cuda-runtime")]
    owned_fast420_packets: VecDeque<CachedOwnedFastPacket<JpegFast420PacketV1>>,
    #[cfg(feature = "cuda-runtime")]
    owned_fast422_packets: VecDeque<CachedOwnedFastPacket<JpegFast422PacketV1>>,
    #[cfg(feature = "cuda-runtime")]
    owned_fast444_packets: VecDeque<CachedOwnedFastPacket<JpegFast444PacketV1>>,
    #[cfg(feature = "cuda-runtime")]
    context: Option<CudaContext>,
    #[cfg(feature = "cuda-runtime")]
    owned_output_pool: Option<CudaBufferPool>,
}

impl CudaSession {
    /// Number of decode submissions recorded through this session.
    pub fn submissions(&self) -> u64 {
        self.submissions
    }

    /// Number of cached J2K-owned CUDA fast JPEG packets.
    pub fn owned_cuda_packet_cache_len(&self) -> usize {
        #[cfg(feature = "cuda-runtime")]
        {
            self.owned_fast420_packets.len()
                + self.owned_fast422_packets.len()
                + self.owned_fast444_packets.len()
        }
        #[cfg(not(feature = "cuda-runtime"))]
        {
            0
        }
    }

    #[cfg(feature = "cuda-runtime")]
    /// Whether a CUDA runtime context has been initialized successfully.
    pub fn is_runtime_initialized(&self) -> bool {
        self.context.is_some()
    }

    #[cfg(feature = "cuda-runtime")]
    /// Borrow or allocate a reusable CUDA output buffer for owned JPEG decode.
    ///
    /// Return buffers to the session with
    /// [`recycle_owned_cuda_output_buffer`](Self::recycle_owned_cuda_output_buffer).
    ///
    /// # Errors
    /// Returns a CUDA adapter error if the runtime is unavailable or the pool
    /// lock is poisoned.
    pub fn take_owned_cuda_output_buffer(
        &mut self,
        byte_len: usize,
    ) -> Result<CudaDeviceBuffer, Error> {
        let buffer = self
            .owned_output_pool()?
            .take(byte_len)
            .map_err(cuda_error)?;
        buffer.into_device_buffer().map_err(cuda_error)
    }

    #[cfg(feature = "cuda-runtime")]
    /// Return a CUDA output buffer to this session's owned JPEG decode pool.
    ///
    /// # Errors
    /// Returns a CUDA adapter error if the pool lock is poisoned.
    pub fn recycle_owned_cuda_output_buffer(
        &mut self,
        buffer: CudaDeviceBuffer,
    ) -> Result<(), Error> {
        self.owned_output_pool()?
            .recycle(buffer)
            .map_err(cuda_error)
    }

    #[cfg(feature = "cuda-runtime")]
    /// Number of reusable owned CUDA output buffers retained by this session.
    pub fn retained_owned_cuda_output_buffers(&self) -> Result<usize, Error> {
        self.owned_output_pool
            .as_ref()
            .map_or(Ok(0), |pool| pool.cached_count().map_err(cuda_error))
    }

    #[cfg(feature = "cuda-runtime")]
    pub(crate) fn resolve_owned_fast420_packet(
        &mut self,
        input: &[u8],
    ) -> Result<Arc<JpegFast420PacketV1>, Error> {
        resolve_owned_packet(&mut self.owned_fast420_packets, input, build_fast420_packet)
    }

    #[cfg(feature = "cuda-runtime")]
    pub(crate) fn resolve_owned_fast422_packet(
        &mut self,
        input: &[u8],
    ) -> Result<Arc<JpegFast422PacketV1>, Error> {
        resolve_owned_packet(&mut self.owned_fast422_packets, input, build_fast422_packet)
    }

    #[cfg(feature = "cuda-runtime")]
    pub(crate) fn resolve_owned_fast444_packet(
        &mut self,
        input: &[u8],
    ) -> Result<Arc<JpegFast444PacketV1>, Error> {
        resolve_owned_packet(&mut self.owned_fast444_packets, input, build_fast444_packet)
    }

    #[cfg(feature = "cuda-runtime")]
    pub(crate) fn cuda_context(&mut self) -> Result<CudaContext, Error> {
        if self.context.is_none() {
            self.context = Some(CudaContext::system_default().map_err(cuda_error)?);
        }
        self.context.clone().ok_or(Error::CudaUnavailable)
    }

    #[cfg(feature = "cuda-runtime")]
    fn owned_output_pool(&mut self) -> Result<CudaBufferPool, Error> {
        if let Some(pool) = &self.owned_output_pool {
            return Ok(pool.clone());
        }
        let pool = self.cuda_context()?.buffer_pool();
        self.owned_output_pool = Some(pool.clone());
        Ok(pool)
    }
}

impl j2k_core::DeviceSubmitSession for CudaSession {
    fn record_submit(&mut self) {
        self.submissions = self.submissions.saturating_add(1);
    }
}

impl j2k_core::AcceleratorSession for CudaSession {
    fn backend_kind(&self) -> j2k_core::BackendKind {
        j2k_core::BackendKind::Cuda
    }

    fn execution_stats(&self) -> j2k_core::ExecutionStats {
        j2k_core::ExecutionStats {
            submissions: self.submissions,
            ..j2k_core::ExecutionStats::default()
        }
    }
}

impl std::fmt::Debug for CudaSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut debug = f.debug_struct("CudaSession");
        debug.field("submissions", &self.submissions);
        debug.field(
            "owned_cuda_packet_cache_len",
            &self.owned_cuda_packet_cache_len(),
        );
        #[cfg(feature = "cuda-runtime")]
        debug.field("runtime_initialized", &self.is_runtime_initialized());
        #[cfg(feature = "cuda-runtime")]
        debug.field(
            "retained_owned_cuda_output_buffers",
            &self.retained_owned_cuda_output_buffers().ok(),
        );
        debug.finish_non_exhaustive()
    }
}

#[cfg(feature = "cuda-runtime")]
fn owned_packet_error(error: FastPacketError) -> Error {
    match error {
        FastPacketError::Decode(error) => Error::Decode(error),
        _ => Error::UnsupportedCudaRequest {
            reason: "J2K CUDA JPEG decode currently supports baseline 8-bit YCbCr 4:2:0, 4:2:2, or 4:4:4 RGB8 output",
        },
    }
}

#[cfg(feature = "cuda-runtime")]
fn resolve_owned_packet<T>(
    cache: &mut VecDeque<CachedOwnedFastPacket<T>>,
    input: &[u8],
    build: fn(&[u8]) -> Result<T, FastPacketError>,
) -> Result<Arc<T>, Error> {
    let digest = digest_bytes(input);
    if let Some(entry) = cache
        .iter()
        .find(|entry| entry.digest == digest && entry.input.as_ref() == input)
    {
        return Ok(Arc::clone(&entry.packet));
    }

    let packet = build(input).map_err(owned_packet_error)?;
    let input = Arc::<[u8]>::from(input);
    let packet = Arc::new(packet);
    if cache.len() == OWNED_PACKET_CACHE_SLOTS {
        cache.pop_front();
    }
    cache.push_back(CachedOwnedFastPacket {
        digest,
        input,
        packet: Arc::clone(&packet),
    });
    Ok(packet)
}

#[cfg(feature = "cuda-runtime")]
fn digest_bytes(bytes: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}
