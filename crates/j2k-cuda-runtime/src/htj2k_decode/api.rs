// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use crate::{
    bytes::u16_slice_as_bytes,
    context::CudaContext,
    error::CudaError,
    execution::CudaExecutionStats,
    memory::{pooled_device_buffer, CudaBufferPool},
};

use super::{
    output_regions::validate_htj2k_output_layout,
    types::{
        CudaHtj2kCodeBlockJob, CudaHtj2kDecodeOutput, CudaHtj2kDecodePayload,
        CudaHtj2kDecodeResources, CudaHtj2kDecodeStageTimings, CudaHtj2kDecodeTableResourceInner,
        CudaHtj2kDecodeTableResources, CudaHtj2kDecodeTables, CudaPooledHtj2kDecodeOutput,
    },
};

impl CudaContext {
    /// Decode HTJ2K code blocks into a device-resident f32 coefficient plane.
    #[doc(hidden)]
    pub fn decode_htj2k_codeblocks(
        &self,
        payload: &[u8],
        jobs: &[CudaHtj2kCodeBlockJob],
        tables: CudaHtj2kDecodeTables<'_>,
        output_words: usize,
    ) -> Result<CudaHtj2kDecodeOutput, CudaError> {
        if jobs.is_empty() {
            return self.decode_empty_htj2k_codeblocks(jobs, output_words);
        }
        let resources = self.upload_htj2k_decode_resources(payload, tables)?;
        self.decode_htj2k_codeblocks_with_resources(&resources, jobs, output_words)
    }

    /// Upload HTJ2K decode payload and lookup tables once for reuse by sub-band dispatches.
    fn upload_htj2k_decode_resources(
        &self,
        payload: &[u8],
        tables: CudaHtj2kDecodeTables<'_>,
    ) -> Result<CudaHtj2kDecodeResources, CudaError> {
        let tables = self.upload_htj2k_decode_table_resources(tables)?;
        self.upload_htj2k_decode_resources_with_tables(payload, &tables)
    }

    /// Upload static HTJ2K cleanup decode lookup tables once for reuse.
    #[doc(hidden)]
    pub fn upload_htj2k_decode_table_resources(
        &self,
        tables: CudaHtj2kDecodeTables<'_>,
    ) -> Result<CudaHtj2kDecodeTableResources, CudaError> {
        self.inner.set_current()?;
        Ok(CudaHtj2kDecodeTableResources {
            inner: Arc::new(CudaHtj2kDecodeTableResourceInner {
                vlc_table0: self.upload(u16_slice_as_bytes(tables.vlc_table0))?,
                vlc_table1: self.upload(u16_slice_as_bytes(tables.vlc_table1))?,
                uvlc_table0: self.upload(u16_slice_as_bytes(tables.uvlc_table0))?,
                uvlc_table1: self.upload(u16_slice_as_bytes(tables.uvlc_table1))?,
            }),
        })
    }

    /// Upload an HTJ2K decode payload while reusing already resident cleanup tables.
    #[doc(hidden)]
    pub fn upload_htj2k_decode_resources_with_tables(
        &self,
        payload: &[u8],
        tables: &CudaHtj2kDecodeTableResources,
    ) -> Result<CudaHtj2kDecodeResources, CudaError> {
        if !tables.is_owned_by(self) {
            return Err(CudaError::InvalidArgument {
                message: "HTJ2K decode tables must belong to the upload context".to_string(),
            });
        }
        self.inner.set_current()?;
        Ok(CudaHtj2kDecodeResources {
            payload: CudaHtj2kDecodePayload::Owned(self.upload_pinned(payload)?),
            payload_len: payload.len(),
            tables: Some(tables.clone()),
        })
    }

    /// Upload a classic-only J2K payload without HTJ2K lookup tables.
    #[doc(hidden)]
    pub fn upload_j2k_decode_payload(
        &self,
        payload: &[u8],
    ) -> Result<CudaHtj2kDecodeResources, CudaError> {
        self.inner.set_current()?;
        Ok(CudaHtj2kDecodeResources {
            payload: CudaHtj2kDecodePayload::Owned(self.upload_pinned(payload)?),
            payload_len: payload.len(),
            tables: None,
        })
    }

    /// Upload an HTJ2K decode payload into a pooled buffer while reusing already resident cleanup tables.
    #[doc(hidden)]
    pub fn upload_htj2k_decode_resources_with_tables_and_pool(
        &self,
        payload: &[u8],
        tables: &CudaHtj2kDecodeTableResources,
        pool: &CudaBufferPool,
    ) -> Result<CudaHtj2kDecodeResources, CudaError> {
        if !tables.is_owned_by(self) || !pool.is_owned_by(self) {
            return Err(CudaError::InvalidArgument {
                message: "HTJ2K decode tables and pool must belong to the upload context"
                    .to_string(),
            });
        }
        self.inner.set_current()?;
        Ok(CudaHtj2kDecodeResources {
            payload: CudaHtj2kDecodePayload::Pooled(pool.upload_pinned(payload)?),
            payload_len: payload.len(),
            tables: Some(tables.clone()),
        })
    }

    /// Upload a classic-only J2K payload into a pooled buffer without HTJ2K lookup tables.
    #[doc(hidden)]
    pub fn upload_j2k_decode_payload_with_pool(
        &self,
        payload: &[u8],
        pool: &CudaBufferPool,
    ) -> Result<CudaHtj2kDecodeResources, CudaError> {
        if !pool.is_owned_by(self) {
            return Err(CudaError::InvalidArgument {
                message: "J2K decode payload pool must belong to the upload context".to_string(),
            });
        }
        self.inner.set_current()?;
        Ok(CudaHtj2kDecodeResources {
            payload: CudaHtj2kDecodePayload::Pooled(pool.upload_pinned(payload)?),
            payload_len: payload.len(),
            tables: None,
        })
    }

    /// Decode HTJ2K code blocks using already resident payload and lookup tables.
    pub(crate) fn decode_htj2k_codeblocks_with_resources(
        &self,
        resources: &CudaHtj2kDecodeResources,
        jobs: &[CudaHtj2kCodeBlockJob],
        output_words: usize,
    ) -> Result<CudaHtj2kDecodeOutput, CudaError> {
        self.decode_htj2k_codeblocks_with_resources_impl(resources, jobs, output_words, true)
    }

    /// Allocate and initialize an HTJ2K coefficient output buffer without
    /// launching entropy cleanup decode. This is used when cleanup work is
    /// batched across multiple output buffers.
    #[doc(hidden)]
    pub fn allocate_htj2k_codeblock_coefficients_with_pool(
        &self,
        jobs: &[CudaHtj2kCodeBlockJob],
        output_words: usize,
        pool: &CudaBufferPool,
    ) -> Result<CudaPooledHtj2kDecodeOutput, CudaError> {
        if !pool.is_owned_by(self) {
            return Err(CudaError::InvalidArgument {
                message: "HTJ2K coefficient pool must belong to the allocation context".to_string(),
            });
        }
        let output_layout = validate_htj2k_output_layout(jobs, output_words)?;
        self.inner.set_current()?;
        let coefficients = pool.take(output_layout.output_bytes)?;
        let coefficient_buffer = pooled_device_buffer(&coefficients)?;
        if output_layout.needs_zero_fill {
            self.memset_d32(coefficient_buffer, 0, output_words)?;
            self.synchronize()?;
        }
        Ok(CudaPooledHtj2kDecodeOutput {
            coefficients,
            execution: CudaExecutionStats::default(),
            statuses: Vec::new(),
            stage_timings: CudaHtj2kDecodeStageTimings::default(),
        })
    }
}
