// SPDX-License-Identifier: MIT OR Apache-2.0

//! Profile-only attribution for the unchanged CUDA JPEG decode launch graph.

use std::time::Instant;

use crate::{error::CudaError, JpegCudaEngine};
use j2k_cuda_runtime::{elapsed_event_us_ceil, CudaEvent};

use super::super::CudaJpegDecodeStageTimings;

pub(super) struct CudaJpegDecodeStageProfiler {
    collect: bool,
    resource_upload_start: Option<Instant>,
    resource_upload_us: u128,
    fused_start: Option<CudaEvent>,
    fused_end: Option<CudaEvent>,
    conversion_end: Option<CudaEvent>,
    status_readback_start: Option<Instant>,
    status_readback_us: u128,
}

impl CudaJpegDecodeStageProfiler {
    pub(super) fn new(collect: bool) -> Self {
        Self {
            collect,
            resource_upload_start: collect.then(Instant::now),
            resource_upload_us: 0,
            fused_start: None,
            fused_end: None,
            conversion_end: None,
            status_readback_start: None,
            status_readback_us: 0,
        }
    }

    pub(super) fn finish_resource_upload(&mut self) {
        self.resource_upload_us = self
            .resource_upload_start
            .take()
            .map_or(0, |start| start.elapsed().as_micros());
    }

    pub(super) fn begin_fused(&mut self, engine: JpegCudaEngine<'_>) -> Result<(), CudaError> {
        self.fused_start = self.record_event(engine)?;
        Ok(())
    }

    pub(super) fn finish_fused(&mut self, engine: JpegCudaEngine<'_>) -> Result<(), CudaError> {
        self.fused_end = self.record_event(engine)?;
        Ok(())
    }

    pub(super) fn finish_conversion(
        &mut self,
        engine: JpegCudaEngine<'_>,
    ) -> Result<(), CudaError> {
        self.conversion_end = self.record_event(engine)?;
        Ok(())
    }

    pub(super) fn begin_status_readback(&mut self) {
        self.status_readback_start = self.collect.then(Instant::now);
    }

    pub(super) fn synchronize_device_stages(&self) -> Result<(), CudaError> {
        self.conversion_end
            .as_ref()
            .or(self.fused_end.as_ref())
            .map(CudaEvent::synchronize)
            .transpose()
            .map(|_| ())
    }

    pub(super) fn finish_status_readback(&mut self) {
        self.status_readback_us = self
            .status_readback_start
            .take()
            .map_or(0, |start| start.elapsed().as_micros());
    }

    pub(super) fn finish(
        self,
        component_workspace_bytes: usize,
    ) -> Result<CudaJpegDecodeStageTimings, CudaError> {
        let fused_decode_kernel_us = elapsed(self.fused_start.as_ref(), self.fused_end.as_ref())?;
        let conversion_us = elapsed(self.fused_end.as_ref(), self.conversion_end.as_ref())?;
        Ok(CudaJpegDecodeStageTimings {
            resource_upload_us: self.resource_upload_us,
            fused_decode_kernel_us,
            conversion_us,
            status_readback_us: self.status_readback_us,
            component_workspace_bytes,
        })
    }

    fn record_event(&self, engine: JpegCudaEngine<'_>) -> Result<Option<CudaEvent>, CudaError> {
        if !self.collect {
            return Ok(None);
        }
        let event = engine.context().create_event()?;
        event.record_default_stream()?;
        Ok(Some(event))
    }
}

fn elapsed(start: Option<&CudaEvent>, end: Option<&CudaEvent>) -> Result<u128, CudaError> {
    match (start, end) {
        (Some(start), Some(end)) => elapsed_event_us_ceil(start, end),
        _ => Ok(0),
    }
}
