// SPDX-License-Identifier: Apache-2.0

#[cfg(feature = "cuda-runtime")]
use signinum_cuda_runtime::{CudaContext, CudaDwt53Output};
use signinum_j2k_native::{
    EncodedHtJ2kCodeBlock, EncodedJ2kCodeBlock, J2kEncodeDispatchReport, J2kEncodeStageAccelerator,
    J2kForwardDwt53Job, J2kForwardDwt53Output, J2kForwardRctJob, J2kHtCodeBlockEncodeJob,
    J2kPacketizationEncodeJob, J2kTier1CodeBlockEncodeJob,
};

use crate::profile;

#[derive(Debug, Default, Clone)]
pub struct CudaEncodeStageAccelerator {
    #[cfg(feature = "cuda-runtime")]
    context: Option<CudaContext>,
    forward_rct_attempts: usize,
    forward_dwt53_attempts: usize,
    tier1_code_block_attempts: usize,
    ht_code_block_attempts: usize,
    packetization_attempts: usize,
    forward_rct_dispatches: usize,
    forward_dwt53_dispatches: usize,
    tier1_code_block_dispatches: usize,
    ht_code_block_dispatches: usize,
    packetization_dispatches: usize,
}

impl CudaEncodeStageAccelerator {
    #[cfg(feature = "cuda-runtime")]
    fn cuda_context(&mut self) -> core::result::Result<Option<CudaContext>, &'static str> {
        if self.context.is_none() {
            match CudaContext::system_default() {
                Ok(context) => self.context = Some(context),
                Err(_) if cuda_runtime_required() => return Err("CUDA encode stage unavailable"),
                Err(_) => return Ok(None),
            }
        }
        Ok(self.context.clone())
    }

    pub fn forward_rct_attempts(&self) -> usize {
        self.forward_rct_attempts
    }

    pub fn forward_dwt53_attempts(&self) -> usize {
        self.forward_dwt53_attempts
    }

    pub fn tier1_code_block_attempts(&self) -> usize {
        self.tier1_code_block_attempts
    }

    pub fn ht_code_block_attempts(&self) -> usize {
        self.ht_code_block_attempts
    }

    pub fn packetization_attempts(&self) -> usize {
        self.packetization_attempts
    }

    pub fn forward_rct_dispatches(&self) -> usize {
        self.forward_rct_dispatches
    }

    pub fn forward_dwt53_dispatches(&self) -> usize {
        self.forward_dwt53_dispatches
    }

    pub fn tier1_code_block_dispatches(&self) -> usize {
        self.tier1_code_block_dispatches
    }

    pub fn ht_code_block_dispatches(&self) -> usize {
        self.ht_code_block_dispatches
    }

    pub fn packetization_dispatches(&self) -> usize {
        self.packetization_dispatches
    }
}

#[cfg(feature = "cuda-runtime")]
fn cuda_runtime_required() -> bool {
    std::env::var_os("SIGNINUM_REQUIRE_CUDA_RUNTIME").is_some()
}

impl J2kEncodeStageAccelerator for CudaEncodeStageAccelerator {
    fn dispatch_report(&self) -> J2kEncodeDispatchReport {
        J2kEncodeDispatchReport {
            forward_rct: self.forward_rct_dispatches,
            forward_dwt53: self.forward_dwt53_dispatches,
            tier1_code_block: self.tier1_code_block_dispatches,
            ht_code_block: self.ht_code_block_dispatches,
            packetization: self.packetization_dispatches,
        }
    }

    fn encode_forward_rct(
        &mut self,
        job: J2kForwardRctJob<'_>,
    ) -> core::result::Result<bool, &'static str> {
        self.forward_rct_attempts = self.forward_rct_attempts.saturating_add(1);
        #[cfg(feature = "cuda-runtime")]
        if let Some(context) = self.cuda_context()? {
            context
                .j2k_forward_rct(job.plane0, job.plane1, job.plane2)
                .map_err(|_| "CUDA forward RCT encode kernel failed")?;
            self.forward_rct_dispatches = self.forward_rct_dispatches.saturating_add(1);
            if profile::gpu_route_profile_enabled() {
                profile::emit_gpu_route_profile(
                    "j2k",
                    "gpu_route",
                    "cuda",
                    &[
                        ("op", "encode_forward_rct"),
                        ("decision", "cuda_dispatch"),
                        ("dispatches", "1"),
                    ],
                );
            }
            return Ok(true);
        }
        #[cfg(not(feature = "cuda-runtime"))]
        let _ = job;
        if profile::gpu_route_profile_enabled() {
            profile::emit_gpu_route_profile(
                "j2k",
                "gpu_route",
                "cuda",
                &[
                    ("op", "encode_forward_rct"),
                    ("decision", "cpu_fallback"),
                    ("reason", "cuda_unavailable"),
                ],
            );
        }
        Ok(false)
    }

    fn encode_forward_dwt53(
        &mut self,
        job: J2kForwardDwt53Job<'_>,
    ) -> core::result::Result<Option<J2kForwardDwt53Output>, &'static str> {
        self.forward_dwt53_attempts = self.forward_dwt53_attempts.saturating_add(1);
        if job.num_levels == 0 {
            if profile::gpu_route_profile_enabled() {
                profile::emit_gpu_route_profile(
                    "j2k",
                    "gpu_route",
                    "cuda",
                    &[
                        ("op", "encode_forward_dwt53"),
                        ("decision", "cpu_fallback"),
                        ("reason", "zero_levels"),
                    ],
                );
            }
            return Ok(None);
        }
        #[cfg(feature = "cuda-runtime")]
        if let Some(context) = self.cuda_context()? {
            let output = context
                .j2k_forward_dwt53(job.samples, job.width, job.height, job.num_levels)
                .map_err(|_| "CUDA forward 5/3 DWT encode kernel failed")?;
            let dispatches = output.execution().kernel_dispatches();
            self.forward_dwt53_dispatches =
                self.forward_dwt53_dispatches.saturating_add(dispatches);
            if profile::gpu_route_profile_enabled() {
                let width_s = job.width.to_string();
                let height_s = job.height.to_string();
                let levels_s = job.num_levels.to_string();
                let dispatches_s = dispatches.to_string();
                profile::emit_gpu_route_profile(
                    "j2k",
                    "gpu_route",
                    "cuda",
                    &[
                        ("op", "encode_forward_dwt53"),
                        ("decision", "cuda_dispatch"),
                        ("width", width_s.as_str()),
                        ("height", height_s.as_str()),
                        ("levels", levels_s.as_str()),
                        ("dispatches", dispatches_s.as_str()),
                    ],
                );
            }
            return Ok(Some(cuda_dwt53_output_to_j2k(&output)?));
        }
        #[cfg(not(feature = "cuda-runtime"))]
        let _ = job;
        if profile::gpu_route_profile_enabled() {
            profile::emit_gpu_route_profile(
                "j2k",
                "gpu_route",
                "cuda",
                &[
                    ("op", "encode_forward_dwt53"),
                    ("decision", "cpu_fallback"),
                    ("reason", "cuda_unavailable"),
                ],
            );
        }
        Ok(None)
    }

    fn encode_tier1_code_block(
        &mut self,
        _job: J2kTier1CodeBlockEncodeJob<'_>,
    ) -> core::result::Result<Option<EncodedJ2kCodeBlock>, &'static str> {
        self.tier1_code_block_attempts = self.tier1_code_block_attempts.saturating_add(1);
        if profile::gpu_route_profile_enabled() {
            profile::emit_gpu_route_profile(
                "j2k",
                "gpu_route",
                "cuda",
                &[
                    ("op", "encode_tier1_code_block"),
                    ("decision", "cpu_fallback"),
                    ("reason", "unsupported_stage"),
                ],
            );
        }
        Ok(None)
    }

    fn encode_ht_code_block(
        &mut self,
        _job: J2kHtCodeBlockEncodeJob<'_>,
    ) -> core::result::Result<Option<EncodedHtJ2kCodeBlock>, &'static str> {
        self.ht_code_block_attempts = self.ht_code_block_attempts.saturating_add(1);
        if profile::gpu_route_profile_enabled() {
            profile::emit_gpu_route_profile(
                "j2k",
                "gpu_route",
                "cuda",
                &[
                    ("op", "encode_ht_code_block"),
                    ("decision", "cpu_fallback"),
                    ("reason", "unsupported_stage"),
                ],
            );
        }
        Ok(None)
    }

    fn encode_packetization(
        &mut self,
        _job: J2kPacketizationEncodeJob<'_>,
    ) -> core::result::Result<Option<Vec<u8>>, &'static str> {
        self.packetization_attempts = self.packetization_attempts.saturating_add(1);
        if profile::gpu_route_profile_enabled() {
            profile::emit_gpu_route_profile(
                "j2k",
                "gpu_route",
                "cuda",
                &[
                    ("op", "encode_packetization"),
                    ("decision", "cpu_fallback"),
                    ("reason", "unsupported_stage"),
                ],
            );
        }
        Ok(None)
    }
}

#[cfg(feature = "cuda-runtime")]
fn cuda_dwt53_output_to_j2k(
    output: &CudaDwt53Output,
) -> core::result::Result<J2kForwardDwt53Output, &'static str> {
    let (ll_width, ll_height) = output.ll_dimensions();
    let transformed = output.transformed();
    let full_width = output
        .levels()
        .first()
        .map_or(ll_width, |level| level.width) as usize;
    let mut ll = Vec::with_capacity((ll_width as usize) * (ll_height as usize));
    for y in 0..ll_height as usize {
        let row_start = y
            .checked_mul(full_width)
            .ok_or("CUDA DWT LL row offset overflow")?;
        ll.extend_from_slice(&transformed[row_start..row_start + ll_width as usize]);
    }

    let mut levels = Vec::with_capacity(output.levels().len());
    for shape in output.levels() {
        levels.push(signinum_j2k_native::J2kForwardDwt53Level {
            hl: extract_cuda_subband(
                transformed,
                full_width,
                shape.low_width,
                0,
                shape.high_width,
                shape.low_height,
            )?,
            lh: extract_cuda_subband(
                transformed,
                full_width,
                0,
                shape.low_height,
                shape.low_width,
                shape.high_height,
            )?,
            hh: extract_cuda_subband(
                transformed,
                full_width,
                shape.low_width,
                shape.low_height,
                shape.high_width,
                shape.high_height,
            )?,
            width: shape.width,
            height: shape.height,
            low_width: shape.low_width,
            low_height: shape.low_height,
            high_width: shape.high_width,
            high_height: shape.high_height,
        });
    }
    levels.reverse();

    Ok(J2kForwardDwt53Output {
        ll,
        ll_width,
        ll_height,
        levels,
    })
}

#[cfg(feature = "cuda-runtime")]
fn extract_cuda_subband(
    transformed: &[f32],
    full_width: usize,
    x0: u32,
    y0: u32,
    width: u32,
    height: u32,
) -> core::result::Result<Vec<f32>, &'static str> {
    let mut out = Vec::with_capacity((width as usize) * (height as usize));
    for y in 0..height as usize {
        let row_start = (y0 as usize)
            .checked_add(y)
            .and_then(|row| row.checked_mul(full_width))
            .and_then(|row| row.checked_add(x0 as usize))
            .ok_or("CUDA DWT subband offset overflow")?;
        out.extend_from_slice(&transformed[row_start..row_start + width as usize]);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::CudaEncodeStageAccelerator;
    use signinum_j2k_native::{encode_with_accelerator, DecodeSettings, EncodeOptions, Image};

    #[test]
    fn cuda_encode_stage_accelerator_preserves_cpu_codestream_validity() {
        let pixels: Vec<u8> = (0u8..192).collect();
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 1,
            ..EncodeOptions::default()
        };
        let mut accelerator = CudaEncodeStageAccelerator::default();

        let codestream =
            encode_with_accelerator(&pixels, 8, 8, 3, 8, false, &options, &mut accelerator)
                .expect("encode with CUDA stage accelerator");
        let decoded = Image::new(&codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");

        assert_eq!(decoded.width, 8);
        assert_eq!(decoded.height, 8);
        assert_eq!(decoded.num_components, 3);
        assert_eq!(decoded.bit_depth, 8);
        assert_eq!(accelerator.forward_rct_attempts(), 1);
        assert_eq!(accelerator.forward_dwt53_attempts(), 3);
        assert!(accelerator.tier1_code_block_attempts() > 0);
        assert_eq!(accelerator.packetization_attempts(), 1);
    }

    #[cfg(feature = "cuda-runtime")]
    #[test]
    fn cuda_forward_rct_dispatches_when_runtime_required() {
        if std::env::var_os("SIGNINUM_REQUIRE_CUDA_RUNTIME").is_none() {
            return;
        }

        let pixels: Vec<u8> = (0u16..7 * 5 * 3)
            .map(|i| u8::try_from((i * 17) & 0xFF).expect("masked value fits in u8"))
            .collect();
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 0,
            ..EncodeOptions::default()
        };
        let mut accelerator = CudaEncodeStageAccelerator::default();

        let codestream =
            encode_with_accelerator(&pixels, 7, 5, 3, 8, false, &options, &mut accelerator)
                .expect("encode with CUDA forward RCT");
        let decoded = Image::new(&codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");

        assert_eq!(decoded.data, pixels);
        assert_eq!(accelerator.forward_rct_attempts(), 1);
        assert_eq!(accelerator.forward_rct_dispatches(), 1);
    }

    #[cfg(feature = "cuda-runtime")]
    #[test]
    fn cuda_forward_dwt53_dispatches_when_runtime_required() {
        if std::env::var_os("SIGNINUM_REQUIRE_CUDA_RUNTIME").is_none() {
            return;
        }

        let pixels: Vec<u8> = (0u16..8 * 8)
            .map(|i| u8::try_from((i * 5) & 0xFF).expect("masked value fits in u8"))
            .collect();
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 1,
            ..EncodeOptions::default()
        };
        let mut accelerator = CudaEncodeStageAccelerator::default();

        let codestream =
            encode_with_accelerator(&pixels, 8, 8, 1, 8, false, &options, &mut accelerator)
                .expect("encode with CUDA forward DWT 5/3");
        let decoded = Image::new(&codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");

        assert_eq!(decoded.data, pixels);
        assert_eq!(accelerator.forward_dwt53_attempts(), 1);
        assert_eq!(accelerator.forward_dwt53_dispatches(), 2);
    }
}
