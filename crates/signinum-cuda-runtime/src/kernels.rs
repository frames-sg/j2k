use std::os::raw::c_uint;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum CudaKernel {
    CopyU8,
    J2kDeinterleaveToF32,
    J2kForwardRct,
    J2kForwardIct,
    J2kForwardDwt53Horizontal,
    J2kForwardDwt53Vertical,
    J2kForwardDwt97Horizontal,
    J2kForwardDwt97Vertical,
    J2kQuantizeSubband,
    J2kQuantizeSubbandStrided,
    Htj2kDecodeCodeblocks,
    J2kDequantizeHtj2kCodeblocks,
    J2kIdwtInterleave,
    J2kIdwtHorizontal,
    J2kIdwtHorizontal53,
    J2kIdwtHorizontal97,
    J2kIdwtVertical,
    J2kIdwtVertical53,
    J2kIdwtVertical97,
    Htj2kEncodeCodeblock,
    Htj2kEncodeCodeblocks,
    Htj2kPacketizeCleanup,
    J2kInverseDwtSingle,
    J2kInverseMct,
    J2kStoreGray16,
    J2kStoreGray8,
    J2kStoreRgb16,
    J2kStoreRgb8,
    // Coefficient-domain JPEG->HTJ2K transcode (signinum-transcode-cuda).
    TranscodeReversible53Idct,
    TranscodeReversible53VerticalLow,
    TranscodeReversible53VerticalHigh,
    TranscodeReversible53HorizontalLow,
    TranscodeReversible53HorizontalHigh,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CudaLaunchGeometry {
    pub grid: (c_uint, c_uint, c_uint),
    pub block: (c_uint, c_uint, c_uint),
}

impl CudaKernel {
    pub(crate) fn ptx(self) -> &'static [u8] {
        match self {
            Self::CopyU8 => COPY_U8_PTX,
            Self::J2kDeinterleaveToF32
            | Self::J2kForwardRct
            | Self::J2kForwardIct
            | Self::J2kForwardDwt53Horizontal
            | Self::J2kForwardDwt53Vertical
            | Self::J2kForwardDwt97Horizontal
            | Self::J2kForwardDwt97Vertical
            | Self::J2kQuantizeSubband
            | Self::J2kQuantizeSubbandStrided => J2K_ENCODE_PTX,
            Self::Htj2kDecodeCodeblocks
            | Self::J2kDequantizeHtj2kCodeblocks
            | Self::J2kIdwtInterleave
            | Self::J2kIdwtHorizontal
            | Self::J2kIdwtHorizontal53
            | Self::J2kIdwtHorizontal97
            | Self::J2kIdwtVertical
            | Self::J2kIdwtVertical53
            | Self::J2kIdwtVertical97
            | Self::J2kInverseDwtSingle
            | Self::J2kInverseMct
            | Self::J2kStoreGray16
            | Self::J2kStoreGray8
            | Self::J2kStoreRgb16
            | Self::J2kStoreRgb8 => HTJ2K_DECODE_PTX,
            Self::Htj2kEncodeCodeblock
            | Self::Htj2kEncodeCodeblocks
            | Self::Htj2kPacketizeCleanup => HTJ2K_ENCODE_PTX,
            Self::TranscodeReversible53Idct
            | Self::TranscodeReversible53VerticalLow
            | Self::TranscodeReversible53VerticalHigh
            | Self::TranscodeReversible53HorizontalLow
            | Self::TranscodeReversible53HorizontalHigh => TRANSCODE_PTX,
        }
    }

    pub(crate) fn entrypoint(self) -> &'static [u8] {
        match self {
            Self::CopyU8 => b"signinum_copy_u8\0",
            Self::J2kDeinterleaveToF32 => b"signinum_j2k_deinterleave_to_f32\0",
            Self::J2kForwardRct => b"signinum_j2k_forward_rct\0",
            Self::J2kForwardIct => b"signinum_j2k_forward_ict\0",
            Self::J2kForwardDwt53Horizontal => b"signinum_j2k_forward_dwt53_horizontal\0",
            Self::J2kForwardDwt53Vertical => b"signinum_j2k_forward_dwt53_vertical\0",
            Self::J2kForwardDwt97Horizontal => b"signinum_j2k_forward_dwt97_horizontal\0",
            Self::J2kForwardDwt97Vertical => b"signinum_j2k_forward_dwt97_vertical\0",
            Self::J2kQuantizeSubband => b"signinum_j2k_quantize_subband\0",
            Self::J2kQuantizeSubbandStrided => b"signinum_j2k_quantize_subband_strided\0",
            Self::Htj2kDecodeCodeblocks => b"signinum_htj2k_decode_codeblocks\0",
            Self::J2kDequantizeHtj2kCodeblocks => b"signinum_j2k_dequantize_htj2k_codeblocks\0",
            Self::J2kIdwtInterleave => b"signinum_j2k_idwt_interleave\0",
            Self::J2kIdwtHorizontal => b"signinum_j2k_idwt_horizontal\0",
            Self::J2kIdwtHorizontal53 => b"signinum_j2k_idwt_horizontal_53\0",
            Self::J2kIdwtHorizontal97 => b"signinum_j2k_idwt_horizontal_97\0",
            Self::J2kIdwtVertical => b"signinum_j2k_idwt_vertical\0",
            Self::J2kIdwtVertical53 => b"signinum_j2k_idwt_vertical_53\0",
            Self::J2kIdwtVertical97 => b"signinum_j2k_idwt_vertical_97\0",
            Self::Htj2kEncodeCodeblock => b"signinum_htj2k_encode_codeblock\0",
            Self::Htj2kEncodeCodeblocks => b"signinum_htj2k_encode_codeblocks\0",
            Self::Htj2kPacketizeCleanup => b"signinum_htj2k_packetize_cleanup\0",
            Self::J2kInverseDwtSingle => b"signinum_j2k_inverse_dwt_single\0",
            Self::J2kInverseMct => b"signinum_j2k_inverse_mct\0",
            Self::J2kStoreGray16 => b"signinum_j2k_store_gray16\0",
            Self::J2kStoreGray8 => b"signinum_j2k_store_gray8\0",
            Self::J2kStoreRgb16 => b"signinum_j2k_store_rgb16\0",
            Self::J2kStoreRgb8 => b"signinum_j2k_store_rgb8\0",
            Self::TranscodeReversible53Idct => b"transcode_reversible53_idct\0",
            Self::TranscodeReversible53VerticalLow => b"transcode_reversible53_vertical_low\0",
            Self::TranscodeReversible53VerticalHigh => b"transcode_reversible53_vertical_high\0",
            Self::TranscodeReversible53HorizontalLow => b"transcode_reversible53_horizontal_low\0",
            Self::TranscodeReversible53HorizontalHigh => b"transcode_reversible53_horizontal_high\0",
        }
    }
}

pub(crate) fn copy_u8_launch_geometry(len: usize) -> Option<CudaLaunchGeometry> {
    let blocks = c_uint::try_from(len.div_ceil(COPY_U8_THREADS)).ok()?;
    Some(CudaLaunchGeometry {
        grid: (blocks, 1, 1),
        block: (COPY_U8_THREADS_CUDA, 1, 1),
    })
}

const COPY_U8_THREADS: usize = 256;
const COPY_U8_THREADS_CUDA: c_uint = 256;
const J2K_ENCODE_THREADS_X: c_uint = 16;
const J2K_ENCODE_THREADS_Y: c_uint = 16;
const J2K_ENCODE_PTX: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/j2k_encode_kernels.ptx"));
const HTJ2K_DECODE_PTX: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/htj2k_decode_kernels.ptx"));
const HTJ2K_ENCODE_PTX: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/htj2k_encode_kernels.ptx"));
// Always resolves: build.rs writes a placeholder empty module when nvcc is
// absent (the dispatch checks `signinum_cuda_transcode_ptx_built` before load).
const TRANSCODE_PTX: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/transcode_kernels.ptx"));

pub(crate) fn j2k_forward_rct_launch_geometry(len: usize) -> Option<CudaLaunchGeometry> {
    let blocks = c_uint::try_from(len.div_ceil(COPY_U8_THREADS)).ok()?;
    Some(CudaLaunchGeometry {
        grid: (blocks, 1, 1),
        block: (COPY_U8_THREADS_CUDA, 1, 1),
    })
}

pub(crate) fn j2k_dwt53_launch_geometry(width: u32, height: u32) -> Option<CudaLaunchGeometry> {
    let grid_x = c_uint::try_from(width.div_ceil(J2K_ENCODE_THREADS_X)).ok()?;
    let grid_y = c_uint::try_from(height.div_ceil(J2K_ENCODE_THREADS_Y)).ok()?;
    Some(CudaLaunchGeometry {
        grid: (grid_x, grid_y, 1),
        block: (J2K_ENCODE_THREADS_X, J2K_ENCODE_THREADS_Y, 1),
    })
}

pub(crate) fn htj2k_codeblock_launch_geometry(job_count: usize) -> Option<CudaLaunchGeometry> {
    let jobs = c_uint::try_from(job_count).ok()?;
    Some(CudaLaunchGeometry {
        grid: (jobs, 1, 1),
        block: (1, 1, 1),
    })
}

pub(crate) fn htj2k_codeblock_sample_launch_geometry(
    job_count: usize,
) -> Option<CudaLaunchGeometry> {
    let jobs = c_uint::try_from(job_count).ok()?;
    Some(CudaLaunchGeometry {
        grid: (jobs, 1, 1),
        block: (COPY_U8_THREADS_CUDA, 1, 1),
    })
}

pub(crate) fn htj2k_encode_codeblock_launch_geometry(
    job_count: usize,
) -> Option<CudaLaunchGeometry> {
    htj2k_codeblock_sample_launch_geometry(job_count)
}

pub(crate) fn htj2k_packetize_launch_geometry(packet_count: usize) -> Option<CudaLaunchGeometry> {
    htj2k_codeblock_sample_launch_geometry(packet_count)
}

const COPY_U8_PTX: &[u8] = concat!(
    r"
.version 7.0
.target sm_52
.address_size 64

.visible .entry signinum_copy_u8(
    .param .u64 dst,
    .param .u64 src,
    .param .u64 len
)
{
    .reg .pred %p;
    .reg .b32 %r<5>;
    .reg .b64 %rd<7>;
    .reg .b16 %u;

    ld.param.u64 %rd1, [dst];
    ld.param.u64 %rd2, [src];
    ld.param.u64 %rd3, [len];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r4, %r2, %r3, %r1;
    cvt.u64.u32 %rd4, %r4;
    setp.ge.u64 %p, %rd4, %rd3;
    @%p bra DONE;
    add.u64 %rd5, %rd2, %rd4;
    ld.global.u8 %u, [%rd5];
    add.u64 %rd6, %rd1, %rd4;
    st.global.u8 [%rd6], %u;
DONE:
    ret;
}
",
    "\0"
)
.as_bytes();

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copy_u8_kernel_metadata_matches_embedded_ptx() {
        let ptx = CudaKernel::CopyU8.ptx();
        assert_eq!(ptx.last(), Some(&0));
        let source = std::str::from_utf8(&ptx[..ptx.len() - 1]).expect("ptx utf8");
        assert!(source.contains(".visible .entry signinum_copy_u8("));
        assert_eq!(CudaKernel::CopyU8.entrypoint(), b"signinum_copy_u8\0");
    }

    #[test]
    fn htj2k_sample_geometry_uses_threads_with_one_block_per_codeblock() {
        let geometry = htj2k_codeblock_sample_launch_geometry(3).expect("geometry");
        assert_eq!(geometry.grid, (3, 1, 1));
        assert_eq!(geometry.block, (COPY_U8_THREADS_CUDA, 1, 1));
    }

    #[test]
    fn htj2k_encode_geometry_uses_cooperative_threads_per_codeblock() {
        let geometry = htj2k_encode_codeblock_launch_geometry(4).expect("geometry");
        assert_eq!(geometry.grid, (4, 1, 1));
        assert_eq!(geometry.block, (COPY_U8_THREADS_CUDA, 1, 1));
    }

    #[test]
    fn htj2k_packetize_geometry_uses_cooperative_threads_per_packet() {
        let geometry = htj2k_packetize_launch_geometry(5).expect("geometry");
        assert_eq!(geometry.grid, (5, 1, 1));
        assert_eq!(geometry.block, (COPY_U8_THREADS_CUDA, 1, 1));
    }

    #[test]
    fn j2k_encode_kernel_metadata_matches_generated_ptx() {
        assert_eq!(J2K_ENCODE_PTX.last(), Some(&0));
        assert_eq!(
            CudaKernel::J2kDeinterleaveToF32.entrypoint(),
            b"signinum_j2k_deinterleave_to_f32\0"
        );
        assert_eq!(
            CudaKernel::J2kForwardRct.entrypoint(),
            b"signinum_j2k_forward_rct\0"
        );
        assert_eq!(
            CudaKernel::J2kForwardIct.entrypoint(),
            b"signinum_j2k_forward_ict\0"
        );
        assert_eq!(
            CudaKernel::J2kForwardDwt53Horizontal.entrypoint(),
            b"signinum_j2k_forward_dwt53_horizontal\0"
        );
        assert_eq!(
            CudaKernel::J2kForwardDwt53Vertical.entrypoint(),
            b"signinum_j2k_forward_dwt53_vertical\0"
        );
        assert_eq!(
            CudaKernel::J2kForwardDwt97Horizontal.entrypoint(),
            b"signinum_j2k_forward_dwt97_horizontal\0"
        );
        assert_eq!(
            CudaKernel::J2kForwardDwt97Vertical.entrypoint(),
            b"signinum_j2k_forward_dwt97_vertical\0"
        );
        assert_eq!(
            CudaKernel::J2kQuantizeSubband.entrypoint(),
            b"signinum_j2k_quantize_subband\0"
        );
        assert_eq!(
            CudaKernel::J2kQuantizeSubbandStrided.entrypoint(),
            b"signinum_j2k_quantize_subband_strided\0"
        );
        let source =
            std::str::from_utf8(&J2K_ENCODE_PTX[..J2K_ENCODE_PTX.len() - 1]).expect("ptx utf8");
        assert!(source.contains(".visible .entry signinum_j2k_deinterleave_to_f32("));
        assert!(source.contains(".visible .entry signinum_j2k_quantize_subband_strided("));
    }

    #[test]
    fn htj2k_decode_kernel_metadata_matches_generated_ptx() {
        assert_eq!(HTJ2K_DECODE_PTX.last(), Some(&0));
        assert_eq!(
            CudaKernel::Htj2kDecodeCodeblocks.entrypoint(),
            b"signinum_htj2k_decode_codeblocks\0"
        );
        assert_eq!(
            CudaKernel::J2kDequantizeHtj2kCodeblocks.entrypoint(),
            b"signinum_j2k_dequantize_htj2k_codeblocks\0"
        );
        assert_eq!(
            CudaKernel::J2kIdwtInterleave.entrypoint(),
            b"signinum_j2k_idwt_interleave\0"
        );
        assert_eq!(
            CudaKernel::J2kIdwtHorizontal.entrypoint(),
            b"signinum_j2k_idwt_horizontal\0"
        );
        assert_eq!(
            CudaKernel::J2kIdwtVertical.entrypoint(),
            b"signinum_j2k_idwt_vertical\0"
        );
        assert_eq!(
            CudaKernel::J2kInverseDwtSingle.entrypoint(),
            b"signinum_j2k_inverse_dwt_single\0"
        );
        assert_eq!(
            CudaKernel::J2kInverseMct.entrypoint(),
            b"signinum_j2k_inverse_mct\0"
        );
        assert_eq!(
            CudaKernel::J2kStoreGray8.entrypoint(),
            b"signinum_j2k_store_gray8\0"
        );
        assert_eq!(
            CudaKernel::J2kStoreGray16.entrypoint(),
            b"signinum_j2k_store_gray16\0"
        );
        assert_eq!(
            CudaKernel::J2kStoreRgb8.entrypoint(),
            b"signinum_j2k_store_rgb8\0"
        );
        assert_eq!(
            CudaKernel::J2kStoreRgb16.entrypoint(),
            b"signinum_j2k_store_rgb16\0"
        );
        let source =
            std::str::from_utf8(&HTJ2K_DECODE_PTX[..HTJ2K_DECODE_PTX.len() - 1]).expect("ptx utf8");
        assert!(source.contains(".visible .entry signinum_j2k_dequantize_htj2k_codeblocks("));
        assert!(source.contains(".visible .entry signinum_j2k_idwt_interleave("));
        assert!(source.contains(".visible .entry signinum_j2k_idwt_horizontal("));
        assert!(source.contains(".visible .entry signinum_j2k_idwt_vertical("));
        assert!(source.contains(".visible .entry signinum_j2k_idwt_horizontal_53("));
        assert!(source.contains(".visible .entry signinum_j2k_idwt_vertical_53("));
        assert!(source.contains(".visible .entry signinum_j2k_idwt_horizontal_97("));
        assert!(source.contains(".visible .entry signinum_j2k_idwt_vertical_97("));
    }

    #[test]
    fn htj2k_encode_kernel_metadata_matches_generated_ptx() {
        assert_eq!(HTJ2K_ENCODE_PTX.last(), Some(&0));
        assert_eq!(
            CudaKernel::Htj2kEncodeCodeblock.entrypoint(),
            b"signinum_htj2k_encode_codeblock\0"
        );
        assert_eq!(
            CudaKernel::Htj2kEncodeCodeblocks.entrypoint(),
            b"signinum_htj2k_encode_codeblocks\0"
        );
        assert_eq!(
            CudaKernel::Htj2kPacketizeCleanup.entrypoint(),
            b"signinum_htj2k_packetize_cleanup\0"
        );
        let source =
            std::str::from_utf8(&HTJ2K_ENCODE_PTX[..HTJ2K_ENCODE_PTX.len() - 1]).expect("ptx utf8");
        assert!(source.contains(".visible .entry signinum_htj2k_encode_codeblocks("));
        assert!(source.contains(".visible .entry signinum_htj2k_packetize_cleanup("));
        let cuda_source = include_str!("htj2k_encode_kernels.cu");
        assert!(cuda_source.contains("j2k_ht_reduce_max_magnitude_cooperative"));
        assert!(cuda_source.contains("j2k_packet_copy_body_cooperative"));
    }

    #[test]
    fn copy_u8_launch_geometry_rounds_up_to_256_thread_blocks() {
        assert_eq!(copy_u8_launch_geometry(1).unwrap().grid, (1, 1, 1));
        assert_eq!(copy_u8_launch_geometry(256).unwrap().grid, (1, 1, 1));
        assert_eq!(copy_u8_launch_geometry(257).unwrap().grid, (2, 1, 1));
    }

    #[test]
    fn j2k_dwt53_launch_geometry_uses_16_by_16_thread_blocks() {
        let geometry = j2k_dwt53_launch_geometry(17, 33).unwrap();
        assert_eq!(geometry.grid, (2, 3, 1));
        assert_eq!(geometry.block, (16, 16, 1));
    }
}
