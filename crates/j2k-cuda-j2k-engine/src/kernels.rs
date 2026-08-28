// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    build_flags::{
        ensure_htj2k_decode_ptx_built, ensure_htj2k_encode_ptx_built,
        ensure_j2k_decode_store_ptx_built, ensure_j2k_dequantize_ptx_built,
        ensure_j2k_encode_ptx_built, ensure_j2k_idwt_ptx_built,
    },
    error::CudaError,
};
use j2k_cuda_runtime::CudaKernelSpec;

pub(crate) use j2k_cuda_runtime::CudaLaunchGeometry;

const HTJ2K_DECODE_PACKED_BLOCK_MIN_JOBS: usize = 2_048;
const HTJ2K_DECODE_CODEBLOCK_THREADS: u32 = 32;
const HTJ2K_ENCODE_CODEBLOCK_THREADS: u32 = 128;
const SAMPLE_THREADS: u32 = 256;
const J2K_THREADS_X: u32 = 16;
const J2K_THREADS_Y: u32 = 16;
#[cfg(test)]
pub(crate) const CUDA_MAX_GRID_DIM_X: u32 = 2_147_483_647;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CudaKernel {
    Htj2kDecodeCodeblocks,
    Htj2kDecodeCodeblocksMulti,
    Htj2kDecodeCodeblocksMultiCleanupOnly,
    Htj2kDecodeCodeblocksMultiCleanupDequantize,
    J2kDequantizeHtj2kCodeblocks,
    J2kDequantizeHtj2kCodeblocksMulti,
    J2kDequantizeHtj2kCleanupJobsMulti,
    Htj2kEncodeCodeblocks,
    Htj2kEncodeCodeblocksMultiInput,
    Htj2kEncodeCodeblocksMultiInputCleanup,
    Htj2kEncodeCodeblocksMultiInputCleanup64,
    Htj2kCompactCodeblocks,
    Htj2kPacketizeCleanup,
    J2kDeinterleaveToF32,
    J2kDeinterleaveStridedToF32,
    J2kForwardRct,
    J2kForwardIct,
    J2kForwardDwt53Horizontal,
    J2kForwardDwt53Vertical,
    J2kForwardDwt97Horizontal,
    J2kForwardDwt97Vertical,
    J2kQuantizeSubband,
    J2kQuantizeSubbandStrided,
    J2kIdwtInterleave,
    J2kIdwtInterleaveHorizontalMulti,
    J2kIdwtInterleaveHorizontal53Multi,
    J2kIdwtInterleaveHorizontal97Multi,
    J2kIdwtHorizontal53,
    J2kIdwtHorizontal97,
    J2kIdwtVerticalMulti,
    J2kIdwtVertical53Multi,
    J2kIdwtVertical97Multi,
    J2kIdwtVertical97MultiCols4,
    J2kIdwtVertical53,
    J2kIdwtVertical97,
    J2kInverseMct,
    J2kStoreGray16,
    J2kStoreGray16Batch,
    J2kStoreGrayI16Batch,
    J2kStoreGray8,
    J2kStoreGray8Batch,
    J2kStoreRgb16,
    J2kStoreRgb16Mct,
    J2kStoreRgb8,
    J2kStoreRgb8MctBatch,
    J2kStoreRgb8NativeBatch,
    J2kStoreRgb16NativeBatch,
    J2kStoreRgbI16NativeBatch,
    J2kStoreRgba8NativeBatch,
    J2kStoreRgba16NativeBatch,
    J2kStoreRgbaI16NativeBatch,
}

impl CudaKernel {
    pub(crate) fn spec(self) -> Result<CudaKernelSpec, CudaError> {
        let (module_id, ptx) = if self.is_htj2k_decode() {
            ensure_htj2k_decode_ptx_built()?;
            ("htj2k-decode", htj2k_decode_ptx())
        } else if self.is_htj2k_encode() {
            ensure_htj2k_encode_ptx_built()?;
            ("htj2k-encode", htj2k_encode_ptx())
        } else if self.is_j2k_dequantize() {
            ensure_j2k_dequantize_ptx_built()?;
            ("j2k-dequantize", j2k_dequantize_ptx())
        } else if self.is_j2k_encode() {
            ensure_j2k_encode_ptx_built()?;
            ("j2k-encode", j2k_encode_ptx())
        } else if self.is_j2k_idwt() {
            ensure_j2k_idwt_ptx_built()?;
            ("j2k-idwt", j2k_idwt_ptx())
        } else {
            ensure_j2k_decode_store_ptx_built()?;
            ("j2k-decode-store", j2k_decode_store_ptx())
        };
        CudaKernelSpec::new(module_id, ptx, self.entrypoint())
    }

    const fn is_htj2k_decode(self) -> bool {
        matches!(
            self,
            Self::Htj2kDecodeCodeblocks
                | Self::Htj2kDecodeCodeblocksMulti
                | Self::Htj2kDecodeCodeblocksMultiCleanupOnly
                | Self::Htj2kDecodeCodeblocksMultiCleanupDequantize
        )
    }

    const fn is_j2k_dequantize(self) -> bool {
        matches!(
            self,
            Self::J2kDequantizeHtj2kCodeblocks
                | Self::J2kDequantizeHtj2kCodeblocksMulti
                | Self::J2kDequantizeHtj2kCleanupJobsMulti
        )
    }

    pub(crate) const fn is_htj2k_encode(self) -> bool {
        matches!(
            self,
            Self::Htj2kEncodeCodeblocks
                | Self::Htj2kEncodeCodeblocksMultiInput
                | Self::Htj2kEncodeCodeblocksMultiInputCleanup
                | Self::Htj2kEncodeCodeblocksMultiInputCleanup64
        )
    }

    pub(crate) const fn is_j2k_encode(self) -> bool {
        matches!(
            self,
            Self::J2kDeinterleaveToF32
                | Self::J2kDeinterleaveStridedToF32
                | Self::J2kForwardRct
                | Self::J2kForwardIct
                | Self::J2kForwardDwt53Horizontal
                | Self::J2kForwardDwt53Vertical
                | Self::J2kForwardDwt97Horizontal
                | Self::J2kForwardDwt97Vertical
                | Self::J2kQuantizeSubband
                | Self::J2kQuantizeSubbandStrided
                | Self::Htj2kCompactCodeblocks
                | Self::Htj2kPacketizeCleanup
        )
    }

    pub(crate) const fn is_j2k_idwt(self) -> bool {
        matches!(
            self,
            Self::J2kIdwtInterleave
                | Self::J2kIdwtInterleaveHorizontalMulti
                | Self::J2kIdwtInterleaveHorizontal53Multi
                | Self::J2kIdwtInterleaveHorizontal97Multi
                | Self::J2kIdwtHorizontal53
                | Self::J2kIdwtHorizontal97
                | Self::J2kIdwtVerticalMulti
                | Self::J2kIdwtVertical53Multi
                | Self::J2kIdwtVertical97Multi
                | Self::J2kIdwtVertical97MultiCols4
                | Self::J2kIdwtVertical53
                | Self::J2kIdwtVertical97
        )
    }

    const fn entrypoint(self) -> &'static [u8] {
        match self {
            Self::Htj2kDecodeCodeblocks => b"j2k_htj2k_decode_codeblocks\0",
            Self::Htj2kDecodeCodeblocksMulti => b"j2k_htj2k_decode_codeblocks_multi\0",
            Self::Htj2kDecodeCodeblocksMultiCleanupOnly => {
                b"j2k_htj2k_decode_codeblocks_multi_cleanup_only\0"
            }
            Self::Htj2kDecodeCodeblocksMultiCleanupDequantize => {
                b"j2k_htj2k_decode_codeblocks_multi_cleanup_dequantize\0"
            }
            Self::J2kDequantizeHtj2kCodeblocks => b"j2k_dequantize_htj2k_codeblocks\0",
            Self::J2kDequantizeHtj2kCodeblocksMulti => b"j2k_dequantize_htj2k_codeblocks_multi\0",
            Self::J2kDequantizeHtj2kCleanupJobsMulti => {
                b"j2k_dequantize_htj2k_cleanup_jobs_multi\0"
            }
            Self::Htj2kEncodeCodeblocks => b"j2k_htj2k_encode_codeblocks\0",
            Self::Htj2kEncodeCodeblocksMultiInput => b"j2k_htj2k_encode_codeblocks_multi_input\0",
            Self::Htj2kEncodeCodeblocksMultiInputCleanup => {
                b"j2k_htj2k_encode_codeblocks_multi_input_cleanup\0"
            }
            Self::Htj2kEncodeCodeblocksMultiInputCleanup64 => {
                b"j2k_htj2k_encode_codeblocks_multi_input_cleanup_64\0"
            }
            Self::Htj2kCompactCodeblocks => b"j2k_htj2k_compact_codeblocks\0",
            Self::Htj2kPacketizeCleanup => b"j2k_htj2k_packetize_cleanup\0",
            Self::J2kDeinterleaveToF32 => b"j2k_deinterleave_to_f32\0",
            Self::J2kDeinterleaveStridedToF32 => b"j2k_deinterleave_strided_to_f32\0",
            Self::J2kForwardRct => b"j2k_forward_rct\0",
            Self::J2kForwardIct => b"j2k_forward_ict\0",
            Self::J2kForwardDwt53Horizontal => b"j2k_forward_dwt53_horizontal\0",
            Self::J2kForwardDwt53Vertical => b"j2k_forward_dwt53_vertical\0",
            Self::J2kForwardDwt97Horizontal => b"j2k_forward_dwt97_horizontal\0",
            Self::J2kForwardDwt97Vertical => b"j2k_forward_dwt97_vertical\0",
            Self::J2kQuantizeSubband => b"j2k_quantize_subband\0",
            Self::J2kQuantizeSubbandStrided => b"j2k_quantize_subband_strided\0",
            Self::J2kIdwtInterleave => b"j2k_idwt_interleave\0",
            Self::J2kIdwtInterleaveHorizontalMulti => b"j2k_idwt_interleave_horizontal_multi\0",
            Self::J2kIdwtInterleaveHorizontal53Multi => {
                b"j2k_idwt_interleave_horizontal_53_multi\0"
            }
            Self::J2kIdwtInterleaveHorizontal97Multi => {
                b"j2k_idwt_interleave_horizontal_97_multi\0"
            }
            Self::J2kIdwtHorizontal53 => b"j2k_idwt_horizontal_53\0",
            Self::J2kIdwtHorizontal97 => b"j2k_idwt_horizontal_97\0",
            Self::J2kIdwtVerticalMulti => b"j2k_idwt_vertical_multi\0",
            Self::J2kIdwtVertical53Multi => b"j2k_idwt_vertical_53_multi\0",
            Self::J2kIdwtVertical97Multi => b"j2k_idwt_vertical_97_multi\0",
            Self::J2kIdwtVertical97MultiCols4 => b"j2k_idwt_vertical_97_multi_cols4\0",
            Self::J2kIdwtVertical53 => b"j2k_idwt_vertical_53\0",
            Self::J2kIdwtVertical97 => b"j2k_idwt_vertical_97\0",
            Self::J2kInverseMct => b"j2k_inverse_mct\0",
            Self::J2kStoreGray16 => b"j2k_store_gray16\0",
            Self::J2kStoreGray16Batch => b"j2k_store_gray16_batch\0",
            Self::J2kStoreGrayI16Batch => b"j2k_store_grayi16_batch\0",
            Self::J2kStoreGray8 => b"j2k_store_gray8\0",
            Self::J2kStoreGray8Batch => b"j2k_store_gray8_batch\0",
            Self::J2kStoreRgb16 => b"j2k_store_rgb16\0",
            Self::J2kStoreRgb16Mct => b"j2k_store_rgb16_mct\0",
            Self::J2kStoreRgb8 => b"j2k_store_rgb8\0",
            Self::J2kStoreRgb8MctBatch => b"j2k_store_rgb8_mct_batch\0",
            Self::J2kStoreRgb8NativeBatch => b"j2k_store_rgb8_native_batch\0",
            Self::J2kStoreRgb16NativeBatch => b"j2k_store_rgb16_native_batch\0",
            Self::J2kStoreRgbI16NativeBatch => b"j2k_store_rgbi16_native_batch\0",
            Self::J2kStoreRgba8NativeBatch => b"j2k_store_rgba8_native_batch\0",
            Self::J2kStoreRgba16NativeBatch => b"j2k_store_rgba16_native_batch\0",
            Self::J2kStoreRgbaI16NativeBatch => b"j2k_store_rgbai16_native_batch\0",
        }
    }
}

fn x_blocks_launch_geometry(
    work_items: usize,
    grid_y: usize,
    threads: u32,
) -> Option<CudaLaunchGeometry> {
    if threads == 0 {
        return None;
    }
    let work_items = u32::try_from(work_items).ok()?;
    let grid_y = u32::try_from(grid_y).ok()?;
    CudaLaunchGeometry::new((work_items.div_ceil(threads), grid_y, 1), (threads, 1, 1))
}

pub(crate) fn j2k_forward_rct_launch_geometry(len: usize) -> Option<CudaLaunchGeometry> {
    x_blocks_launch_geometry(len, 1, SAMPLE_THREADS)
}

pub(crate) fn j2k_dwt53_launch_geometry(width: u32, height: u32) -> Option<CudaLaunchGeometry> {
    CudaLaunchGeometry::new(
        (
            width.div_ceil(J2K_THREADS_X),
            height.div_ceil(J2K_THREADS_Y),
            1,
        ),
        (J2K_THREADS_X, J2K_THREADS_Y, 1),
    )
}

pub(crate) fn j2k_idwt_multi_1d_launch_geometry(
    max_len: usize,
    job_count: usize,
) -> Option<CudaLaunchGeometry> {
    x_blocks_launch_geometry(max_len, job_count, SAMPLE_THREADS)
}

pub(crate) fn j2k_idwt_multi_coop_axis_launch_geometry(
    work_items: usize,
    lane_count: usize,
    job_count: usize,
) -> Option<CudaLaunchGeometry> {
    let blocks = u32::try_from(work_items).ok()?;
    let jobs = u32::try_from(job_count).ok()?;
    let threads = if lane_count > SAMPLE_THREADS as usize {
        512
    } else {
        256
    };
    CudaLaunchGeometry::new((blocks, jobs, 1), (threads, 1, 1))
}

pub(crate) fn j2k_idwt_multi_coop_columns_launch_geometry(
    columns: usize,
    rows: usize,
    job_count: usize,
    columns_per_block: usize,
) -> Option<CudaLaunchGeometry> {
    if rows == 0 || columns_per_block == 0 || rows.saturating_mul(columns_per_block) > 1024 {
        return None;
    }
    CudaLaunchGeometry::new(
        (
            u32::try_from(columns.div_ceil(columns_per_block)).ok()?,
            u32::try_from(job_count).ok()?,
            1,
        ),
        (
            u32::try_from(columns_per_block).ok()?,
            u32::try_from(rows).ok()?,
            1,
        ),
    )
}

pub(crate) fn j2k_store_batch_launch_geometry(
    max_pixels: usize,
    job_count: usize,
) -> Option<CudaLaunchGeometry> {
    x_blocks_launch_geometry(max_pixels, job_count, SAMPLE_THREADS)
}

pub(crate) fn htj2k_codeblock_launch_geometry(job_count: usize) -> Option<CudaLaunchGeometry> {
    if job_count >= HTJ2K_DECODE_PACKED_BLOCK_MIN_JOBS {
        let threads = usize::try_from(HTJ2K_DECODE_CODEBLOCK_THREADS).ok()?;
        let jobs = u32::try_from(job_count.div_ceil(threads)).ok()?;
        CudaLaunchGeometry::new((jobs, 1, 1), (HTJ2K_DECODE_CODEBLOCK_THREADS, 1, 1))
    } else {
        let jobs = u32::try_from(job_count).ok()?;
        CudaLaunchGeometry::new((jobs, 1, 1), (1, 1, 1))
    }
}

pub(crate) fn htj2k_codeblock_sample_launch_geometry(
    job_count: usize,
) -> Option<CudaLaunchGeometry> {
    let jobs = u32::try_from(job_count).ok()?;
    CudaLaunchGeometry::new((jobs, 1, 1), (SAMPLE_THREADS, 1, 1))
}

pub(crate) fn htj2k_encode_codeblock_launch_geometry(
    job_count: usize,
) -> Option<CudaLaunchGeometry> {
    let jobs = u32::try_from(job_count).ok()?;
    CudaLaunchGeometry::new((jobs, 1, 1), (HTJ2K_ENCODE_CODEBLOCK_THREADS, 1, 1))
}

pub(crate) fn htj2k_packetize_launch_geometry(packet_count: usize) -> Option<CudaLaunchGeometry> {
    htj2k_codeblock_sample_launch_geometry(packet_count)
}

#[cfg(feature = "cuda-oxide-htj2k-decode")]
fn htj2k_decode_ptx() -> &'static [u8] {
    include_bytes!(concat!(env!("OUT_DIR"), "/cuda_oxide_htj2k_decode.ptx"))
}

#[cfg(feature = "cuda-oxide-htj2k-encode")]
fn htj2k_encode_ptx() -> &'static [u8] {
    include_bytes!(concat!(env!("OUT_DIR"), "/cuda_oxide_htj2k_encode.ptx"))
}

#[cfg(not(feature = "cuda-oxide-htj2k-encode"))]
fn htj2k_encode_ptx() -> &'static [u8] {
    b".version 7.0\n.target sm_52\n.address_size 64\n\0"
}

#[cfg(not(feature = "cuda-oxide-htj2k-decode"))]
fn htj2k_decode_ptx() -> &'static [u8] {
    b".version 7.0\n.target sm_52\n.address_size 64\n\0"
}

#[cfg(feature = "cuda-oxide-j2k-dequantize")]
fn j2k_dequantize_ptx() -> &'static [u8] {
    include_bytes!(concat!(env!("OUT_DIR"), "/cuda_oxide_j2k_dequantize.ptx"))
}

#[cfg(not(feature = "cuda-oxide-j2k-dequantize"))]
fn j2k_dequantize_ptx() -> &'static [u8] {
    b".version 7.0\n.target sm_52\n.address_size 64\n\0"
}

#[cfg(feature = "cuda-oxide-j2k-encode")]
fn j2k_encode_ptx() -> &'static [u8] {
    include_bytes!(concat!(env!("OUT_DIR"), "/cuda_oxide_j2k_encode.ptx"))
}

#[cfg(not(feature = "cuda-oxide-j2k-encode"))]
fn j2k_encode_ptx() -> &'static [u8] {
    b".version 7.0\n.target sm_52\n.address_size 64\n\0"
}

#[cfg(feature = "cuda-oxide-j2k-idwt")]
fn j2k_idwt_ptx() -> &'static [u8] {
    include_bytes!(concat!(env!("OUT_DIR"), "/cuda_oxide_j2k_idwt.ptx"))
}

#[cfg(not(feature = "cuda-oxide-j2k-idwt"))]
fn j2k_idwt_ptx() -> &'static [u8] {
    b".version 7.0\n.target sm_52\n.address_size 64\n\0"
}

#[cfg(feature = "cuda-oxide-j2k-decode-store")]
fn j2k_decode_store_ptx() -> &'static [u8] {
    include_bytes!(concat!(env!("OUT_DIR"), "/cuda_oxide_j2k_decode_store.ptx"))
}

#[cfg(not(feature = "cuda-oxide-j2k-decode-store"))]
fn j2k_decode_store_ptx() -> &'static [u8] {
    b".version 7.0\n.target sm_52\n.address_size 64\n\0"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejected_parallel_ht_encode_candidate_is_absent() {
        let sources = [
            include_str!("kernels.rs"),
            include_str!("htj2k_encode/launch.rs"),
            include_str!("cuda_oxide_htj2k_encode/simt/src/main.rs"),
        ];
        for rejected in [
            ["J2", "K_CUDA_HT_ENCODE_", "COOPERATIVE"].concat(),
            ["j2k_htj2k_encode_codeblocks_", "cooperative"].concat(),
            ["j2k_htj2k_encode_codeblocks_multi_input_", "cooperative"].concat(),
        ] {
            assert!(
                sources.iter().all(|source| !source.contains(&rejected)),
                "rejected CUDA HT encode candidate marker remains: {rejected}"
            );
        }
    }

    #[test]
    fn htj2k_launch_geometry_matches_codeblock_work() {
        let samples = htj2k_codeblock_sample_launch_geometry(3).expect("sample geometry");
        assert_eq!(samples.grid(), (3, 1, 1));
        assert_eq!(samples.block(), (SAMPLE_THREADS, 1, 1));

        let encode = htj2k_encode_codeblock_launch_geometry(3).expect("encode geometry");
        assert_eq!(encode.grid(), (3, 1, 1));
        assert_eq!(encode.block(), (HTJ2K_ENCODE_CODEBLOCK_THREADS, 1, 1));

        let small = htj2k_codeblock_launch_geometry(1_200).expect("small geometry");
        assert_eq!(small.grid(), (1_200, 1, 1));
        assert_eq!(small.block(), (1, 1, 1));

        let large = htj2k_codeblock_launch_geometry(2_048).expect("large geometry");
        assert_eq!(large.grid(), (64, 1, 1));
        assert_eq!(large.block(), (32, 1, 1));
    }

    #[test]
    fn htj2k_sigprop_forward_reader_discards_a_set_stuffed_overlap_bit() {
        let data = [0xFF_u8, 0x80, 0x00, 0x00, 0x00];
        let mut tmp = 0_u64;
        let mut bits = 0_u32;
        let mut unstuff = false;
        for byte in data {
            let valid_bits = 8 - u32::from(unstuff);
            let next_unstuff = byte == 0xFF;
            let byte = if unstuff { byte & 0x7F } else { byte };
            tmp |= u64::from(byte) << bits;
            bits += valid_bits;
            unstuff = next_unstuff;
        }
        assert_eq!(u32::try_from(tmp).expect("low reservoir word"), 0x0000_00FF);

        let device = include_str!("cuda_oxide_htj2k_decode/simt/src/main.rs");
        let fill = device
            .split("fn forward_reader_fill")
            .nth(1)
            .expect("forward-reader fill")
            .split("fn forward_reader_fetch")
            .next()
            .expect("forward-reader fill body");
        assert!(
            fill.contains("let byte = if reader.unstuff { byte & 0x7f } else { byte };")
                || fill.contains("let byte = if reader.unstuff { byte & 0x7F } else { byte };")
        );
        assert!(fill.contains("reader.unstuff = next_unstuff;"));
    }

    #[test]
    fn htj2k_sigprop_quad_preserves_the_next_above_stripe_context() {
        let mut previous_row = [0x0000_u16, 0xA5A5];
        let combined_sig = 0x0000_0088_u32 | (0xF00F_0011_u32 & 0xFFFF);
        previous_row[0] = u16::try_from(combined_sig).expect("low significance half");
        assert_eq!(previous_row, [0x0099, 0xA5A5]);

        let device = include_str!("cuda_oxide_htj2k_decode/simt/src/main.rs");
        let sigprop = device
            .split("fn apply_significance_propagation")
            .nth(1)
            .expect("SigProp phase")
            .split("fn apply_magnitude_refinement")
            .next()
            .expect("SigProp phase body");
        assert!(sigprop.contains("let combined_sig = new_sig | (cs & 0xffff);"));
        assert!(!sigprop.contains("prev_row_sig[idx as usize + 1] ="));
    }

    #[test]
    fn htj2k_magref_reverse_reader_discards_a_set_stuffed_overlap_bit() {
        let data = [0x00_u8, 0x00, 0x00, 0x00, 0xFF];
        let mut tmp = 0_u64;
        let mut bits = 0_u32;
        let mut unstuff = true;
        for raw in data.into_iter().rev() {
            let stuffed = unstuff && (raw & 0x7F) == 0x7F;
            let valid_bits = 8 - u32::from(stuffed);
            let next_unstuff = raw > 0x8F;
            let byte = if stuffed { raw & 0x7F } else { raw };
            tmp |= u64::from(byte) << bits;
            bits += valid_bits;
            unstuff = next_unstuff;
        }
        assert_eq!(u32::try_from(tmp).expect("low reservoir word"), 0x0000_007F);

        let device = include_str!("cuda_oxide_htj2k_decode/simt/src/main.rs");
        let fill = device
            .split("fn reverse_reader_fill")
            .nth(1)
            .expect("reverse-reader fill")
            .split("fn reverse_reader_fetch")
            .next()
            .expect("reverse-reader fill body");
        assert!(fill.contains("let stuffed = reader.unstuff && (byte & 0x7f) == 0x7f;"));
        assert!(fill.contains("let byte = if stuffed { byte & 0x7f } else { byte };"));
        assert!(fill.contains("reader.unstuff = next_unstuff;"));
    }

    #[test]
    fn htj2k_decode_and_dequantize_entrypoints_are_stable() {
        let cases = [
            (
                CudaKernel::Htj2kDecodeCodeblocks,
                "j2k_htj2k_decode_codeblocks",
            ),
            (
                CudaKernel::Htj2kDecodeCodeblocksMulti,
                "j2k_htj2k_decode_codeblocks_multi",
            ),
            (
                CudaKernel::Htj2kDecodeCodeblocksMultiCleanupOnly,
                "j2k_htj2k_decode_codeblocks_multi_cleanup_only",
            ),
            (
                CudaKernel::Htj2kDecodeCodeblocksMultiCleanupDequantize,
                "j2k_htj2k_decode_codeblocks_multi_cleanup_dequantize",
            ),
            (
                CudaKernel::J2kDequantizeHtj2kCodeblocks,
                "j2k_dequantize_htj2k_codeblocks",
            ),
            (
                CudaKernel::J2kDequantizeHtj2kCodeblocksMulti,
                "j2k_dequantize_htj2k_codeblocks_multi",
            ),
            (
                CudaKernel::J2kDequantizeHtj2kCleanupJobsMulti,
                "j2k_dequantize_htj2k_cleanup_jobs_multi",
            ),
        ];
        for (kernel, expected) in cases {
            let entrypoint = kernel.entrypoint();
            assert_eq!(&entrypoint[..entrypoint.len() - 1], expected.as_bytes());
            assert_eq!(entrypoint.last(), Some(&0));
        }
    }

    #[test]
    fn j2k_encode_and_forward_transform_entrypoints_are_stable() {
        let cases = [
            (
                CudaKernel::Htj2kEncodeCodeblocks,
                "j2k_htj2k_encode_codeblocks",
            ),
            (
                CudaKernel::Htj2kEncodeCodeblocksMultiInput,
                "j2k_htj2k_encode_codeblocks_multi_input",
            ),
            (
                CudaKernel::Htj2kEncodeCodeblocksMultiInputCleanup,
                "j2k_htj2k_encode_codeblocks_multi_input_cleanup",
            ),
            (
                CudaKernel::Htj2kEncodeCodeblocksMultiInputCleanup64,
                "j2k_htj2k_encode_codeblocks_multi_input_cleanup_64",
            ),
            (
                CudaKernel::Htj2kCompactCodeblocks,
                "j2k_htj2k_compact_codeblocks",
            ),
            (
                CudaKernel::Htj2kPacketizeCleanup,
                "j2k_htj2k_packetize_cleanup",
            ),
            (CudaKernel::J2kDeinterleaveToF32, "j2k_deinterleave_to_f32"),
            (
                CudaKernel::J2kDeinterleaveStridedToF32,
                "j2k_deinterleave_strided_to_f32",
            ),
            (CudaKernel::J2kForwardRct, "j2k_forward_rct"),
            (CudaKernel::J2kForwardIct, "j2k_forward_ict"),
            (
                CudaKernel::J2kForwardDwt53Horizontal,
                "j2k_forward_dwt53_horizontal",
            ),
            (
                CudaKernel::J2kForwardDwt53Vertical,
                "j2k_forward_dwt53_vertical",
            ),
            (
                CudaKernel::J2kForwardDwt97Horizontal,
                "j2k_forward_dwt97_horizontal",
            ),
            (
                CudaKernel::J2kForwardDwt97Vertical,
                "j2k_forward_dwt97_vertical",
            ),
            (CudaKernel::J2kQuantizeSubband, "j2k_quantize_subband"),
            (
                CudaKernel::J2kQuantizeSubbandStrided,
                "j2k_quantize_subband_strided",
            ),
        ];
        assert_stable_entrypoints(&cases);
    }

    #[test]
    fn j2k_idwt_and_store_entrypoints_are_stable() {
        let cases = [
            (CudaKernel::J2kIdwtInterleave, "j2k_idwt_interleave"),
            (
                CudaKernel::J2kIdwtInterleaveHorizontalMulti,
                "j2k_idwt_interleave_horizontal_multi",
            ),
            (
                CudaKernel::J2kIdwtInterleaveHorizontal53Multi,
                "j2k_idwt_interleave_horizontal_53_multi",
            ),
            (
                CudaKernel::J2kIdwtInterleaveHorizontal97Multi,
                "j2k_idwt_interleave_horizontal_97_multi",
            ),
            (CudaKernel::J2kIdwtHorizontal53, "j2k_idwt_horizontal_53"),
            (CudaKernel::J2kIdwtHorizontal97, "j2k_idwt_horizontal_97"),
            (CudaKernel::J2kIdwtVerticalMulti, "j2k_idwt_vertical_multi"),
            (
                CudaKernel::J2kIdwtVertical53Multi,
                "j2k_idwt_vertical_53_multi",
            ),
            (
                CudaKernel::J2kIdwtVertical97Multi,
                "j2k_idwt_vertical_97_multi",
            ),
            (
                CudaKernel::J2kIdwtVertical97MultiCols4,
                "j2k_idwt_vertical_97_multi_cols4",
            ),
            (CudaKernel::J2kIdwtVertical53, "j2k_idwt_vertical_53"),
            (CudaKernel::J2kIdwtVertical97, "j2k_idwt_vertical_97"),
            (CudaKernel::J2kInverseMct, "j2k_inverse_mct"),
            (CudaKernel::J2kStoreGray16, "j2k_store_gray16"),
            (CudaKernel::J2kStoreGray16Batch, "j2k_store_gray16_batch"),
            (CudaKernel::J2kStoreGrayI16Batch, "j2k_store_grayi16_batch"),
            (CudaKernel::J2kStoreGray8, "j2k_store_gray8"),
            (CudaKernel::J2kStoreGray8Batch, "j2k_store_gray8_batch"),
            (CudaKernel::J2kStoreRgb16, "j2k_store_rgb16"),
            (CudaKernel::J2kStoreRgb16Mct, "j2k_store_rgb16_mct"),
            (CudaKernel::J2kStoreRgb8, "j2k_store_rgb8"),
            (CudaKernel::J2kStoreRgb8MctBatch, "j2k_store_rgb8_mct_batch"),
            (
                CudaKernel::J2kStoreRgb8NativeBatch,
                "j2k_store_rgb8_native_batch",
            ),
            (
                CudaKernel::J2kStoreRgb16NativeBatch,
                "j2k_store_rgb16_native_batch",
            ),
            (
                CudaKernel::J2kStoreRgbI16NativeBatch,
                "j2k_store_rgbi16_native_batch",
            ),
            (
                CudaKernel::J2kStoreRgba8NativeBatch,
                "j2k_store_rgba8_native_batch",
            ),
            (
                CudaKernel::J2kStoreRgba16NativeBatch,
                "j2k_store_rgba16_native_batch",
            ),
            (
                CudaKernel::J2kStoreRgbaI16NativeBatch,
                "j2k_store_rgbai16_native_batch",
            ),
        ];
        assert_stable_entrypoints(&cases);
    }

    fn assert_stable_entrypoints(cases: &[(CudaKernel, &str)]) {
        for (kernel, expected) in cases {
            let entrypoint = kernel.entrypoint();
            assert_eq!(&entrypoint[..entrypoint.len() - 1], expected.as_bytes());
            assert_eq!(entrypoint.last(), Some(&0));
        }
    }

    #[cfg(all(feature = "cuda-oxide-htj2k-decode", j2k_cuda_oxide_htj2k_decode_built))]
    #[test]
    fn htj2k_decode_kernel_metadata_matches_generated_ptx() {
        assert_ptx_entrypoints(
            htj2k_decode_ptx(),
            &[
                CudaKernel::Htj2kDecodeCodeblocks,
                CudaKernel::Htj2kDecodeCodeblocksMulti,
                CudaKernel::Htj2kDecodeCodeblocksMultiCleanupOnly,
                CudaKernel::Htj2kDecodeCodeblocksMultiCleanupDequantize,
            ],
        );
    }

    #[cfg(all(
        feature = "cuda-oxide-j2k-dequantize",
        j2k_cuda_oxide_j2k_dequantize_built
    ))]
    #[test]
    fn j2k_dequantize_kernel_metadata_matches_generated_ptx() {
        assert_ptx_entrypoints(
            j2k_dequantize_ptx(),
            &[
                CudaKernel::J2kDequantizeHtj2kCodeblocks,
                CudaKernel::J2kDequantizeHtj2kCodeblocksMulti,
                CudaKernel::J2kDequantizeHtj2kCleanupJobsMulti,
            ],
        );
    }

    #[cfg(all(feature = "cuda-oxide-htj2k-encode", j2k_cuda_oxide_htj2k_encode_built))]
    #[test]
    fn htj2k_encode_kernel_metadata_matches_generated_ptx() {
        assert_ptx_entrypoints(
            htj2k_encode_ptx(),
            &[
                CudaKernel::Htj2kEncodeCodeblocks,
                CudaKernel::Htj2kEncodeCodeblocksMultiInput,
                CudaKernel::Htj2kEncodeCodeblocksMultiInputCleanup,
                CudaKernel::Htj2kEncodeCodeblocksMultiInputCleanup64,
            ],
        );
    }

    #[cfg(all(feature = "cuda-oxide-j2k-encode", j2k_cuda_oxide_j2k_encode_built))]
    #[test]
    fn j2k_encode_kernel_metadata_matches_generated_ptx() {
        assert_ptx_entrypoints(
            j2k_encode_ptx(),
            &[
                CudaKernel::Htj2kCompactCodeblocks,
                CudaKernel::Htj2kPacketizeCleanup,
                CudaKernel::J2kDeinterleaveToF32,
                CudaKernel::J2kDeinterleaveStridedToF32,
                CudaKernel::J2kForwardRct,
                CudaKernel::J2kForwardIct,
                CudaKernel::J2kForwardDwt53Horizontal,
                CudaKernel::J2kForwardDwt53Vertical,
                CudaKernel::J2kForwardDwt97Horizontal,
                CudaKernel::J2kForwardDwt97Vertical,
                CudaKernel::J2kQuantizeSubband,
                CudaKernel::J2kQuantizeSubbandStrided,
            ],
        );
    }

    #[cfg(all(feature = "cuda-oxide-j2k-idwt", j2k_cuda_oxide_j2k_idwt_built))]
    #[test]
    fn j2k_idwt_kernel_metadata_matches_generated_ptx() {
        assert_ptx_entrypoints(
            j2k_idwt_ptx(),
            &[
                CudaKernel::J2kIdwtInterleave,
                CudaKernel::J2kIdwtInterleaveHorizontalMulti,
                CudaKernel::J2kIdwtInterleaveHorizontal53Multi,
                CudaKernel::J2kIdwtInterleaveHorizontal97Multi,
                CudaKernel::J2kIdwtHorizontal53,
                CudaKernel::J2kIdwtHorizontal97,
                CudaKernel::J2kIdwtVerticalMulti,
                CudaKernel::J2kIdwtVertical53Multi,
                CudaKernel::J2kIdwtVertical97Multi,
                CudaKernel::J2kIdwtVertical97MultiCols4,
                CudaKernel::J2kIdwtVertical53,
                CudaKernel::J2kIdwtVertical97,
            ],
        );
    }

    #[cfg(all(
        feature = "cuda-oxide-j2k-decode-store",
        j2k_cuda_oxide_j2k_decode_store_built
    ))]
    #[test]
    fn j2k_decode_store_kernel_metadata_matches_generated_ptx() {
        assert_ptx_entrypoints(
            j2k_decode_store_ptx(),
            &[
                CudaKernel::J2kInverseMct,
                CudaKernel::J2kStoreGray16,
                CudaKernel::J2kStoreGray16Batch,
                CudaKernel::J2kStoreGrayI16Batch,
                CudaKernel::J2kStoreGray8,
                CudaKernel::J2kStoreGray8Batch,
                CudaKernel::J2kStoreRgb16,
                CudaKernel::J2kStoreRgb16Mct,
                CudaKernel::J2kStoreRgb8,
                CudaKernel::J2kStoreRgb8MctBatch,
                CudaKernel::J2kStoreRgb8NativeBatch,
                CudaKernel::J2kStoreRgb16NativeBatch,
                CudaKernel::J2kStoreRgbI16NativeBatch,
                CudaKernel::J2kStoreRgba8NativeBatch,
                CudaKernel::J2kStoreRgba16NativeBatch,
                CudaKernel::J2kStoreRgbaI16NativeBatch,
            ],
        );
    }

    #[cfg(any(
        all(feature = "cuda-oxide-htj2k-decode", j2k_cuda_oxide_htj2k_decode_built),
        all(
            feature = "cuda-oxide-j2k-dequantize",
            j2k_cuda_oxide_j2k_dequantize_built
        ),
        all(feature = "cuda-oxide-htj2k-encode", j2k_cuda_oxide_htj2k_encode_built),
        all(feature = "cuda-oxide-j2k-encode", j2k_cuda_oxide_j2k_encode_built),
        all(feature = "cuda-oxide-j2k-idwt", j2k_cuda_oxide_j2k_idwt_built),
        all(
            feature = "cuda-oxide-j2k-decode-store",
            j2k_cuda_oxide_j2k_decode_store_built
        )
    ))]
    fn assert_ptx_entrypoints(ptx: &[u8], kernels: &[CudaKernel]) {
        assert_eq!(ptx.last(), Some(&0));
        let source = std::str::from_utf8(&ptx[..ptx.len() - 1]).expect("PTX UTF-8");
        for kernel in kernels {
            let raw = kernel.entrypoint();
            let entrypoint = std::str::from_utf8(&raw[..raw.len() - 1]).expect("entrypoint UTF-8");
            assert!(source.contains(&format!(".visible .entry {entrypoint}(")));
        }
    }
}
