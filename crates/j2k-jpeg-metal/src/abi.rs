// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(target_os = "macos")]
use j2k_jpeg::adapter::{JpegEntropyCheckpointV1, JpegHuffmanTable as PacketHuffmanTable};
#[cfg(target_os = "macos")]
use metal::Buffer;

#[cfg(target_os = "macos")]
pub(crate) const MODE_GRAY: u32 = 0;
#[cfg(target_os = "macos")]
pub(crate) const MODE_YCBCR: u32 = 1;
#[cfg(target_os = "macos")]
pub(crate) const MODE_RGB: u32 = 2;

#[cfg(target_os = "macos")]
pub(crate) const OUT_GRAY: u32 = 0;
#[cfg(target_os = "macos")]
pub(crate) const OUT_RGB: u32 = 1;
#[cfg(target_os = "macos")]
pub(crate) const OUT_RGBA: u32 = 2;

pub(crate) const JPEG_BASELINE_ENCODE_STATUS_OK: u32 = 0;
#[cfg(target_os = "macos")]
pub(crate) const JPEG_BASELINE_ENCODE_STATUS_OVERFLOW: u32 = 1;
#[cfg(target_os = "macos")]
pub(crate) const JPEG_BASELINE_ENCODE_STATUS_MISSING_HUFFMAN: u32 = 2;
#[cfg(target_os = "macos")]
pub(crate) const JPEG_BASELINE_ENCODE_STATUS_INVALID_PARAMS: u32 = 3;

#[cfg(target_os = "macos")]
pub(crate) const FAST420_STATUS_OK: u32 = 0;
#[cfg(target_os = "macos")]
pub(crate) const FAST420_STATUS_TRUNCATED: u32 = 1;
#[cfg(target_os = "macos")]
pub(crate) const FAST420_STATUS_HUFFMAN: u32 = 2;

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct JpegPackParams {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) out_stride: u32,
    pub(crate) alpha: u32,
    pub(crate) mode: u32,
    pub(crate) out_format: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct JpegBaselineEncodeParams {
    pub(crate) input_offset_bytes: u32,
    pub(crate) input_width: u32,
    pub(crate) input_height: u32,
    pub(crate) output_width: u32,
    pub(crate) output_height: u32,
    pub(crate) pitch_bytes: u32,
    pub(crate) mcus_per_row: u32,
    pub(crate) mcu_rows: u32,
    pub(crate) restart_interval_mcus: u32,
    pub(crate) format: u32,
    pub(crate) components: u32,
    pub(crate) max_h: u32,
    pub(crate) max_v: u32,
    pub(crate) h0: u32,
    pub(crate) v0: u32,
    pub(crate) h1: u32,
    pub(crate) v1: u32,
    pub(crate) h2: u32,
    pub(crate) v2: u32,
    pub(crate) entropy_offset_bytes: u32,
    pub(crate) entropy_capacity: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct JpegBaselineEncodeHuffmanTable {
    pub(crate) codes: [u16; 256],
    pub(crate) lens: [u8; 256],
}

#[cfg(target_os = "macos")]
impl Default for JpegBaselineEncodeHuffmanTable {
    fn default() -> Self {
        Self {
            codes: [0; 256],
            lens: [0; 256],
        }
    }
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct JpegBaselineEncodeStatus {
    pub(crate) code: u32,
    pub(crate) entropy_len: u32,
    pub(crate) detail: u32,
    pub(crate) reserved: u32,
}

#[cfg(target_os = "macos")]
pub(crate) struct JpegBaselineEntropyEncodeJob<'a> {
    pub(crate) input: &'a Buffer,
    pub(crate) input_offset: usize,
    pub(crate) params: JpegBaselineEncodeParams,
    pub(crate) q_luma: [u8; 64],
    pub(crate) q_chroma: [u8; 64],
    pub(crate) huff_dc_luma: JpegBaselineEncodeHuffmanTable,
    pub(crate) huff_ac_luma: JpegBaselineEncodeHuffmanTable,
    pub(crate) huff_dc_chroma: JpegBaselineEncodeHuffmanTable,
    pub(crate) huff_ac_chroma: JpegBaselineEncodeHuffmanTable,
    pub(crate) entropy_capacity: usize,
}

#[cfg(target_os = "macos")]
pub(crate) struct JpegBaselineEntropyEncodeBatchJob<'a> {
    pub(crate) input: &'a Buffer,
    pub(crate) params: Vec<JpegBaselineEncodeParams>,
    pub(crate) q_luma: [u8; 64],
    pub(crate) q_chroma: [u8; 64],
    pub(crate) huff_dc_luma: JpegBaselineEncodeHuffmanTable,
    pub(crate) huff_ac_luma: JpegBaselineEncodeHuffmanTable,
    pub(crate) huff_dc_chroma: JpegBaselineEncodeHuffmanTable,
    pub(crate) huff_ac_chroma: JpegBaselineEncodeHuffmanTable,
    pub(crate) entropy_capacity: usize,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct JpegFast420Params {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) chroma_width: u32,
    pub(crate) chroma_height: u32,
    pub(crate) mcus_per_row: u32,
    pub(crate) mcu_rows: u32,
    pub(crate) restart_interval_mcus: u32,
    pub(crate) restart_offset_count: u32,
    pub(crate) restart_start_mcu: u32,
    pub(crate) entropy_len: u32,
    pub(crate) out_stride: u32,
    pub(crate) alpha: u32,
    pub(crate) out_format: u32,
    pub(crate) origin_x: u32,
    pub(crate) origin_y: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct JpegFast420ScaledParams {
    pub(crate) scaled_width: u32,
    pub(crate) scaled_height: u32,
    pub(crate) chroma_width: u32,
    pub(crate) chroma_height: u32,
    pub(crate) mcus_per_row: u32,
    pub(crate) mcu_rows: u32,
    pub(crate) restart_interval_mcus: u32,
    pub(crate) restart_offset_count: u32,
    pub(crate) restart_start_mcu: u32,
    pub(crate) entropy_len: u32,
    pub(crate) scale_shift: u32,
    pub(crate) origin_x: u32,
    pub(crate) origin_y: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct JpegFast444Params {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) mcus_per_row: u32,
    pub(crate) mcu_rows: u32,
    pub(crate) restart_interval_mcus: u32,
    pub(crate) restart_offset_count: u32,
    pub(crate) restart_start_mcu: u32,
    pub(crate) entropy_len: u32,
    pub(crate) origin_x: u32,
    pub(crate) origin_y: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct JpegFast444ScaledParams {
    pub(crate) scaled_width: u32,
    pub(crate) scaled_height: u32,
    pub(crate) mcus_per_row: u32,
    pub(crate) mcu_rows: u32,
    pub(crate) restart_interval_mcus: u32,
    pub(crate) restart_offset_count: u32,
    pub(crate) restart_start_mcu: u32,
    pub(crate) entropy_len: u32,
    pub(crate) scale_shift: u32,
    pub(crate) origin_x: u32,
    pub(crate) origin_y: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct JpegFast420WindowedPackParams {
    pub(crate) src_width: u32,
    pub(crate) src_height: u32,
    pub(crate) chroma_width: u32,
    pub(crate) chroma_height: u32,
    pub(crate) src_x: u32,
    pub(crate) src_y: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) out_stride: u32,
    pub(crate) alpha: u32,
    pub(crate) out_format: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct JpegFast420BatchParams {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) chroma_width: u32,
    pub(crate) chroma_height: u32,
    pub(crate) mcus_per_row: u32,
    pub(crate) mcu_rows: u32,
    pub(crate) segment_count: u32,
    pub(crate) tile_count: u32,
    pub(crate) out_stride: u32,
    pub(crate) alpha: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct JpegFastRegionScaledBatchParams {
    pub(crate) scaled_width: u32,
    pub(crate) scaled_height: u32,
    pub(crate) chroma_width: u32,
    pub(crate) chroma_height: u32,
    pub(crate) mcus_per_row: u32,
    pub(crate) mcu_rows: u32,
    pub(crate) segment_count: u32,
    pub(crate) tile_count: u32,
    pub(crate) scale_shift: u32,
    pub(crate) origin_x: u32,
    pub(crate) origin_y: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct JpegFast444TextureBatchParams {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) mcus_per_row: u32,
    pub(crate) mcu_rows: u32,
    pub(crate) segment_count: u32,
    pub(crate) tile_index: u32,
    pub(crate) alpha: u32,
    pub(crate) mode: u32,
}

#[cfg(target_os = "macos")]
pub(crate) const FAST422_TEXTURE_BOUNDARY_META_WORDS: usize = 4;
#[cfg(target_os = "macos")]
pub(crate) const FAST422_TEXTURE_BOUNDARY_SAMPLE_BYTES: usize = 48;
#[cfg(target_os = "macos")]
pub(crate) const FAST420_TEXTURE_BOUNDARY_META_WORDS: usize = 4;
#[cfg(target_os = "macos")]
pub(crate) const FAST420_TEXTURE_BOUNDARY_SAMPLE_BYTES: usize = 64;
#[cfg(target_os = "macos")]
pub(crate) const FAST420_TEXTURE_VERTICAL_META_WORDS: usize = 4;
#[cfg(target_os = "macos")]
pub(crate) const FAST420_TEXTURE_VERTICAL_SAMPLE_BYTES: usize = 64;

/// Direct-to-texture decode params shared by the 4:2:0 and 4:2:2 texture
/// kernels (identical layout; both kernel families read the same fields).
#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct JpegFast420TextureBatchParams {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) chroma_width: u32,
    pub(crate) chroma_height: u32,
    pub(crate) mcus_per_row: u32,
    pub(crate) mcu_rows: u32,
    pub(crate) segment_count: u32,
    pub(crate) tile_index: u32,
    pub(crate) alpha: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct JpegWindowedPackBatchParams {
    pub(crate) src_width: u32,
    pub(crate) src_height: u32,
    pub(crate) chroma_width: u32,
    pub(crate) chroma_height: u32,
    pub(crate) src_x: u32,
    pub(crate) src_y: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) tile_count: u32,
    pub(crate) out_stride: u32,
    pub(crate) alpha: u32,
    pub(crate) mode: u32,
    pub(crate) out_format: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct JpegWindowedTexturePackBatchParams {
    pub(crate) src_width: u32,
    pub(crate) src_height: u32,
    pub(crate) chroma_width: u32,
    pub(crate) chroma_height: u32,
    pub(crate) src_x: u32,
    pub(crate) src_y: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) tile_index: u32,
    pub(crate) alpha: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct JpegTexturePackBatchParams {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) chroma_width: u32,
    pub(crate) chroma_height: u32,
    pub(crate) tile_index: u32,
    pub(crate) alpha: u32,
    pub(crate) mode: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct JpegRgb8ToRgbaTextureParams {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) in_stride: u32,
    pub(crate) alpha: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct PreparedHuffmanHost {
    pub(crate) min_code: [i32; 17],
    pub(crate) max_code: [i32; 17],
    pub(crate) val_offset: [i32; 17],
    pub(crate) values: [u8; 256],
    pub(crate) fast_symbol: [u8; 512],
    pub(crate) fast_len: [u8; 512],
    pub(crate) values_len: u16,
    pub(crate) reserved: u16,
}

#[cfg(target_os = "macos")]
impl From<&PacketHuffmanTable> for PreparedHuffmanHost {
    fn from(value: &PacketHuffmanTable) -> Self {
        let canonical = value
            .derive_canonical()
            .expect("backend packet Huffman table must be canonicalizable");
        let mut values = [0u8; 256];
        let mut fast_symbol = [0u8; 512];
        let mut fast_len = [0u8; 512];
        let values_len = usize::from(value.values_len);
        values[..values_len].copy_from_slice(&value.values[..values_len]);

        for (idx, &symbol) in values.iter().enumerate().take(canonical.huffsize_len) {
            let len = usize::from(canonical.huffsize[idx]);
            if len == 0 || len > 9 {
                continue;
            }
            let code = usize::from(canonical.huffcode[idx]);
            let prefix = code << (9 - len);
            let fill = 1usize << (9 - len);
            for suffix in 0..fill {
                fast_symbol[prefix | suffix] = symbol;
                fast_len[prefix | suffix] = canonical.huffsize[idx];
            }
        }

        Self {
            min_code: canonical.min_code,
            max_code: canonical.max_code,
            val_offset: canonical.val_offset,
            values,
            fast_symbol,
            fast_len,
            values_len: value.values_len,
            reserved: 0,
        }
    }
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct JpegDecodeStatus {
    pub(crate) code: u32,
    pub(crate) detail: u32,
    pub(crate) position: u32,
    pub(crate) reserved: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct JpegEntropyCheckpointHost {
    pub(crate) mcu_index: u32,
    pub(crate) entropy_pos: u32,
    pub(crate) bit_acc: u64,
    pub(crate) bit_count: u32,
    pub(crate) y_prev_dc: i32,
    pub(crate) cb_prev_dc: i32,
    pub(crate) cr_prev_dc: i32,
    pub(crate) reserved: u32,
    pub(crate) reserved_tail: u32,
}

#[cfg(target_os = "macos")]
impl From<JpegEntropyCheckpointV1> for JpegEntropyCheckpointHost {
    fn from(value: JpegEntropyCheckpointV1) -> Self {
        Self {
            mcu_index: value.mcu_index,
            entropy_pos: value.entropy_pos,
            bit_acc: value.bit_acc,
            bit_count: value.bit_count,
            y_prev_dc: value.y_prev_dc,
            cb_prev_dc: value.cb_prev_dc,
            cr_prev_dc: value.cr_prev_dc,
            reserved: value.reserved,
            reserved_tail: 0,
        }
    }
}

#[cfg(target_os = "macos")]
macro_rules! prove_gpu_readback_layout {
    ($ty:ty, $offset:expr;) => {
        let _: [(); core::mem::size_of::<$ty>()] = [(); $offset];
    };
    (
        $ty:ty,
        $offset:expr;
        $field:ident: $field_ty:ty
        $(, $remaining_field:ident: $remaining_field_ty:ty)*
    ) => {
        let _: [(); core::mem::offset_of!($ty, $field)] = [(); $offset];
        prove_gpu_readback_layout!(
            $ty,
            $offset + core::mem::size_of::<$field_ty>();
            $($remaining_field: $remaining_field_ty),*
        );
    };
}

#[cfg(target_os = "macos")]
macro_rules! impl_gpu_readback_abi {
    ($(
        $ty:ty {
            $first_field:ident: $first_field_ty:ty
            $(, $field:ident: $field_ty:ty)*
            $(,)?
        }
    ),+ $(,)?) => {
        $(
            const _: () = {
                fn assert_field_types(value: &$ty) {
                    let _: &$first_field_ty = &value.$first_field;
                    $(let _: &$field_ty = &value.$field;)*
                }
                let _ = assert_field_types;

                prove_gpu_readback_layout!(
                    $ty,
                    0;
                    $first_field: $first_field_ty
                    $(, $field: $field_ty)*
                );
            };

            // SAFETY: Each listed type is `#[repr(C)]`, `Copy`, accepts every
            // bit pattern, and the compile-time field walk proves its complete
            // object representation contains no padding bytes.
            unsafe impl j2k_core::accelerator::GpuAbi for $ty {
                const NAME: &'static str = stringify!($ty);
            }
        )+
    };
}

#[cfg(target_os = "macos")]
impl_gpu_readback_abi!(
    JpegBaselineEncodeParams {
        input_offset_bytes: u32,
        input_width: u32,
        input_height: u32,
        output_width: u32,
        output_height: u32,
        pitch_bytes: u32,
        mcus_per_row: u32,
        mcu_rows: u32,
        restart_interval_mcus: u32,
        format: u32,
        components: u32,
        max_h: u32,
        max_v: u32,
        h0: u32,
        v0: u32,
        h1: u32,
        v1: u32,
        h2: u32,
        v2: u32,
        entropy_offset_bytes: u32,
        entropy_capacity: u32,
    },
    JpegBaselineEncodeStatus {
        code: u32,
        entropy_len: u32,
        detail: u32,
        reserved: u32,
    },
    JpegDecodeStatus {
        code: u32,
        detail: u32,
        position: u32,
        reserved: u32,
    },
    JpegEntropyCheckpointHost {
        mcu_index: u32,
        entropy_pos: u32,
        bit_acc: u64,
        bit_count: u32,
        y_prev_dc: i32,
        cb_prev_dc: i32,
        cr_prev_dc: i32,
        reserved: u32,
        reserved_tail: u32,
    },
);

#[cfg(all(test, target_os = "macos"))]
mod gpu_readback_abi_tests {
    use core::mem::{align_of, offset_of, size_of};

    use j2k_core::accelerator::GpuAbi;
    use j2k_jpeg::adapter::JpegEntropyCheckpointV1;

    use super::{
        JpegBaselineEncodeParams, JpegBaselineEncodeStatus, JpegDecodeStatus,
        JpegEntropyCheckpointHost,
    };

    #[test]
    fn status_layouts_match_metal_shader_abi() {
        assert_eq!(size_of::<JpegBaselineEncodeParams>(), 84);
        assert_eq!(align_of::<JpegBaselineEncodeParams>(), 4);
        assert_eq!(offset_of!(JpegBaselineEncodeParams, input_offset_bytes), 0);
        assert_eq!(offset_of!(JpegBaselineEncodeParams, entropy_capacity), 80);
        assert_eq!(size_of::<JpegBaselineEncodeStatus>(), 16);
        assert_eq!(align_of::<JpegBaselineEncodeStatus>(), 4);
        assert_eq!(offset_of!(JpegBaselineEncodeStatus, code), 0);
        assert_eq!(offset_of!(JpegBaselineEncodeStatus, entropy_len), 4);
        assert_eq!(offset_of!(JpegBaselineEncodeStatus, detail), 8);
        assert_eq!(offset_of!(JpegBaselineEncodeStatus, reserved), 12);

        assert_eq!(size_of::<JpegDecodeStatus>(), 16);
        assert_eq!(align_of::<JpegDecodeStatus>(), 4);
        assert_eq!(offset_of!(JpegDecodeStatus, code), 0);
        assert_eq!(offset_of!(JpegDecodeStatus, detail), 4);
        assert_eq!(offset_of!(JpegDecodeStatus, position), 8);
        assert_eq!(offset_of!(JpegDecodeStatus, reserved), 12);

        assert_eq!(size_of::<JpegEntropyCheckpointHost>(), 40);
        assert_eq!(align_of::<JpegEntropyCheckpointHost>(), 8);
        assert_eq!(offset_of!(JpegEntropyCheckpointHost, mcu_index), 0);
        assert_eq!(offset_of!(JpegEntropyCheckpointHost, entropy_pos), 4);
        assert_eq!(offset_of!(JpegEntropyCheckpointHost, bit_acc), 8);
        assert_eq!(offset_of!(JpegEntropyCheckpointHost, bit_count), 16);
        assert_eq!(offset_of!(JpegEntropyCheckpointHost, y_prev_dc), 20);
        assert_eq!(offset_of!(JpegEntropyCheckpointHost, cb_prev_dc), 24);
        assert_eq!(offset_of!(JpegEntropyCheckpointHost, cr_prev_dc), 28);
        assert_eq!(offset_of!(JpegEntropyCheckpointHost, reserved), 32);
        assert_eq!(offset_of!(JpegEntropyCheckpointHost, reserved_tail), 36);
    }

    #[test]
    fn checkpoint_conversion_initializes_the_complete_gpu_abi() {
        let checkpoint = JpegEntropyCheckpointHost::from(JpegEntropyCheckpointV1 {
            mcu_index: 1,
            entropy_pos: 2,
            bit_acc: 3,
            bit_count: 4,
            y_prev_dc: 5,
            cb_prev_dc: 6,
            cr_prev_dc: 7,
            reserved: 8,
        });

        assert_eq!(checkpoint.reserved_tail, 0);
        let bytes = JpegEntropyCheckpointHost::as_bytes(&checkpoint);
        assert_eq!(bytes.len(), size_of::<JpegEntropyCheckpointHost>());
        assert_eq!(&bytes[36..], &[0, 0, 0, 0]);
    }
}
