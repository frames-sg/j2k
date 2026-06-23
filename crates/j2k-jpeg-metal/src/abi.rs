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

#[cfg(target_os = "macos")]
pub(crate) const JPEG_BASELINE_ENCODE_FORMAT_GRAY8: u32 = 0;
#[cfg(target_os = "macos")]
pub(crate) const JPEG_BASELINE_ENCODE_FORMAT_RGB8: u32 = 1;
#[cfg(target_os = "macos")]
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
        let mut min_code = [i32::MAX; 17];
        let mut max_code = [-1i32; 17];
        let mut val_offset = [0i32; 17];
        let mut values = [0u8; 256];
        let mut fast_symbol = [0u8; 512];
        let mut fast_len = [0u8; 512];
        let values_len = usize::from(value.values_len);
        values[..values_len].copy_from_slice(&value.values[..values_len]);

        let mut huffsize = [0u8; 256];
        let mut huffsize_len = 0usize;
        for (len_minus_1, &count) in value.bits.iter().enumerate() {
            let len = u8::try_from(len_minus_1 + 1).expect("JPEG Huffman code length fits in u8");
            for _ in 0..count {
                huffsize[huffsize_len] = len;
                huffsize_len += 1;
            }
        }

        let mut huffcode = [0u16; 256];
        let mut code = 0u32;
        let mut si = huffsize.first().copied().unwrap_or(0);
        for (idx, &size) in huffsize[..huffsize_len].iter().enumerate() {
            while size != si {
                code <<= 1;
                si += 1;
            }
            huffcode[idx] = u16::try_from(code).expect("JPEG Huffman code fits in u16");
            code += 1;
        }

        let mut idx = 0usize;
        for (len_minus_1, &count) in value.bits.iter().enumerate() {
            let len = len_minus_1 + 1;
            let count = usize::from(count);
            if count == 0 {
                continue;
            }
            min_code[len] = i32::from(huffcode[idx]);
            max_code[len] = i32::from(huffcode[idx + count - 1]);
            val_offset[len] =
                i32::try_from(idx).expect("JPEG Huffman value index fits in i32") - min_code[len];
            idx += count;
        }

        for idx in 0..huffsize_len {
            let len = usize::from(huffsize[idx]);
            if len == 0 || len > 9 {
                continue;
            }
            let code = usize::from(huffcode[idx]);
            let prefix = code << (9 - len);
            let fill = 1usize << (9 - len);
            for suffix in 0..fill {
                fast_symbol[prefix | suffix] = values[idx];
                fast_len[prefix | suffix] = huffsize[idx];
            }
        }

        Self {
            min_code,
            max_code,
            val_offset,
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
        }
    }
}
