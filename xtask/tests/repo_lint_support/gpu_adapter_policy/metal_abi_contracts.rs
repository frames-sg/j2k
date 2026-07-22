// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use crate::repo_lint_support::{assert_pattern_checks, read_source_files, repo_root, PatternCheck};

#[test]
fn jpeg_metal_huffman_derivation_uses_shared_entropy_canonical_tables() {
    let root = repo_root();
    let codec_math = fs::read_to_string(root.join("crates/j2k-codec-math/src/jpeg.rs"))
        .expect("read codec-math JPEG helpers");
    let entropy_huffman = fs::read_to_string(root.join("crates/j2k-jpeg/src/entropy/huffman.rs"))
        .expect("read JPEG entropy Huffman implementation");
    let fast_packet_types =
        fs::read_to_string(root.join("crates/j2k-jpeg/src/adapter/fast_packet/types.rs"))
            .expect("read JPEG fast packet type module");
    let metal_abi = fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/abi.rs"))
        .expect("read JPEG Metal ABI");
    let cuda_runtime = read_source_files(
        root,
        &[
            "crates/j2k-cuda-runtime/src/jpeg.rs",
            "crates/j2k-cuda-runtime/src/jpeg/types.rs",
        ],
    );

    assert!(
        codec_math.contains("pub fn derive_canonical_huffman")
            && codec_math.contains("pub struct CanonicalHuffmanDerivation")
            && codec_math.contains("let mut huffsize")
            && codec_math.contains("let mut huffcode"),
        "j2k-codec-math must own the Annex C canonical Huffman derivation"
    );
    assert!(
        entropy_huffman.contains("pub(crate) fn derive_canonical_huffman")
            && entropy_huffman.contains("derive_canonical_huffman(raw)?"),
        "j2k-jpeg entropy must expose and use one shared Annex C canonical Huffman derivation"
    );
    assert!(
        fast_packet_types.contains("pub struct JpegCanonicalHuffmanTable")
            && fast_packet_types.contains("pub fn derive_canonical(&self)")
            && fast_packet_types.contains("derive_canonical_huffman(&raw)?"),
        "j2k-jpeg adapter must expose backend-facing canonical Huffman derivation"
    );
    assert!(
        metal_abi.contains(".derive_canonical()")
            && !metal_abi.contains("let mut huffsize")
            && !metal_abi.contains("let mut huffcode")
            && !metal_abi.contains("let mut code = 0u32")
            && !metal_abi.contains("for (len_minus_1, &count) in value.bits.iter().enumerate()"),
        "JPEG Metal ABI must pack shared canonical Huffman tables instead of deriving Annex C locally"
    );
    assert!(
        cuda_runtime.contains("j2k_codec_math::jpeg::derive_canonical_huffman")
            && !cuda_runtime.contains("let mut huffsize")
            && !cuda_runtime.contains("let mut huffcode")
            && !cuda_runtime.contains("let mut code = 0u32"),
        "CUDA JPEG runtime must use shared codec-math canonical Huffman derivation"
    );
}

#[test]
fn jpeg_metal_gpu_abi_uploads_are_padding_free() {
    let root = repo_root();
    let abi = fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/abi.rs"))
        .expect("read JPEG Metal ABI");
    let buffers = fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/buffers.rs"))
        .expect("read JPEG Metal buffers");
    let params =
        fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/compute/fast_packets/params.rs"))
            .expect("read JPEG Metal fast-packet params");
    let status = fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/compute/status.rs"))
        .expect("read JPEG Metal status");
    let shader = fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/shaders_shared.metal"))
        .expect("read JPEG Metal shared shader ABI");

    assert_pattern_checks(&[
        PatternCheck::new("JPEG Metal padding-free ABI proof", &abi).required(&[
            "pub(crate) reserved_tail: u32",
            "macro_rules! prove_gpu_readback_layout",
            "let _: [(); core::mem::size_of::<$ty>()] = [(); $offset];",
            "core::mem::offset_of!($ty, $field)",
            "$offset + core::mem::size_of::<$field_ty>();",
            "prove_gpu_readback_layout!(",
            "JpegEntropyCheckpointHost {",
            "reserved_tail: u32",
        ]),
        PatternCheck::new("JPEG Metal typed upload boundary", &buffers)
            .required(&[
                "pub(crate) fn shared_buffer_with_slice<T: GpuAbi>",
                "let bytes = T::slice_as_bytes(values);",
            ])
            .forbidden(&["from_raw_parts(values.as_ptr().cast::<u8>()"]),
        PatternCheck::new("JPEG Metal checkpoint staging", &params)
            .required(&[
                "<u32 as GpuAbi>::slice_as_bytes(restart_offsets)",
                "checked_count_product(",
                "let buffer = new_shared_buffer(device, total_bytes)?;",
                "for (index, checkpoint) in entropy_checkpoints.iter().copied().enumerate()",
                "JpegEntropyCheckpointHost::as_bytes(&checkpoint)",
                "checked_copy_bytes_to_buffer_at(",
            ])
            .forbidden(&["from_raw_parts("]),
        PatternCheck::new("JPEG Metal status staging", &status)
            .required(&[
                "checked_count_product(",
                "core::mem::size_of::<JpegDecodeStatus>()",
                "let buffer = new_shared_buffer(device, bytes)?;",
                "checked_fill_buffer_u8(&buffer, bytes, 0",
                "checked_buffer_slice::<JpegDecodeStatus>(",
            ])
            .forbidden(&["from_raw_parts("]),
        PatternCheck::new("JPEG Metal checkpoint shader padding", &shader)
            .required(&["uint reserved_tail;"]),
    ]);
}

#[test]
fn j2k_metal_ht_uvlc_upload_uses_a_local_padding_free_abi_row() {
    let root = repo_root();
    let abi = fs::read_to_string(root.join("crates/j2k-metal/src/compute/abi.rs"))
        .expect("read J2K Metal ABI");
    let runtime = fs::read_to_string(root.join("crates/j2k-metal/src/compute/runtime.rs"))
        .expect("read J2K Metal runtime");
    let shader = fs::read_to_string(root.join("crates/j2k-metal/src/encode_bitstream_ht.metal"))
        .expect("read J2K Metal HT encoder shader");

    assert_pattern_checks(&[
        PatternCheck::new("J2K Metal HT UVLC padding-free upload row", &abi).required(&[
            "pub(crate) struct J2kHtUvlcEncodeTableEntry",
            "core::mem::offset_of!(J2kHtUvlcEncodeTableEntry, ext_len)",
            "core::mem::size_of::<J2kHtUvlcEncodeTableEntry>()",
            "unsafe impl j2k_core::accelerator::GpuAbi for J2kHtUvlcEncodeTableEntry",
            "ht_uvlc_upload_rows_match_the_canonical_packed_table",
            "j2k_native::ht_uvlc_encode_table_bytes()",
        ]),
        PatternCheck::new("J2K Metal typed HT UVLC upload", &runtime)
            .required(&[
                "(*ht_uvlc_encode_table()).map(J2kHtUvlcEncodeTableEntry::from)",
                "checked_shared_buffer_with_slice(",
                "&ht_uvlc_encode_rows",
            ])
            .forbidden(&[
                "ht_uvlc_encode_table_bytes",
                "checked_shared_buffer_with_bytes",
            ]),
        PatternCheck::new("J2K Metal byte-addressed HT UVLC shader ABI", &shader)
            .required(&["return table[index * 6u + field];"]),
    ]);
}
