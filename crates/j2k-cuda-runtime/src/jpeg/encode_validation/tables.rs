// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    allocation::HostPhaseBudget,
    error::CudaError,
    jpeg::{CudaJpegBaselineEncodeHuffmanTable, CudaJpegBaselineEncodeParams},
};

use super::invalid_request;

#[derive(Clone, Copy)]
pub(in crate::jpeg) struct CudaJpegBaselineEncodeTableRefs<'a> {
    pub(in crate::jpeg) q_luma: &'a [u8; 64],
    pub(in crate::jpeg) q_chroma: &'a [u8; 64],
    pub(in crate::jpeg) huff_dc_luma: &'a CudaJpegBaselineEncodeHuffmanTable,
    pub(in crate::jpeg) huff_ac_luma: &'a CudaJpegBaselineEncodeHuffmanTable,
    pub(in crate::jpeg) huff_dc_chroma: &'a CudaJpegBaselineEncodeHuffmanTable,
    pub(in crate::jpeg) huff_ac_chroma: &'a CudaJpegBaselineEncodeHuffmanTable,
}

#[cfg(any(feature = "cuda-oxide-jpeg-encode", test))]
pub(in crate::jpeg) fn jpeg_encode_table_validation_host_bytes() -> usize {
    crate::allocation::host_element_bytes::<(u8, u32, usize)>(256)
}

fn validate_quant_table(name: &str, table: &[u8; 64]) -> Result<(), CudaError> {
    if let Some(index) = table.iter().position(|value| *value == 0) {
        return Err(invalid_request(format!(
            "JPEG CUDA encode {name} quantization table entry {index} must be nonzero"
        )));
    }
    Ok(())
}

fn validate_huffman_table(
    name: &str,
    table: &CudaJpegBaselineEncodeHuffmanTable,
    retained_host_bytes: usize,
) -> Result<(), CudaError> {
    let mut host_budget = HostPhaseBudget::new("JPEG baseline Huffman table validation");
    host_budget.account_bytes(retained_host_bytes)?;
    let mut entries = host_budget.try_vec_with_capacity(table.lens.len())?;
    for (symbol, (&code, &len)) in table.codes.iter().zip(&table.lens).enumerate() {
        if len == 0 {
            continue;
        }
        if len > 16 {
            return Err(invalid_request(format!(
                "JPEG CUDA encode {name} Huffman symbol {symbol} has code length {len}, above 16"
            )));
        }
        let code = u32::from(code);
        let code_space = 1u32 << u32::from(len);
        if code >= code_space {
            return Err(invalid_request(format!(
                "JPEG CUDA encode {name} Huffman symbol {symbol} code does not fit length {len}"
            )));
        }
        if code == code_space - 1 {
            return Err(invalid_request(format!(
                "JPEG CUDA encode {name} Huffman symbol {symbol} uses a JPEG-prohibited all-ones code"
            )));
        }
        entries.push((len, code, symbol));
    }

    entries.sort_unstable();
    let mut bits = [0u8; 16];
    for &(len, _, symbol) in &entries {
        let count = &mut bits[usize::from(len - 1)];
        *count = (*count).checked_add(1).ok_or_else(|| {
            invalid_request(format!(
                "JPEG CUDA encode {name} Huffman length {len} has too many symbols at symbol {symbol}"
            ))
        })?;
    }
    let canonical = j2k_codec_math::jpeg::derive_canonical_huffman(&bits, entries.len())
        .map_err(|error| invalid_request(format!("JPEG CUDA encode {name} Huffman {error}")))?;
    for (index, &(len, code, symbol)) in entries.iter().enumerate() {
        if canonical.huffsize[index] != len || u32::from(canonical.huffcode[index]) != code {
            return Err(invalid_request(format!(
                "JPEG CUDA encode {name} Huffman symbol {symbol} does not follow canonical prefix-free code progression"
            )));
        }
    }
    Ok(())
}

pub(super) fn validate_encode_tables(
    params: &[CudaJpegBaselineEncodeParams],
    tables: CudaJpegBaselineEncodeTableRefs<'_>,
    retained_host_bytes: usize,
) -> Result<(), CudaError> {
    validate_quant_table("luma", tables.q_luma)?;
    validate_huffman_table("luma DC", tables.huff_dc_luma, retained_host_bytes)?;
    validate_huffman_table("luma AC", tables.huff_ac_luma, retained_host_bytes)?;
    if params.iter().any(|params| params.components == 3) {
        validate_quant_table("chroma", tables.q_chroma)?;
        validate_huffman_table("chroma DC", tables.huff_dc_chroma, retained_host_bytes)?;
        validate_huffman_table("chroma AC", tables.huff_ac_chroma, retained_host_bytes)?;
    }
    Ok(())
}
