// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    error::CudaError,
    jpeg::{CudaJpegChunkedEntropyPlan, CudaJpegHuffmanTable, CudaJpegRgb8DecodePlan},
};

const JPEG_HUFFMAN_CAPACITY: u32 = 256;

#[derive(Clone, Copy)]
enum HuffmanRole {
    Dc,
    Ac,
}

pub(super) fn validate_rgb8_huffman_tables(
    plan: &CudaJpegRgb8DecodePlan<'_>,
) -> Result<(), CudaError> {
    validate_huffman_table(&plan.y_dc_table, HuffmanRole::Dc, "Y DC")?;
    validate_huffman_table(&plan.y_ac_table, HuffmanRole::Ac, "Y AC")?;
    validate_huffman_table(&plan.cb_dc_table, HuffmanRole::Dc, "Cb DC")?;
    validate_huffman_table(&plan.cb_ac_table, HuffmanRole::Ac, "Cb AC")?;
    validate_huffman_table(&plan.cr_dc_table, HuffmanRole::Dc, "Cr DC")?;
    validate_huffman_table(&plan.cr_ac_table, HuffmanRole::Ac, "Cr AC")
}

pub(super) fn validate_entropy_huffman_tables(
    plan: &CudaJpegChunkedEntropyPlan<'_>,
) -> Result<(), CudaError> {
    validate_huffman_table(&plan.y_dc_table, HuffmanRole::Dc, "Y DC")?;
    validate_huffman_table(&plan.y_ac_table, HuffmanRole::Ac, "Y AC")?;
    validate_huffman_table(&plan.cb_dc_table, HuffmanRole::Dc, "Cb DC")?;
    validate_huffman_table(&plan.cb_ac_table, HuffmanRole::Ac, "Cb AC")?;
    validate_huffman_table(&plan.cr_dc_table, HuffmanRole::Dc, "Cr DC")?;
    validate_huffman_table(&plan.cr_ac_table, HuffmanRole::Ac, "Cr AC")
}

fn validate_huffman_table(
    table: &CudaJpegHuffmanTable,
    role: HuffmanRole,
    label: &str,
) -> Result<(), CudaError> {
    if table.values_len == 0 || table.values_len > JPEG_HUFFMAN_CAPACITY {
        return Err(invalid_huffman(
            label,
            format_args!(
                "value count {} is outside the supported range 1..={JPEG_HUFFMAN_CAPACITY}",
                table.values_len
            ),
        ));
    }
    if table.max_code[0] != -1 || table.val_offset[0] != 0 {
        return Err(invalid_huffman(label, "index-zero sentinel is invalid"));
    }

    let values_len = usize::try_from(table.values_len)
        .map_err(|_| invalid_huffman(label, "value count cannot be represented on this host"))?;
    let mut next_code = 0i64;
    let mut value_cursor = 0i64;
    for len in 1usize..=16 {
        let max_code = i64::from(table.max_code[len]);
        let val_offset = i64::from(table.val_offset[len]);
        if max_code == -1 {
            if val_offset != 0 {
                return Err(invalid_huffman(
                    label,
                    format_args!("length {len} has no codes but a nonzero value offset"),
                ));
            }
        } else {
            let code_limit = 1i64 << len;
            if max_code < next_code || max_code >= code_limit {
                return Err(invalid_huffman(
                    label,
                    format_args!("length {len} has non-canonical code bounds"),
                ));
            }
            if max_code == code_limit - 1 {
                return Err(invalid_huffman(
                    label,
                    format_args!("length {len} assigns the forbidden all-ones code"),
                ));
            }
            if val_offset != value_cursor - next_code {
                return Err(invalid_huffman(
                    label,
                    format_args!("length {len} has a non-canonical value offset"),
                ));
            }
            value_cursor += max_code - next_code + 1;
            if value_cursor > i64::from(table.values_len) {
                return Err(invalid_huffman(
                    label,
                    format_args!("length {len} addresses beyond the declared values"),
                ));
            }
            next_code = max_code + 1;
        }
        next_code <<= 1;
    }
    if value_cursor != i64::from(table.values_len) {
        return Err(invalid_huffman(
            label,
            "canonical code count does not match the declared value count",
        ));
    }

    for (index, &symbol) in table.values[..values_len].iter().enumerate() {
        let valid = match role {
            HuffmanRole::Dc => symbol <= 11,
            HuffmanRole::Ac => {
                let size = symbol & 0x0f;
                size <= 10 && (size != 0 || matches!(symbol, 0x00 | 0xf0))
            }
        };
        if !valid {
            return Err(invalid_huffman(
                label,
                format_args!("symbol {index} has invalid baseline value 0x{symbol:02x}"),
            ));
        }
    }
    Ok(())
}

fn invalid_huffman(label: &str, reason: impl std::fmt::Display) -> CudaError {
    CudaError::InvalidArgument {
        message: format!("JPEG CUDA {label} Huffman table {reason}"),
    }
}
