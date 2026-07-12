// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    bytes::u16_slice_as_bytes,
    context::{CudaContext, HTJ2K_UVLC_ENCODE_TABLE_BYTES},
    error::CudaError,
};

use super::super::types::{CudaHtj2kEncodeResources, CudaHtj2kEncodeTables};

impl CudaContext {
    /// Upload static HTJ2K cleanup encoder lookup tables once for reuse.
    #[doc(hidden)]
    pub fn upload_htj2k_encode_resources(
        &self,
        tables: CudaHtj2kEncodeTables<'_>,
    ) -> Result<CudaHtj2kEncodeResources, CudaError> {
        if tables.uvlc_table.len() != HTJ2K_UVLC_ENCODE_TABLE_BYTES {
            return Err(CudaError::LengthTooLarge {
                len: tables.uvlc_table.len(),
            });
        }
        self.inner.set_current()?;
        Ok(CudaHtj2kEncodeResources {
            vlc_table0: self.upload(u16_slice_as_bytes(tables.vlc_table0))?,
            vlc_table1: self.upload(u16_slice_as_bytes(tables.vlc_table1))?,
            uvlc_table: self.upload(tables.uvlc_table)?,
        })
    }
}
