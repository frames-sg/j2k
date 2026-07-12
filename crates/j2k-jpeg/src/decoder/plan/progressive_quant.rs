// SPDX-License-Identifier: MIT OR Apache-2.0

//! First-use quantization-table binding for progressive frame components.

use crate::error::{JpegError, MarkerKind};
use crate::parse::header::ParsedHeader;

const MAX_FRAME_COMPONENTS: usize = 4;

pub(super) struct LatchedProgressiveQuantTables {
    tables: [[u16; 64]; MAX_FRAME_COMPONENTS],
}

impl LatchedProgressiveQuantTables {
    pub(super) fn table(&self, component_index: usize) -> Option<[u16; 64]> {
        self.tables.get(component_index).copied()
    }
}

pub(super) fn latch_progressive_quant_tables(
    header: &ParsedHeader,
) -> Result<LatchedProgressiveQuantTables, JpegError> {
    let mut tables = [[0; 64]; MAX_FRAME_COMPONENTS];
    let mut latched = [false; MAX_FRAME_COMPONENTS];

    for parsed in &header.progressive_scans {
        for scan_component in &parsed.scan.components {
            let component_index = header
                .component_ids
                .iter()
                .position(|&id| id == scan_component.id)
                .ok_or(JpegError::UnknownScanComponent {
                    offset: parsed.entropy_offset,
                    component: scan_component.id,
                })?;
            let table_id =
                *header
                    .quant_table_ids
                    .get(component_index)
                    .ok_or(JpegError::MissingMarker {
                        marker: MarkerKind::Sof,
                    })?;
            let resolved = *header
                .quant_tables
                .resolve(&parsed.table_state, table_id)
                .ok_or(JpegError::MissingQuantTable {
                    component: scan_component.id,
                    table_id,
                })?;
            if latched[component_index] && tables[component_index] != resolved {
                return Err(JpegError::ProgressiveQuantTableChanged {
                    offset: parsed.entropy_offset,
                    component: scan_component.id,
                    table_id,
                });
            }
            if !latched[component_index] {
                tables[component_index] = resolved;
                latched[component_index] = true;
            }
        }
    }

    for (component_index, &component) in header.component_ids.iter().enumerate() {
        if !latched[component_index] {
            let table_id = header.quant_table_ids[component_index];
            return Err(JpegError::MissingQuantTable {
                component,
                table_id,
            });
        }
    }
    Ok(LatchedProgressiveQuantTables { tables })
}

#[cfg(test)]
mod tests;
