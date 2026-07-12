// SPDX-License-Identifier: MIT OR Apache-2.0

//! Incomplete progressive scan metadata that becomes retained only at a boundary.

use super::super::types::ParsedProgressiveScan;
use crate::parse::scan::ParsedScan;
use crate::parse::tables::ProgressiveTableState;

pub(super) struct PendingProgressiveScan {
    scan: ParsedScan,
    entropy_offset: usize,
    table_state: ProgressiveTableState,
    restart_interval: Option<u16>,
}

impl PendingProgressiveScan {
    pub(super) fn new(
        scan: ParsedScan,
        entropy_offset: usize,
        table_state: ProgressiveTableState,
        restart_interval: Option<u16>,
    ) -> Self {
        Self {
            scan,
            entropy_offset,
            table_state,
            restart_interval,
        }
    }

    pub(super) fn finish(self, terminal_offset: usize, terminal_code: u8) -> ParsedProgressiveScan {
        ParsedProgressiveScan {
            scan: self.scan,
            entropy_offset: self.entropy_offset,
            terminal_offset,
            terminal_code,
            table_state: self.table_state,
            restart_interval: self.restart_interval,
        }
    }
}
