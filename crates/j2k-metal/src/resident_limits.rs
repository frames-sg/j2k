// SPDX-License-Identifier: MIT OR Apache-2.0

/// Default resident chunk used to size warm private-buffer retention.
pub(crate) const DEFAULT_RESIDENT_CHUNK_TILES: usize = 512;

const MAX_RESIDENT_COMPONENTS: usize = 3;
const PRIVATE_COMPONENT_PLANES_PER_TILE: usize = MAX_RESIDENT_COMPONENTS;
const PRIVATE_DWT_SCRATCH_PER_COMPONENT: usize = 1;
const MAX_PRIVATE_BUFFERS_PER_RESIDENT_TILE: usize =
    PRIVATE_COMPONENT_PLANES_PER_TILE + MAX_RESIDENT_COMPONENTS * PRIVATE_DWT_SCRATCH_PER_COMPONENT;

const BASE_PRIVATE_BUFFERS_PER_RESIDENT_BATCH: usize = 7;
const CLASSIC_SPLIT_TOKEN_PRIVATE_BUFFERS_PER_BATCH: usize = 4;
const MAX_PRIVATE_BUFFERS_PER_RESIDENT_BATCH: usize =
    BASE_PRIVATE_BUFFERS_PER_RESIDENT_BATCH + CLASSIC_SPLIT_TOKEN_PRIVATE_BUFFERS_PER_BATCH;

pub(crate) const DEFAULT_RESIDENT_PRIVATE_WORKING_SET_BUFFERS: usize = DEFAULT_RESIDENT_CHUNK_TILES
    * MAX_PRIVATE_BUFFERS_PER_RESIDENT_TILE
    + MAX_PRIVATE_BUFFERS_PER_RESIDENT_BATCH;

const fn private_pool_record_limit(working_set: usize) -> usize {
    working_set.next_power_of_two()
}

/// Power-of-two structural ceiling above the default resident working set.
pub(crate) const RESIDENT_PRIVATE_POOL_BUFFER_LIMIT: usize =
    private_pool_record_limit(DEFAULT_RESIDENT_PRIVATE_WORKING_SET_BUFFERS);

#[cfg(test)]
mod tests {
    use super::{
        private_pool_record_limit, DEFAULT_RESIDENT_PRIVATE_WORKING_SET_BUFFERS,
        RESIDENT_PRIVATE_POOL_BUFFER_LIMIT,
    };

    #[test]
    fn private_pool_record_limit_covers_default_resident_chunk_working_set() {
        assert_eq!(DEFAULT_RESIDENT_PRIVATE_WORKING_SET_BUFFERS, 3_083);
        assert_eq!(RESIDENT_PRIVATE_POOL_BUFFER_LIMIT, 4_096);
        assert_eq!(
            private_pool_record_limit(std::hint::black_box(3_083)),
            4_096
        );
    }
}
