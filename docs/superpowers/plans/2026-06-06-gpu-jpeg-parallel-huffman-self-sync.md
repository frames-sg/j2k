# GPU JPEG Parallel Huffman Self-Sync Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an experimental CUDA diagnostic path that tests JPEG Huffman self-synchronization on generated baseline 4:2:0 JPEG entropy streams.

**Architecture:** Keep the production owned CUDA decode path unchanged. Add a separate experimental chunked entropy diagnostic API in `signinum-cuda-runtime`, expose it through a narrow `signinum-jpeg-cuda::Codec` method, and validate it with generated 4:2:0 JPEGs before any coefficient-write or routing changes. The first pass measures whether GPU arbitrary-bit subsequence synchronization is viable.

**Tech Stack:** Rust, CUDA C++, Signinum JPEG fast packets, Signinum CUDA Driver API runtime, generated baseline JPEG fixtures, Criterion for benchmark timing.

---

## Files

- Modify: `crates/signinum-cuda-runtime/src/lib.rs`
  Add chunk config/report structs, validation helpers, upload/launch/download methods, and tests.
- Modify: `crates/signinum-cuda-runtime/src/kernels.rs`
  Register new diagnostic kernel entrypoints.
- Modify: `crates/signinum-cuda-runtime/src/jpeg_decode_kernels.cu`
  Add self-sync diagnostic kernels that reuse the existing JPEG Huffman helpers.
- Modify: `crates/signinum-jpeg-cuda/src/owned_decode.rs`
  Build a `CudaJpegChunkedEntropyPlan` from a cached 4:2:0 packet.
- Modify: `crates/signinum-jpeg-cuda/src/codec.rs`
  Expose `Codec::diagnose_tile_rgb8_chunked_entropy_with_session`.
- Modify: `crates/signinum-jpeg-cuda/tests/host_surface.rs`
  Add runtime-gated generated 4:2:0 diagnostic tests.
- Modify: `crates/signinum-jpeg-cuda/benches/device_decode.rs`
  Add a narrow benchmark for CPU checkpoint planning vs GPU sync diagnostics.
- Modify: `docs/bench.md`
  Document the diagnostic env knobs and expected interpretation.
- Modify: `docs/stable-api-1.0.public-api.txt`
  Regenerate via `cargo xtask stable-api --write` if public API changes are accepted.

---

## Task 1: Host-Side Chunk Plan Math

**Files:**
- Modify: `crates/signinum-cuda-runtime/src/lib.rs`

- [ ] **Step 1: Write failing tests for chunk math**

Add tests near the existing CUDA runtime unit tests in `crates/signinum-cuda-runtime/src/lib.rs`:

```rust
#[test]
fn jpeg_chunked_entropy_config_counts_bit_subsequences() {
    let config = CudaJpegChunkedEntropyConfig {
        subsequence_words: 4,
        sequence_len: 8,
        max_overflow_subsequences: 2,
    };

    assert_eq!(config.subsequence_bits(), 128);
    assert_eq!(config.subsequence_count_for_entropy_bytes(0).unwrap(), 0);
    assert_eq!(config.subsequence_count_for_entropy_bytes(1).unwrap(), 1);
    assert_eq!(config.subsequence_count_for_entropy_bytes(16).unwrap(), 1);
    assert_eq!(config.subsequence_count_for_entropy_bytes(17).unwrap(), 2);
}

#[test]
fn jpeg_chunked_entropy_config_rejects_zero_subsequence_or_sequence() {
    let zero_words = CudaJpegChunkedEntropyConfig {
        subsequence_words: 0,
        ..CudaJpegChunkedEntropyConfig::default()
    };
    let zero_sequence = CudaJpegChunkedEntropyConfig {
        sequence_len: 0,
        ..CudaJpegChunkedEntropyConfig::default()
    };

    assert!(zero_words.validate().is_err());
    assert!(zero_sequence.validate().is_err());
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cargo test -p signinum-cuda-runtime jpeg_chunked_entropy_config --lib
```

Expected: FAIL with missing `CudaJpegChunkedEntropyConfig`.

- [ ] **Step 3: Add minimal config/report host types**

Add near `CudaJpegRgb8Sampling` in `crates/signinum-cuda-runtime/src/lib.rs`:

```rust
/// Experimental JPEG entropy chunking parameters for CUDA self-sync diagnostics.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaJpegChunkedEntropyConfig {
    /// Subsequence size in 32-bit words.
    pub subsequence_words: u32,
    /// Number of adjacent subsequences handled as one synchronization sequence.
    pub sequence_len: u32,
    /// Maximum adjacent subsequences an overflow decoder may scan.
    pub max_overflow_subsequences: u32,
}

impl Default for CudaJpegChunkedEntropyConfig {
    fn default() -> Self {
        Self {
            subsequence_words: 1024,
            sequence_len: 128,
            max_overflow_subsequences: 4,
        }
    }
}

impl CudaJpegChunkedEntropyConfig {
    /// Return one subsequence size in bits.
    pub fn subsequence_bits(self) -> u32 {
        self.subsequence_words.saturating_mul(32)
    }

    /// Validate parameters before launching diagnostic kernels.
    pub fn validate(self) -> Result<(), CudaError> {
        if self.subsequence_words == 0 {
            return Err(CudaError::InvalidArgument {
                message: "JPEG entropy subsequence_words must be nonzero".to_string(),
            });
        }
        if self.sequence_len == 0 {
            return Err(CudaError::InvalidArgument {
                message: "JPEG entropy sequence_len must be nonzero".to_string(),
            });
        }
        Ok(())
    }

    /// Count fixed-size bit subsequences needed for an entropy payload.
    pub fn subsequence_count_for_entropy_bytes(self, entropy_len: usize) -> Result<usize, CudaError> {
        self.validate()?;
        let entropy_bits = entropy_len.checked_mul(8).ok_or(CudaError::LengthTooLarge {
            len: entropy_len,
        })?;
        let bits = self.subsequence_bits() as usize;
        Ok(entropy_bits.div_ceil(bits))
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
cargo test -p signinum-cuda-runtime jpeg_chunked_entropy_config --lib
```

Expected: PASS.

---

## Task 2: Runtime ABI for Diagnostic Sync States

**Files:**
- Modify: `crates/signinum-cuda-runtime/src/lib.rs`
- Modify: `crates/signinum-cuda-runtime/src/kernels.rs`

- [ ] **Step 1: Write failing tests for report summary**

Add tests in `crates/signinum-cuda-runtime/src/lib.rs`:

```rust
#[test]
fn jpeg_chunked_entropy_report_summarizes_sync_quality() {
    let report = CudaJpegChunkedEntropyReport {
        config: CudaJpegChunkedEntropyConfig {
            subsequence_words: 4,
            sequence_len: 8,
            max_overflow_subsequences: 2,
        },
        entropy_bytes: 4096,
        states: vec![
            CudaJpegEntropySyncState {
                code: 0,
                start_bit: 0,
                end_bit: 128,
                bit_pos: 128,
                symbol_count: 10,
                block_phase: 0,
                zigzag_index: 0,
                reserved: 0,
            },
            CudaJpegEntropySyncState {
                code: 0,
                start_bit: 128,
                end_bit: 256,
                bit_pos: 256,
                symbol_count: 9,
                block_phase: 3,
                zigzag_index: 12,
                reserved: 0,
            },
        ],
        overflows: vec![CudaJpegEntropyOverflowState {
            code: 0,
            from_subsequence: 0,
            to_subsequence: 1,
            overflow_bits: 96,
            synchronized: 1,
            reserved: [0; 3],
        }],
        execution: CudaExecutionStats {
            kernel_dispatches: 2,
            copy_kernel_dispatches: 0,
            decode_kernel_dispatches: 0,
            hardware_decode: false,
        },
    };

    assert_eq!(report.subsequence_count(), 2);
    assert_eq!(report.synchronized_overflow_count(), 1);
    assert_eq!(report.max_overflow_bits(), Some(96));
    assert_eq!(report.failed_state_count(), 0);
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cargo test -p signinum-cuda-runtime jpeg_chunked_entropy_report --lib
```

Expected: FAIL with missing report/state types.

- [ ] **Step 3: Add repr(C) state and report types**

Add near the config type:

```rust
/// Device-written state for one entropy subsequence self-sync diagnostic.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CudaJpegEntropySyncState {
    /// Zero means success; nonzero maps to diagnostic kernel status.
    pub code: u32,
    /// Subsequence start bit offset.
    pub start_bit: u32,
    /// Subsequence exclusive end bit offset.
    pub end_bit: u32,
    /// Decoder bit position after scanning this subsequence.
    pub bit_pos: u32,
    /// Decoded coefficient-slot count.
    pub symbol_count: u32,
    /// 4:2:0 block phase: 0..=3 for Y blocks, 4 Cb, 5 Cr.
    pub block_phase: u32,
    /// Zig-zag coefficient index inside the current block.
    pub zigzag_index: u32,
    /// Reserved for ABI-compatible expansion.
    pub reserved: u32,
}

/// Device-written overflow result for adjacent subsequence synchronization.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CudaJpegEntropyOverflowState {
    /// Zero means success; nonzero maps to diagnostic kernel status.
    pub code: u32,
    /// Source subsequence index.
    pub from_subsequence: u32,
    /// Target subsequence index.
    pub to_subsequence: u32,
    /// Bits scanned after the target subsequence start before synchronization.
    pub overflow_bits: u32,
    /// One when synchronization was detected.
    pub synchronized: u32,
    /// Reserved for ABI-compatible expansion.
    pub reserved: [u32; 3],
}

/// Host-side report returned by experimental JPEG entropy self-sync diagnostics.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaJpegChunkedEntropyReport {
    /// Diagnostic chunk configuration.
    pub config: CudaJpegChunkedEntropyConfig,
    /// Entropy payload length in bytes.
    pub entropy_bytes: usize,
    /// Per-subsequence first-pass states.
    pub states: Vec<CudaJpegEntropySyncState>,
    /// Per-adjacent-subsequence overflow states.
    pub overflows: Vec<CudaJpegEntropyOverflowState>,
    /// Runtime dispatch stats for diagnostic kernels.
    pub execution: CudaExecutionStats,
}

impl CudaJpegChunkedEntropyReport {
    /// Number of subsequences examined.
    pub fn subsequence_count(&self) -> usize {
        self.states.len()
    }

    /// Number of overflow records that synchronized.
    pub fn synchronized_overflow_count(&self) -> usize {
        self.overflows
            .iter()
            .filter(|overflow| overflow.synchronized != 0)
            .count()
    }

    /// Maximum overflow scan length in bits.
    pub fn max_overflow_bits(&self) -> Option<u32> {
        self.overflows.iter().map(|overflow| overflow.overflow_bits).max()
    }

    /// Number of first-pass states with nonzero status.
    pub fn failed_state_count(&self) -> usize {
        self.states.iter().filter(|state| state.code != 0).count()
    }
}
```

- [ ] **Step 4: Add kernel enum variants**

In `crates/signinum-cuda-runtime/src/kernels.rs`, add variants:

```rust
#[cfg_attr(not(signinum_cuda_jpeg_decode_ptx_built), allow(dead_code))]
JpegEntropySync420,
#[cfg_attr(not(signinum_cuda_jpeg_decode_ptx_built), allow(dead_code))]
JpegEntropyOverflow420,
```

Map both variants to `JPEG_DECODE_PTX`, with entrypoints:

```rust
Self::JpegEntropySync420 => b"signinum_jpeg_entropy_sync420\0",
Self::JpegEntropyOverflow420 => b"signinum_jpeg_entropy_overflow420\0",
```

- [ ] **Step 5: Run tests**

Run:

```bash
cargo test -p signinum-cuda-runtime jpeg_chunked_entropy_report --lib
cargo test -p signinum-cuda-runtime kernels::tests::htj2k_decode_kernel_metadata_matches_generated_ptx --lib
```

Expected: report test PASS; PTX metadata test may fail until Task 3 adds CUDA entrypoints.

---

## Task 3: CUDA First-Pass Self-Sync Kernel

**Files:**
- Modify: `crates/signinum-cuda-runtime/src/jpeg_decode_kernels.cu`
- Modify: `crates/signinum-cuda-runtime/src/lib.rs`

- [ ] **Step 1: Write compile/runtime test shell in Rust**

Add a runtime-gated unit test in `crates/signinum-cuda-runtime/src/lib.rs`:

```rust
#[test]
fn jpeg_entropy_self_sync_returns_empty_report_for_empty_entropy_when_runtime_required() {
    if std::env::var_os("SIGNINUM_REQUIRE_CUDA_RUNTIME").is_none() {
        return;
    }

    let context = CudaContext::system_default().expect("cuda context");
    let plan = CudaJpegChunkedEntropyPlan {
        config: CudaJpegChunkedEntropyConfig::default(),
        entropy_bytes: &[],
        y_dc_table: CudaJpegHuffmanTable::from_jpeg_bits_values([0; 16], 0, [0; 256])
            .expect("empty huffman table"),
        y_ac_table: CudaJpegHuffmanTable::from_jpeg_bits_values([0; 16], 0, [0; 256])
            .expect("empty huffman table"),
        cb_dc_table: CudaJpegHuffmanTable::from_jpeg_bits_values([0; 16], 0, [0; 256])
            .expect("empty huffman table"),
        cb_ac_table: CudaJpegHuffmanTable::from_jpeg_bits_values([0; 16], 0, [0; 256])
            .expect("empty huffman table"),
        cr_dc_table: CudaJpegHuffmanTable::from_jpeg_bits_values([0; 16], 0, [0; 256])
            .expect("empty huffman table"),
        cr_ac_table: CudaJpegHuffmanTable::from_jpeg_bits_values([0; 16], 0, [0; 256])
            .expect("empty huffman table"),
    };

    let report = context
        .diagnose_jpeg_420_entropy_self_sync(&plan)
        .expect("empty diagnostic report");
    assert_eq!(report.subsequence_count(), 0);
    assert_eq!(report.overflows.len(), 0);
}
```

- [ ] **Step 2: Add the diagnostic plan type**

Add near `CudaJpegRgb8DecodePlan`:

```rust
/// Experimental Signinum-owned CUDA JPEG entropy self-sync diagnostic plan.
#[derive(Debug)]
pub struct CudaJpegChunkedEntropyPlan<'a> {
    /// Chunking configuration.
    pub config: CudaJpegChunkedEntropyConfig,
    /// Entropy-coded scan payload with byte stuffing/restart markers removed.
    pub entropy_bytes: &'a [u8],
    /// Y DC Huffman table.
    pub y_dc_table: CudaJpegHuffmanTable,
    /// Y AC Huffman table.
    pub y_ac_table: CudaJpegHuffmanTable,
    /// Cb DC Huffman table.
    pub cb_dc_table: CudaJpegHuffmanTable,
    /// Cb AC Huffman table.
    pub cb_ac_table: CudaJpegHuffmanTable,
    /// Cr DC Huffman table.
    pub cr_dc_table: CudaJpegHuffmanTable,
    /// Cr AC Huffman table.
    pub cr_ac_table: CudaJpegHuffmanTable,
}
```

- [ ] **Step 3: Add the empty-report runtime method**

Add on `impl CudaContext`:

```rust
/// Run experimental 4:2:0 JPEG entropy self-sync diagnostics.
pub fn diagnose_jpeg_420_entropy_self_sync(
    &self,
    plan: &CudaJpegChunkedEntropyPlan<'_>,
) -> Result<CudaJpegChunkedEntropyReport, CudaError> {
    plan.config.validate()?;
    let subsequences = plan
        .config
        .subsequence_count_for_entropy_bytes(plan.entropy_bytes.len())?;
    if subsequences == 0 {
        return Ok(CudaJpegChunkedEntropyReport {
            config: plan.config,
            entropy_bytes: plan.entropy_bytes.len(),
            states: Vec::new(),
            overflows: Vec::new(),
            execution: CudaExecutionStats {
                kernel_dispatches: 0,
                copy_kernel_dispatches: 0,
                decode_kernel_dispatches: 0,
                hardware_decode: false,
            },
        });
    }

    #[cfg(not(signinum_cuda_jpeg_decode_ptx_built))]
    {
        let _ = subsequences;
        Err(CudaError::InvalidArgument {
            message: "Signinum CUDA JPEG decode PTX was not built from jpeg_decode_kernels.cu"
                .to_string(),
        })
    }

    #[cfg(signinum_cuda_jpeg_decode_ptx_built)]
    {
        self.diagnose_jpeg_420_entropy_self_sync_nonempty(plan, subsequences)
    }
}
```

- [ ] **Step 4: Add CUDA ABI structs and helpers**

In `jpeg_decode_kernels.cu`, add ABI structs matching Rust:

```cpp
struct SigninumJpegEntropyChunkParams {
    unsigned int entropy_len;
    unsigned int entropy_bits;
    unsigned int subsequence_bits;
    unsigned int subsequence_count;
    unsigned int sequence_len;
    unsigned int max_overflow_subsequences;
    unsigned int reserved0;
    unsigned int reserved1;
};

struct SigninumJpegEntropySyncState {
    unsigned int code;
    unsigned int start_bit;
    unsigned int end_bit;
    unsigned int bit_pos;
    unsigned int symbol_count;
    unsigned int block_phase;
    unsigned int zigzag_index;
    unsigned int reserved;
};

struct SigninumJpegEntropyOverflowState {
    unsigned int code;
    unsigned int from_subsequence;
    unsigned int to_subsequence;
    unsigned int overflow_bits;
    unsigned int synchronized;
    unsigned int reserved[3];
};
```

Add bit-reader initializer:

```cpp
__device__ SigninumJpegBitReader signinum_jpeg_bit_reader_at_bit(
    const unsigned char *entropy,
    unsigned int entropy_len,
    unsigned int bit_pos
) {
    SigninumJpegBitReader reader;
    reader.pos = bit_pos / 8u;
    reader.acc = 0ull;
    reader.bits = 0u;
    const unsigned int skip = bit_pos & 7u;
    if (skip != 0u && reader.pos < entropy_len) {
        reader.acc = static_cast<unsigned long long>(entropy[reader.pos]) << 56u;
        reader.pos += 1u;
        reader.bits = 8u;
        signinum_jpeg_consume_bits(reader, skip);
    }
    return reader;
}
```

- [ ] **Step 5: Add first-pass sync kernel**

Add kernel:

```cpp
extern "C" __global__ void signinum_jpeg_entropy_sync420(
    const unsigned char *entropy,
    SigninumJpegEntropyChunkParams params,
    const SigninumJpegHuffmanTable *y_dc,
    const SigninumJpegHuffmanTable *y_ac,
    const SigninumJpegHuffmanTable *cb_dc,
    const SigninumJpegHuffmanTable *cb_ac,
    const SigninumJpegHuffmanTable *cr_dc,
    const SigninumJpegHuffmanTable *cr_ac,
    SigninumJpegEntropySyncState *states
) {
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= params.subsequence_count) {
        return;
    }

    SigninumJpegEntropySyncState state;
    state.code = JPEG_STATUS_OK;
    state.start_bit = gid * params.subsequence_bits;
    state.end_bit = min(state.start_bit + params.subsequence_bits, params.entropy_bits);
    state.bit_pos = state.start_bit;
    state.symbol_count = 0u;
    state.block_phase = 0u;
    state.zigzag_index = 0u;
    state.reserved = 0u;

    SigninumJpegBitReader reader =
        signinum_jpeg_bit_reader_at_bit(entropy, params.entropy_len, state.start_bit);
    SigninumJpegDecodeStatus status;
    status.code = JPEG_STATUS_OK;
    status.detail = 0u;
    status.position = 0u;
    status.reserved = 0u;

    while (state.bit_pos < state.end_bit && status.code == JPEG_STATUS_OK) {
        const bool dc = state.zigzag_index == 0u;
        const SigninumJpegHuffmanTable *table =
            state.block_phase < 4u
                ? (dc ? y_dc : y_ac)
                : (state.block_phase == 4u ? (dc ? cb_dc : cb_ac) : (dc ? cr_dc : cr_ac));
        unsigned char symbol = 0u;
        const unsigned int before_pos = reader.pos;
        const unsigned int before_bits = reader.bits;
        if (!signinum_jpeg_decode_symbol(reader, entropy, params.entropy_len, table, &status, symbol)) {
            break;
        }
        unsigned int coeff_bits = dc ? symbol : (symbol & 0x0Fu);
        if (coeff_bits > 15u) {
            signinum_jpeg_set_error(&status, JPEG_STATUS_HUFFMAN, coeff_bits, reader.pos);
            break;
        }
        if (!signinum_jpeg_ensure_bits(reader, entropy, params.entropy_len, coeff_bits)) {
            signinum_jpeg_set_error(&status, JPEG_STATUS_TRUNCATED, coeff_bits, reader.pos);
            break;
        }
        signinum_jpeg_consume_bits(reader, coeff_bits);
        const unsigned int consumed = (reader.pos - before_pos) * 8u + before_bits - reader.bits;
        state.bit_pos += consumed;
        if (dc) {
            state.zigzag_index = 1u;
            state.symbol_count += 1u;
            continue;
        }
        const unsigned int run = symbol >> 4u;
        const unsigned int ssss = symbol & 0x0Fu;
        if (ssss == 0u && run != 15u) {
            state.symbol_count += 64u - state.zigzag_index;
            state.zigzag_index = 0u;
            state.block_phase = (state.block_phase + 1u) % 6u;
            continue;
        }
        state.zigzag_index += run + 1u;
        state.symbol_count += run + 1u;
        if (state.zigzag_index >= 64u) {
            state.zigzag_index = 0u;
            state.block_phase = (state.block_phase + 1u) % 6u;
        }
    }
    state.code = status.code;
    states[gid] = state;
}
```

- [ ] **Step 6: Implement Rust launch/download for first-pass only**

Add this Rust ABI struct near `CudaJpeg420Params`:

```rust
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[cfg_attr(not(signinum_cuda_jpeg_decode_ptx_built), allow(dead_code))]
struct CudaJpegEntropyChunkParams {
    entropy_len: u32,
    entropy_bits: u32,
    subsequence_bits: u32,
    subsequence_count: u32,
    sequence_len: u32,
    max_overflow_subsequences: u32,
    reserved0: u32,
    reserved1: u32,
}
```

Add state byte helpers near `cuda_jpeg_decode_statuses_as_bytes_mut`:

```rust
#[cfg_attr(not(signinum_cuda_jpeg_decode_ptx_built), allow(dead_code))]
fn cuda_jpeg_entropy_sync_states_as_bytes(states: &[CudaJpegEntropySyncState]) -> &[u8] {
    // SAFETY: CudaJpegEntropySyncState is repr(C), plain integer data copied to CUDA.
    unsafe {
        std::slice::from_raw_parts(
            states.as_ptr().cast::<u8>(),
            std::mem::size_of_val(states),
        )
    }
}

#[cfg_attr(not(signinum_cuda_jpeg_decode_ptx_built), allow(dead_code))]
fn cuda_jpeg_entropy_sync_states_as_bytes_mut(
    states: &mut [CudaJpegEntropySyncState],
) -> &mut [u8] {
    // SAFETY: the returned byte slice covers exactly the same initialized state storage.
    unsafe {
        std::slice::from_raw_parts_mut(
            states.as_mut_ptr().cast::<u8>(),
            std::mem::size_of_val(states),
        )
    }
}
```

Add validation and launch helpers on the existing CUDA JPEG runtime path:

```rust
#[cfg(signinum_cuda_jpeg_decode_ptx_built)]
fn validate_jpeg_entropy_chunk_plan(
    plan: &CudaJpegChunkedEntropyPlan<'_>,
    subsequences: usize,
) -> Result<CudaJpegEntropyChunkParams, CudaError> {
    let entropy_len =
        u32::try_from(plan.entropy_bytes.len()).map_err(|_| CudaError::LengthTooLarge {
            len: plan.entropy_bytes.len(),
        })?;
    let entropy_bits = entropy_len.checked_mul(8).ok_or(CudaError::LengthTooLarge {
        len: plan.entropy_bytes.len(),
    })?;
    let subsequence_count =
        u32::try_from(subsequences).map_err(|_| CudaError::LengthTooLarge { len: subsequences })?;

    Ok(CudaJpegEntropyChunkParams {
        entropy_len,
        entropy_bits,
        subsequence_bits: plan.config.subsequence_bits(),
        subsequence_count,
        sequence_len: plan.config.sequence_len,
        max_overflow_subsequences: plan.config.max_overflow_subsequences,
        reserved0: 0,
        reserved1: 0,
    })
}

#[cfg(signinum_cuda_jpeg_decode_ptx_built)]
#[allow(clippy::too_many_arguments)]
fn launch_jpeg_entropy_sync420(
    &self,
    entropy: &CudaDeviceBuffer,
    mut params: CudaJpegEntropyChunkParams,
    y_dc: &CudaDeviceBuffer,
    y_ac: &CudaDeviceBuffer,
    cb_dc: &CudaDeviceBuffer,
    cb_ac: &CudaDeviceBuffer,
    cr_dc: &CudaDeviceBuffer,
    cr_ac: &CudaDeviceBuffer,
    states: &CudaDeviceBuffer,
) -> Result<(), CudaError> {
    let function = self.inner.kernel_function(CudaKernel::JpegEntropySync420)?;
    let mut entropy_ptr = entropy.device_ptr();
    let mut y_dc_ptr = y_dc.device_ptr();
    let mut y_ac_ptr = y_ac.device_ptr();
    let mut cb_dc_ptr = cb_dc.device_ptr();
    let mut cb_ac_ptr = cb_ac.device_ptr();
    let mut cr_dc_ptr = cr_dc.device_ptr();
    let mut cr_ac_ptr = cr_ac.device_ptr();
    let mut states_ptr = states.device_ptr();
    let mut kernel_params = [
        (&raw mut entropy_ptr).cast::<c_void>(),
        (&raw mut params).cast::<c_void>(),
        (&raw mut y_dc_ptr).cast::<c_void>(),
        (&raw mut y_ac_ptr).cast::<c_void>(),
        (&raw mut cb_dc_ptr).cast::<c_void>(),
        (&raw mut cb_ac_ptr).cast::<c_void>(),
        (&raw mut cr_dc_ptr).cast::<c_void>(),
        (&raw mut cr_ac_ptr).cast::<c_void>(),
        (&raw mut states_ptr).cast::<c_void>(),
    ];
    let geometry = CudaLaunchGeometry {
        grid: (params.subsequence_count.div_ceil(128), 1, 1),
        block: (128, 1, 1),
    };

    self.launch_kernel(function, geometry, &mut kernel_params)
}
```

Add the nonempty method body:

```rust
#[cfg(signinum_cuda_jpeg_decode_ptx_built)]
fn diagnose_jpeg_420_entropy_self_sync_nonempty(
    &self,
    plan: &CudaJpegChunkedEntropyPlan<'_>,
    subsequences: usize,
) -> Result<CudaJpegChunkedEntropyReport, CudaError> {
    let params = validate_jpeg_entropy_chunk_plan(plan, subsequences)?;
    self.inner.set_current()?;
    let entropy = self.upload_pinned(plan.entropy_bytes)?;
    let y_dc = self.upload(cuda_jpeg_huffman_table_as_bytes(&plan.y_dc_table))?;
    let y_ac = self.upload(cuda_jpeg_huffman_table_as_bytes(&plan.y_ac_table))?;
    let cb_dc = self.upload(cuda_jpeg_huffman_table_as_bytes(&plan.cb_dc_table))?;
    let cb_ac = self.upload(cuda_jpeg_huffman_table_as_bytes(&plan.cb_ac_table))?;
    let cr_dc = self.upload(cuda_jpeg_huffman_table_as_bytes(&plan.cr_dc_table))?;
    let cr_ac = self.upload(cuda_jpeg_huffman_table_as_bytes(&plan.cr_ac_table))?;

    let mut states = vec![CudaJpegEntropySyncState::default(); subsequences];
    let states_buffer = self.upload(cuda_jpeg_entropy_sync_states_as_bytes(&states))?;
    self.launch_jpeg_entropy_sync420(
        &entropy,
        params,
        &y_dc,
        &y_ac,
        &cb_dc,
        &cb_ac,
        &cr_dc,
        &cr_ac,
        &states_buffer,
    )?;
    states_buffer.copy_to_host(cuda_jpeg_entropy_sync_states_as_bytes_mut(&mut states))?;

    Ok(CudaJpegChunkedEntropyReport {
        config: plan.config,
        entropy_bytes: plan.entropy_bytes.len(),
        states,
        overflows: Vec::new(),
        execution: CudaExecutionStats {
            kernel_dispatches: 1,
            copy_kernel_dispatches: 0,
            decode_kernel_dispatches: 0,
            hardware_decode: false,
        },
    })
}
```

Run:

```bash
cargo test -p signinum-cuda-runtime jpeg_entropy_self_sync_returns_empty_report --lib
cargo clippy -p signinum-cuda-runtime --tests -- -D warnings
```

Expected locally: PASS without CUDA env; remote runtime test should reach the method.

---

## Task 4: Overflow Synchronization Kernel

**Files:**
- Modify: `crates/signinum-cuda-runtime/src/jpeg_decode_kernels.cu`
- Modify: `crates/signinum-cuda-runtime/src/lib.rs`

- [ ] **Step 1: Write failing report test for overflow allocation**

Add a non-runtime test:

```rust
#[test]
fn jpeg_chunked_entropy_report_has_one_less_overflow_than_subsequence_count() {
    let config = CudaJpegChunkedEntropyConfig {
        subsequence_words: 1,
        sequence_len: 8,
        max_overflow_subsequences: 2,
    };
    let subsequences = config.subsequence_count_for_entropy_bytes(16).unwrap();

    assert_eq!(subsequences, 4);
    assert_eq!(jpeg_entropy_overflow_count(subsequences), 3);
    assert_eq!(jpeg_entropy_overflow_count(0), 0);
}
```

- [ ] **Step 2: Add overflow count helper**

Add:

```rust
fn jpeg_entropy_overflow_count(subsequence_count: usize) -> usize {
    subsequence_count.saturating_sub(1)
}
```

- [ ] **Step 3: Add overflow CUDA kernel**

Add `signinum_jpeg_entropy_overflow420` in `jpeg_decode_kernels.cu`:

```cpp
extern "C" __global__ void signinum_jpeg_entropy_overflow420(
    const unsigned char *entropy,
    SigninumJpegEntropyChunkParams params,
    const SigninumJpegHuffmanTable *y_dc,
    const SigninumJpegHuffmanTable *y_ac,
    const SigninumJpegHuffmanTable *cb_dc,
    const SigninumJpegHuffmanTable *cb_ac,
    const SigninumJpegHuffmanTable *cr_dc,
    const SigninumJpegHuffmanTable *cr_ac,
    const SigninumJpegEntropySyncState *states,
    SigninumJpegEntropyOverflowState *overflows
) {
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid + 1u >= params.subsequence_count) {
        return;
    }

    const SigninumJpegEntropySyncState target = states[gid + 1u];
    SigninumJpegEntropySyncState state = states[gid];
    SigninumJpegEntropyOverflowState out;
    out.code = JPEG_STATUS_OK;
    out.from_subsequence = gid;
    out.to_subsequence = gid + 1u;
    out.overflow_bits = 0u;
    out.synchronized = 0u;
    out.reserved[0] = 0u;
    out.reserved[1] = 0u;
    out.reserved[2] = 0u;

    if (state.code != JPEG_STATUS_OK || target.code != JPEG_STATUS_OK) {
        out.code = state.code != JPEG_STATUS_OK ? state.code : target.code;
        overflows[gid] = out;
        return;
    }

    const unsigned int overflow_limit =
        params.max_overflow_subsequences * params.subsequence_bits;
    const unsigned int stop_bit =
        min(params.entropy_bits, state.bit_pos + overflow_limit);
    SigninumJpegBitReader reader =
        signinum_jpeg_bit_reader_at_bit(entropy, params.entropy_len, state.bit_pos);
    SigninumJpegDecodeStatus status;
    status.code = JPEG_STATUS_OK;
    status.detail = 0u;
    status.position = 0u;
    status.reserved = 0u;

    while (state.bit_pos < stop_bit && status.code == JPEG_STATUS_OK) {
        const bool dc = state.zigzag_index == 0u;
        const SigninumJpegHuffmanTable *table =
            state.block_phase < 4u
                ? (dc ? y_dc : y_ac)
                : (state.block_phase == 4u ? (dc ? cb_dc : cb_ac) : (dc ? cr_dc : cr_ac));
        unsigned char symbol = 0u;
        const unsigned int before_pos = reader.pos;
        const unsigned int before_bits = reader.bits;
        if (!signinum_jpeg_decode_symbol(reader, entropy, params.entropy_len, table, &status, symbol)) {
            break;
        }
        const unsigned int coeff_bits = dc ? symbol : (symbol & 0x0Fu);
        if (coeff_bits > 15u) {
            signinum_jpeg_set_error(&status, JPEG_STATUS_HUFFMAN, coeff_bits, reader.pos);
            break;
        }
        if (!signinum_jpeg_ensure_bits(reader, entropy, params.entropy_len, coeff_bits)) {
            signinum_jpeg_set_error(&status, JPEG_STATUS_TRUNCATED, coeff_bits, reader.pos);
            break;
        }
        signinum_jpeg_consume_bits(reader, coeff_bits);
        const unsigned int consumed = (reader.pos - before_pos) * 8u + before_bits - reader.bits;
        state.bit_pos += consumed;
        if (dc) {
            state.zigzag_index = 1u;
            state.symbol_count += 1u;
        } else {
            const unsigned int run = symbol >> 4u;
            const unsigned int ssss = symbol & 0x0Fu;
            if (ssss == 0u && run != 15u) {
                state.symbol_count += 64u - state.zigzag_index;
                state.zigzag_index = 0u;
                state.block_phase = (state.block_phase + 1u) % 6u;
            } else {
                state.zigzag_index += run + 1u;
                state.symbol_count += run + 1u;
                if (state.zigzag_index >= 64u) {
                    state.zigzag_index = 0u;
                    state.block_phase = (state.block_phase + 1u) % 6u;
                }
            }
        }
        if (state.bit_pos == target.bit_pos
            && state.block_phase == target.block_phase
            && state.zigzag_index == target.zigzag_index) {
            out.synchronized = 1u;
            out.overflow_bits =
                state.bit_pos > target.start_bit ? state.bit_pos - target.start_bit : 0u;
            break;
        }
    }

    out.code = status.code;
    overflows[gid] = out;
}
```

- [ ] **Step 4: Launch overflow from Rust**

Add byte helpers near the sync-state helpers:

```rust
#[cfg_attr(not(signinum_cuda_jpeg_decode_ptx_built), allow(dead_code))]
fn cuda_jpeg_entropy_overflow_states_as_bytes(
    states: &[CudaJpegEntropyOverflowState],
) -> &[u8] {
    // SAFETY: CudaJpegEntropyOverflowState is repr(C), plain integer data copied to CUDA.
    unsafe {
        std::slice::from_raw_parts(
            states.as_ptr().cast::<u8>(),
            std::mem::size_of_val(states),
        )
    }
}

#[cfg_attr(not(signinum_cuda_jpeg_decode_ptx_built), allow(dead_code))]
fn cuda_jpeg_entropy_overflow_states_as_bytes_mut(
    states: &mut [CudaJpegEntropyOverflowState],
) -> &mut [u8] {
    // SAFETY: the returned byte slice covers exactly the same initialized overflow storage.
    unsafe {
        std::slice::from_raw_parts_mut(
            states.as_mut_ptr().cast::<u8>(),
            std::mem::size_of_val(states),
        )
    }
}
```

Add the overflow launcher:

```rust
#[cfg(signinum_cuda_jpeg_decode_ptx_built)]
#[allow(clippy::too_many_arguments)]
fn launch_jpeg_entropy_overflow420(
    &self,
    entropy: &CudaDeviceBuffer,
    mut params: CudaJpegEntropyChunkParams,
    y_dc: &CudaDeviceBuffer,
    y_ac: &CudaDeviceBuffer,
    cb_dc: &CudaDeviceBuffer,
    cb_ac: &CudaDeviceBuffer,
    cr_dc: &CudaDeviceBuffer,
    cr_ac: &CudaDeviceBuffer,
    states: &CudaDeviceBuffer,
    overflows: &CudaDeviceBuffer,
) -> Result<(), CudaError> {
    let function = self.inner.kernel_function(CudaKernel::JpegEntropyOverflow420)?;
    let mut entropy_ptr = entropy.device_ptr();
    let mut y_dc_ptr = y_dc.device_ptr();
    let mut y_ac_ptr = y_ac.device_ptr();
    let mut cb_dc_ptr = cb_dc.device_ptr();
    let mut cb_ac_ptr = cb_ac.device_ptr();
    let mut cr_dc_ptr = cr_dc.device_ptr();
    let mut cr_ac_ptr = cr_ac.device_ptr();
    let mut states_ptr = states.device_ptr();
    let mut overflows_ptr = overflows.device_ptr();
    let mut kernel_params = [
        (&raw mut entropy_ptr).cast::<c_void>(),
        (&raw mut params).cast::<c_void>(),
        (&raw mut y_dc_ptr).cast::<c_void>(),
        (&raw mut y_ac_ptr).cast::<c_void>(),
        (&raw mut cb_dc_ptr).cast::<c_void>(),
        (&raw mut cb_ac_ptr).cast::<c_void>(),
        (&raw mut cr_dc_ptr).cast::<c_void>(),
        (&raw mut cr_ac_ptr).cast::<c_void>(),
        (&raw mut states_ptr).cast::<c_void>(),
        (&raw mut overflows_ptr).cast::<c_void>(),
    ];
    let geometry = CudaLaunchGeometry {
        grid: ((params.subsequence_count.saturating_sub(1)).div_ceil(128), 1, 1),
        block: (128, 1, 1),
    };

    self.launch_kernel(function, geometry, &mut kernel_params)
}
```

In `diagnose_jpeg_420_entropy_self_sync_nonempty`, replace the empty overflow
tail with this block after the first-pass `states_buffer.copy_to_host(...)`:

```rust
let mut overflows = vec![CudaJpegEntropyOverflowState::default(); jpeg_entropy_overflow_count(subsequences)];
if !overflows.is_empty() {
    let overflow_buffer = self.upload(cuda_jpeg_entropy_overflow_states_as_bytes(&overflows))?;
    self.launch_jpeg_entropy_overflow420(
        &entropy,
        params,
        &y_dc,
        &y_ac,
        &cb_dc,
        &cb_ac,
        &cr_dc,
        &cr_ac,
        &states_buffer,
        &overflow_buffer,
    )?;
    overflow_buffer.copy_to_host(cuda_jpeg_entropy_overflow_states_as_bytes_mut(&mut overflows))?;
}

Ok(CudaJpegChunkedEntropyReport {
    config: plan.config,
    entropy_bytes: plan.entropy_bytes.len(),
    states,
    overflows,
    execution: CudaExecutionStats {
        kernel_dispatches: 1 + usize::from(subsequences > 1),
        copy_kernel_dispatches: 0,
        decode_kernel_dispatches: 0,
        hardware_decode: false,
    },
})
```

- [ ] **Step 5: Verify locally**

Run:

```bash
cargo test -p signinum-cuda-runtime jpeg_chunked_entropy_report --lib
cargo clippy -p signinum-cuda-runtime --tests -- -D warnings
```

Expected: PASS locally.

---

## Task 5: JPEG CUDA Adapter Diagnostic API

**Files:**
- Modify: `crates/signinum-jpeg-cuda/src/owned_decode.rs`
- Modify: `crates/signinum-jpeg-cuda/src/codec.rs`
- Modify: `crates/signinum-jpeg-cuda/tests/host_surface.rs`

- [ ] **Step 1: Write failing integration test**

Add in `host_surface.rs` under `#[cfg(feature = "cuda-runtime")]`:

```rust
#[cfg(feature = "cuda-runtime")]
#[test]
fn generated_420_chunked_entropy_diagnostic_runs_when_runtime_required() {
    if !runtime_required() {
        return;
    }

    let input = generated_rgb_jpeg(signinum_jpeg::JpegSubsampling::Ybr420, 256, 256);
    let mut session = CudaSession::default();
    let report = Codec::diagnose_tile_rgb8_chunked_entropy_with_session(
        &input,
        signinum_cuda_runtime::CudaJpegChunkedEntropyConfig {
            subsequence_words: 64,
            sequence_len: 32,
            max_overflow_subsequences: 4,
        },
        &mut session,
    )
    .expect("chunked entropy diagnostic");

    assert!(report.subsequence_count() > 0);
    assert_eq!(report.failed_state_count(), 0);
}
```

Add helper:

```rust
fn generated_rgb_jpeg(
    subsampling: signinum_jpeg::JpegSubsampling,
    width: u32,
    height: u32,
) -> Vec<u8> {
    let rgb = signinum_test_support::gpu_bench_rgb8(width, height);
    signinum_jpeg::encode_jpeg_baseline(
        signinum_jpeg::JpegSamples::Rgb8 {
            data: &rgb,
            width,
            height,
        },
        signinum_jpeg::JpegEncodeOptions {
            quality: 90,
            subsampling,
            restart_interval: None,
            backend: signinum_jpeg::JpegBackend::Cpu,
        },
    )
    .expect("generated JPEG")
    .data
}
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
cargo test -p signinum-jpeg-cuda --features cuda-runtime --test host_surface generated_420_chunked_entropy_diagnostic_runs_when_runtime_required -- --exact
```

Expected: FAIL with missing `diagnose_tile_rgb8_chunked_entropy_with_session`.

- [ ] **Step 3: Add adapter plan builder**

In `owned_decode.rs`, add:

```rust
#[cfg(feature = "cuda-runtime")]
pub(crate) fn diagnose_owned_cuda_420_entropy(
    bytes: &[u8],
    config: signinum_cuda_runtime::CudaJpegChunkedEntropyConfig,
    session: &mut CudaSession,
) -> Result<signinum_cuda_runtime::CudaJpegChunkedEntropyReport, Error> {
    let packet = session.resolve_owned_fast420_packet(bytes)?;
    let plan = signinum_cuda_runtime::CudaJpegChunkedEntropyPlan {
        config,
        entropy_bytes: &packet.entropy_bytes,
        y_dc_table: cuda_huffman_table(&packet.y_dc_table)?,
        y_ac_table: cuda_huffman_table(&packet.y_ac_table)?,
        cb_dc_table: cuda_huffman_table(&packet.cb_dc_table)?,
        cb_ac_table: cuda_huffman_table(&packet.cb_ac_table)?,
        cr_dc_table: cuda_huffman_table(&packet.cr_dc_table)?,
        cr_ac_table: cuda_huffman_table(&packet.cr_ac_table)?,
    };
    session
        .cuda_context()?
        .diagnose_jpeg_420_entropy_self_sync(&plan)
        .map_err(cuda_owned_decode_error)
}
```

- [ ] **Step 4: Add public experimental Codec method**

In `codec.rs`, add:

```rust
#[cfg(feature = "cuda-runtime")]
/// Run experimental chunked JPEG entropy self-sync diagnostics for a 4:2:0 RGB8 tile.
///
/// This does not decode pixels and does not affect production CUDA routing.
pub fn diagnose_tile_rgb8_chunked_entropy_with_session(
    input: &[u8],
    config: signinum_cuda_runtime::CudaJpegChunkedEntropyConfig,
    session: &mut CudaSession,
) -> Result<signinum_cuda_runtime::CudaJpegChunkedEntropyReport, Error> {
    crate::owned_decode::diagnose_owned_cuda_420_entropy(input, config, session)
}
```

- [ ] **Step 5: Verify locally**

Run:

```bash
cargo test -p signinum-jpeg-cuda --features cuda-runtime --test host_surface generated_420_chunked_entropy_diagnostic_runs_when_runtime_required -- --exact
cargo clippy -p signinum-jpeg-cuda --features cuda-runtime --tests --benches -- -D warnings
```

Expected locally: PASS when runtime env is absent; compile with feature enabled.

---

## Task 6: Bench, Docs, and Remote CUDA Decision Gate

**Files:**
- Modify: `crates/signinum-jpeg-cuda/benches/device_decode.rs`
- Modify: `docs/bench.md`
- Modify: `docs/stable-api-1.0.public-api.txt`

- [ ] **Step 1: Add benchmark case**

In `device_decode.rs`, add a function under `#[cfg(feature = "cuda-runtime")]`:

```rust
#[cfg(feature = "cuda-runtime")]
fn bench_chunked_entropy_diagnostic(c: &mut Criterion) {
    let (width, height) = generated_dimensions();
    let input = generated_jpeg(width, height);
    let mut group = c.benchmark_group("jpeg_cuda_chunked_entropy");
    group.sample_size(10);

    group.bench_function("cpu_fast_packet_planning", |b| {
        b.iter(|| {
            let packet = signinum_jpeg::adapter::build_metal_fast420_packet(&input)
                .expect("fast420 packet");
            std::hint::black_box(packet.entropy_checkpoints.len())
        });
    });

    group.bench_function("cuda_chunked_entropy_sync", |b| {
        let mut session = CudaSession::default();
        b.iter(|| {
            let report = CudaCodec::diagnose_tile_rgb8_chunked_entropy_with_session(
                &input,
                signinum_cuda_runtime::CudaJpegChunkedEntropyConfig::default(),
                &mut session,
            )
            .expect("chunked entropy diagnostic");
            std::hint::black_box(report.synchronized_overflow_count())
        });
    });

    group.finish();
}
```

Call it from `bench_device_decode` after `bench_batch_decode(c)`.

- [ ] **Step 2: Document benchmark knobs**

Add to `docs/bench.md` near CUDA JPEG benchmark instructions:

```markdown
The experimental `jpeg_cuda_chunked_entropy` group measures the parallel
Huffman self-synchronization spike. It compares current CPU fast-packet planning
against GPU subsequence synchronization diagnostics. It does not decode pixels
and should not be reported as user-visible JPEG decode speed.
```

- [ ] **Step 3: Regenerate stable API snapshot**

Run:

```bash
cargo xtask stable-api --write
cargo xtask stable-api
```

Expected: both commands succeed.

- [ ] **Step 4: Run local verification**

Run:

```bash
cargo test -p signinum-cuda-runtime --lib --no-fail-fast
cargo test -p signinum-jpeg-cuda --features cuda-runtime --test host_surface --no-fail-fast
cargo clippy -p signinum-cuda-runtime --tests -- -D warnings
cargo clippy -p signinum-jpeg-cuda --features cuda-runtime --tests --benches -- -D warnings
cargo fmt --check
git diff --check
```

Expected: all pass locally. Runtime-gated CUDA tests may skip if env is absent.

- [ ] **Step 5: Sync and run remote CUDA tests**

Run from `/Users/user/Bench/signinum`:

```bash
mkdir -p .tmp
REMOTE_DIR=/home/jcwal/codex-runs/signinum-jpeg-self-sync-$(date +%Y%m%d-%H%M%S)
printf '%s\n' "$REMOTE_DIR" > .tmp/signinum-jpeg-self-sync-remote-dir
ssh jcwal@cuda-wsl "mkdir -p '$REMOTE_DIR'"
rsync -a --delete --exclude .git --exclude target --exclude tests/nvidia-baseline/target --exclude .venv ./ jcwal@cuda-wsl:"$REMOTE_DIR"/
ssh jcwal@cuda-wsl "cd '$REMOTE_DIR' && source ~/.cargo/env && SIGNINUM_REQUIRE_CUDA_RUNTIME=1 cargo test -p signinum-jpeg-cuda --features cuda-runtime --test host_surface generated_420_chunked_entropy_diagnostic_runs_when_runtime_required -- --exact --nocapture"
```

Expected: test passes and reports nonzero subsequences with zero failed states.

- [ ] **Step 6: Run remote benchmarks**

Run:

```bash
REMOTE_DIR=$(cat .tmp/signinum-jpeg-self-sync-remote-dir)
ssh jcwal@cuda-wsl "cd '$REMOTE_DIR' && source ~/.cargo/env && SIGNINUM_REQUIRE_CUDA_BENCH=1 SIGNINUM_GPU_BENCH_DIM=1024 cargo bench -p signinum-jpeg-cuda --bench device_decode --features cuda-runtime jpeg_cuda_chunked_entropy -- --noplot --sample-size 10 --warm-up-time 1 --measurement-time 2"
ssh jcwal@cuda-wsl "cd '$REMOTE_DIR' && source ~/.cargo/env && SIGNINUM_REQUIRE_CUDA_BENCH=1 SIGNINUM_GPU_BENCH_DIM=2048 cargo bench -p signinum-jpeg-cuda --bench device_decode --features cuda-runtime jpeg_cuda_chunked_entropy -- --noplot --sample-size 10 --warm-up-time 1 --measurement-time 2"
ssh jcwal@cuda-wsl "cd '$REMOTE_DIR' && source ~/.cargo/env && SIGNINUM_REQUIRE_CUDA_BENCH=1 SIGNINUM_GPU_BENCH_DIM=4096 cargo bench -p signinum-jpeg-cuda --bench device_decode --features cuda-runtime jpeg_cuda_chunked_entropy -- --noplot --sample-size 10 --warm-up-time 1 --measurement-time 2"
```

Expected: capture mean timing for `cpu_fast_packet_planning` and
`cuda_chunked_entropy_sync` at all three dimensions.

- [ ] **Step 7: Decide whether to continue to WSI realism**

Continue to second slice only if:

- Remote diagnostic test passes without failed states.
- At `2048x2048` or `4096x4096`, `cuda_chunked_entropy_sync` is faster than
  `cpu_fast_packet_planning`.
- Overflow synchronization succeeds for at least 99% of adjacent subsequences.

If any condition fails, stop and document the failed condition in `docs/bench.md`
instead of wiring this path into production decode.
