// SPDX-License-Identifier: MIT OR Apache-2.0

use super::read;
use crate::repo_lint_support::{assert_pattern_checks, PatternCheck};

fn assert_below(path: &str, source: &str, limit: usize) {
    let lines = source.lines().count();
    assert!(
        lines < limit,
        "{path} has {lines} lines; expected fewer than {limit}"
    );
}

#[test]
fn j2k_ml_is_a_thin_persistent_batch_adapter() {
    let library_path = "crates/j2k-ml/src/lib.rs";
    let cpu_module_path = "crates/j2k-ml/src/cpu.rs";
    let cpu_batch_path = "crates/j2k-ml/src/cpu/batch.rs";
    let cuda_module_path = "crates/j2k-ml/src/cuda.rs";
    let cuda_batch_path = "crates/j2k-ml/src/cuda/batch.rs";
    let metal_module_path = "crates/j2k-ml/src/metal.rs";
    let metal_batch_path = "crates/j2k-ml/src/metal/batch.rs";
    let library = read(library_path);
    let cpu_module = read(cpu_module_path);
    let cpu_batch = read(cpu_batch_path);
    let cuda_module = read(cuda_module_path);
    let cuda_batch = read(cuda_batch_path);
    let metal_module = read(metal_module_path);
    let metal_batch = read(metal_batch_path);
    let all_adapter_sources = format!(
        "{library}\n{cpu_module}\n{cpu_batch}\n{cuda_module}\n{cuda_batch}\n{metal_module}\n{metal_batch}"
    );

    assert_below(library_path, &library, 180);
    assert_below(cpu_module_path, &cpu_module, 30);
    assert_below(cuda_module_path, &cuda_module, 40);
    assert_below(metal_module_path, &metal_module, 40);
    assert_pattern_checks(&[
        PatternCheck::new("shared Burn batch output", &library).required(&[
            "pub enum BurnBatchTensor",
            "pub struct BurnBatchGroup",
            "pub struct BurnBatchDecode",
        ]),
        PatternCheck::new("CPU session adapter", &cpu_batch)
            .required(&[
                "pub struct CpuBurnDecoder",
                "j2k::CpuBatchDecoder",
                "pub fn decode_prepared",
                "TensorData::new",
            ])
            .forbidden(&["J2kDecoder::new", "decode_components_with_context"]),
        PatternCheck::new("CUDA session adapter", &cuda_batch)
            .required(&[
                "pub struct CudaBurnDecoder",
                "CudaBatchDecoder as CodecDecoder",
                ".context_for_device_interop(self.device.index)",
                "submit_batch_into",
            ])
            .forbidden(&[
                "context: Option<CudaContext>",
                "fn ensure_context(",
                "J2kDecoder::new",
                "decode_request_to_device_with_session",
            ]),
        PatternCheck::new("Metal session adapter", &metal_batch)
            .required(&[
                "pub struct MetalBurnDecoder",
                "MetalBatchDecoder as CodecDecoder",
                "submit_prepared_group_into_for_consumer_queue(",
            ])
            .forbidden(&["J2kDecoder::new", "decode_request_to_device_with_session"]),
        PatternCheck::new("training policy stays outside j2k-ml", &all_adapter_sources).forbidden(
            &[
                "FloatNormalization",
                "PanicOnDecodeError",
                "decode_float",
                "MeanStd",
                "Dataset",
                "augmentation",
                "prefetch",
            ],
        ),
    ]);
}

#[test]
fn metal_burn_decoder_keeps_batch_options_in_the_codec_session_only() {
    let metal_batch = read("crates/j2k-ml/src/metal/batch.rs");

    assert!(
        !metal_batch.contains("\n    options: BatchDecodeOptions,\n"),
        "MetalBurnDecoder must not duplicate options already retained by CodecDecoder"
    );
    assert!(
        metal_batch.contains(".field(\"options\", &self.codec.options())"),
        "MetalBurnDecoder Debug must read the codec-owned options"
    );
}
