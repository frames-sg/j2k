# J2K Adoption Benchmark Report

Status: diagnostic only. Do not use for marketing claims.

Blocking issues:
- cpu-fixture-compare publication_eligible=false blockers=generated-fixtures-included
- cpu-fixture-compare publication_blockers=generated-fixtures-included
- cpu-encode-compare publication_eligible=false blockers=generated-fixtures-included
- cpu-encode-compare publication_blockers=generated-fixtures-included
- run mode is not full
- generated fixtures included

## Bundle

- `run_dir`: `$RUN_DIR`
- `mode`: `quick`
- `include_generated`: `true`
- `input_dirs`: `not-recorded`
- `manifest`: `not-recorded`
- `encode_input_dirs`: `not-recorded`
- `encode_manifest`: `not-recorded`
- `cuda_decode_batch_sizes`: `not-recorded`
- `cuda_requested`: `not-recorded`
- `metal_requested`: `not-recorded`
- `require_cuda`: `not-recorded`
- `require_metal`: `not-recorded`

## Publication Gates

| section | publication_eligible | publication_blockers | benchmark_complete | case_batch_sizes | mixed_batch_sizes | external_unique_input_count | external_native_case_count | external_materialized_case_count | external_native_unique_input_count | mixed_external_batch_group_count | mixed_external_min_distinct_inputs | mixed_external_max_distinct_inputs | mixed_external_group_distinct_inputs | publication_gate_skipped_comparators |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cpu-fixture-compare | false | generated-fixtures-included | true | not-recorded | not-recorded | not-recorded | not-recorded | not-recorded | not-recorded | not-recorded | not-recorded | not-recorded | not-recorded | not-recorded |
| cpu-encode-compare | false | generated-fixtures-included | true | not-recorded | not-recorded | not-recorded | not-recorded | not-recorded | not-recorded | not-recorded | not-recorded | not-recorded | not-recorded | not-recorded |

## Methodology

Fixture comparability scope is pinned for this report.

Publication note preserves the recorded bundle context.

| section | benchmark_mode | build_profile | debug_assertions | git_revision | git_dirty | host_hardware | openjpeg_version | grok_version | openjpeg_compress_available | grok_compress_available | kakadu_included |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cpu-fixture-compare | not-recorded | not-recorded | not-recorded | not-recorded | not-recorded | not-recorded | not-recorded | not-recorded | not-recorded | not-recorded | not-recorded |
| cpu-encode-compare | not-recorded | not-recorded | not-recorded | not-recorded | not-recorded | not-recorded | not-recorded | not-recorded | not-recorded | not-recorded | not-recorded |

## CPU Decode Rows

Missing expected raw columns: `corpus_name`, `license_status`, `encode_command`, `manifest_status`, `source_fnv1a64`.

| decoder | case | benchmark_mode | decode_method | input_source | corpus_category | corpus_name | license_status | encode_command | manifest_status | codec | container | operation | format | dimensions | source_fnv1a64 | batch_size | median_us | tiles_per_second_median | decoded_mib_per_second_median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| j2k | case_a | portable-native | native | external:case-a | natural-image | NA | NA | NA | NA | j2k | jp2 | full | rgb8 | 128x128 | NA | 1 | 10.0 | 100.0 | 20.0 |
| j2k | external_mixed_decode | portable-native | native-mixed-external-batch | external:mixed | natural-image | NA | NA | NA | NA | mixed | mixed | full | rgb8 | mixed | NA | 16 | 20.0 | 800.0 | 200.0 |
| openjpeg | external_mixed_decode | portable-native | native-mixed-external-batch | external:mixed | natural-image | NA | NA | NA | NA | mixed | mixed | full | rgb8 | mixed | NA | 16 | 22.0 | 700.0 | 180.0 |
| grok | external_mixed_decode | portable-native | native-mixed-external-batch | external:mixed | natural-image | NA | NA | NA | NA | mixed | mixed | full | rgb8 | mixed | NA | 16 | 26.0 | 600.0 | 150.0 |
| openjph | external_mixed_decode | portable-native | openjph-cli-process-output-pnm | external:mixed | natural-image | NA | NA | NA | NA | mixed | mixed | full | rgb8 | mixed | NA | 16 | 5.0 | 1000.0 | 400.0 |

## CPU Decode Mixed Winner Summary

Winner eligibility is limited to first-class comparable rows: `j2k`, `openjpeg`, and `grok`. Optional CLI context rows such as OpenJPH or Kakadu remain in raw tables but do not decide this summary.

| case | batch_size | j2k_mib_per_s | openjpeg_mib_per_s | grok_mib_per_s | winner | winner_mib_per_s | j2k_vs_winner |
| --- | --- | --- | --- | --- | --- | --- | --- |
| external_mixed_decode | 16 | 200.000 | 180.000 | 150.000 | j2k | 200.000 | 1.000x |

## CPU Decode Mixed Batch Rows

Missing expected raw columns: `corpus_name`, `license_status`.

| decoder | case | benchmark_mode | decode_method | corpus_category | corpus_name | license_status | codec | container | operation | format | dimensions | batch_size | input_bytes | median_us | tiles_per_second_median | decoded_mib_per_second_median | decoded_bytes_per_repeat |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| j2k | external_mixed_decode | portable-native | native-mixed-external-batch | natural-image | NA | NA | mixed | mixed | full | rgb8 | mixed | 16 | 16384 | 20.0 | 800.0 | 200.0 | 32768 |
| openjpeg | external_mixed_decode | portable-native | native-mixed-external-batch | natural-image | NA | NA | mixed | mixed | full | rgb8 | mixed | 16 | 16384 | 22.0 | 700.0 | 180.0 | 32768 |
| grok | external_mixed_decode | portable-native | native-mixed-external-batch | natural-image | NA | NA | mixed | mixed | full | rgb8 | mixed | 16 | 16384 | 26.0 | 600.0 | 150.0 | 32768 |
| openjph | external_mixed_decode | portable-native | openjph-cli-process-output-pnm | natural-image | NA | NA | mixed | mixed | full | rgb8 | mixed | 16 | 16384 | 5.0 | 1000.0 | 400.0 | 32768 |

## CPU Encode Rows

Missing expected raw columns: `corpus_name`, `license_status`, `source_command`, `manifest_status`.

| encoder | case | benchmark_mode | encode_method | input_source | corpus_category | corpus_name | license_status | source_command | manifest_status | format | dimensions | batch_size | median_us | images_per_second_median | input_mib_per_second_median | encoded_bytes_per_repeat |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| j2k | case_a | classic-lossless-cli | pnm-input-cli-process-output-jp2 | external:case-a | natural-image | NA | NA | NA | NA | png | 128x128 | 1 | 10.0 | 100.0 | 20.0 | 1234 |
| j2k | external_mixed_encode | classic-lossless-cli | pnm-input-cli-process-output-jp2 | external:mixed | natural-image | NA | NA | NA | NA | mixed | mixed | 16 | 20.0 | 800.0 | 200.0 | 12345 |
| openjpeg | external_mixed_encode | classic-lossless-cli | pnm-input-cli-process-output-jp2 | external:mixed | natural-image | NA | NA | NA | NA | mixed | mixed | 16 | 28.0 | 500.0 | 140.0 | 12345 |
| grok | external_mixed_encode | classic-lossless-cli | pnm-input-cli-process-output-jp2 | external:mixed | natural-image | NA | NA | NA | NA | mixed | mixed | 16 | 18.0 | 900.0 | 220.0 | 12345 |
| kakadu | external_mixed_encode | classic-lossless-cli | kakadu-cli-process-output-jp2 | external:mixed | natural-image | NA | NA | NA | NA | mixed | mixed | 16 | 10.0 | 1600.0 | 390.0 | 12345 |

## CPU Encode Mixed Winner Summary

Winner eligibility is limited to first-class comparable rows: `j2k`, `openjpeg`, and `grok`. Optional CLI context rows such as OpenJPH or Kakadu remain in raw tables but do not decide this summary.

| case | batch_size | j2k_mib_per_s | openjpeg_mib_per_s | grok_mib_per_s | winner | winner_mib_per_s | j2k_vs_winner |
| --- | --- | --- | --- | --- | --- | --- | --- |
| external_mixed_encode | 16 | 200.000 | 140.000 | 220.000 | grok | 220.000 | 0.909x |

## CPU Encode Mixed Batch Rows

Missing expected raw columns: `corpus_name`, `license_status`.

| encoder | case | benchmark_mode | encode_method | corpus_category | corpus_name | license_status | format | dimensions | batch_size | input_bytes | median_us | images_per_second_median | input_mib_per_second_median | encoded_bytes_per_repeat |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| j2k | external_mixed_encode | classic-lossless-cli | pnm-input-cli-process-output-jp2 | natural-image | NA | NA | mixed | mixed | 16 | 16384 | 20.0 | 800.0 | 200.0 | 12345 |
| openjpeg | external_mixed_encode | classic-lossless-cli | pnm-input-cli-process-output-jp2 | natural-image | NA | NA | mixed | mixed | 16 | 16384 | 28.0 | 500.0 | 140.0 | 12345 |
| grok | external_mixed_encode | classic-lossless-cli | pnm-input-cli-process-output-jp2 | natural-image | NA | NA | mixed | mixed | 16 | 16384 | 18.0 | 900.0 | 220.0 | 12345 |
| kakadu | external_mixed_encode | classic-lossless-cli | kakadu-cli-process-output-jp2 | natural-image | NA | NA | mixed | mixed | 16 | 16384 | 10.0 | 1600.0 | 390.0 | 12345 |

## Skipped And Context Rows

- decode: openjpeg-unavailable (1 rows)
- encode: openjpeg-compress-unavailable (1 rows)

## Hybrid Summary

| section | j2k_cuda_decode_batch_sizes | j2k_cuda_decode_io_policy | j2k_cuda_decode_external_case_count | j2k_cuda_decode_external_skipped_non_htj2k_count | j2k_cuda_encode_io_policy | j2k_cuda_encode_external_case_count | j2k_cuda_encode_external_input_format |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cuda-htj2k-decode | not-recorded | not-recorded | not-recorded | not-recorded | not-recorded | not-recorded | not-recorded |
| cuda-htj2k-encode | not-recorded | not-recorded | not-recorded | not-recorded | not-recorded | not-recorded | not-recorded |

CUDA Criterion estimate rows:

| step | id | median_ms | median_lower_ms | median_upper_ms |
| --- | --- | --- | --- | --- |
| cuda-htj2k-decode | cuda_decode_external_gray8 | 1.500 | 1.400 | 1.600 |
| cuda-htj2k-encode | cuda_encode_external_rgb8 | 2.500 | 2.400 | 2.600 |

Metal decode benchmark summary:

- status: ran
- `j2k_metal_decode_io_policy`: `generated-fixtures-and-preloaded-external-codestreams;timed-full-rows-include-decode-work;metal_resident_ms-does-not-readback;metal_readback_ms-includes-host-visible-byte-access`
- `j2k_metal_decode_external_case_count`: `0`
- `j2k_metal_decode_generated_included`: `true`
- `bench_count`: `1`
- `skipped_bench_count`: `0`
- `verified_bench_count`: `1`
- `skipped_case_count`: `0`

Metal decode row summary:

| source | codec | container | operation | fmt | size | rows | cpu_ms_avg | metal_resident_ms_avg | metal_readback_ms_avg | readback_vs_cpu | winner |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| generated | j2k | raw-codestream | full | gray8 | 512x512 | 1 | 1.000 | 0.500 | 0.750 | 0.750x | metal-readback |

Metal auto-routing summary:

- status: ran
- `j2k_metal_encode_io_policy`: `staged-pnm-pixels-preloaded-no-filesystem-io-in-timed-loop;auto-rows-include-public-api-host-submission-and-metal-auto-route-work`
- `j2k_metal_encode_external_case_count`: `2`
- `j2k_metal_encode_external_input_format`: `staged-pnm-p5-p6`
- `j2k_metal_encode_resident_batch_sizes`: `not-recorded`
- `auto_bench_count`: `2`
- `skipped_auto_bench_count`: `0`
- `probe_error_count`: `0`
- `resident_bench_count`: `1`
- `skipped_resident_bench_count`: `0`
- `resident_verified_bench_count`: `1`

Metal auto external row summary:

| mode | codec | components | size | rows | cpu_ms_avg | metal_auto_ms_avg | metal_vs_cpu | winner |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lossless_external | htj2k | gray8 | 512x512 | 2 | 12.500 | 5.000 | 0.400x | metal-auto |

Metal resident packetization summary:

| mode | codec | components | size | batch_size | rows | cpu_ms_avg | hybrid_cpu_packet_ms_avg | resident_host_ms_avg | resident_buffer_ms_avg | host_readback_ms_avg | resident_host_vs_cpu | winner |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lossless_external | htj2k | gray8 | 512x512 | 16 | 1 | 10.000 | 6.000 | 4.000 | 3.000 | 1.000 | 0.400x | resident-host |

Metal transcode benchmark summary:

- status: ran
- `bench_filter`: `jpeg_to_htj2k_wsi_integer_53_tile_batch/srgb_ybr420_224_batch_128`
- `profile_count`: `2`
- `verified_profile_count`: `1`
- `comparison_context_count`: `1`
- `auto_metal_profile_count`: `1`
- `explicit_metal_profile_count`: `0`

Metal transcode profile summary:

| context | request | transform_processor | pipeline | rows | total_ms_avg | successful_tiles | dct_handoffs | dwt_handoffs | accelerator_dispatches | transfer_bytes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| srgb_ybr420_224_batch_128 | cpu | cpu | jpeg_to_htj2k | 1 | 86.000 | 128 | 0 | 0 | 0 | 0 |
| srgb_ybr420_224_batch_128 | metal_auto | metal | jpeg_to_htj2k | 1 | 57.000 | 128 | 384 | 1536 | 1 | 131072 |
