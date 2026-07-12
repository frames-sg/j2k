use std::fs;

mod existing;
use self::existing::existing_steps;
mod options;
use self::options::help_text;
pub(crate) use self::options::AdoptionBenchmarkOptions;
mod parsing;
#[cfg(test)]
use self::parsing::{
    parse_metal_auto_bench_line, parse_metal_auto_probe_line, parse_metal_decode_bench_line,
    parse_metal_resident_bench_line, parse_metal_stage_bench_line,
    parse_metal_transcode_profile_line, read_metal_decode_summary, read_metal_encode_summary,
    read_metal_transcode_summary,
};
mod readme;
use self::readme::write_readme;
mod runner;
#[cfg(test)]
use self::runner::display_command;
use self::runner::{
    run_cpu_encode_compare, run_cpu_fixture_compare, run_cpu_public_api_decode,
    run_cpu_public_api_encode, run_cuda_htj2k_decode, run_cuda_htj2k_encode,
    run_metal_decode_benchmark, run_metal_encode_auto_routing, run_metal_transcode_benchmark,
    skipped_step,
};
mod summary;
use self::summary::write_summary;
#[cfg(test)]
use self::summary::{AdoptionStep, StepStatus};
mod support;
use self::support::enforce_publication_gate;

pub(crate) fn adoption_benchmark(args: impl Iterator<Item = String>) -> Result<(), String> {
    let args = args.collect::<Vec<_>>();
    if args
        .iter()
        .any(|arg| matches!(arg.as_str(), "--help" | "-h"))
    {
        println!("{}", help_text());
        return Ok(());
    }
    let options = AdoptionBenchmarkOptions::parse(args.into_iter())?;
    fs::create_dir_all(&options.out_dir)
        .map_err(|err| format!("failed to create {}: {err}", options.out_dir.display()))?;

    if options.finalize_existing {
        let steps = existing_steps(&options)?;
        write_summary(&options, &steps)?;
        write_readme(&options, &steps)?;
        enforce_publication_gate(&options)?;
        eprintln!(
            "finalized existing adoption benchmark artifacts under {}",
            options.out_dir.display()
        );
        return Ok(());
    }

    let mut steps = vec![
        run_cpu_fixture_compare(&options)?,
        run_cpu_encode_compare(&options)?,
        run_cpu_public_api_encode(&options)?,
        run_cpu_public_api_decode(&options)?,
    ];

    if options.cuda {
        steps.push(run_cuda_htj2k_decode(&options)?);
        steps.push(run_cuda_htj2k_encode(&options)?);
    } else {
        steps.push(skipped_step(
            "cuda-htj2k-decode",
            "not requested; pass --cuda for CUDA decode/encode Criterion benches",
            &options.out_dir,
        ));
        steps.push(skipped_step(
            "cuda-htj2k-encode",
            "not requested; pass --cuda for CUDA decode/encode Criterion benches",
            &options.out_dir,
        ));
    }

    if options.metal {
        steps.push(run_metal_decode_benchmark(&options)?);
        steps.push(run_metal_encode_auto_routing(&options)?);
        steps.push(run_metal_transcode_benchmark(&options)?);
    } else {
        steps.push(skipped_step(
            "metal-decode-benchmark",
            "not requested; pass --metal for Metal decode benchmark",
            &options.out_dir,
        ));
        steps.push(skipped_step(
            "metal-encode-auto-routing",
            "not requested; pass --metal for Metal hybrid encode routing benchmark",
            &options.out_dir,
        ));
        steps.push(skipped_step(
            "metal-transcode-benchmark",
            "not requested; pass --metal for Metal transcode benchmark",
            &options.out_dir,
        ));
    }

    write_summary(&options, &steps)?;
    write_readme(&options, &steps)?;
    enforce_publication_gate(&options)?;
    eprintln!(
        "wrote adoption benchmark artifacts under {}",
        options.out_dir.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        display_command, enforce_publication_gate, parse_metal_auto_bench_line,
        parse_metal_auto_probe_line, parse_metal_decode_bench_line,
        parse_metal_resident_bench_line, parse_metal_stage_bench_line,
        parse_metal_transcode_profile_line, read_metal_decode_summary, read_metal_encode_summary,
        read_metal_transcode_summary, AdoptionBenchmarkOptions, AdoptionStep, StepStatus,
    };
    use std::ffi::OsString;

    #[test]
    fn generated_smoke_requires_explicit_flag() {
        let error = AdoptionBenchmarkOptions::parse(std::iter::empty())
            .expect_err("default external-only run must require fixtures");

        assert!(error.contains("external-only benchmark requires --fixtures and --encode-fixtures"));
    }

    #[test]
    fn manifest_requires_fixture_dirs() {
        let error = AdoptionBenchmarkOptions::parse(
            ["--manifest", "fixtures.tsv", "--include-generated"]
                .map(str::to_string)
                .into_iter(),
        )
        .expect_err("manifest without fixture dirs must fail");

        assert!(error.contains("--manifest requires --fixtures"));
    }

    #[test]
    fn encode_manifest_requires_encode_fixture_dirs() {
        let error = AdoptionBenchmarkOptions::parse(
            ["--encode-manifest", "encode.tsv", "--include-generated"]
                .map(str::to_string)
                .into_iter(),
        )
        .expect_err("encode manifest without encode dirs must fail");

        assert!(error.contains("--encode-manifest requires --encode-fixtures"));
    }

    #[test]
    fn external_only_requires_decode_and_encode_fixture_dirs() {
        let error = AdoptionBenchmarkOptions::parse(
            ["--fixtures", "decode-fixtures"]
                .map(str::to_string)
                .into_iter(),
        )
        .expect_err("decode-only external run must fail");

        assert!(error.contains("--fixtures and --encode-fixtures"));
    }

    #[test]
    fn parses_external_decode_and_encode_fixture_dirs() {
        let options = AdoptionBenchmarkOptions::parse(
            [
                "--fixtures",
                "decode-fixtures",
                "--encode-fixtures",
                "source-images",
                "--manifest",
                "decode.tsv",
                "--encode-manifest",
                "encode.tsv",
            ]
            .map(str::to_string)
            .into_iter(),
        )
        .expect("valid external options");

        assert_eq!(options.input_dirs.as_deref(), Some("decode-fixtures"));
        assert_eq!(options.encode_input_dirs.as_deref(), Some("source-images"));
        assert_eq!(
            options.manifest.as_deref(),
            Some(std::path::Path::new("decode.tsv"))
        );
        assert_eq!(
            options.encode_manifest.as_deref(),
            Some(std::path::Path::new("encode.tsv"))
        );
    }

    #[test]
    fn full_external_run_fails_when_comparator_publication_gate_fails() {
        let out_dir = std::env::current_dir()
            .expect("current dir")
            .join("target")
            .join("j2k-adoption-gate-test")
            .join(std::process::id().to_string());
        std::fs::create_dir_all(&out_dir).expect("create out dir");
        for name in ["cpu-fixture-compare.out", "cpu-encode-compare.out"] {
            std::fs::write(
                out_dir.join(name),
                "publication_eligible\tfalse\npublication_blockers\tgenerated-fixtures-included\nbenchmark_complete\ttrue\n",
            )
            .expect("write metadata");
        }
        let options = AdoptionBenchmarkOptions {
            out_dir,
            input_dirs: Some("decode-fixtures".to_string()),
            manifest: Some("fixtures.tsv".into()),
            encode_input_dirs: Some("source-images".to_string()),
            encode_manifest: Some("encode.tsv".into()),
            cuda_decode_batch_sizes: None,
            include_generated: false,
            quick: false,
            cuda: false,
            metal: false,
            openjph: false,
            kakadu: false,
            require_cuda: false,
            require_metal: false,
            require_openjph: false,
            require_kakadu: false,
            finalize_existing: false,
        };

        let error = enforce_publication_gate(&options).expect_err("gate must fail");

        assert!(error.contains("adoption benchmark is not publishable"));
        assert!(error.contains("cpu-fixture-compare publication_eligible=false"));
        assert!(error.contains("cpu-encode-compare publication_blockers"));
    }

    #[test]
    fn displayed_commands_show_benchmark_env_scrub() {
        let command = display_command(
            &OsString::from("cargo"),
            &["run".to_string()],
            &[("J2K_FIXTURE_COMPARE_REPEATS".to_string(), "1".to_string())],
            None,
        );

        assert!(command.starts_with("env -u J2K_FIXTURE_COMPARE_MODE"));
        assert!(command.contains("-u J2K_INCLUDE_OPENJPH"));
        assert!(command.contains("-u J2K_REQUIRE_OPENJPH"));
        assert!(command.contains("-u J2K_OPENJPH_EXPAND_BIN"));
        assert!(command.contains("-u J2K_INCLUDE_KAKADU"));
        assert!(command.contains("-u J2K_REQUIRE_KAKADU"));
        assert!(command.contains("-u J2K_KDU_EXPAND_BIN"));
        assert!(command.contains("-u J2K_KDU_COMPRESS_BIN"));
        assert!(command.contains("-u J2K_ENCODE_COMPARE_ENCODERS"));
        assert!(command.contains("-u J2K_CUDA_DECODE_INPUT_DIRS"));
        assert!(command.contains("-u J2K_CUDA_ENCODE_INPUT_DIRS"));
        assert!(command.contains("-u J2K_CUDA_ENCODE_MANIFEST"));
        assert!(command.contains("-u J2K_METAL_DECODE_INPUT_DIRS"));
        assert!(command.contains("-u J2K_METAL_DECODE_MANIFEST"));
        assert!(command.contains("-u J2K_METAL_ENCODE_INPUT_DIRS"));
        assert!(command.contains("-u J2K_METAL_ENCODE_MANIFEST"));
        assert!(command.contains("-u J2K_TRANSCODE_METAL_PROFILE_STAGES"));
        assert!(command.contains("J2K_FIXTURE_COMPARE_REPEATS=1"));
        assert!(command.ends_with("cargo run"));
    }

    #[test]
    fn parses_metal_decode_bench_row() {
        let row = parse_metal_decode_bench_line(
            "j2k_metal_decode_bench case=generated_htj2k_gray8_512 source=generated codec=htj2k container=raw-codestream operation=region_scaled fmt=gray8 size=256x256 cpu_ms=1.250 metal_resident_ms=0.500 metal_readback_ms=0.750 output_bytes=65536",
        )
        .expect("valid Metal decode row");

        assert_eq!(row["case"], "generated_htj2k_gray8_512");
        assert_eq!(row["codec"], "htj2k");
        assert_eq!(row["operation"], "region_scaled");
        assert_eq!(row["fmt"], "gray8");
        assert_eq!(row["cpu_ms"], 1.25);
        assert_eq!(row["metal_resident_ms"], 0.5);
        assert_eq!(row["metal_readback_ms"], 0.75);
        assert_eq!(row["output_bytes"], 65_536);
    }

    #[test]
    fn parses_metal_transcode_profile_row() {
        let row = parse_metal_transcode_profile_line(
            "j2k_profile codec=transcode op=transcode_batch request=metal_explicit path=metal pipeline=jpeg_to_htj2k context=srgb_ybr420_224_batch_128 coefficient_path=dct97 extract_processor=cpu transform_processor=metal encode_processor=cpu tile_count=128 successful_tiles=128 failed_tiles=0 transformed_components=384 total_us=57500 extract_us=2100 transform_us=33100 encode_us=22300 dct_to_wavelet_total_us=33100 dct_to_wavelet_accelerator_us=30000 dct_to_wavelet_cpu_fallback_us=0 dwt97_batch_pack_upload_transfers=1 dwt97_batch_pack_upload_bytes=65536 dwt97_batch_resident_dct_handoff_count=384 dwt97_batch_resident_dwt_handoff_count=1536 dwt97_batch_readback_transfers=1 dwt97_batch_readback_bytes=65536 host_to_device_transfer_count=1 host_to_device_transfer_bytes=65536 device_to_host_transfer_count=1 device_to_host_transfer_bytes=65536 component_count=384 batch_count=1 batch_jobs=384 accelerator_attempts=384 accelerator_jobs=384 accelerator_dispatches=1 accelerator_dispatched_jobs=384 cpu_fallback_jobs=0",
        )
        .expect("valid Metal transcode profile row");

        assert_eq!(row["request"], "metal_explicit");
        assert_eq!(row["context"], "srgb_ybr420_224_batch_128");
        assert_eq!(row["transform_processor"], "metal");
        assert_eq!(row["tile_count"], 128);
        assert_eq!(row["accelerator_dispatches"], 1);
        assert_eq!(row["host_to_device_transfer_bytes"], 65_536);
        assert_eq!(row["dwt97_batch_resident_dct_handoff_count"], 384);
        assert_eq!(row["dwt97_batch_resident_dwt_handoff_count"], 1536);
    }

    #[test]
    fn metal_decode_summary_counts_verified_and_skipped_rows() {
        let out_dir = std::env::current_dir()
            .expect("current dir")
            .join("target")
            .join("j2k-metal-decode-summary-test")
            .join(std::process::id().to_string());
        std::fs::create_dir_all(&out_dir).expect("create out dir");
        let stdout = out_dir.join("metal-decode-benchmark.out");
        std::fs::write(
            &stdout,
            concat!(
                "j2k_metal_decode_bench case=a source=generated codec=j2k container=raw-codestream operation=full fmt=gray8 size=512x512 cpu_ms=1.000 metal_resident_ms=0.500 metal_readback_ms=0.700 output_bytes=262144\n",
                "j2k_metal_decode_bench case=b source=generated codec=htj2k container=raw-codestream operation=region_scaled fmt=rgb8 size=256x256 cpu_ms=skipped metal_resident_ms=skipped metal_readback_ms=skipped output_bytes=skipped error=unsupported\n",
                "j2k_metal_decode_skipped_case path=/tmp/wrapped.jph reason=wrapper_container_not_claimed_for_metal_decode container=jph\n",
                "j2k_metal_decode_generated_case_count\t3\n",
            ),
        )
        .expect("write Metal decode stdout");
        let step = AdoptionStep {
            name: "metal-decode-benchmark",
            command: "cargo test".to_string(),
            stdout: stdout.clone(),
            stderr: out_dir.join("metal-decode-benchmark.err"),
            criterion_root: None,
            status: StepStatus::Ran,
        };

        let summary = read_metal_decode_summary(&stdout, &[step]);

        assert_eq!(summary["bench_count"], 2);
        assert_eq!(summary["skipped_bench_count"], 1);
        assert_eq!(summary["verified_bench_count"], 1);
        assert_eq!(summary["skipped_case_count"], 1);
        assert_eq!(
            summary["metadata"]["j2k_metal_decode_generated_case_count"],
            "3"
        );
    }

    #[test]
    fn metal_transcode_summary_counts_comparable_cpu_and_metal_contexts() {
        let out_dir = std::env::current_dir()
            .expect("current dir")
            .join("target")
            .join("j2k-metal-transcode-summary-test")
            .join(std::process::id().to_string());
        std::fs::create_dir_all(&out_dir).expect("create out dir");
        let stdout = out_dir.join("metal-transcode-benchmark.out");
        let stderr = out_dir.join("metal-transcode-benchmark.err");
        std::fs::write(&stdout, "criterion output\n").expect("write Metal transcode stdout");
        std::fs::write(
            &stderr,
            concat!(
                "j2k_profile codec=transcode op=transcode_batch request=cpu path=cpu pipeline=jpeg_to_htj2k context=srgb_ybr420_224_batch_128 coefficient_path=dct97 extract_processor=cpu transform_processor=cpu encode_processor=cpu tile_count=128 successful_tiles=128 failed_tiles=0 transformed_components=384 total_us=86000 extract_us=2000 transform_us=62000 encode_us=22000 dct_to_wavelet_total_us=62000 dct_to_wavelet_accelerator_us=0 dct_to_wavelet_cpu_fallback_us=62000 dwt97_batch_pack_upload_transfers=0 dwt97_batch_pack_upload_bytes=0 dwt97_batch_resident_dct_handoff_count=0 dwt97_batch_resident_dwt_handoff_count=0 dwt97_batch_readback_transfers=0 dwt97_batch_readback_bytes=0 host_to_device_transfer_count=0 host_to_device_transfer_bytes=0 device_to_host_transfer_count=0 device_to_host_transfer_bytes=0 component_count=384 batch_count=1 batch_jobs=384 accelerator_attempts=0 accelerator_jobs=0 accelerator_dispatches=0 accelerator_dispatched_jobs=0 cpu_fallback_jobs=384\n",
                "j2k_profile codec=transcode op=transcode_batch request=metal_auto path=auto pipeline=jpeg_to_htj2k context=srgb_ybr420_224_batch_128 coefficient_path=dct97 extract_processor=cpu transform_processor=metal encode_processor=cpu tile_count=128 successful_tiles=128 failed_tiles=0 transformed_components=384 total_us=57000 extract_us=2000 transform_us=33000 encode_us=22000 dct_to_wavelet_total_us=33000 dct_to_wavelet_accelerator_us=30000 dct_to_wavelet_cpu_fallback_us=0 dwt97_batch_pack_upload_transfers=1 dwt97_batch_pack_upload_bytes=65536 dwt97_batch_resident_dct_handoff_count=384 dwt97_batch_resident_dwt_handoff_count=1536 dwt97_batch_readback_transfers=1 dwt97_batch_readback_bytes=65536 host_to_device_transfer_count=1 host_to_device_transfer_bytes=65536 device_to_host_transfer_count=1 device_to_host_transfer_bytes=65536 component_count=384 batch_count=1 batch_jobs=384 accelerator_attempts=384 accelerator_jobs=384 accelerator_dispatches=1 accelerator_dispatched_jobs=384 cpu_fallback_jobs=0\n",
                "j2k_profile codec=transcode op=transcode_batch request=metal_explicit path=metal pipeline=jpeg_to_htj2k context=srgb_ybr420_224_batch_128 coefficient_path=dct97 extract_processor=cpu transform_processor=metal encode_processor=cpu tile_count=128 successful_tiles=128 failed_tiles=0 transformed_components=384 total_us=58000 extract_us=2000 transform_us=34000 encode_us=22000 dct_to_wavelet_total_us=34000 dct_to_wavelet_accelerator_us=31000 dct_to_wavelet_cpu_fallback_us=0 dwt97_batch_pack_upload_transfers=1 dwt97_batch_pack_upload_bytes=65536 dwt97_batch_resident_dct_handoff_count=384 dwt97_batch_resident_dwt_handoff_count=1536 dwt97_batch_readback_transfers=1 dwt97_batch_readback_bytes=65536 host_to_device_transfer_count=1 host_to_device_transfer_bytes=65536 device_to_host_transfer_count=1 device_to_host_transfer_bytes=65536 component_count=384 batch_count=1 batch_jobs=384 accelerator_attempts=384 accelerator_jobs=384 accelerator_dispatches=1 accelerator_dispatched_jobs=384 cpu_fallback_jobs=0\n",
            ),
        )
        .expect("write Metal transcode stderr");
        let step = AdoptionStep {
            name: "metal-transcode-benchmark",
            command: "cargo bench".to_string(),
            stdout: stdout.clone(),
            stderr: stderr.clone(),
            criterion_root: None,
            status: StepStatus::Ran,
        };

        let summary = read_metal_transcode_summary(&stdout, &stderr, &[step]);

        assert_eq!(summary["profile_count"], 3);
        assert_eq!(summary["verified_profile_count"], 2);
        assert_eq!(summary["cpu_profile_count"], 1);
        assert_eq!(summary["auto_metal_profile_count"], 1);
        assert_eq!(summary["explicit_metal_profile_count"], 1);
        assert_eq!(summary["comparison_context_count"], 1);
    }

    #[test]
    fn require_cuda_enables_cuda_benches() {
        let options = AdoptionBenchmarkOptions::parse(
            ["--include-generated", "--require-cuda"]
                .map(str::to_string)
                .into_iter(),
        )
        .expect("valid generated CUDA smoke options");

        assert!(options.cuda);
        assert!(options.require_cuda);
        assert!(!options.metal);
    }

    #[test]
    fn parses_cuda_decode_huge_batch_sizes() {
        let options = AdoptionBenchmarkOptions::parse(
            [
                "--include-generated",
                "--cuda",
                "--cuda-decode-batch-sizes",
                "1,16,256,1024,256",
            ]
            .map(str::to_string)
            .into_iter(),
        )
        .expect("valid generated CUDA huge-batch smoke options");

        assert!(options.cuda);
        assert_eq!(
            options.cuda_decode_batch_sizes.as_deref(),
            Some("1,16,256,1024")
        );
    }

    #[test]
    fn rejects_invalid_cuda_decode_batch_sizes() {
        let error = AdoptionBenchmarkOptions::parse(
            [
                "--include-generated",
                "--cuda",
                "--cuda-decode-batch-sizes",
                "8,0",
            ]
            .map(str::to_string)
            .into_iter(),
        )
        .expect_err("zero CUDA batch size must fail");

        assert!(error.contains("--cuda-decode-batch-sizes entries must be greater than zero"));
    }

    #[test]
    fn require_openjph_enables_openjph_comparator() {
        let options = AdoptionBenchmarkOptions::parse(
            ["--include-generated", "--require-openjph"]
                .map(str::to_string)
                .into_iter(),
        )
        .expect("valid generated OpenJPH smoke options");

        assert!(options.openjph);
        assert!(options.require_openjph);
        assert!(!options.cuda);
        assert!(!options.metal);
    }

    #[test]
    fn require_kakadu_enables_kakadu_comparator() {
        let options = AdoptionBenchmarkOptions::parse(
            ["--include-generated", "--require-kakadu"]
                .map(str::to_string)
                .into_iter(),
        )
        .expect("valid generated Kakadu smoke options");

        assert!(options.kakadu);
        assert!(options.require_kakadu);
        assert!(!options.cuda);
        assert!(!options.metal);
    }

    #[test]
    fn parses_metal_auto_bench_row() {
        let row = parse_metal_auto_bench_line(
            "j2k_metal_encode_auto_bench mode=lossless codec=htj2k components=rgb8 size=1024x1024 cpu_ms=12.345 auto_ms=6.789",
        )
        .expect("valid auto bench row");

        assert_eq!(row["mode"], "lossless");
        assert_eq!(row["codec"], "htj2k");
        assert_eq!(row["components"], "rgb8");
        assert_eq!(row["size"], "1024x1024");
        assert_eq!(row["cpu_ms"], 12.345);
        assert_eq!(row["auto_ms"], 6.789);
    }

    #[test]
    fn parses_metal_stage_skip_with_error() {
        let row = parse_metal_stage_bench_line(
            "j2k_metal_encode_stage_bench stage=forward_dwt97 size=512x512 cpu_ms=1.250 metal_ms=skipped error=Metal device unavailable",
        )
        .expect("valid stage bench row");

        assert_eq!(row["stage"], "forward_dwt97");
        assert_eq!(row["metal_ms"], "skipped");
        assert_eq!(row["error"], "Metal device unavailable");
    }

    #[test]
    fn parses_metal_resident_bench_row() {
        let row = parse_metal_resident_bench_line(
            "j2k_metal_encode_resident_bench mode=lossless_external codec=htj2k components=rgb8 size=1024x768 batch_size=256 fixture_count=24 resident_input_storage=private resident_staging=already_padded_contiguous cpu_ms=120.000 hybrid_cpu_packet_ms=81.250 resident_host_ms=44.500 resident_buffer_ms=39.250 packetization_used=true codestream_assembly_used=true host_readback_ms=5.125 gpu_ms=33.750 encoded_host_bytes=123456 encoded_buffer_bytes=123456",
        )
        .expect("valid resident bench row");

        assert_eq!(row["mode"], "lossless_external");
        assert_eq!(row["codec"], "htj2k");
        assert_eq!(row["batch_size"], 256);
        assert_eq!(row["fixture_count"], 24);
        assert_eq!(row["resident_host_ms"], 44.5);
        assert_eq!(row["resident_buffer_ms"], 39.25);
        assert_eq!(row["packetization_used"], true);
        assert_eq!(row["codestream_assembly_used"], true);
        assert_eq!(row["encoded_host_bytes"], 123_456);
        assert_eq!(row["resident_input_storage"], "private");
        assert_eq!(row["resident_staging"], "already_padded_contiguous");
    }

    #[test]
    fn metal_resident_summary_counts_only_full_resident_rows_as_verified() {
        let out_dir = std::env::current_dir()
            .expect("current dir")
            .join("target")
            .join("j2k-metal-resident-summary-test")
            .join(std::process::id().to_string());
        std::fs::create_dir_all(&out_dir).expect("create out dir");
        let stdout = out_dir.join("metal-encode-auto-routing.out");
        std::fs::write(
            &stdout,
            concat!(
                "j2k_metal_encode_resident_bench mode=lossless_external codec=htj2k components=rgb8 size=64x64 batch_size=16 fixture_count=1 cpu_ms=1.000 hybrid_cpu_packet_ms=skipped resident_host_ms=0.500 resident_buffer_ms=0.400 packetization_used=true codestream_assembly_used=true host_readback_ms=0.050 gpu_ms=not-recorded encoded_host_bytes=128 encoded_buffer_bytes=128\n",
                "j2k_metal_encode_resident_bench mode=lossless_external codec=htj2k components=rgb8 size=64x64 batch_size=256 fixture_count=1 cpu_ms=10.000 hybrid_cpu_packet_ms=skipped resident_host_ms=4.500 resident_buffer_ms=3.900 packetization_used=true codestream_assembly_used=false host_readback_ms=0.300 gpu_ms=not-recorded encoded_host_bytes=2048 encoded_buffer_bytes=2048\n",
                "j2k_metal_encode_resident_bench mode=lossless_external codec=htj2k components=rgb8 size=64x64 batch_size=1024 fixture_count=1 cpu_ms=skipped hybrid_cpu_packet_ms=skipped resident_host_ms=skipped resident_buffer_ms=skipped packetization_used=false codestream_assembly_used=false host_readback_ms=skipped gpu_ms=skipped encoded_host_bytes=skipped encoded_buffer_bytes=skipped error=memory budget prevented resident batch\n",
                "j2k_metal_encode_resident_batch_sizes\t1,16,256,1024\n",
            ),
        )
        .expect("write Metal stdout");
        let step = AdoptionStep {
            name: "metal-encode-auto-routing",
            command: "cargo test".to_string(),
            stdout: stdout.clone(),
            stderr: out_dir.join("metal-encode-auto-routing.err"),
            criterion_root: None,
            status: StepStatus::Ran,
        };

        let summary = read_metal_encode_summary(&stdout, &[step]);

        assert_eq!(summary["resident_bench_count"], 3);
        assert_eq!(summary["skipped_resident_bench_count"], 1);
        assert_eq!(summary["resident_verified_bench_count"], 1);
        assert_eq!(
            summary["metadata"]["j2k_metal_encode_resident_batch_sizes"],
            "1,16,256,1024"
        );
    }

    #[test]
    fn parses_metal_probe_dispatch_suffix() {
        let row = parse_metal_auto_probe_line(
            "j2k_metal_encode_auto_probe mode=lossy codec=htj2k components=gray8 size=512x512 dispatch=J2kEncodeDispatchReport { forward_dwt97: Some(1) }",
        )
        .expect("valid probe row");

        assert_eq!(row["mode"], "lossy");
        assert_eq!(
            row["dispatch"],
            "J2kEncodeDispatchReport { forward_dwt97: Some(1) }"
        );
    }
}

#[cfg(test)]
mod artifact_tests;
