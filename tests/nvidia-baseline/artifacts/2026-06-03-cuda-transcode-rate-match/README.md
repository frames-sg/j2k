# CUDA JPEG to HTJ2K Rate-Matched Transcode Artifacts

These reports were collected on the CUDA WSL runner as user `jcwal` on
2026-06-03 from branch `codex/cuda-htj2k-batch-callers`.

Input corpus:

```sh
export SIGNINUM_BENCH_JPEG_DIR=tests/nvidia-baseline/benchtiles/pancreas
```

The JSON files are the source of truth. CSV files are included for quick review.

| File | Workload | Summary |
| --- | --- | --- |
| `level1-rate-match.json` | 9/7, 1 decomposition level, quant scale 1.90 | Signinum CUDA HT: 441.0 MP/s, NVIDIA reused-session serial: 49.4 MP/s |
| `level2-rate-match.json` | 9/7, 2 decomposition levels, quant scale 1.00 | Signinum CUDA HT: 67.7 MP/s, NVIDIA reused-session serial: 51.3 MP/s |

The Signinum rows are coefficient-domain JPEG to HTJ2K transforms. The NVIDIA
row is nvJPEG decode to pixels followed by nvJPEG2000 HT encode. nvJPEG2000
does not expose the same coefficient-domain transform, so the comparison is
end-to-end JPEG to HTJ2K throughput at a similar output byte size.
