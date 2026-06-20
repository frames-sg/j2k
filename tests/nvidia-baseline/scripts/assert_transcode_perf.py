#!/usr/bin/env python3

import argparse
import json
import math
import sys
from pathlib import Path


def positive_float(value, name):
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return float(value)


def load_report(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    if not isinstance(report, dict):
        raise ValueError("report root must be a JSON object")
    return report


def result_mps(report, result, name):
    megapixels = positive_float(report.get("megapixels"), "megapixels")
    wall_ms = positive_float(result.get("wall_ms"), f"{name} wall_ms")
    return megapixels / (wall_ms / 1000.0)


def require_result(report, key, label):
    result = report.get(key)
    if not isinstance(result, dict):
        raise ValueError(f"missing {label} result `{key}`")
    if result.get("ran") is not True:
        raise ValueError(f"{label} did not run")
    return result


def assert_gate(args):
    report = load_report(args.json)
    failures = []

    try:
        cuda_ht = require_result(
            report, "j2k_cuda_ht_experimental", "j2k CUDA HT"
        )
    except ValueError as error:
        failures.append(str(error))
        cuda_ht = None

    nvidia = None
    if args.min_cuda_ht_speedup_vs_nvidia is not None:
        try:
            nvidia = require_result(
                report, "nvidia_reused_session_serial", "NVIDIA reused-session serial"
            )
            if nvidia.get("status") != "ok":
                failures.append(f"NVIDIA status is {nvidia.get('status')!r}, expected 'ok'")
        except ValueError as error:
            failures.append(str(error))

    cuda_mps = None
    if cuda_ht is not None:
        if cuda_ht.get("used_gpu") is not True:
            failures.append("j2k CUDA HT did not report used_gpu=true")
        try:
            cuda_mps = result_mps(report, cuda_ht, "j2k CUDA HT")
        except ValueError as error:
            failures.append(str(error))

        if args.min_cuda_ht_mps is not None and cuda_mps is not None:
            if cuda_mps < args.min_cuda_ht_mps:
                failures.append(
                    "j2k CUDA HT MP/s "
                    f"{cuda_mps:.3f} below threshold {args.min_cuda_ht_mps:.3f}"
                )

        if args.max_byte_delta_abs is not None:
            byte_delta = cuda_ht.get("byte_delta_vs_nvidia")
            if not isinstance(byte_delta, (int, float)) or not math.isfinite(byte_delta):
                failures.append("j2k CUDA HT byte_delta_vs_nvidia is missing or non-finite")
            elif abs(float(byte_delta)) > args.max_byte_delta_abs:
                failures.append(
                    "j2k CUDA HT byte delta "
                    f"{float(byte_delta):.6f} outside +/-{args.max_byte_delta_abs:.6f}"
                )

        dispatches = cuda_ht.get("ht_codeblock_dispatches")
        if not isinstance(dispatches, int):
            failures.append("j2k CUDA HT code-block dispatch count is missing")
        elif dispatches < args.min_ht_codeblock_dispatches:
            failures.append(
                "j2k CUDA HT code-block dispatches "
                f"{dispatches} below threshold {args.min_ht_codeblock_dispatches}"
            )

    if (
        args.min_cuda_ht_speedup_vs_nvidia is not None
        and cuda_mps is not None
        and nvidia is not None
    ):
        try:
            nvidia_mps = result_mps(report, nvidia, "NVIDIA")
        except ValueError as error:
            failures.append(str(error))
        else:
            speedup = cuda_mps / nvidia_mps
            if speedup < args.min_cuda_ht_speedup_vs_nvidia:
                failures.append(
                    "j2k CUDA HT speedup vs NVIDIA "
                    f"{speedup:.3f} below threshold "
                    f"{args.min_cuda_ht_speedup_vs_nvidia:.3f}"
                )

    if failures:
        for failure in failures:
            print(f"{args.label}: FAIL: {failure}", file=sys.stderr)
        return 1

    parts = [f"{args.label}: PASS"]
    if cuda_mps is not None:
        parts.append(f"j2k_cuda_ht_mps={cuda_mps:.3f}")
    if nvidia is not None:
        parts.append(f"nvidia_mps={result_mps(report, nvidia, 'NVIDIA'):.3f}")
    print(" ".join(parts))
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        description="Fail-closed perf gate for transcode_compare JSON reports."
    )
    parser.add_argument("--json", required=True, help="transcode_compare JSON report")
    parser.add_argument("--label", required=True, help="gate label printed in diagnostics")
    parser.add_argument("--min-cuda-ht-mps", type=float)
    parser.add_argument("--min-cuda-ht-speedup-vs-nvidia", type=float)
    parser.add_argument("--max-byte-delta-abs", type=float)
    parser.add_argument("--min-ht-codeblock-dispatches", type=int, default=1)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.min_ht_codeblock_dispatches < 0:
        parser.error("--min-ht-codeblock-dispatches must be >= 0")
    for option in [
        args.min_cuda_ht_mps,
        args.min_cuda_ht_speedup_vs_nvidia,
        args.max_byte_delta_abs,
    ]:
        if option is not None and (
            not math.isfinite(option) or option < 0
        ):
            parser.error("numeric thresholds must be finite and >= 0")
    return assert_gate(args)


if __name__ == "__main__":
    raise SystemExit(main())
