#!/usr/bin/env python3

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("assert_transcode_perf.py")


def write_report(report):
    temp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    with temp:
        json.dump(report, temp)
    return Path(temp.name)


def base_report(cuda_wall_ms=16.196893, cuda_dispatches=1, nvidia_wall_ms=144.689665):
    return {
        "tile_count": 109,
        "megapixels": 7.143424,
        "match_nvidia_bytes": True,
        "match_tolerance": 0.2,
        "rd_points": [
            {
                "scale": 1.9,
                "ran": True,
                "used_gpu": True,
                "bytes": 6199631,
                "byte_delta_vs_nvidia": -0.00536860,
                "wall_ms": 67.994635,
                "encode_dispatches": 0,
                "ht_codeblock_dispatches": 0,
            }
        ],
        "j2k_cuda_ht_experimental": {
            "scale": 1.9,
            "ran": True,
            "used_gpu": True,
            "bytes": 6199631,
            "byte_delta_vs_nvidia": -0.00536860,
            "wall_ms": cuda_wall_ms,
            "encode_dispatches": cuda_dispatches,
            "ht_codeblock_dispatches": cuda_dispatches,
        },
        "nvidia_reused_session_serial": {
            "ran": True,
            "status": "ok",
            "bytes": 6233094,
            "wall_ms": nvidia_wall_ms,
        },
    }


class AssertTranscodePerfTest(unittest.TestCase):
    def run_gate(self, report, *extra_args):
        report_path = write_report(report)
        self.addCleanup(lambda: report_path.unlink(missing_ok=True))
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--json",
                str(report_path),
                "--label",
                "unit",
                *extra_args,
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def test_passes_when_cuda_ht_beats_absolute_and_relative_thresholds(self):
        result = self.run_gate(
            base_report(),
            "--min-cuda-ht-mps",
            "400",
            "--min-cuda-ht-speedup-vs-nvidia",
            "5.0",
            "--max-byte-delta-abs",
            "0.02",
            "--min-ht-codeblock-dispatches",
            "1",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("unit: PASS", result.stdout)

    def test_fails_when_cuda_ht_is_below_absolute_threshold(self):
        result = self.run_gate(
            base_report(cuda_wall_ms=200.0),
            "--min-cuda-ht-mps",
            "400",
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("j2k CUDA HT MP/s", result.stderr)

    def test_fails_when_cuda_ht_did_not_dispatch(self):
        result = self.run_gate(
            base_report(cuda_dispatches=0),
            "--min-ht-codeblock-dispatches",
            "1",
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("HT code-block dispatches", result.stderr)


if __name__ == "__main__":
    unittest.main()
