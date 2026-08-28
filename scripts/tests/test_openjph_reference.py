# SPDX-License-Identifier: MIT OR Apache-2.0

import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "prepare-openjph-reference.sh"
COMMON = ROOT / "scripts" / "reference-build-common.sh"


class OpenJphReferenceTests(unittest.TestCase):
    def test_shared_reference_build_functions_load_in_strict_bash(self):
        subprocess.run(
            [
                "bash",
                "-euo",
                "pipefail",
                "-c",
                'source "$1"; declare -F reference_prepare_checkout '
                "reference_find_artifact reference_canonical_file reference_emit_env >/dev/null",
                "bash",
                str(COMMON),
            ],
            check=True,
        )

    def test_prepare_script_pins_library_and_cli_from_official_source(self):
        source = SCRIPT.read_text(encoding="utf-8")

        self.assertIn("https://github.com/aous72/OpenJPH.git", source)
        self.assertIn("c68064d0e4cad8e96bab9a068f6cc4e7799744fc", source)
        self.assertIn('version="0.31.0"', source)
        self.assertIn("J2K_OPENJPH_EXPAND_BIN", source)
        self.assertIn("J2K_OPENJPH_COMPRESS_BIN", source)
        self.assertIn("--target ojph_expand ojph_compress", source)
        self.assertIn("J2K_OPENJPH_SOURCE_DIR", source)
        self.assertIn("J2K_OPENJPH_LIB_DIR", source)
        subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


if __name__ == "__main__":
    unittest.main()
