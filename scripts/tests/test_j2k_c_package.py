# SPDX-License-Identifier: MIT OR Apache-2.0

from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "crates" / "j2k-c" / "scripts" / "package.py"
CMAKE_TEMPLATE = ROOT / "crates" / "j2k-c" / "cmake" / "J2KConfig.cmake.in"
PKGCONFIG_TEMPLATE = ROOT / "crates" / "j2k-c" / "pkgconfig" / "j2k.pc.in"
SPEC = importlib.util.spec_from_file_location("j2k_c_package", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
package = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(package)


class ParseNativeStaticLibsTest(unittest.TestCase):
    def test_strips_ansi_reset_from_the_final_flag(self) -> None:
        stderr = (
            "    Finished `release` profile\n"
            "note: native-static-libs: -lgcc_s -lutil -lrt -lpthread -lm -ldl -lc\x1b[0m\n"
        )

        self.assertEqual(
            package.parse_native_static_libs(stderr),
            "-lgcc_s -lutil -lrt -lpthread -lm -ldl -lc",
        )

    def test_strips_colored_note_prefix(self) -> None:
        stderr = "\x1b[1mnote\x1b[0m: native-static-libs: -liconv -lSystem -lc -lm\x1b[0m\r\n"

        self.assertEqual(
            package.parse_native_static_libs(stderr),
            "-liconv -lSystem -lc -lm",
        )

    def test_rejects_missing_line(self) -> None:
        with self.assertRaises(RuntimeError):
            package.parse_native_static_libs("    Finished `release` profile\n")

    def test_rejects_empty_value(self) -> None:
        with self.assertRaises(RuntimeError):
            package.parse_native_static_libs("note: native-static-libs: \x1b[0m\n")

    def test_does_not_capture_the_next_line_as_libraries(self) -> None:
        stderr = (
            "note: native-static-libs:\n"
            "    Finished `release` profile [optimized] target(s)\n"
        )

        with self.assertRaisesRegex(RuntimeError, "empty"):
            package.parse_native_static_libs(stderr)

    def test_rejects_remaining_control_characters(self) -> None:
        with self.assertRaises(RuntimeError):
            package.parse_native_static_libs("note: native-static-libs: -lm\x07\n")


class StageStaticArchiveTest(unittest.TestCase):
    def test_copies_archive_to_lib_j2k_a(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            built = tmp_path / "target" / "libj2k_c.a"
            built.parent.mkdir(parents=True)
            built.write_bytes(b"!<arch>\nfake archive bytes\n")
            output = tmp_path / "dist" / "j2k-c"

            staged = package.stage_static_archive(built, output)

            self.assertEqual(staged, output / "lib" / "libj2k.a")
            self.assertEqual(staged.read_bytes(), b"!<arch>\nfake archive bytes\n")

    def test_rejects_missing_archive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            missing = tmp_path / "libj2k_c.a"
            output = tmp_path / "dist" / "j2k-c"

            with self.assertRaisesRegex(RuntimeError, "Expected static archive was not built"):
                package.stage_static_archive(missing, output)
            self.assertFalse((output / "lib" / "libj2k.a").exists())


class RenderCmakeConfigTest(unittest.TestCase):
    def test_substitutes_library_and_static_libs(self) -> None:
        template = CMAKE_TEMPLATE.read_text(encoding="utf-8")
        library = package.library_for("x86_64-unknown-linux-gnu", "0.14.0", 1)

        rendered = package.render_cmake_config(template, library, "-lgcc_s -lpthread -lm -lc")

        self.assertIn('set(_j2k_library "${_j2k_prefix}/lib/libj2k.so.1")', rendered)
        self.assertIn('IMPORTED_SONAME "libj2k.so.1"', rendered)
        self.assertIn('set(_j2k_static_libs "-lgcc_s -lpthread -lm -lc")', rendered)
        self.assertNotIn("@LIBRARY_FILE@", rendered)
        self.assertNotIn("@LIBRARY_SONAME@", rendered)
        self.assertNotIn("@STATIC_LIBS@", rendered)

    def test_rejects_unexpanded_placeholder(self) -> None:
        library = package.library_for("aarch64-apple-darwin", "0.14.0", 1)
        template = "set(A @LIBRARY_FILE@)\nset(B @EXTRA_PLACEHOLDER@)\n"

        with self.assertRaisesRegex(RuntimeError, "Unexpanded placeholder"):
            package.render_cmake_config(template, library, "-liconv -lSystem")


class RenderPkgconfigTest(unittest.TestCase):
    def test_substitutes_version_url_and_private_libs(self) -> None:
        template = PKGCONFIG_TEMPLATE.read_text(encoding="utf-8")

        rendered = package.render_pkgconfig(
            template,
            "0.14.0",
            "https://example.invalid/j2k",
            "-liconv -lSystem -lc -lm",
        )

        self.assertIn("Version: 0.14.0\n", rendered)
        self.assertIn("URL: https://example.invalid/j2k\n", rendered)
        self.assertIn("Libs.private: -liconv -lSystem -lc -lm\n", rendered)
        self.assertIn("Cflags.private: -DJ2K_STATIC", rendered)
        self.assertNotIn("@VERSION@", rendered)
        self.assertNotIn("@URL@", rendered)
        self.assertNotIn("@PRIVATE_LIBS@", rendered)

    def test_allows_none_url(self) -> None:
        template = "URL: @URL@\nVersion: @VERSION@\nLibs.private: @PRIVATE_LIBS@\n"

        rendered = package.render_pkgconfig(template, "0.14.0", None, "-lm")

        self.assertEqual(rendered, "URL: \nVersion: 0.14.0\nLibs.private: -lm\n")

    def test_rejects_empty_private_libs(self) -> None:
        template = PKGCONFIG_TEMPLATE.read_text(encoding="utf-8")

        with self.assertRaisesRegex(RuntimeError, "Derived PRIVATE_LIBS is empty"):
            package.render_pkgconfig(template, "0.14.0", "https://example.invalid/j2k", "")

    def test_rejects_whitespace_only_private_libs(self) -> None:
        template = PKGCONFIG_TEMPLATE.read_text(encoding="utf-8")

        with self.assertRaisesRegex(RuntimeError, "Derived PRIVATE_LIBS is empty"):
            package.render_pkgconfig(template, "0.14.0", "https://example.invalid/j2k", "  \t\n ")

    def test_rejects_unexpanded_placeholder(self) -> None:
        template = "Version: @VERSION@\nLibs.private: @PRIVATE_LIBS@\nExtra: @EXTRA_VAR@\n"

        with self.assertRaisesRegex(RuntimeError, "Unexpanded placeholder"):
            package.render_pkgconfig(template, "0.14.0", "https://example.invalid/j2k", "-lm")


if __name__ == "__main__":
    unittest.main()
