# SPDX-License-Identifier: MIT OR Apache-2.0
"""Tests for crates/j2k-c/scripts/package.py and test_package.py helpers."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import os
import pathlib
import re
import tarfile
import tempfile
import unittest
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
CRATE = ROOT / "crates" / "j2k-c"


def load(name: str):
    spec = importlib.util.spec_from_file_location(f"j2k_c_{name}", CRATE / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


package = load("package")
test_package = load("test_package")

DUMPBIN = """Microsoft (R) COFF/PE Dumper Version 14.44.35207.1

Dump of file j2k_c.dll

File Type: DLL

  Section contains the following exports for j2k_c.dll

    00000000 characteristics
    FFFFFFFF time date stamp
        0.00 version
           1 ordinal base
           2 number of functions
           2 number of names

    ordinal hint RVA      name

          1    0 00012340 j2k_abi_version
          2    1 0001A2B0 j2k_version

  Summary

        1000 .data
"""

# J2KConfig.cmake for a GNU/Linux package before Windows support; Unix
# packages must keep producing these bytes.
LINUX_CONFIG = """# SPDX-License-Identifier: MIT OR Apache-2.0
# Relocatable package; consumers require no Rust toolchain.
get_filename_component(_j2k_prefix "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
# The real file is named by C ABI revision; lib/ also holds an unversioned symlink.
set(_j2k_library "${_j2k_prefix}/lib/libj2k.so.1")
# macOS and GNU/Linux packages also ship a static archive.
set(_j2k_static_library "${_j2k_prefix}/lib/libj2k.a")
if(NOT EXISTS "${_j2k_library}" AND NOT EXISTS "${_j2k_static_library}")
    set(J2K_FOUND FALSE)
    set(J2K_NOT_FOUND_MESSAGE "Neither the J2K shared library nor the static archive was found in ${_j2k_prefix}")
elseif(EXISTS "${_j2k_library}" AND NOT TARGET J2K::j2k)
    add_library(J2K::j2k SHARED IMPORTED)
    set_target_properties(J2K::j2k PROPERTIES
        IMPORTED_LOCATION "${_j2k_library}"
        IMPORTED_SONAME "libj2k.so.1"
        INTERFACE_INCLUDE_DIRECTORIES "${_j2k_prefix}/include")
endif()
if(EXISTS "${_j2k_static_library}" AND NOT TARGET J2K::j2k_static)
    add_library(J2K::j2k_static STATIC IMPORTED)
    set(_j2k_static_libs "-lgcc_s -lpthread -lm -lc")
    separate_arguments(_j2k_static_libs NATIVE_COMMAND "${_j2k_static_libs}")
    set_target_properties(J2K::j2k_static PROPERTIES
        IMPORTED_LOCATION "${_j2k_static_library}"
        INTERFACE_INCLUDE_DIRECTORIES "${_j2k_prefix}/include"
        INTERFACE_COMPILE_DEFINITIONS J2K_STATIC
        INTERFACE_LINK_LIBRARIES "${_j2k_static_libs}")
    unset(_j2k_static_libs)
endif()
unset(_j2k_library)
unset(_j2k_static_library)
unset(_j2k_prefix)
"""


def can_symlink(directory: pathlib.Path) -> bool:
    try:
        (directory / "probe-link").symlink_to("probe")
    except (OSError, NotImplementedError):
        return False
    (directory / "probe-link").unlink()
    return True


class LibraryLayoutTest(unittest.TestCase):
    def test_header_declares_exactly_the_rust_exports(self) -> None:
        rust = set()
        for source in (CRATE / "src").glob("*.rs"):
            rust.update(re.findall(
                r"#\[(?:unsafe\()?no_mangle\)?\]\s*pub (?:unsafe )?extern \"C\" fn (j2k_\w+)",
                source.read_text(encoding="utf-8"),
            ))
        declared = package.header_functions()
        self.assertEqual(len(declared), len(set(declared)))
        self.assertEqual(set(declared), rust)

    def test_header_parsing_rejects_repeated_declarations(self) -> None:
        header = "J2K_API uint32_t j2k_abi_version(void);\nJ2K_API uint32_t j2k_abi_version(void);\n"
        with self.assertRaises(RuntimeError):
            package.header_functions(header)

    def test_dumpbin_exports_are_parsed_by_name(self) -> None:
        self.assertEqual(package.dll_exports(DUMPBIN), ["j2k_abi_version", "j2k_version"])

    def test_module_definition_names_the_packaged_dll(self) -> None:
        self.assertEqual(
            package.module_definition("j2k.dll", ["j2k_abi_version", "j2k_version"]),
            'LIBRARY "j2k.dll"\nEXPORTS\n    j2k_abi_version\n    j2k_version\n',
        )

    def test_msvc_toolset_versions_order_numerically(self) -> None:
        paths = [pathlib.PureWindowsPath(rf"C:\VS\VC\Tools\MSVC\{version}\bin\Hostx64\x64\lib.exe")
                 for version in ("14.9.1", "14.44.35207", "14.38.33130")]
        self.assertEqual(max(paths, key=package.msvc_version).parents[3].name, "14.44.35207")

    def test_library_names_per_host(self) -> None:
        linux = package.library_for("x86_64-unknown-linux-gnu", "1.2.3", 1)
        self.assertEqual((linux.packaged, linux.link, linux.implib, linux.soname),
                         ("lib/libj2k.so.1", "lib/libj2k.so", None, "libj2k.so.1"))
        macos = package.library_for("aarch64-apple-darwin", "1.2.3", 1)
        self.assertEqual((macos.packaged, macos.link, macos.soname),
                         ("lib/libj2k.1.dylib", "lib/libj2k.dylib", "@rpath/libj2k.1.dylib"))
        windows = package.library_for("x86_64-pc-windows-msvc", "1.2.3", 1)
        self.assertEqual((windows.built, windows.packaged, windows.link, windows.implib),
                         ("j2k_c.dll", "bin/j2k.dll", None, "lib/j2k.dll.lib"))
        for host in ("x86_64-pc-windows-gnu", "aarch64-pc-windows-msvc", "x86_64-unknown-linux-musl"):
            with self.subTest(host=host), self.assertRaises(RuntimeError):
                package.library_for(host, "1.2.3", 1)

    def test_cmake_config_imports_the_unix_libraries_and_the_dll_on_windows(self) -> None:
        template = (CRATE / "cmake" / "J2KConfig.cmake.in").read_text(encoding="utf-8")
        linux = package.library_for("x86_64-unknown-linux-gnu", "1.2.3", 1)
        self.assertEqual(package.render_cmake_config(template, linux, "-lgcc_s -lpthread -lm -lc"),
                         LINUX_CONFIG)

        windows = package.render_cmake_config(
            template, package.library_for("x86_64-pc-windows-msvc", "1.2.3", 1), "")
        self.assertIn('set(_j2k_library "${_j2k_prefix}/bin/j2k.dll")', windows)
        self.assertIn('IMPORTED_IMPLIB "${_j2k_prefix}/lib/j2k.dll.lib"', windows)
        self.assertNotIn("IMPORTED_SONAME", windows)
        self.assertNotRegex(windows, r"@[A-Z_]+@")


class ArchiveTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = pathlib.Path(temporary.name)
        self.package_dir = self.root / "build-dir"
        (self.package_dir / "lib" / "cmake").mkdir(parents=True)
        (self.package_dir / "lib" / "libj2k.so.1").write_bytes(b"\x7fELF library")
        (self.package_dir / "lib" / "cmake" / "J2KConfig.cmake").write_text("config\n", encoding="utf-8")
        (self.package_dir / "README.md").write_text("readme\n", encoding="utf-8")

    def archive(self, archive_format: str) -> tuple[pathlib.Path, pathlib.Path, str]:
        archive, checksum = package.archive_paths(self.package_dir, "j2k-c-1.2.3-test", archive_format)
        digest = package.write_archive(self.package_dir, "j2k-c-1.2.3-test", archive_format, archive, checksum)
        return archive, checksum, digest

    def test_tar_gz_has_a_sha256sum_file_and_neutral_ownership(self) -> None:
        symlinks = can_symlink(self.root)
        if symlinks:
            (self.package_dir / "lib" / "libj2k.so").symlink_to("libj2k.so.1")
        archive, checksum, digest = self.archive("tar.gz")

        self.assertEqual(archive.name, "j2k-c-1.2.3-test.tar.gz")
        self.assertEqual(digest, hashlib.sha256(archive.read_bytes()).hexdigest())
        self.assertEqual(checksum.read_bytes(), f"{digest}  {archive.name}\n".encode())
        with tarfile.open(archive, "r:gz") as tar:
            members = {member.name: member for member in tar.getmembers()}
        self.assertEqual(sorted(members)[0], "j2k-c-1.2.3-test")
        self.assertTrue(all(name.split("/")[0] == "j2k-c-1.2.3-test" for name in members))
        self.assertTrue(all((m.uid, m.gid, m.uname, m.gname) == (0, 0, "", "") for m in members.values()))
        if symlinks:
            link = members["j2k-c-1.2.3-test/lib/libj2k.so"]
            self.assertTrue(link.issym())
            self.assertEqual(link.linkname, "libj2k.so.1")

        self.assertEqual(test_package.verify_checksum(archive), digest)
        unpacked = test_package.extract(archive, "j2k-c-1.2.3-test", self.root / "unpacked")
        self.assertEqual((unpacked / "lib" / "libj2k.so.1").read_bytes(), b"\x7fELF library")
        if symlinks:
            self.assertEqual(os.readlink(unpacked / "lib" / "libj2k.so"), "libj2k.so.1")

    def test_zip_round_trips_and_rejects_symlinks(self) -> None:
        archive, _, digest = self.archive("zip")
        self.assertEqual(test_package.verify_checksum(archive), digest)
        with zipfile.ZipFile(archive) as zipped:
            self.assertIn("j2k-c-1.2.3-test/lib/cmake/J2KConfig.cmake", zipped.namelist())
        unpacked = test_package.extract(archive, "j2k-c-1.2.3-test", self.root / "unpacked")
        self.assertEqual((unpacked / "README.md").read_text(encoding="utf-8"), "readme\n")

        if can_symlink(self.root):
            (self.package_dir / "lib" / "libj2k.so").symlink_to("libj2k.so.1")
            other = self.root / "other.zip"
            with self.assertRaisesRegex(RuntimeError, "not a regular file"):
                package.write_archive(self.package_dir, "x", "zip", other, self.root / "other.zip.sha256")

    def test_archives_are_never_overwritten(self) -> None:
        archive, checksum, _ = self.archive("tar.gz")
        before = archive.read_bytes()
        with self.assertRaises(FileExistsError):
            package.write_archive(self.package_dir, "j2k-c-1.2.3-test", "tar.gz", archive, checksum)
        self.assertEqual(archive.read_bytes(), before)

    def test_checksum_verification_rejects_a_changed_archive(self) -> None:
        archive, checksum, _ = self.archive("tar.gz")
        archive.write_bytes(archive.read_bytes() + b"\0")
        with self.assertRaisesRegex(RuntimeError, "SHA-256"):
            test_package.verify_checksum(archive)
        checksum.write_text(f"{'0' * 64}  other.tar.gz\n", encoding="ascii")
        with self.assertRaisesRegex(RuntimeError, "must hold one"):
            test_package.verify_checksum(archive)


class ConsumerTestHelpersTest(unittest.TestCase):
    def test_package_name_comes_from_the_archive_name(self) -> None:
        self.assertEqual(test_package.package_name(pathlib.Path("a/j2k-c-1-x.tar.gz")), "j2k-c-1-x")
        self.assertEqual(test_package.package_name(pathlib.Path("j2k-c-1-w.zip")), "j2k-c-1-w")
        with self.assertRaises(RuntimeError):
            test_package.package_name(pathlib.Path("j2k-c-1-x.tar"))

    def test_tar_members_must_stay_in_the_package(self) -> None:
        def member(name: str, kind: bytes = tarfile.REGTYPE, link: str = "") -> tarfile.TarInfo:
            info = tarfile.TarInfo(name)
            info.type = kind
            info.linkname = link
            return info

        test_package.check_tar_members(
            [member("p", tarfile.DIRTYPE), member("p/lib/libj2k.so", tarfile.SYMTYPE, "libj2k.so.1")], "p")
        for bad in (
            member("/p/x"),
            member("p/../x"),
            member("q/x"),
            member("p/lib/x", tarfile.SYMTYPE, "../../etc/passwd"),
            member("p/lib/x", tarfile.SYMTYPE, "/etc/passwd"),
            member("p/lib/x", tarfile.LNKTYPE, "p/lib/y"),
            member("p/dev", tarfile.CHRTYPE),
        ):
            with self.subTest(name=bad.name, link=bad.linkname), self.assertRaises(RuntimeError):
                test_package.check_tar_members([bad], "p")

    def test_zip_members_must_stay_in_the_package(self) -> None:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zipped:
            zipped.writestr("p/../evil.txt", "x")
        with zipfile.ZipFile(buffer) as zipped, self.assertRaises(RuntimeError):
            test_package.check_zip_members(zipped.infolist(), "p")

    def test_platform_floors_are_parsed(self) -> None:
        objdump = (
            "0000000000000000      DF *UND*  0000000000000000 (GLIBC_2.2.5) memcpy\n"
            "0000000000000000      DF *UND*  0000000000000000 (GLIBC_2.28) fcntl64\n"
            "0000000000000000      DF *UND*  0000000000000000 (GLIBC_2.3) __tls_get_addr\n"
        )
        self.assertEqual(test_package.newest_glibc(objdump), (2, 28))
        otool = (
            "Load command 9\n      cmd LC_BUILD_VERSION\n  cmdsize 32\n platform 1\n"
            "    minos 11.0\n      sdk 27.2\n"
        )
        self.assertEqual(test_package.macos_minimum(otool), (11, 0))
        with self.assertRaises(RuntimeError):
            test_package.newest_glibc("no versions")
        with self.assertRaises(RuntimeError):
            test_package.macos_minimum("Load command 1\n      cmd LC_UUID\n")


if __name__ == "__main__":
    unittest.main()
