# SPDX-License-Identifier: MIT OR Apache-2.0

from __future__ import annotations

import contextlib
import hashlib
import io
import pathlib
import sys
import tempfile
import unittest

from scripts import c_release_assets

VERSION = "1.2.3"


def write_archives(directory: pathlib.Path, version: str = VERSION) -> dict[str, bytes]:
    """One fake archive per platform, each with a matching .sha256 file."""
    contents = {}
    for name in c_release_assets.archive_names(version):
        data = f"archive {name}".encode()
        (directory / name).write_bytes(data)
        digest = hashlib.sha256(data).hexdigest()
        (directory / f"{name}.sha256").write_text(f"{digest}  {name}\n", encoding="ascii")
        contents[name] = data
    return contents


class CReleaseAssetsTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self.directory = pathlib.Path(self._temporary.name)

    def test_release_names_cover_every_platform(self) -> None:
        self.assertEqual(
            c_release_assets.archive_names(VERSION),
            [
                "j2k-c-1.2.3-linux-x86_64.tar.gz",
                "j2k-c-1.2.3-linux-aarch64.tar.gz",
                "j2k-c-1.2.3-macos-aarch64.tar.gz",
                "j2k-c-1.2.3-windows-x86_64.zip",
            ],
        )
        self.assertEqual(len(c_release_assets.release_files(VERSION)), 9)
        for version in ("1.2", "../1.2.3", "1.2.3/x", "v1.2.3"):
            with self.subTest(version=version), self.assertRaises(c_release_assets.AssetError):
                c_release_assets.archive_names(version)

    def test_manifest_lists_each_verified_archive(self) -> None:
        contents = write_archives(self.directory)
        manifest, digest = c_release_assets.write_manifest(self.directory, VERSION)

        self.assertEqual(manifest.name, "j2k-c-1.2.3-SHA256SUMS")
        self.assertEqual(digest, hashlib.sha256(manifest.read_bytes()).hexdigest())
        self.assertEqual(
            c_release_assets.read_checksums(manifest),
            {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()},
        )
        files = c_release_assets.verify_release(self.directory, VERSION, digest, tag="v1.2.3")
        self.assertEqual([path.name for path in files], c_release_assets.release_files(VERSION))

    def test_manifest_requires_every_platform_and_nothing_else(self) -> None:
        write_archives(self.directory)
        (self.directory / "j2k-c-1.2.3-windows-x86_64.zip").unlink()
        with self.assertRaisesRegex(c_release_assets.AssetError, "missing.*windows"):
            c_release_assets.write_manifest(self.directory, VERSION)

        write_archives(self.directory)
        (self.directory / "c-package-test-fixture.j2c").write_bytes(b"x")
        with self.assertRaisesRegex(c_release_assets.AssetError, "unexpected.*fixture"):
            c_release_assets.write_manifest(self.directory, VERSION)

    def test_manifest_rejects_an_archive_that_differs_from_its_checksum_file(self) -> None:
        write_archives(self.directory)
        (self.directory / "j2k-c-1.2.3-macos-aarch64.tar.gz").write_bytes(b"changed")
        with self.assertRaisesRegex(c_release_assets.AssetError, "macos-aarch64"):
            c_release_assets.write_manifest(self.directory, VERSION)

    def test_checksum_files_must_be_exact_sha256sum_lines(self) -> None:
        write_archives(self.directory)
        target = self.directory / "j2k-c-1.2.3-linux-x86_64.tar.gz.sha256"
        digest = target.read_text(encoding="ascii").split()[0]
        for text in (
            f"{digest} j2k-c-1.2.3-linux-x86_64.tar.gz\n",
            f"{digest}  j2k-c-1.2.3-linux-x86_64.tar.gz",
            f"{digest.upper()}  j2k-c-1.2.3-linux-x86_64.tar.gz\n",
            f"{digest}  j2k-c-1.2.3-linux-aarch64.tar.gz\n",
        ):
            with self.subTest(text=text):
                target.write_text(text, encoding="ascii")
                with self.assertRaises(c_release_assets.AssetError):
                    c_release_assets.write_manifest(self.directory, VERSION)

    def test_verify_rejects_files_changed_after_the_manifest(self) -> None:
        write_archives(self.directory)
        manifest, digest = c_release_assets.write_manifest(self.directory, VERSION)

        with self.assertRaisesRegex(c_release_assets.AssetError, "recorded"):
            c_release_assets.verify_release(self.directory, VERSION, "0" * 64)

        # A swapped archive with a matching .sha256 still differs from the manifest.
        swapped = "j2k-c-1.2.3-linux-aarch64.tar.gz"
        (self.directory / swapped).write_bytes(b"swapped")
        (self.directory / f"{swapped}.sha256").write_text(
            f"{hashlib.sha256(b'swapped').hexdigest()}  {swapped}\n", encoding="ascii"
        )
        with self.assertRaisesRegex(c_release_assets.AssetError, "differ"):
            c_release_assets.verify_release(self.directory, VERSION, digest)

        (self.directory / "extra.txt").write_text("x", encoding="ascii")
        with self.assertRaisesRegex(c_release_assets.AssetError, "unexpected"):
            c_release_assets.verify_release(self.directory, VERSION, digest)
        self.assertTrue(manifest.is_file())

    def test_verify_requires_the_matching_tag(self) -> None:
        write_archives(self.directory)
        _, digest = c_release_assets.write_manifest(self.directory, VERSION)
        with self.assertRaisesRegex(c_release_assets.AssetError, "tag v1.2.4"):
            c_release_assets.verify_release(self.directory, VERSION, digest, tag="v1.2.4")

    def test_manifest_is_never_rewritten(self) -> None:
        write_archives(self.directory)
        manifest, _ = c_release_assets.write_manifest(self.directory, VERSION)
        before = manifest.read_bytes()
        with self.assertRaisesRegex(c_release_assets.AssetError, "unexpected.*SHA256SUMS"):
            c_release_assets.write_manifest(self.directory, VERSION)
        self.assertEqual(manifest.read_bytes(), before)

    def test_main_writes_github_outputs_and_reports_errors(self) -> None:
        write_archives(self.directory)
        outputs = self.directory.parent / f"{self.directory.name}-outputs"
        summary = self.directory.parent / f"{self.directory.name}-summary"
        self.addCleanup(lambda: [path.unlink() for path in (outputs, summary) if path.exists()])
        with contextlib.redirect_stdout(io.StringIO()):
            status = c_release_assets.main(
                ["manifest", "--version", VERSION, "--dir", str(self.directory),
                 "--github-output", str(outputs), "--summary", str(summary)]
            )
        self.assertEqual(status, 0)
        digest = hashlib.sha256(
            (self.directory / c_release_assets.manifest_name(VERSION)).read_bytes()
        ).hexdigest()
        self.assertEqual(outputs.read_text(encoding="utf-8"), f"sha256={digest}\n")
        self.assertIn(digest, summary.read_text(encoding="utf-8"))
        with contextlib.redirect_stderr(io.StringIO()) as errors:
            status = c_release_assets.main(
                ["verify", "--version", VERSION, "--dir", str(self.directory),
                 "--manifest-sha256", "f" * 64]
            )
        self.assertEqual(status, 1)
        self.assertIn("recorded", errors.getvalue())

    @unittest.skipUnless(sys.version_info >= (3, 11), "tomllib needs Python 3.11")
    def test_workspace_version_reads_the_workspace_package(self) -> None:
        cargo_toml = self.directory / "Cargo.toml"
        cargo_toml.write_text('[workspace.package]\nversion = "4.5.6"\n', encoding="utf-8")
        self.assertEqual(c_release_assets.workspace_version(cargo_toml), "4.5.6")
        cargo_toml.write_text('[workspace]\nmembers = []\n', encoding="utf-8")
        with self.assertRaises(c_release_assets.AssetError):
            c_release_assets.workspace_version(cargo_toml)


if __name__ == "__main__":
    unittest.main()
