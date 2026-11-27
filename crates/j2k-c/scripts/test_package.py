#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Test a j2k-c package archive the way a consumer uses it, in a new directory.

Checks the archive against its .sha256 file and unpacks it. Then builds and
runs the native consumer tests (through CMake, and through pkg-config outside
Windows) and the packaged examples against the unpacked package. Needs CMake
3.20 or newer and C and C++ compilers, and pkg-config outside Windows.
"""

import argparse
import hashlib
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import tarfile
import zipfile

CRATE = Path(__file__).resolve().parents[1]
NATIVE_TESTS = CRATE / "native" / "tests"
ARCHIVE_SUFFIXES = (".tar.gz", ".zip")
CHECKSUM_LINE = re.compile(r"([0-9a-f]{64})  ([^\s/\\]+)\n")
VERSION = re.compile(r"\d+(?:\.\d+)*")


def run(*args):
    """Run a command, echoing it; a nonzero exit raises CalledProcessError."""
    print("+ " + " ".join(str(arg) for arg in args), flush=True)
    subprocess.run([str(arg) for arg in args], check=True)


def command_output(*args):
    return subprocess.run([str(arg) for arg in args], check=True, text=True,
                          stdout=subprocess.PIPE).stdout


def package_name(archive):
    """The package's top directory, which is the archive's name without its suffix."""
    for suffix in ARCHIVE_SUFFIXES:
        if archive.name.endswith(suffix) and len(archive.name) > len(suffix):
            return archive.name[:-len(suffix)]
    raise RuntimeError(f"{archive.name} is not a .tar.gz or .zip archive")


def sha256_file(path):
    """Hex SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as binary:
        while True:
            block = binary.read(1 << 20)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def verify_checksum(archive):
    """Require ARCHIVE.sha256 to hold one `sha256sum` line for the archive; return the digest."""
    checksum = archive.with_name(f"{archive.name}.sha256")
    match = CHECKSUM_LINE.fullmatch(checksum.read_text(encoding="ascii"))
    if not match or match.group(2) != archive.name:
        raise RuntimeError(f"{checksum.name} must hold one '<sha256>  {archive.name}' line")
    actual = sha256_file(archive)
    if actual != match.group(1):
        raise RuntimeError(f"{archive.name} has SHA-256 {actual}, not the {match.group(1)} "
                           f"in {checksum.name}")
    return actual


def member_path(name, top):
    """Reject an archive member outside the top directory."""
    path = PurePosixPath(name)
    if path.is_absolute() or "\\" in name or not path.parts or path.parts[0] != top \
            or ".." in path.parts:
        raise RuntimeError(f"Archive member is outside {top}/: {name!r}")
    return path


def check_tar_members(members, top):
    """Allow files, directories, and symlinks to a file in the same directory."""
    for member in members:
        member_path(member.name, top)
        if member.issym():
            target = PurePosixPath(member.linkname)
            if len(target.parts) != 1 or target.name in ("", ".", "..") or "\\" in member.linkname:
                raise RuntimeError(f"Archive symlink leaves its directory: {member.name} -> {member.linkname}")
        elif not (member.isreg() or member.isdir()):
            raise RuntimeError(f"Archive member is not a file, directory or symlink: {member.name}")


def check_zip_members(infos, top):
    for info in infos:
        member_path(info.filename, top)
        if stat.S_ISLNK(info.external_attr >> 16):
            raise RuntimeError(f"Zip archives must not hold symlinks: {info.filename}")


def extract(archive, top, destination):
    """Unpack the archive into the new directory destination; return the package directory."""
    destination.mkdir()
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zipped:
            check_zip_members(zipped.infolist(), top)
            zipped.extractall(destination)
    else:
        with tarfile.open(archive, "r:gz") as tar:
            check_tar_members(tar.getmembers(), top)
            # Python 3.12 and later also apply their own extraction checks.
            if hasattr(tarfile, "data_filter"):
                tar.extractall(destination, filter="data")
            else:
                tar.extractall(destination)
    package = destination / top
    if [entry.name for entry in destination.iterdir()] != [top] or not package.is_dir():
        raise RuntimeError(f"{archive.name} must hold exactly one top directory, {top}/")
    return package


def version_tuple(text):
    if not VERSION.fullmatch(text):
        raise RuntimeError(f"Not a dotted version: {text!r}")
    return tuple(int(part) for part in text.split("."))


def newest_glibc(objdump_output):
    """The newest GLIBC_x.y symbol version `objdump -T` reports."""
    versions = [version_tuple(found) for found in re.findall(r"\bGLIBC_(\d+(?:\.\d+)*)\b", objdump_output)]
    if not versions:
        raise RuntimeError("objdump -T reported no GLIBC symbol versions")
    return max(versions)


def macos_minimum(otool_output):
    """The minimum macOS that LC_BUILD_VERSION in `otool -l` output records."""
    found = re.findall(r"cmd LC_BUILD_VERSION\n(?:[ \t]*\w+ [^\n]*\n)*?[ \t]*minos (\S+)", otool_output)
    if len(found) != 1:
        raise RuntimeError("otool -l must report exactly one LC_BUILD_VERSION minos")
    return version_tuple(found[0])


def only_file(package, pattern):
    matches = [path for path in package.glob(pattern) if path.is_file() and not path.is_symlink()]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {pattern} in the package, found {len(matches)}")
    return matches[0]


def check_platform_floor(package, max_glibc, max_macos):
    """Fail when the library needs a newer glibc or macOS than the stated floor."""
    if max_glibc:
        library = only_file(package, "lib/libj2k.so.*")
        needed = newest_glibc(command_output("objdump", "-T", library))
        print(f"{library.name} needs GLIBC_{'.'.join(map(str, needed))}")
        if needed > version_tuple(max_glibc):
            raise RuntimeError(f"{library.name} needs glibc newer than {max_glibc}")
    if max_macos:
        library = only_file(package, "lib/libj2k.*.dylib")
        needed = macos_minimum(command_output("otool", "-l", library))
        print(f"{library.name} needs macOS {'.'.join(map(str, needed))}")
        if needed > version_tuple(max_macos):
            raise RuntimeError(f"{library.name} needs macOS newer than {max_macos}")


def cmake_build(source, build, package, *definitions):
    run("cmake", "-S", source, "-B", build, f"-DCMAKE_PREFIX_PATH={package}",
        "-DCMAKE_BUILD_TYPE=Release", *definitions)
    run("cmake", "--build", build, "--config", "Release")


def program(build, name):
    """A built program: Visual Studio's multi-config layout, or a single-config one."""
    for candidate in (build / "Release" / f"{name}.exe", build / f"{name}.exe", build / name):
        if candidate.is_file():
            return candidate
    raise RuntimeError(f"{name} was not built in {build}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="NAME.tar.gz or NAME.zip, beside NAME.<format>.sha256")
    parser.add_argument("--fixture", type=Path, required=True,
                        help="Image written by `cargo run -p j2k-c --example make_fixture`")
    parser.add_argument("--work", type=Path, required=True, help="New directory for the unpacked package and builds")
    parser.add_argument("--max-glibc", help="Fail if the Linux library needs a newer glibc, such as 2.28")
    parser.add_argument("--max-macos", help="Fail if the macOS library needs a newer macOS, such as 11.0")
    return parser.parse_args()


def main():
    args = parse_args()
    archive, fixture, work = args.archive.resolve(), args.fixture.resolve(), args.work.resolve()
    if work.exists() or work.is_symlink():
        raise RuntimeError(f"Work directory already exists; choose a new one: {work}")
    if not fixture.is_file():
        raise RuntimeError(f"Fixture not found: {fixture}")
    name = package_name(archive)
    print(f"{archive.name}: SHA-256 {verify_checksum(archive)} matches its .sha256 file")
    work.mkdir(parents=True)
    package = extract(archive, name, work / "unpacked")
    check_platform_floor(package, args.max_glibc, args.max_macos)
    tests = work / "tests"
    cmake_build(NATIVE_TESTS, tests, package, f"-DJ2K_FIXTURE={fixture}")
    run("ctest", "--test-dir", tests, "-C", "Release", "--output-on-failure", "--no-tests=error")
    examples = work / "examples"
    cmake_build(package / "examples", examples, package)
    for example in ("j2k_decode", "j2k_decode_cpp"):
        run(program(examples, example), fixture)
    print(f"{name}: consumer tests and examples passed")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, subprocess.CalledProcessError, tarfile.TarError,
            zipfile.BadZipFile) as error:
        print(f"test_package: {error}", file=sys.stderr)
        sys.exit(1)
