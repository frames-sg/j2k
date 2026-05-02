#!/usr/bin/env bash
set -euo pipefail

manifest="${SIGNINUM_PARITY_CORPUS_MANIFEST:-${1:-corpus/wsi-samples/manifest.json}}"
out_dir="${SIGNINUM_PARITY_CORPUS_DIR:-${2:-corpus/wsi-samples}}"

python3 - "$manifest" "$out_dir" <<'PY'
import hashlib
import json
import os
import pathlib
import sys
import urllib.parse
import urllib.request


def fail(message):
    print(f"parity-corpus-fetch: {message}", file=sys.stderr)
    sys.exit(1)


def entry_value(entry, *keys):
    for key in keys:
        value = entry.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def safe_relative_path(raw, url):
    if raw:
        path = pathlib.PurePosixPath(raw)
    else:
        parsed = urllib.parse.urlparse(url)
        name = pathlib.PurePosixPath(parsed.path).name
        if not name:
            fail(f"entry {url!r} has no filename; add path, file, or filename")
        path = pathlib.PurePosixPath(name)

    if path.is_absolute() or ".." in path.parts:
        fail(f"unsafe output path {str(path)!r}")
    return pathlib.Path(*path.parts)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


manifest_path = pathlib.Path(sys.argv[1])
out_dir = pathlib.Path(sys.argv[2])

if not manifest_path.exists():
    fail(f"manifest not found: {manifest_path}")

with manifest_path.open("r", encoding="utf-8") as handle:
    manifest = json.load(handle)

entries = manifest.get("fixtures") or manifest.get("samples")
if not isinstance(entries, list):
    fail("manifest must contain a fixtures or samples list")

out_dir.mkdir(parents=True, exist_ok=True)

for entry in entries:
    if not isinstance(entry, dict):
        fail("manifest entries must be objects")

    url = entry_value(entry, "url", "source_url", "download_url")
    expected = entry_value(entry, "sha256", "sha-256")
    if not url or not expected:
        fail("each entry must contain url and sha256")

    relative = safe_relative_path(entry_value(entry, "path", "file", "filename") or "", url)
    destination = out_dir / relative
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and sha256_file(destination).lower() == expected.lower():
        print(f"ok {destination}")
        continue

    tmp = destination.with_suffix(destination.suffix + ".part")
    print(f"fetch {url} -> {destination}")
    with urllib.request.urlopen(url, timeout=60) as response:
        tmp.write_bytes(response.read())

    actual = sha256_file(tmp)
    if actual.lower() != expected.lower():
        tmp.unlink(missing_ok=True)
        fail(f"sha256 mismatch for {destination}: expected {expected}, got {actual}")

    os.replace(tmp, destination)
    print(f"ok {destination}")
PY
