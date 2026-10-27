"""Shared helpers for the dataset download scripts.

Each script lists its remote files with the provider's MD5 when one is
published. SHA-256 checksums are pinned in scripts/download/SHA256SUMS
(sha256sum format, paths relative to data/raw/). A file without a pinned
checksum is recorded on first download; a pinned checksum that does not match
aborts the script.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "data" / "raw"
SUMS = Path(__file__).with_name("SHA256SUMS")
CHUNK = 1 << 20


@dataclass(frozen=True)
class Remote:
    url: str
    name: str
    size: int | None = None
    md5: str | None = None


def read_sums() -> dict[str, str]:
    if not SUMS.exists():
        return {}
    sums = {}
    for line in SUMS.read_text().splitlines():
        if line.strip():
            digest, path = line.split(maxsplit=1)
            sums[path.strip()] = digest
    return sums


def record_sum(key: str, digest: str) -> None:
    sums = read_sums()
    sums[key] = digest
    SUMS.write_text("".join(f"{sums[k]}  {k}\n" for k in sorted(sums)))


def digests(path: Path) -> tuple[str, str]:
    md5, sha = hashlib.md5(), hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(CHUNK):
            md5.update(block)
            sha.update(block)
    return md5.hexdigest(), sha.hexdigest()


def _download(url: str, part: Path, size: int | None) -> None:
    done = part.stat().st_size if part.exists() else 0
    request = urllib.request.Request(url, headers={"User-Agent": "cfc-download"})
    if done:
        request.add_header("Range", f"bytes={done}-")
    with urllib.request.urlopen(request, timeout=120) as response:
        mode = "ab" if done and response.status == 206 else "wb"
        if mode == "wb":
            done = 0
        total = size or int(response.headers.get("Content-Length", 0)) + done
        next_report = 0.0
        with part.open(mode) as fh:
            while block := response.read(CHUNK):
                fh.write(block)
                done += len(block)
                if total and done / total >= next_report:
                    print(f"  {done / 1e6:9.1f} / {total / 1e6:.1f} MB", file=sys.stderr)
                    next_report += 0.1


def fetch(dataset: str, remote: Remote, extract: bool = False) -> Path:
    """Download one file into data/raw/<dataset>/, verify it, optionally unzip it."""
    dest = RAW / dataset / remote.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    key = f"{dataset}/{remote.name}"
    pinned = read_sums().get(key)

    if not (dest.exists() and pinned and digests(dest)[1] == pinned):
        print(f"{key}: downloading {remote.url}", file=sys.stderr)
        part = dest.with_name(dest.name + ".part")
        _download(remote.url, part, remote.size)
        if remote.size is not None and part.stat().st_size != remote.size:
            raise SystemExit(f"{key}: size {part.stat().st_size} != {remote.size}")
        md5, sha = digests(part)
        if remote.md5 is not None and md5 != remote.md5:
            raise SystemExit(f"{key}: MD5 {md5} != published {remote.md5}")
        if pinned is not None and sha != pinned:
            raise SystemExit(f"{key}: SHA-256 {sha} != pinned {pinned}")
        part.replace(dest)
        if pinned is None:
            record_sum(key, sha)
            print(f"{key}: recorded SHA-256 {sha} in {SUMS.name}", file=sys.stderr)
    print(f"{key}: OK", file=sys.stderr)

    if extract and zipfile.is_zipfile(dest):
        target = dest.with_suffix("")
        if not target.exists():
            with zipfile.ZipFile(dest) as zf:
                zf.extractall(target)
            print(f"{key}: extracted to {target.relative_to(REPO)}", file=sys.stderr)
    return dest
