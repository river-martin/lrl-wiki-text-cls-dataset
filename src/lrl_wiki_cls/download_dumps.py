"""Download the Wikimedia dump files needed to build the dataset.

Downloads into ``config.DATA_DIR`` using the pinned ``config.DUMP_DATE``.
Downloads are resumable (via HTTP ``Range``), validated against the advertised
``Content-Length``, and (optionally) checked for compression integrity, so an
interrupted transfer cannot silently produce a truncated dump.

Usage::

    python -m lrl_wiki_cls.download_dumps            # download missing files
    python -m lrl_wiki_cls.download_dumps --force    # re-download everything
    python -m lrl_wiki_cls.download_dumps --verify   # also verify compression
"""

from __future__ import annotations

import argparse
import bz2
import gzip
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

from tqdm import tqdm

from lrl_wiki_cls.config import (
    CACHE_DIR,
    DATA_DIR,
    DUMP_DATE,
    DUMP_FILES,
    dump_path,
    dump_url,
)

# Wikimedia requires a descriptive User-Agent; include a contact address.
USER_AGENT = (
    "lrl-wiki-text-cls-dataset/0.1 "
    "(https://github.com/river-martin; mailto:rivermartin128@gmail.com)"
)
MANIFEST_PATH = DATA_DIR / "dump_manifest.json"
_CHUNK = 1 << 16


def _remote_size(url: str) -> int | None:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            length = response.headers.get("Content-Length")
            return int(length) if length is not None else None
    except urllib.error.URLError as exc:  # pragma: no cover - network errors
        raise RuntimeError(f"Could not query {url}: {exc}") from exc


def verify_compression(path: Path) -> None:
    """Fully read a .gz/.bz2 file to confirm its CRC / stream terminator is intact.

    The format is detected from the file's magic bytes so this also works for
    in-progress ``.part`` files.
    """
    with open(path, "rb") as raw:
        magic = raw.read(3)
    if magic.startswith(b"\x1f\x8b"):
        opener = gzip.open
    elif magic.startswith(b"BZh"):
        opener = bz2.open
    else:
        raise ValueError(f"{path} is neither gzip nor bzip2 (magic={magic!r})")
    with opener(path, "rb") as f:
        for _ in f:
            pass


def download_file(url: str, dest: Path, force: bool = False, verify: bool = False) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    expected = _remote_size(url)

    if dest.exists() and not force:
        have = dest.stat().st_size
        if expected is None or have == expected:
            print(f"✓ {dest.name} already present ({have:,} bytes)")
            return dest
        print(f"↻ {dest.name} is {have:,} bytes, expected {expected:,}; re-downloading")

    part = dest.with_name(dest.name + ".part")
    start = part.stat().st_size if part.exists() and not force else 0

    # A previous run may have finished the transfer but crashed before
    # validating/renaming the .part file.
    if start and expected is not None and start == expected:
        if verify:
            print(f"  verifying {part.name} …")
            verify_compression(part)
        os.replace(part, dest)
        print(f"  saved {dest} ({start:,} bytes)")
        return dest

    headers = {"User-Agent": USER_AGENT}
    if start:
        headers["Range"] = f"bytes={start}-"

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            resumed = start > 0 and response.status == 206
            mode = "ab" if resumed else "wb"
            if not resumed:
                start = 0
            remaining = (expected - start) if expected else None
            with open(part, mode) as f, tqdm(
                desc=dest.name,
                total=remaining,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                initial=start,
            ) as bar:
                while True:
                    chunk = response.read(_CHUNK)
                    if not chunk:
                        break
                    f.write(chunk)
                    bar.update(len(chunk))
    except urllib.error.HTTPError as exc:  # pragma: no cover - network errors
        raise RuntimeError(f"Download failed for {url}: {exc}") from exc

    size = part.stat().st_size
    if expected is not None and size != expected:
        raise RuntimeError(
            f"{part.name} is {size:,} bytes but the server advertised {expected:,}; "
            "the download is incomplete. Re-run to resume."
        )

    if verify:
        print(f"  verifying {part.name} …")
        verify_compression(part)

    os.replace(part, dest)
    print(f"  saved {dest} ({size:,} bytes)")
    return dest


def download_all(files: tuple[str, ...] = DUMP_FILES, force: bool = False, verify: bool = False) -> dict:
    manifest = {"dump_date": DUMP_DATE, "files": {}}
    for name in files:
        url = dump_url(name)
        dest = download_file(url, dump_path(name), force=force, verify=verify)
        manifest["files"][name] = {
            "path": str(dest.relative_to(DATA_DIR.parent)),
            "bytes": dest.stat().st_size,
            "url": url,
        }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest: {MANIFEST_PATH}")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download files even if present")
    parser.add_argument("--verify", action="store_true", help="verify compression integrity after download")
    parser.add_argument("--files", nargs="*", default=None, help="subset of DUMP_FILES to fetch")
    args = parser.parse_args(argv)
    files = tuple(args.files) if args.files else DUMP_FILES
    download_all(files=files, force=args.force, verify=args.verify)


if __name__ == "__main__":
    main()
