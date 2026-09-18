from __future__ import annotations

import hashlib
import shutil
import tempfile
import time
from pathlib import Path

import requests

_MAGIC = {".gz": b"\x1f\x8b", ".zip": b"PK", ".xlsx": b"PK"}


class DownloadError(RuntimeError):
    pass


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def check_magic(tmp_path: Path, dest: Path) -> None:
    expected = _MAGIC.get(dest.suffix)
    if expected is None:
        return
    with open(tmp_path, "rb") as fh:
        head = fh.read(len(expected))
    if head != expected:
        tmp_path.unlink(missing_ok=True)
        raise DownloadError(f"{dest.name}: unexpected file signature {head!r}")


def download(url: str, dest: str | Path, retries: int = 3, timeout: int = 120) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=timeout) as resp:
                resp.raise_for_status()
                with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as tmp:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        tmp.write(chunk)
                    tmp_path = Path(tmp.name)
            check_magic(tmp_path, dest)
            shutil.move(tmp_path, dest)
            Path(str(dest) + ".sha256").write_text(f"{sha256(dest)}  {dest.name}\n")
            return dest
        except (requests.RequestException, DownloadError) as err:
            last_err = err
            if attempt < retries:
                time.sleep(2**attempt)
    raise DownloadError(f"failed to download {url}: {last_err}")
