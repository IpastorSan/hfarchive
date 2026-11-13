"""Append-only JSONL record of what landed on the drive."""

from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path

from lmb.download import DownloadResult
from lmb.hub import RepoStatus

MANIFEST_NAME = "manifest.jsonl"


def lmb_dir(dest: Path) -> Path:
    """Bookkeeping lives in $DEST/_lmb/, next to the category folders."""
    return Path(dest) / "_lmb"


def append(dest: Path, results: list[DownloadResult], statuses: dict[str, RepoStatus]) -> Path:
    d = lmb_dir(dest)
    d.mkdir(parents=True, exist_ok=True)
    path = d / MANIFEST_NAME
    now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    host = platform.node()
    with path.open("a", encoding="utf-8") as fh:
        for res in results:
            m = res.model
            st = statuses.get(m.id)
            record = {
                "timestamp": now,
                "host": host,
                "id": m.id,
                "hf_repo": m.hf_repo,
                "revision": m.revision,
                "commit": res.commit,
                "category": m.category,
                "status": res.status,
                "dest": str(m.dest_dir(dest)),
                "rel_dest": os.path.relpath(m.dest_dir(dest), dest),
                "files": st.matched_files if st else None,
                "bytes_matched": st.matched_bytes if st else None,
                "bytes_transferred_estimate": res.bytes_expected,
                "license": m.license,
                "gated": m.gated,
                "message": res.message or None,
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def read(dest: Path) -> list[dict]:
    path = lmb_dir(dest) / MANIFEST_NAME
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out
