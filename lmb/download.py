"""Repo downloads, driven by the official `hf download` CLI."""

from __future__ import annotations

import concurrent.futures as cf
import logging
import os
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

from lmb.catalog import Model
from lmb.hub import RepoStatus

log = logging.getLogger("lmb")


def resolve_hf_bin() -> str | None:
    """Locate the `hf` CLI: $LMB_HF_BIN, then this interpreter's venv, then PATH.

    The venv sibling is preferred so the CLI matches the huggingface_hub version
    we used to resolve metadata.
    """
    explicit = os.environ.get("LMB_HF_BIN")
    if explicit:
        return explicit if Path(explicit).exists() else None
    sibling = Path(sys.executable).with_name("hf")
    if sibling.exists():
        return str(sibling)
    return shutil.which("hf")


@dataclass
class DownloadResult:
    model: Model
    status: str  # downloaded | skipped-gated | failed | up-to-date
    bytes_expected: int = 0
    commit: str | None = None
    message: str = ""
    returncode: int | None = None


def build_command(model: Model, dest: Path, hf_bin: str, max_workers: int) -> list[str]:
    cmd = [hf_bin, "download", model.hf_repo, "--local-dir", str(model.dest_dir(dest))]
    if model.revision:
        cmd += ["--revision", model.revision]
    for pat in model.include:
        cmd += ["--include", pat]
    for pat in model.exclude:
        cmd += ["--exclude", pat]
    cmd += ["--max-workers", str(max_workers)]
    return cmd


def download_one(
    st: RepoStatus,
    dest: Path,
    hf_bin: str,
    max_workers: int,
    have_token: bool,
    print_lock: threading.Lock,
) -> DownloadResult:
    m = st.model
    if not st.exists:
        return DownloadResult(m, "failed", message=st.error or "repo unavailable")
    if st.gated and not have_token:
        return DownloadResult(
            m,
            "skipped-gated",
            message="gated repo and no HF token found; run `hf auth login` and accept "
            f"the licence at https://huggingface.co/{m.hf_repo}",
        )

    here, missing = st.local_state(dest)
    if st.files and missing == 0:
        log.info("%s: already complete (%s)", m.id, human(here))
        return DownloadResult(m, "up-to-date", bytes_expected=0, commit=st.commit)

    target = m.dest_dir(dest)
    target.mkdir(parents=True, exist_ok=True)
    cmd = build_command(m, dest, hf_bin, max_workers)
    with print_lock:
        log.info(
            "%s -> %s (%s to fetch, %s already there)",
            m.id, target, human(missing), human(here),
        )
        log.debug("exec: %s", " ".join(cmd))

    # `hf download` resumes partial files and skips complete ones by itself.
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-5:]
        return DownloadResult(
            m, "failed",
            bytes_expected=missing,
            commit=st.commit,
            message=" | ".join(tail) or f"hf download exited {proc.returncode}",
            returncode=proc.returncode,
        )
    return DownloadResult(m, "downloaded", bytes_expected=missing, commit=st.commit)


def download_many(
    statuses: list[RepoStatus],
    dest: Path,
    hf_bin: str,
    jobs: int,
    max_workers: int,
    have_token: bool,
) -> list[DownloadResult]:
    lock = threading.Lock()
    results: list[DownloadResult] = []
    if not statuses:
        return results
    with cf.ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
        futures = {
            ex.submit(download_one, st, dest, hf_bin, max_workers, have_token, lock): st
            for st in statuses
        }
        for fut in cf.as_completed(futures):
            st = futures[fut]
            try:
                res = fut.result()
            except Exception as exc:  # pragma: no cover - defensive
                res = DownloadResult(st.model, "failed", message=f"{type(exc).__name__}: {exc}")
            if res.status == "failed":
                log.error("%s: FAILED — %s", res.model.id, res.message)
            elif res.status == "skipped-gated":
                log.warning("%s: skipped (gated) — %s", res.model.id, res.message)
            else:
                log.info("%s: %s", res.model.id, res.status)
            results.append(res)
    order = {st.model.id: i for i, st in enumerate(statuses)}
    results.sort(key=lambda r: order[r.model.id])
    return results


def human(n: float) -> str:
    """Bytes as a decimal-unit string (drive vendors and the Hub both use GB=1e9)."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1000 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.2f} {unit}"
        n /= 1000.0
    return f"{n:.2f} TB"
