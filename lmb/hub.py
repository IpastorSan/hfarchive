"""Hub metadata: existence checks, matched-file resolution and size accounting."""

from __future__ import annotations

import concurrent.futures as cf
from dataclasses import dataclass, field
from pathlib import Path

from huggingface_hub import HfApi, get_token
from huggingface_hub.utils import (
    EntryNotFoundError,
    GatedRepoError,
    RepositoryNotFoundError,
    filter_repo_objects,
)

from lmb.catalog import Model


def token_available() -> bool:
    """True if an HF token is reachable (HF_TOKEN env or `hf auth login`).

    The token itself is never returned, logged or printed.
    """
    try:
        return bool(get_token())
    except Exception:
        return False


@dataclass
class RepoStatus:
    model: Model
    exists: bool = False
    gated: bool = False
    commit: str | None = None
    files: list[tuple[str, int]] = field(default_factory=list)
    total_repo_bytes: int = 0
    error: str | None = None

    @property
    def matched_files(self) -> int:
        return len(self.files)

    @property
    def matched_bytes(self) -> int:
        return sum(size for _, size in self.files)

    def local_state(self, dest: Path) -> tuple[int, int]:
        """(bytes already present and complete, bytes still missing).

        A file counts as complete when it exists locally with the exact size
        the Hub reports, which is what `hf download` itself uses to skip work.
        """
        here = missing = 0
        root = self.model.dest_dir(dest)
        for rel, size in self.files:
            p = root / rel
            try:
                if p.is_file() and p.stat().st_size == size:
                    here += size
                    continue
            except OSError:
                pass
            missing += size
        return here, missing


def _match(model: Model, names: list[str]) -> list[str]:
    return list(
        filter_repo_objects(
            names,
            allow_patterns=model.include or None,
            ignore_patterns=model.exclude or None,
        )
    )


def probe(model: Model, api: HfApi | None = None) -> RepoStatus:
    """Resolve one repo against the Hub. Never raises."""
    api = api or HfApi()
    st = RepoStatus(model=model)
    try:
        info = api.model_info(
            model.hf_repo, revision=model.revision, files_metadata=True
        )
    except RepositoryNotFoundError:
        st.error = "repo not found (or private)"
        return st
    except GatedRepoError:
        # Metadata itself is gated: we know it exists but cannot size it.
        st.exists = True
        st.gated = True
        st.error = "gated: metadata requires an accepted licence + token"
        return st
    except EntryNotFoundError:
        st.error = f"revision not found: {model.revision}"
        return st
    except Exception as exc:  # network, rate limit, ...
        st.error = f"{type(exc).__name__}: {exc}"
        return st

    st.exists = True
    st.gated = bool(info.gated)
    st.commit = info.sha
    sizes = {s.rfilename: (s.size or 0) for s in info.siblings}
    st.total_repo_bytes = sum(sizes.values())
    st.files = [(n, sizes[n]) for n in _match(model, list(sizes))]
    if not st.files:
        st.error = "include/exclude globs matched no files"
    return st


def probe_all(models: list[Model], workers: int = 8) -> list[RepoStatus]:
    """Probe repos concurrently, preserving input order."""
    api = HfApi()
    if not models:
        return []
    with cf.ThreadPoolExecutor(max_workers=max(1, min(workers, len(models)))) as ex:
        return list(ex.map(lambda m: probe(m, api), models))
