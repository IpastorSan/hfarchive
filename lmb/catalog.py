"""Catalog loading and profile resolution."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULTS = {
    "min_free_gb": 50,
    "large_repo_gb": 100,
    "jobs": 2,
    "max_workers_per_repo": 8,
}


class CatalogError(Exception):
    """Raised when the YAML catalog is malformed or a selection is impossible."""


@dataclass
class Model:
    id: str
    hf_repo: str
    category: str
    revision: str | None = None
    include: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    license: str | None = None
    gated: bool = False
    approx_bytes: int = 0
    enabled: bool = True
    notes: str | None = None

    @property
    def org(self) -> str:
        return self.hf_repo.split("/", 1)[0]

    @property
    def name(self) -> str:
        return self.hf_repo.split("/", 1)[1]

    def dest_dir(self, dest: Path) -> Path:
        """Layout on the drive: $DEST/<category>/<repo_org>/<repo_name>/."""
        return Path(dest) / self.category / self.org / self.name


@dataclass
class Profile:
    name: str
    description: str = ""
    any_tags: list[str] = field(default_factory=list)
    all_tags: list[str] = field(default_factory=list)
    ids: list[str] = field(default_factory=list)
    all: bool = False
    include_disabled: bool = False

    def matches(self, m: Model) -> bool:
        if self.all:
            return True
        if m.id in self.ids:
            return True
        tags = set(m.tags)
        if self.all_tags and tags.issuperset(self.all_tags):
            return True
        if self.any_tags and tags & set(self.any_tags):
            return True
        return False


@dataclass
class Catalog:
    path: Path
    defaults: dict
    profiles: dict[str, Profile]
    models: list[Model]

    def by_id(self, model_id: str) -> Model:
        for m in self.models:
            if m.id == model_id:
                return m
        raise CatalogError(f"unknown model id: {model_id}")

    def select(
        self,
        profile: str,
        only: list[str] | None = None,
        exclude: list[str] | None = None,
        enable: list[str] | None = None,
        categories: list[str] | None = None,
    ) -> list[Model]:
        if profile not in self.profiles:
            raise CatalogError(
                f"unknown profile {profile!r}; available: {', '.join(sorted(self.profiles))}"
            )
        prof = self.profiles[profile]
        forced = set(enable or [])
        for fid in forced:
            self.by_id(fid)  # validate early

        picked: list[Model] = []
        for m in self.models:
            if m.id in forced:
                picked.append(m)
                continue
            if not prof.matches(m):
                continue
            if not m.enabled and not prof.include_disabled:
                continue
            picked.append(m)

        if only:
            wanted = set(only)
            for oid in wanted:
                self.by_id(oid)
            # --only may name models the profile did not select; honour it.
            picked = [m for m in self.models if m.id in wanted]
        if exclude:
            for eid in exclude:
                self.by_id(eid)
            picked = [m for m in picked if m.id not in set(exclude)]
        if categories:
            picked = [m for m in picked if m.category in set(categories)]
        return picked


def load_catalog(path: str | Path) -> Catalog:
    path = Path(path)
    if not path.is_file():
        raise CatalogError(f"catalog not found: {path}")
    raw = yaml.safe_load(path.read_text()) or {}

    defaults = dict(DEFAULTS)
    defaults.update(raw.get("defaults") or {})

    profiles: dict[str, Profile] = {}
    for name, body in (raw.get("profiles") or {}).items():
        body = body or {}
        unknown = set(body) - {
            "description", "any_tags", "all_tags", "ids", "all", "include_disabled",
        }
        if unknown:
            raise CatalogError(f"profile {name!r} has unknown keys: {sorted(unknown)}")
        profiles[name] = Profile(name=name, **body)
    if not profiles:
        raise CatalogError("catalog defines no profiles")

    models: list[Model] = []
    seen: set[str] = set()
    known = set(Model.__dataclass_fields__)
    for entry in raw.get("models") or []:
        unknown = set(entry) - known
        if unknown:
            raise CatalogError(f"model {entry.get('id')!r} has unknown keys: {sorted(unknown)}")
        m = Model(**entry)
        if m.id in seen:
            raise CatalogError(f"duplicate model id: {m.id}")
        if "/" not in m.hf_repo:
            raise CatalogError(f"{m.id}: hf_repo must be 'org/name', got {m.hf_repo!r}")
        seen.add(m.id)
        models.append(m)
    if not models:
        raise CatalogError("catalog defines no models")

    return Catalog(path=path, defaults=defaults, profiles=profiles, models=models)


_ID_LINE = re.compile(r"^\s*-\s+id:\s*(\S+)\s*$")
_SIZE_LINE = re.compile(r"^(\s*)approx_bytes:\s*\d+\s*$")


def write_approx_bytes(path: str | Path, sizes: dict[str, int]) -> int:
    """Rewrite `approx_bytes:` in place, preserving comments and formatting.

    Returns the number of entries updated.
    """
    path = Path(path)
    lines = path.read_text().splitlines(keepends=True)
    current: str | None = None
    updated = 0
    for i, line in enumerate(lines):
        m = _ID_LINE.match(line)
        if m:
            current = m.group(1)
            continue
        s = _SIZE_LINE.match(line)
        if s and current in sizes:
            new = f"{s.group(1)}approx_bytes: {sizes[current]}\n"
            if new != line:
                updated += 1
            lines[i] = new
    path.write_text("".join(lines))
    return updated
