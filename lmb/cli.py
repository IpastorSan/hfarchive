"""Command line interface for local-model-backup."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

from lmb import __version__, manifest
from lmb.catalog import Catalog, CatalogError, Model, load_catalog, write_approx_bytes
from lmb.download import download_many, human, resolve_hf_bin
from lmb.hub import RepoStatus, probe_all, token_available

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CATALOG = PROJECT_ROOT / "models.yaml"
LOG_DIR = PROJECT_ROOT / "logs"
DEST_ENV = "LOCAL_MODEL_BACKUP_DEST"

log = logging.getLogger("lmb")


# --------------------------------------------------------------------- args
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m lmb",
        description="Mirror a curated set of Hugging Face models onto an external drive.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python -m lmb --list --profile tweet\n"
            "  python -m lmb --verify-only --profile tweet-gguf\n"
            "  python -m lmb --dry-run --dest /run/media/ignacio/BACKUP1 --profile current-box\n"
            "  python -m lmb --dest /run/media/ignacio/BACKUP1 --profile domestic-coding\n"
            "  python -m lmb --dest /run/media/ignacio/BACKUP1 --profile tweet --allow-large\n"
        ),
    )
    p.add_argument("--version", action="version", version=f"local-model-backup {__version__}")
    p.add_argument("--catalog", default=str(DEFAULT_CATALOG), help="path to models.yaml")
    p.add_argument("--profile", default="tweet", help="profile to resolve (default: tweet)")

    p.add_argument("--dest", default=os.environ.get(DEST_ENV),
                   help=f"destination root, e.g. /run/media/ignacio/BACKUP1 (env: {DEST_ENV})")

    mode = p.add_argument_group("modes")
    mode.add_argument("--list", action="store_true", help="print the resolved catalog, no network")
    mode.add_argument("--verify-only", action="store_true",
                      help="check every repo exists on the Hub, download nothing")
    mode.add_argument("--dry-run", action="store_true",
                      help="resolve sizes and target paths, write nothing")
    mode.add_argument("--refresh-sizes", action="store_true",
                      help="write measured approx_bytes back into the catalog")

    sel = p.add_argument_group("selection")
    sel.add_argument("--only", default="", help="comma-separated model ids to use instead of the profile")
    sel.add_argument("--exclude", default="", help="comma-separated model ids to drop")
    sel.add_argument("--enable", default="", help="comma-separated ids to force in even if disabled")
    sel.add_argument("--category", default="", help="comma-separated categories to keep")

    beh = p.add_argument_group("download behaviour")
    beh.add_argument("--jobs", type=int, default=None, help="repos downloaded in parallel")
    beh.add_argument("--max-workers", type=int, default=None, help="files in parallel per repo")
    beh.add_argument("--allow-large", "--i-know", dest="allow_large", action="store_true",
                     help="permit entries at or above the large-repo threshold")
    beh.add_argument("--large-repo-gb", type=float, default=None,
                     help="large-repo threshold in GB (default from catalog)")
    beh.add_argument("--min-free-gb", type=float, default=None,
                     help="required free headroom on the destination")
    beh.add_argument("--ignore-space", action="store_true",
                     help="proceed even if the plan does not fit in the free space")
    beh.add_argument("--allow-home-dest", action="store_true",
                     help="allow a destination inside /home (normally refused)")

    out = p.add_argument_group("output")
    out.add_argument("--json", action="store_true", help="machine-readable output")
    out.add_argument("--verbose", "-v", action="store_true", help="debug logging")
    return p


def csv_list(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


# ------------------------------------------------------------------ logging
def setup_logging(verbose: bool) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / f"lmb-{time.strftime('%Y%m%d-%H%M%S')}.log"
    log.setLevel(logging.DEBUG)
    log.handlers.clear()

    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
    log.addHandler(fh)

    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(ch)
    return path


# ------------------------------------------------------------------- tables
def render_table(headers: list[str], rows: list[list[str]], aligns: str = "") -> str:
    if not rows:
        return "(nothing selected)"
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    aligns = (aligns + "l" * len(headers))[: len(headers)]

    def fmt(cells: list[str]) -> str:
        parts = []
        for cell, w, a in zip(cells, widths, aligns):
            parts.append(cell.rjust(w) if a == "r" else cell.ljust(w))
        return "  ".join(parts).rstrip()

    sep = "  ".join("-" * w for w in widths)
    return "\n".join([fmt(headers), sep] + [fmt(r) for r in rows])


def flag_str(m: Model) -> str:
    bits = []
    if m.gated:
        bits.append("gated")
    if not m.enabled:
        bits.append("off")
    if m.include or m.exclude:
        bits.append("globs")
    return ",".join(bits) or "-"


# -------------------------------------------------------------------- modes
def cmd_list(cat: Catalog, models: list[Model], args) -> int:
    prof = cat.profiles[args.profile]
    if args.json:
        print(json.dumps(
            {
                "profile": args.profile,
                "description": prof.description.strip(),
                "count": len(models),
                "models": [
                    {
                        "id": m.id, "hf_repo": m.hf_repo, "category": m.category,
                        "tags": m.tags, "license": m.license, "gated": m.gated,
                        "approx_bytes": m.approx_bytes, "enabled": m.enabled,
                        "include": m.include, "exclude": m.exclude, "notes": m.notes,
                    }
                    for m in models
                ],
            },
            indent=2,
        ))
        return 0

    print(f"profile: {args.profile} — {prof.description.strip() or '(no description)'}")
    print(f"catalog: {cat.path}\n")
    rows = [
        [m.id, m.hf_repo, m.category,
         human(m.approx_bytes) if m.approx_bytes else "?",
         m.license or "?", flag_str(m)]
        for m in models
    ]
    print(render_table(
        ["ID", "HF REPO", "CATEGORY", "APPROX", "LICENSE", "FLAGS"], rows, "lllrll"))
    known = sum(m.approx_bytes for m in models)
    unknown = sum(1 for m in models if not m.approx_bytes)
    print(f"\n{len(models)} models, approx {human(known)}"
          + (f" (+{unknown} unmeasured; run --refresh-sizes)" if unknown else ""))
    return 0


def cmd_verify(models: list[Model], args) -> int:
    log.info("checking %d repos against the Hub...", len(models))
    statuses = probe_all(models)
    ok = [s for s in statuses if s.exists and not s.error]
    gated_ok = [s for s in statuses if s.exists and s.gated]
    bad = [s for s in statuses if not s.exists or s.error]

    if args.json:
        print(json.dumps([
            {
                "id": s.model.id, "hf_repo": s.model.hf_repo, "exists": s.exists,
                "gated": s.gated, "commit": s.commit, "files": s.matched_files,
                "bytes": s.matched_bytes, "repo_bytes": s.total_repo_bytes,
                "error": s.error,
            } for s in statuses
        ], indent=2))
    else:
        rows = []
        for s in statuses:
            if not s.exists:
                verdict = "MISSING"
            elif s.error:
                verdict = "WARN"
            elif s.gated:
                verdict = "OK (gated)"
            else:
                verdict = "OK"
            rows.append([
                s.model.id, s.model.hf_repo, verdict,
                (s.commit or "")[:8],
                str(s.matched_files) if s.exists else "-",
                human(s.matched_bytes) if s.matched_files else "-",
                s.error or "",
            ])
        print(render_table(
            ["ID", "HF REPO", "STATUS", "COMMIT", "FILES", "MATCHED", "NOTE"],
            rows, "llllrrl"))
        print(f"\n{len(ok)} ok, {len(gated_ok)} gated, {len(bad)} problematic "
              f"({len(statuses)} checked)")
    return 1 if any(not s.exists for s in statuses) else 0


def resolve_dest(args) -> tuple[Path | None, list[str], bool]:
    """Return (dest, problems, fatal).

    `fatal` separates "the drive is not plugged in yet" — which degrades to a
    dry run, because planning ahead is useful — from "this destination is wrong",
    which must fail loudly rather than quietly downloading nothing.
    """
    if not args.dest:
        return None, [f"no destination given (--dest or ${DEST_ENV})"], False
    dest = Path(args.dest).expanduser()
    if not dest.exists():
        return dest, [f"destination does not exist: {dest}"], False

    problems: list[str] = []
    if not dest.is_dir():
        return dest, [f"destination is not a directory: {dest}"], True
    resolved = dest.resolve()
    if not args.allow_home_dest and (
        resolved == Path.home() or Path.home() in resolved.parents
    ):
        problems.append(
            f"destination {resolved} is inside {Path.home()}; this backup is meant for an "
            "external drive. Use --allow-home-dest to override."
        )
    if not os.access(dest, os.W_OK):
        problems.append(f"destination is not writable: {dest}")
    return dest, problems, bool(problems)


def plan_rows(statuses: list[RepoStatus], dest: Path | None) -> list[dict]:
    rows = []
    for s in statuses:
        here = missing = 0
        if dest is not None and s.files:
            here, missing = s.local_state(dest)
        elif s.files:
            missing = s.matched_bytes
        rows.append({
            "status": s,
            "model": s.model,
            "present": here,
            "missing": missing,
        })
    return rows


def cmd_plan(models: list[Model], args, defaults: dict, dest: Path | None,
             dest_problems: list[str], forced_dry: bool,
             fatal_dest: bool) -> tuple[int, bool]:
    """Shared path for --dry-run and the real download.

    Returns (exit code, whether any download was actually attempted). Nothing is
    written to the destination unless that second value is True.
    """
    large_gb = args.large_repo_gb if args.large_repo_gb is not None else defaults["large_repo_gb"]
    min_free_gb = args.min_free_gb if args.min_free_gb is not None else defaults["min_free_gb"]
    large_bytes = large_gb * 1e9

    log.info("resolving %d repos against the Hub...", len(models))
    statuses = probe_all(models)
    rows = plan_rows(statuses, dest)

    have_token = token_available()
    dry = args.dry_run or forced_dry

    table = []
    for r in rows:
        s: RepoStatus = r["status"]
        m: Model = r["model"]
        if not s.exists:
            note = s.error or "unavailable"
        elif s.gated and not have_token:
            note = "gated: will skip (no token)"
        elif s.gated:
            note = "gated: token present"
        elif s.matched_bytes >= large_bytes and not args.allow_large:
            note = f"large: needs --allow-large"
        elif r["missing"] == 0 and s.files:
            note = "already complete"
        else:
            note = s.error or ""
        table.append([
            m.id, m.hf_repo,
            str(s.matched_files) if s.exists else "-",
            human(s.matched_bytes) if s.exists else "-",
            human(r["missing"]) if s.exists else "-",
            "yes" if m.gated else "no",
            str(m.dest_dir(dest)) if dest else f"<DEST>/{m.category}/{m.org}/{m.name}",
            note,
        ])

    if args.json:
        print(json.dumps([
            {
                "id": r["model"].id, "hf_repo": r["model"].hf_repo,
                "exists": r["status"].exists, "gated": r["model"].gated,
                "files": r["status"].matched_files,
                "bytes_matched": r["status"].matched_bytes,
                "bytes_missing": r["missing"], "bytes_present": r["present"],
                "commit": r["status"].commit,
                "dest": str(r["model"].dest_dir(dest)) if dest else None,
            } for r in rows
        ], indent=2))
    else:
        print(render_table(
            ["ID", "HF REPO", "FILES", "MATCHED", "TO FETCH", "GATED", "DEST", "NOTE"],
            table, "lllrrlll"))

    total_matched = sum(s.matched_bytes for s in statuses)
    total_missing = sum(r["missing"] for r in rows)
    oversized = [r for r in rows
                 if r["status"].matched_bytes >= large_bytes and r["missing"] > 0]
    gated_skips = [r for r in rows if r["status"].gated and not have_token]
    broken = [r for r in rows if not r["status"].exists]

    print()
    log.info("catalogued %s across %d repos; %s still to fetch",
             human(total_matched), len(rows), human(total_missing))
    if gated_skips:
        log.warning("%d gated repo(s) will be skipped (no HF token): %s",
                    len(gated_skips), ", ".join(r["model"].id for r in gated_skips))
        log.warning("  -> run `hf auth login` and accept each licence on the model page")
    if broken:
        log.error("%d repo(s) could not be resolved: %s",
                  len(broken), ", ".join(r["model"].id for r in broken))
    if oversized:
        log.warning("%d repo(s) are >= %s each: %s",
                    len(oversized), human(large_bytes),
                    ", ".join(r["model"].id for r in oversized))

    if dry:
        if forced_dry and not args.dry_run:
            level = log.error if fatal_dest else log.warning
            level("cannot download: %s", "; ".join(dest_problems))
        log.info("dry run: nothing was written")
        return (2 if fatal_dest else 0), False

    # ---- from here on we are actually going to write to the drive ----
    if dest_problems:
        for problem in dest_problems:
            log.error("%s", problem)
        return 2, False

    assert dest is not None
    free = shutil.disk_usage(dest).free
    log.info("destination %s: %s free", dest, human(free))
    if free < min_free_gb * 1e9:
        log.error("only %s free, below the %s headroom required (--min-free-gb)",
                  human(free), human(min_free_gb * 1e9))
        return 2, False

    to_run = [r for r in rows if r["status"].exists]
    if oversized and not args.allow_large:
        blocked = ", ".join(r["model"].id for r in oversized)
        log.error("refusing to download %d repo(s) of %s or more without --allow-large: %s",
                  len(oversized), human(large_bytes), blocked)
        log.error("re-run with --allow-large (alias --i-know), or narrow with --only/--exclude")
        return 3, False

    budget = free - min_free_gb * 1e9
    if total_missing > budget and not args.ignore_space:
        log.error("plan needs %s but only %s is usable (%s free minus %s headroom)",
                  human(total_missing), human(budget), human(free), human(min_free_gb * 1e9))
        log.error("narrow the selection, use a bigger drive, or pass --ignore-space")
        return 2, False

    jobs = args.jobs if args.jobs is not None else defaults["jobs"]
    max_workers = (args.max_workers if args.max_workers is not None
                   else defaults["max_workers_per_repo"])
    hf_bin = resolve_hf_bin()
    if not hf_bin:
        log.error("could not find the `hf` CLI; install huggingface_hub or set $LMB_HF_BIN")
        return 2, False
    log.info("using %s with %d parallel repo(s), %d file worker(s) each",
             hf_bin, jobs, max_workers)

    results = download_many(
        [r["status"] for r in to_run], dest, hf_bin, jobs, max_workers, have_token)

    mpath = manifest.append(dest, results, {s.model.id: s for s in statuses})
    log.info("manifest updated: %s", mpath)

    done = sum(1 for r in results if r.status == "downloaded")
    fresh = sum(1 for r in results if r.status == "up-to-date")
    skipped = sum(1 for r in results if r.status == "skipped-gated")
    failed = [r for r in results if r.status == "failed"]
    log.info("done: %d downloaded, %d already current, %d gated-skipped, %d failed",
             done, fresh, skipped, len(failed))
    for r in failed:
        log.error("  %s: %s", r.model.id, r.message)
    return (1 if failed else 0), True


def cmd_refresh_sizes(cat: Catalog, models: list[Model], args) -> int:
    log.info("measuring %d repos...", len(models))
    statuses = probe_all(models)
    sizes = {s.model.id: s.matched_bytes for s in statuses if s.exists and s.files}
    failed = [s for s in statuses if not s.exists or not s.files]
    updated = write_approx_bytes(cat.path, sizes)
    log.info("measured %d repos, updated %d approx_bytes entries in %s",
             len(sizes), updated, cat.path)
    for s in failed:
        log.warning("  %s: %s", s.model.id, s.error or "no files matched")
    return 1 if failed else 0


# --------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    log_path = setup_logging(args.verbose)
    log.debug("argv: %s", argv if argv is not None else sys.argv[1:])

    try:
        cat = load_catalog(args.catalog)
        models = cat.select(
            profile=args.profile,
            only=csv_list(args.only),
            exclude=csv_list(args.exclude),
            enable=csv_list(args.enable),
            categories=csv_list(args.category),
        )
    except CatalogError as exc:
        log.error("%s", exc)
        return 2

    if not models:
        log.error("profile %r selected no models", args.profile)
        return 2

    if args.list:
        return cmd_list(cat, models, args)
    if args.refresh_sizes:
        return cmd_refresh_sizes(cat, models, args)
    if args.verify_only:
        return cmd_verify(models, args)

    dest, dest_problems, fatal_dest = resolve_dest(args)
    # An absent destination degrades to a dry run rather than failing: planning
    # is useful before the drive is plugged in. A *wrong* one is an error.
    forced_dry = bool(dest_problems)

    rc, downloaded = cmd_plan(
        models, args, cat.defaults, dest, dest_problems, forced_dry, fatal_dest)

    # Only touch the drive when we actually ran downloads: a refused run
    # (too large, no room, bad destination) must leave it exactly as it was.
    if downloaded and dest is not None and dest.is_dir():
        try:
            logs_on_dest = manifest.lmb_dir(dest) / "logs"
            logs_on_dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(log_path, logs_on_dest / log_path.name)
        except OSError as exc:
            log.warning("could not copy the log to the drive: %s", exc)
    log.info("log: %s", log_path)
    return rc
