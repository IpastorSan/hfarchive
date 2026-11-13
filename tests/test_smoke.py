"""Smoke tests for local-model-backup.

Deliberately cheap: the only network transfer is a ~2 KB config.json, so this
never pulls multi-GB weights. Run either way:

    python tests/test_smoke.py          # standalone, no pytest needed
    python -m pytest tests/ -q          # if pytest is installed
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lmb import manifest  # noqa: E402
from lmb.catalog import CatalogError, Model, load_catalog, write_approx_bytes  # noqa: E402
from lmb.cli import main  # noqa: E402
from lmb.download import build_command, download_one, human, resolve_hf_bin  # noqa: E402
from lmb.hub import RepoStatus, probe, token_available  # noqa: E402

CATALOG = PROJECT_ROOT / "models.yaml"

# A repo with a tiny, always-present config.json. Used for the one live test.
TINY_REPO = "openai/whisper-large-v3-turbo"
TINY_FILE = "config.json"
TINY_MAX_BYTES = 200_000


def _online() -> bool:
    import urllib.request

    try:
        urllib.request.urlopen("https://huggingface.co/api/models/" + TINY_REPO, timeout=15)
        return True
    except Exception:
        return False


ONLINE = _online()


def _skip(reason: str):
    print(f"  SKIP: {reason}")


# ------------------------------------------------------------------ catalog
def test_catalog_loads_and_is_consistent():
    cat = load_catalog(CATALOG)
    assert len(cat.models) >= 55, "catalog looks truncated"

    ids = [m.id for m in cat.models]
    assert len(ids) == len(set(ids)), "duplicate model ids"

    repos = [m.hf_repo for m in cat.models]
    assert len(repos) == len(set(repos)), "duplicate hf_repo entries"

    for m in cat.models:
        assert m.hf_repo.count("/") == 1, f"{m.id}: hf_repo must be org/name"
        assert m.category, f"{m.id}: missing category"
        assert m.tags, f"{m.id}: no tags, no profile can reach it"
        assert m.approx_bytes > 0, f"{m.id}: unmeasured, run --refresh-sizes"


def test_every_tweet_model_is_present():
    """The tweet catalog must survive edits to models.yaml."""
    cat = load_catalog(CATALOG)
    present = {m.hf_repo for m in cat.models}
    expected = {
        # text / reasoning
        "moonshotai/Kimi-K3", "deepseek-ai/DeepSeek-V4.1-Flash", "zai-org/GLM-5.3",
        "zai-org/GLM-5.3-Flash", "Qwen/Qwen3.8-27B", "google/gemma-4-31B-it",
        "google/gemma-4-26B-A4B-it", "openai/gpt-oss-120b",
        # runnable GGUF
        "unsloth/Qwen3.8-27B-GGUF", "unsloth/GLM-5.3-Flash-GGUF",
        "unsloth/Kimi-K3-GGUF", "unsloth/gpt-oss-120b-GGUF",
        # coding
        "moonshotai/Kimi-K2.7-Code", "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF",
        "mistralai/Devstral-Small-2-24B-Instruct-2512",
        # small / on-device
        "openbmb/MiniCPM5-2B", "google/gemma-4-E4B-it",
        "mistralai/Ministral-3-3B-Instruct-2512-GGUF",
        # image
        "black-forest-labs/FLUX.2-dev", "black-forest-labs/FLUX.2-klein-9B",
        "black-forest-labs/FLUX.2-klein-4B", "Qwen/Qwen-Image-2512",
        "Qwen/Qwen-Image-Edit-2511", "Qwen/Qwen-Image-Layered",
        # video
        "Lightricks/LTX-2.5", "MiniMaxAI/MiniMax-H3", "Wan-AI/Wan2.2-Animate-14B",
        "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        "meituan-longcat/LongCat-Video-Avatar-1.5", "krea/krea-realtime-video",
        # asr
        "mistralai/Voxtral-Mini-4B-Realtime-2602",
        "nvidia/nemotron-speech-streaming-en-0.6b", "openai/whisper-large-v3-turbo",
        # tts
        "tencent/AuK", "mistralai/Voxtral-4B-TTS-2603", "Edge0/Audio8-TTS-Preview-0.6b",
        # music
        "m-a-p/YuE2-3B", "audio-cpp/Yue2-3B-GGUF", "audio-cpp/MiniMax-Music3-GGUF",
        "stabilityai/stable-audio-3-medium", "facebook/sam-audio-large",
        # ocr
        "baidu/Unlimited-OCR", "zai-org/GLM-OCR", "deepseek-ai/DeepSeek-OCR-2",
        "datalab-to/chandra-ocr-2",
        # embeddings / rerank
        "jinaai/jina-embeddings-v5-omni-small", "Qwen/Qwen3-Embedding-8B",
        "Qwen/Qwen3-Embedding-0.6B", "google/embeddinggemma-300m",
        "Qwen/Qwen3-Reranker-8B", "BAAI/bge-reranker-v2-m3",
    }
    assert len(expected) == 52, "the tweet list has 52 rows"
    missing = expected - present
    assert not missing, f"tweet models dropped from the catalog: {sorted(missing)}"

    tweet_tagged = {m.hf_repo for m in cat.models if "tweet" in m.tags}
    untagged = expected - tweet_tagged
    assert not untagged, f"tweet models missing the `tweet` tag: {sorted(untagged)}"


def test_extra_coding_models_present_but_off_by_default():
    cat = load_catalog(CATALOG)
    extras = {
        "unsloth/DeepSeek-R1-Distill-Qwen-14B-GGUF",
        "unsloth/DeepSeek-R1-Distill-Qwen-32B-GGUF",
        "bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF",
        "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
        "Qwen/Qwen2.5-Coder-32B-Instruct-GGUF",
        "bartowski/Qwen2.5-Coder-14B-Instruct-GGUF",
        "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
        "TheBloke/deepseek-coder-6.7B-instruct-GGUF",
    }
    by_repo = {m.hf_repo: m for m in cat.models}
    assert extras <= set(by_repo), f"missing extras: {sorted(extras - set(by_repo))}"
    for repo in extras:
        m = by_repo[repo]
        assert not m.enabled, f"{m.id} must be disabled by default"
        assert "coding-small" in m.tags, f"{m.id} must carry the coding-small tag"

    # A 7B-class coder must fit the current 8 GB card.
    current = [m for m in cat.models if "current-box" in m.tags and "coding" in m.tags]
    small = [m for m in current if m.approx_bytes < 8e9]
    assert small, "no coding model small enough for the current 8 GB card"


def test_profiles_resolve():
    cat = load_catalog(CATALOG)
    for name in ("tweet", "tweet-gguf", "domestic-coding", "current-box", "all"):
        assert name in cat.profiles, f"missing profile {name}"
        assert cat.select(name), f"profile {name} selected nothing"

    # tweet is the default and must cover the whole tweet list
    assert len(cat.select("tweet")) == 52

    # tweet-gguf is a strict subset of tweet, all GGUF
    gguf = cat.select("tweet-gguf")
    tweet_ids = {m.id for m in cat.select("tweet")}
    for m in gguf:
        assert m.id in tweet_ids, f"{m.id} is not a tweet model"
        assert "gguf" in m.tags

    # disabled extras only appear in the profiles that opt in
    assert any(not m.enabled for m in cat.select("domestic-coding"))
    assert all(m.enabled for m in cat.select("tweet"))
    assert all(m.enabled for m in cat.select("all"))


def test_selection_filters():
    cat = load_catalog(CATALOG)
    only = cat.select("tweet", only=["whisper-large-v3-turbo"])
    assert [m.id for m in only] == ["whisper-large-v3-turbo"]

    dropped = cat.select("tweet", exclude=["whisper-large-v3-turbo"])
    assert "whisper-large-v3-turbo" not in {m.id for m in dropped}

    # --enable reaches a model no profile would otherwise select
    forced = cat.select("tweet", enable=["qwen2.5-coder-7b-gguf"])
    assert "qwen2.5-coder-7b-gguf" in {m.id for m in forced}

    cats = cat.select("tweet", categories=["ocr"])
    assert cats and {m.category for m in cats} == {"ocr"}

    for bad in (dict(profile="nope"), dict(profile="tweet", only=["nope"])):
        try:
            cat.select(**bad)
        except CatalogError:
            pass
        else:
            raise AssertionError(f"expected CatalogError for {bad}")


def test_dest_layout():
    cat = load_catalog(CATALOG)
    m = cat.by_id("gpt-oss-120b")
    assert m.dest_dir(Path("/run/media/x/B1")) == Path(
        "/run/media/x/B1/text-llm/openai/gpt-oss-120b")


def test_include_globs_are_written_for_multi_quant_repos():
    """GGUF repos hold every quant; without globs a single repo is terabytes."""
    cat = load_catalog(CATALOG)
    for mid in ("kimi-k3-gguf", "glm-5.3-flash-gguf", "gpt-oss-120b-gguf",
                "qwen3.8-27b-gguf", "qwen3-coder-30b-a3b-gguf"):
        m = cat.by_id(mid)
        assert m.include, f"{mid} must pin a quant with include globs"


def test_glob_semantics_match_the_hub():
    """`*` crosses `/` in Hub patterns — the catalog relies on this."""
    from huggingface_hub.utils import filter_repo_objects

    names = ["a.safetensors", "transformer/b.safetensors", "original/c.safetensors"]
    assert list(filter_repo_objects(names, allow_patterns=["*.safetensors"])) == names
    # so excluding one root-level file needs its exact name
    kept = list(filter_repo_objects(names, ignore_patterns=["a.safetensors"]))
    assert kept == ["transformer/b.safetensors", "original/c.safetensors"]


def test_build_command():
    cat = load_catalog(CATALOG)
    cmd = build_command(cat.by_id("gpt-oss-120b"), Path("/mnt/d"), "hf", 4)
    assert cmd[:3] == ["hf", "download", "openai/gpt-oss-120b"]
    assert "--local-dir" in cmd and "/mnt/d/text-llm/openai/gpt-oss-120b" in cmd
    assert cmd.count("--exclude") == 2
    assert "--max-workers" in cmd


def test_refresh_sizes_preserves_formatting():
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / "models.yaml"
        shutil.copy2(CATALOG, copy)
        before = copy.read_text()
        cat = load_catalog(copy)
        target = cat.models[0]
        write_approx_bytes(copy, {target.id: 123})
        after = copy.read_text()
        assert len(before.splitlines()) == len(after.splitlines())
        assert "# local-model-backup — model catalog" in after
        assert load_catalog(copy).by_id(target.id).approx_bytes == 123


def test_human():
    assert human(0) == "0 B"
    assert human(1_500_000_000) == "1.50 GB"
    assert human(2_800_000_000_000) == "2.80 TB"


def test_manifest_roundtrip():
    from lmb.download import DownloadResult

    cat = load_catalog(CATALOG)
    m = cat.by_id("whisper-large-v3-turbo")
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp)
        st = RepoStatus(model=m, exists=True, commit="deadbeef",
                        files=[("config.json", 2048)])
        res = DownloadResult(m, "downloaded", bytes_expected=2048, commit="deadbeef")
        path = manifest.append(dest, [res], {m.id: st})
        assert path.is_file()
        rows = manifest.read(dest)
        assert len(rows) == 1
        row = rows[0]
        assert row["hf_repo"] == m.hf_repo
        assert row["commit"] == "deadbeef"
        assert row["status"] == "downloaded"
        assert row["rel_dest"] == "asr/openai/whisper-large-v3-turbo"
        # appends, never rewrites
        manifest.append(dest, [res], {m.id: st})
        assert len(manifest.read(dest)) == 2


# -------------------------------------------------------------- cli guards
def test_cli_list_needs_no_network():
    assert main(["--list", "--profile", "tweet"]) == 0
    assert main(["--list", "--profile", "current-box", "--json"]) == 0


def test_cli_rejects_unknown_profile():
    assert main(["--list", "--profile", "does-not-exist"]) == 2


def test_cli_refuses_home_destination():
    target = Path.home() / ".lmb-smoke-tmp"
    target.mkdir(exist_ok=True)
    try:
        assert main(["--dest", str(target), "--only", "whisper-large-v3-turbo"]) == 2
        # ... and never wrote anything into it
        assert not any(target.iterdir())
    finally:
        target.rmdir()


def test_cli_missing_dest_degrades_to_dry_run():
    if not ONLINE:
        return _skip("offline")
    with tempfile.TemporaryDirectory() as tmp:
        ghost = Path(tmp) / "not-plugged-in"
        assert main(["--dest", str(ghost), "--only", "whisper-large-v3-turbo"]) == 0
        assert not ghost.exists(), "dry run must not create the destination"


def test_cli_large_repo_needs_explicit_flag():
    if not ONLINE:
        return _skip("offline")
    with tempfile.TemporaryDirectory() as tmp:
        rc = main(["--dest", tmp, "--only", "kimi-k3-gguf", "--min-free-gb", "0"])
        assert rc == 3, f"expected the large-repo refusal, got {rc}"
        assert not any(Path(tmp).iterdir()), "nothing may be written when refused"


def test_cli_dry_run_writes_nothing():
    if not ONLINE:
        return _skip("offline")
    with tempfile.TemporaryDirectory() as tmp:
        assert main(["--dry-run", "--dest", tmp, "--profile", "current-box"]) == 0
        assert not any(Path(tmp).iterdir())


# ------------------------------------------------------------ live (tiny)
def test_hub_probe_is_accurate():
    if not ONLINE:
        return _skip("offline")
    m = Model(id="tiny", hf_repo=TINY_REPO, category="asr", include=[TINY_FILE])
    st = probe(m)
    assert st.exists and st.error is None
    assert st.matched_files == 1, f"expected just {TINY_FILE}, got {st.files}"
    assert 0 < st.matched_bytes < TINY_MAX_BYTES
    assert st.commit
    assert st.total_repo_bytes > st.matched_bytes


def test_end_to_end_tiny_download():
    """Exercise the real `hf download` path — a couple of KB, not a model."""
    if not ONLINE:
        return _skip("offline")
    hf_bin = resolve_hf_bin()
    if not hf_bin:
        return _skip("no `hf` CLI on this machine")

    m = Model(id="tiny", hf_repo=TINY_REPO, category="asr", include=[TINY_FILE])
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp)
        st = probe(m)
        assert st.matched_bytes < TINY_MAX_BYTES, "refusing to download anything large"

        res = download_one(st, dest, hf_bin, 2, token_available(), threading.Lock())
        assert res.status == "downloaded", f"{res.status}: {res.message}"

        landed = m.dest_dir(dest) / TINY_FILE
        assert landed.is_file(), f"{landed} missing"
        assert landed.stat().st_size == st.files[0][1]

        # second pass must recognise the file as complete and skip it
        again = download_one(st, dest, hf_bin, 2, token_available(), threading.Lock())
        assert again.status == "up-to-date", f"resume check failed: {again.status}"

        here, missing = st.local_state(dest)
        assert missing == 0 and here == st.matched_bytes

        manifest.append(dest, [res], {m.id: st})
        rows = manifest.read(dest)
        assert rows and rows[-1]["hf_repo"] == TINY_REPO


# --------------------------------------------------------------- standalone
def _run_standalone() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failures = []
    print(f"local-model-backup smoke test ({'online' if ONLINE else 'OFFLINE'})\n")
    for name, fn in tests:
        try:
            fn()
        except AssertionError as exc:
            failures.append((name, exc))
            print(f"FAIL {name}\n     {exc}")
        except Exception as exc:  # noqa: BLE001
            failures.append((name, exc))
            print(f"ERROR {name}\n     {type(exc).__name__}: {exc}")
        else:
            print(f"ok   {name}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_standalone())
