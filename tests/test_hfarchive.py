"""Tests for the hfarchive script.

Deliberately cheap: the only network transfer is a ~2 KB config.json, so this
never pulls model weights. Run with:

    uv run --no-project --with pytest --with huggingface_hub --with pyyaml pytest -q

Runtimes are exercised against a local git repository; GitHub, workflow URLs
and the curated-config download are mocked.
"""

from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = PROJECT_ROOT / "hfarchive"
EXAMPLE = PROJECT_ROOT / "examples" / "tweet-2026-09.yaml"


def _load_script():
    loader = importlib.machinery.SourceFileLoader("hfarchive", str(SCRIPT))
    spec = importlib.util.spec_from_loader("hfarchive", loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules["hfarchive"] = module  # dataclasses resolve annotations via sys.modules
    loader.exec_module(module)
    return module


hfa = _load_script()

# A repo with a tiny, always-present config.json. Used for the live tests.
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
needs_network = pytest.mark.skipif(not ONLINE, reason="offline")


@pytest.fixture(autouse=True)
def _isolated_logs(tmp_path, monkeypatch):
    """Keep run logs out of the real ~/.local/state."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.delenv(hfa.DEST_ENV, raising=False)


def write_config(path: Path, body: str) -> Path:
    path.write_text(body)
    return path


MINIMAL = """\
models:
  - id: a
    hf_repo: org/a
    category: text
    tags: [x]
  - id: b
    hf_repo: org/b
    category: text
    tags: [y]
  - id: c
    hf_repo: org/c
    category: text
    enabled: false
"""


# ---------------------------------------------------------- example config
def test_example_loads_and_is_consistent():
    cat = hfa.load_catalog(EXAMPLE)
    assert len(cat.models) >= 55, "example looks truncated"

    ids = [m.id for m in cat.models]
    assert len(ids) == len(set(ids)), "duplicate model ids"

    repos = [m.hf_repo for m in cat.models]
    assert len(repos) == len(set(repos)), "duplicate hf_repo entries"

    for m in cat.models:
        assert m.hf_repo.count("/") == 1, f"{m.id}: hf_repo must be org/name"
        assert m.category, f"{m.id}: missing category"
        assert m.tags, f"{m.id}: no tags, no profile can reach it"
        assert m.approx_bytes > 0, f"{m.id}: unmeasured, run --refresh-sizes"


def test_example_keeps_every_listed_model():
    cat = hfa.load_catalog(EXAMPLE)
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
    assert len(expected) == 52
    missing = expected - present
    assert not missing, f"models dropped from the example: {sorted(missing)}"

    tagged = {m.hf_repo for m in cat.models if "tweet" in m.tags}
    untagged = expected - tagged
    assert not untagged, f"models missing the `tweet` tag: {sorted(untagged)}"


def test_example_extra_coding_models_are_off_by_default():
    cat = hfa.load_catalog(EXAMPLE)
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

    small_gpu = [m for m in cat.models if "8gb-vram" in m.tags and "coding" in m.tags]
    assert any(m.approx_bytes < 8e9 for m in small_gpu), "no coding model fits 8 GB"


def test_example_profiles_resolve():
    cat = hfa.load_catalog(EXAMPLE)
    for name in ("tweet", "tweet-gguf", "domestic-coding", "8gb-vram", "all"):
        assert name in cat.profiles, f"missing profile {name}"
        assert cat.select(name), f"profile {name} selected nothing"

    assert cat.default_profile is None
    assert len(cat.select("tweet")) == 52
    # no default_profile: every enabled entry, which here is the same 52
    assert {m.id for m in cat.select()} == {m.id for m in cat.select("tweet")}

    gguf = cat.select("tweet-gguf")
    tweet_ids = {m.id for m in cat.select("tweet")}
    for m in gguf:
        assert m.id in tweet_ids, f"{m.id} is not in the tweet profile"
        assert "gguf" in m.tags

    assert any(not m.enabled for m in cat.select("domestic-coding"))
    assert all(m.enabled for m in cat.select("tweet"))
    assert all(m.enabled for m in cat.select("all"))


def test_selection_filters():
    cat = hfa.load_catalog(EXAMPLE)
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
        with pytest.raises(hfa.CatalogError):
            cat.select(**bad)


def test_include_globs_are_written_for_multi_quant_repos():
    """GGUF repos hold every quant; without globs a single repo is terabytes."""
    cat = hfa.load_catalog(EXAMPLE)
    for mid in ("kimi-k3-gguf", "glm-5.3-flash-gguf", "gpt-oss-120b-gguf",
                "qwen3.8-27b-gguf", "qwen3-coder-30b-a3b-gguf"):
        assert cat.by_id(mid).include, f"{mid} must pin a quant with include globs"


# ------------------------------------------------------------ config rules
def test_without_default_profile_all_enabled_entries_are_selected(tmp_path):
    cat = hfa.load_catalog(write_config(tmp_path / "m.yaml", MINIMAL))
    assert cat.profiles == {}
    assert [m.id for m in cat.select()] == ["a", "b"]


def test_default_profile_is_used_when_no_profile_is_given(tmp_path):
    body = "default_profile: ys\nprofiles:\n  ys:\n    any_tags: [y]\n" + MINIMAL
    cat = hfa.load_catalog(write_config(tmp_path / "m.yaml", body))
    assert [m.id for m in cat.select()] == ["b"]
    assert [m.id for m in cat.select("ys")] == ["b"]


def test_default_profile_must_exist(tmp_path):
    body = "default_profile: ghost\n" + MINIMAL
    with pytest.raises(hfa.CatalogError, match="ghost"):
        hfa.load_catalog(write_config(tmp_path / "m.yaml", body))


@pytest.mark.parametrize("body, message", [
    ("models:\n  - id: a\n    hf_repo: org/a\n    category: t\n    colour: red\n", "unknown keys"),
    ("models:\n  - id: a\n    hf_repo: org/a\n", "missing"),
    ("models:\n  - id: a\n    hf_repo: noslash\n    category: t\n", "org/name"),
    ("models:\n  - {id: a, hf_repo: o/a, category: t}\n  - {id: a, hf_repo: o/b, category: t}\n",
     "duplicate"),
    ("profiles:\n  p:\n    tagz: [x]\n" + MINIMAL, "unknown keys"),
    ("defaults: {}\n", "no models"),
])
def test_malformed_configs_are_rejected(tmp_path, body, message):
    with pytest.raises(hfa.CatalogError, match=message):
        hfa.load_catalog(write_config(tmp_path / "m.yaml", body))


def test_missing_config_points_to_init(tmp_path):
    with pytest.raises(hfa.CatalogError, match="--init"):
        hfa.load_catalog(tmp_path / "nope.yaml")


def test_init_writes_a_loadable_starter_and_never_overwrites(tmp_path):
    target = tmp_path / "models.yaml"
    assert hfa.main(["--init", "--minimal", "--config", str(target)]) == 0
    cat = hfa.load_catalog(target)
    assert 2 <= len(cat.models) <= 3
    assert all(m.approx_bytes < 10e9 for m in cat.models), "starter entries must be small"
    assert cat.select()

    target.write_text("mine")
    assert hfa.main(["--init", "--minimal", "--config", str(target)]) == 2
    assert hfa.main(["--init", "--config", str(target)]) == 2
    assert target.read_text() == "mine"


def test_minimal_starter_shows_usage_and_runtimes_commented_out(tmp_path):
    target = tmp_path / "models.yaml"
    hfa.main(["--init", "--minimal", "--config", str(target)])
    text = target.read_text()
    assert "# usage:" in text and "# runtimes:" in text
    # uncommented, the examples are valid schema
    out, prefix = [], None
    for line in text.splitlines():
        if line.lstrip().startswith(("# usage:", "# runtimes:")):
            prefix = line[: line.index("#")] + "# "
        if prefix and line.startswith(prefix):
            line = line[: len(prefix) - 2] + line[len(prefix):]
        elif prefix and not line.startswith(prefix.rstrip()):
            prefix = None
        out.append(line)
    cat = hfa.load_catalog(write_config(tmp_path / "u.yaml", "\n".join(out)))
    assert cat.by_id("qwen2.5-coder-7b-gguf").usage.runtime == "llama.cpp"
    assert [rt.id for rt in cat.runtimes] == ["llama.cpp", "ComfyUI"]
    assert cat.runtime_by_id("ComfyUI").wheels.platforms == ["manylinux_2_28_x86_64", "win_amd64"]


CURATED = PROJECT_ROOT / "examples" / hfa.CURATED_CONFIG


def test_init_copies_the_curated_selection_from_a_clone(tmp_path, monkeypatch):
    monkeypatch.setattr(hfa, "fetch_text", lambda url: pytest.fail("must not fetch"))
    target = tmp_path / "models.yaml"
    assert hfa.main(["--init", "--config", str(target)]) == 0
    assert target.read_text() == CURATED.read_text()
    assert hfa.load_catalog(target).models


def test_init_fetches_the_curated_selection_when_standalone(tmp_path, monkeypatch):
    tried = []

    def fake_fetch(url):
        tried.append(url)
        if f"/v{hfa.__version__}/" in url:
            raise hfa.urllib.error.URLError("404: no such tag yet")
        return CURATED.read_text()

    monkeypatch.setattr(hfa, "curated_in_clone", lambda: None)
    monkeypatch.setattr(hfa, "fetch_text", fake_fetch)
    target = tmp_path / "models.yaml"
    assert hfa.main(["--init", "--config", str(target)]) == 0
    assert tried == [hfa.CURATED_URL.format(ref=f"v{hfa.__version__}"),
                     hfa.CURATED_URL.format(ref="main")]
    assert target.read_text() == CURATED.read_text()

    def offline(url):
        raise hfa.urllib.error.URLError("offline")

    monkeypatch.setattr(hfa, "fetch_text", offline)
    other = tmp_path / "other.yaml"
    assert hfa.main(["--init", "--config", str(other)]) == 2
    assert not other.exists()


def test_without_config_a_clone_falls_back_to_the_curated_selection(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path, notice = hfa.resolve_config(None)
    assert path == CURATED.resolve() and "curated" in notice
    assert hfa.main(["--list"]) == 0

    (tmp_path / "models.yaml").write_text(MINIMAL)
    assert hfa.resolve_config(None) == (Path("models.yaml"), None)

    (tmp_path / "models.yaml").unlink()
    monkeypatch.setattr(hfa, "curated_in_clone", lambda: None)
    assert hfa.main(["--list"]) == 2


# ------------------------------------------------------------------ pieces
def test_dest_layout():
    m = hfa.load_catalog(EXAMPLE).by_id("gpt-oss-120b")
    assert m.dest_dir(Path("/mnt/b1")) == Path("/mnt/b1/text-llm/openai/gpt-oss-120b")


def test_glob_semantics_match_the_hub():
    """`*` crosses `/` in Hub patterns; the example config relies on this."""
    from huggingface_hub.utils import filter_repo_objects

    names = ["a.safetensors", "transformer/b.safetensors", "original/c.safetensors"]
    assert list(filter_repo_objects(names, allow_patterns=["*.safetensors"])) == names
    kept = list(filter_repo_objects(names, ignore_patterns=["a.safetensors"]))
    assert kept == ["transformer/b.safetensors", "original/c.safetensors"]


def test_build_command():
    cat = hfa.load_catalog(EXAMPLE)
    cmd = hfa.build_command(cat.by_id("gpt-oss-120b"), Path("/mnt/d"), "hf", 4)
    assert cmd[:3] == ["hf", "download", "openai/gpt-oss-120b"]
    assert "--local-dir" in cmd and "/mnt/d/text-llm/openai/gpt-oss-120b" in cmd
    assert cmd.count("--exclude") == 2
    assert "--max-workers" in cmd


def test_hf_bin_prefers_the_script_environment(tmp_path, monkeypatch):
    env_bin = tmp_path / "bin"
    env_bin.mkdir()
    (env_bin / "python").touch()
    (env_bin / "hf").touch()
    other = tmp_path / "other-hf"
    other.touch()
    monkeypatch.setattr(hfa.sys, "executable", str(env_bin / "python"))
    monkeypatch.setenv(hfa.HF_BIN_ENV, str(other))
    assert hfa.resolve_hf_bin() == str(env_bin / "hf")

    (env_bin / "hf").unlink()
    assert hfa.resolve_hf_bin() == str(other)

    monkeypatch.setenv(hfa.HF_BIN_ENV, str(tmp_path / "missing"))
    assert hfa.resolve_hf_bin() is None

    monkeypatch.delenv(hfa.HF_BIN_ENV)
    monkeypatch.setenv("PATH", str(tmp_path))
    assert hfa.resolve_hf_bin() is None


def test_refresh_sizes_preserves_formatting(tmp_path):
    copy = tmp_path / "models.yaml"
    shutil.copy2(EXAMPLE, copy)
    before = copy.read_text()
    target = hfa.load_catalog(copy).models[0]
    hfa.write_approx_bytes(copy, {target.id: 123})
    after = copy.read_text()
    assert len(before.splitlines()) == len(after.splitlines())
    assert after.splitlines()[0] == before.splitlines()[0]
    assert hfa.load_catalog(copy).by_id(target.id).approx_bytes == 123


def test_human():
    assert hfa.human(0) == "0 B"
    assert hfa.human(1_500_000_000) == "1.50 GB"
    assert hfa.human(2_800_000_000_000) == "2.80 TB"


def test_manifest_roundtrip(tmp_path):
    m = hfa.load_catalog(EXAMPLE).by_id("whisper-large-v3-turbo")
    st = hfa.RepoStatus(model=m, exists=True, commit="deadbeef", files=[("config.json", 2048)])
    res = hfa.DownloadResult(m, "downloaded", bytes_expected=2048, commit="deadbeef")
    path = hfa.manifest_append(tmp_path, [res], {m.id: st})
    assert path == tmp_path / "_hfarchive" / "manifest.jsonl"
    rows = hfa.manifest_read(tmp_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["hf_repo"] == m.hf_repo
    assert row["commit"] == "deadbeef"
    assert row["status"] == "downloaded"
    assert row["rel_dest"] == "asr/openai/whisper-large-v3-turbo"
    # appends, never rewrites
    hfa.manifest_append(tmp_path, [res], {m.id: st})
    assert len(hfa.manifest_read(tmp_path)) == 2


# -------------------------------------------------------------- cli guards
def test_cli_list_needs_no_network():
    assert hfa.main(["--config", str(EXAMPLE), "--list", "--profile", "tweet"]) == 0
    assert hfa.main(["--config", str(EXAMPLE), "--list", "--profile", "8gb-vram", "--json"]) == 0
    assert hfa.main(["--config", str(EXAMPLE), "--list"]) == 0


def test_cli_rejects_unknown_profile():
    assert hfa.main(["--config", str(EXAMPLE), "--list", "--profile", "does-not-exist"]) == 2


def test_cli_missing_config_is_an_error(tmp_path):
    assert hfa.main(["--config", str(tmp_path / "nope.yaml"), "--list"]) == 2


def test_cli_refuses_home_destination():
    target = Path.home() / ".hfarchive-test-tmp"
    target.mkdir(exist_ok=True)
    try:
        rc = hfa.main(["--config", str(EXAMPLE), "--dest", str(target),
                       "--only", "whisper-large-v3-turbo"])
        assert rc == 2
        assert not any(target.iterdir()), "a refused run must write nothing"
    finally:
        target.rmdir()


def test_home_filesystem_detection(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / "sub").mkdir(parents=True)
    monkeypatch.setattr(hfa.Path, "home", classmethod(lambda cls: home))
    assert hfa.on_home_filesystem(home)
    assert hfa.on_home_filesystem(home / "sub")
    # outside $HOME but on the same filesystem (both live under tmp_path)
    sibling = tmp_path / "sibling"
    sibling.mkdir()
    assert hfa.on_home_filesystem(sibling)

    args = hfa.build_parser().parse_args(["--dest", str(sibling)])
    _, problems, fatal = hfa.resolve_dest(args)
    assert fatal and "same filesystem" in problems[0]
    args = hfa.build_parser().parse_args(["--dest", str(sibling), "--allow-home-dest"])
    assert hfa.resolve_dest(args)[1:] == ([], False)


def test_home_filesystem_detection_other_device(tmp_path, monkeypatch):
    home_dev = os.stat(tmp_path).st_dev
    real_stat = os.stat

    def fake_stat(path, *a, **kw):
        st = real_stat(path, *a, **kw)
        if Path(path) == tmp_path / "drive":
            return os.stat_result((st.st_mode, st.st_ino, home_dev + 1) + tuple(st[3:]))
        return st

    (tmp_path / "home").mkdir()
    (tmp_path / "drive").mkdir()
    monkeypatch.setattr(hfa.Path, "home", classmethod(lambda cls: tmp_path / "home"))
    monkeypatch.setattr(hfa.os, "stat", fake_stat)
    assert not hfa.on_home_filesystem(tmp_path / "drive")


@needs_network
def test_cli_missing_dest_degrades_to_dry_run(tmp_path):
    ghost = tmp_path / "not-plugged-in"
    rc = hfa.main(["--config", str(EXAMPLE), "--dest", str(ghost),
                   "--only", "whisper-large-v3-turbo"])
    assert rc == 0
    assert not ghost.exists(), "dry run must not create the destination"


@needs_network
def test_cli_large_repo_needs_explicit_flag(tmp_path):
    dest = tmp_path / "drive"
    dest.mkdir()
    rc = hfa.main(["--config", str(EXAMPLE), "--dest", str(dest), "--allow-home-dest",
                   "--only", "kimi-k3-gguf", "--min-free-gb", "0"])
    assert rc == 3, f"expected the large-repo refusal, got {rc}"
    assert not any(dest.iterdir()), "nothing may be written when refused"


@needs_network
def test_cli_dry_run_writes_nothing(tmp_path):
    dest = tmp_path / "drive"
    dest.mkdir()
    rc = hfa.main(["--config", str(EXAMPLE), "--dry-run", "--dest", str(dest),
                   "--allow-home-dest", "--profile", "8gb-vram"])
    assert rc == 0
    assert not any(dest.iterdir())


# ------------------------------------------------------------ live (tiny)
@needs_network
def test_hub_probe_is_accurate():
    m = hfa.Model(id="tiny", hf_repo=TINY_REPO, category="asr", include=[TINY_FILE])
    st = hfa.probe(m)
    assert st.exists and st.error is None
    # the model card always comes along with the globbed files
    assert [n for n, _ in st.files] == [TINY_FILE, "README.md"], st.files
    assert st.hashes[TINY_FILE].get("git_sha1")
    assert 0 < st.matched_bytes < TINY_MAX_BYTES
    assert st.commit
    assert st.total_repo_bytes > st.matched_bytes


@needs_network
def test_end_to_end_tiny_download(tmp_path):
    """Exercise the real `hf download` path: a couple of KB, not a model."""
    hf_bin = hfa.resolve_hf_bin()
    if not hf_bin:
        pytest.skip("no `hf` CLI available")

    m = hfa.Model(id="tiny", hf_repo=TINY_REPO, category="asr", include=[TINY_FILE])
    st = hfa.probe(m)
    assert st.matched_bytes < TINY_MAX_BYTES, "refusing to download anything large"

    res = hfa.download_one(st, tmp_path, hf_bin, 2, hfa.token_available(), threading.Lock())
    assert res.status == "downloaded", f"{res.status}: {res.message}"

    landed = m.dest_dir(tmp_path) / TINY_FILE
    assert landed.is_file()
    assert landed.stat().st_size == st.files[0][1]

    # second pass must recognise the file as complete and skip it
    again = hfa.download_one(st, tmp_path, hf_bin, 2, hfa.token_available(), threading.Lock())
    assert again.status == "up-to-date", f"resume check failed: {again.status}"

    here, missing = st.local_state(tmp_path)
    assert missing == 0 and here == st.matched_bytes

    hfa.manifest_append(tmp_path, [res], {m.id: st})
    assert hfa.manifest_read(tmp_path)[-1]["hf_repo"] == TINY_REPO


# ------------------------------------------------------------- as a script
def test_script_runs_standalone(tmp_path):
    """The file itself (no uv) answers --help with the current interpreter."""
    out = subprocess.run([sys.executable, str(SCRIPT), "--help"],
                         capture_output=True, text=True, check=True)
    assert "--config" in out.stdout and "--init" in out.stdout


# ============================================================ schema 1.1
USAGE_CONFIG = """\
profiles:
  llm:
    any_tags: [llm]
  img:
    any_tags: [image]
  pick:
    ids: [comfy-nodes]
  everything:
    all: true
    include_disabled: true
runtimes:
  - id: llama.cpp
    git: https://github.com/ggml-org/llama.cpp
    ref: b6100
    release_assets: ["*ubuntu-x64*"]
    tags: [llm]
  - id: comfyui
    git: https://github.com/comfyanonymous/ComfyUI
    ref: v0.3.50
  - id: comfy-nodes
    git: https://github.com/city96/ComfyUI-GGUF
    ref: main
    tags: [image]
  - id: old
    git: https://example.org/old.git
    tags: [llm]
    enabled: false
models:
  - id: gguf-a
    hf_repo: org/a-GGUF
    category: llm
    tags: [llm]
    include: ["Q4/*"]
    usage:
      runtime: llama.cpp
      notes: |
        --temp 0.6 --top-p 0.95
  - id: gguf-a-q8
    hf_repo: org/a-GGUF
    category: llm
    tags: [big]
    include: ["Q8/*"]
    usage:
      runtime: llama.cpp
      notes: Q8 for a bigger card.
  - id: flux
    hf_repo: org/flux
    category: image
    tags: [image]
    usage:
      runtime: comfyui
      notes: euler, 20 steps
      workflows:
        - name: t2i.json
          url: https://example.org/t2i.json
        - name: edit.json
          url: https://example.org/edit.json
  - id: plain
    hf_repo: org/plain
    category: text
    tags: [other]
"""


def usage_catalog(tmp_path):
    return hfa.load_catalog(write_config(tmp_path / "u.yaml", USAGE_CONFIG))


def test_usage_and_runtimes_parse(tmp_path):
    cat = usage_catalog(tmp_path)
    flux = cat.by_id("flux")
    assert flux.usage.runtime == "comfyui"
    assert [w.name for w in flux.usage.workflows] == ["t2i.json", "edit.json"]
    assert cat.by_id("plain").usage is None
    llama = cat.runtime_by_id("llama.cpp")
    assert (llama.ref, llama.release_assets, llama.tags) == ("b6100", ["*ubuntu-x64*"], ["llm"])
    assert llama.github_repo == ("ggml-org", "llama.cpp")
    assert cat.runtime_by_id("old").ref == "HEAD" and not cat.runtime_by_id("old").enabled
    assert hfa.Runtime(id="x", git="git@github.com:o/r.git", ref="feat/x").safe_ref == "feat_x"
    assert hfa.Runtime(id="x", git="git@github.com:o/r.git").github_repo == ("o", "r")


@pytest.mark.parametrize("snippet, message", [
    ("    usage: {runtme: x}\n", "unknown keys"),
    ("    usage: {workflows: [{name: a.json}]}\n", "missing"),
    ("    usage: {workflows: [{name: ../a.json, url: u}]}\n", "plain file name"),
    ("    usage: {workflows: [{name: a, url: u}, {name: a, url: v}]}\n", "duplicate"),
])
def test_malformed_usage_is_rejected(tmp_path, snippet, message):
    body = "models:\n  - id: a\n    hf_repo: o/a\n    category: t\n" + snippet
    with pytest.raises(hfa.CatalogError, match=message):
        hfa.load_catalog(write_config(tmp_path / "m.yaml", body))


@pytest.mark.parametrize("runtimes, message", [
    ("  - {id: x}\n", "missing"),
    ("  - {id: x, git: g, colour: red}\n", "unknown keys"),
    ("  - {id: a/b, git: g}\n", "plain file name"),
    ("  - {id: x, git: https://gitlab.com/o/r, ref: v1, release_assets: ['*']}\n", "github"),
    ("  - {id: x, git: https://github.com/o/r, release_assets: ['*']}\n", "tag"),
    ("  - {id: x, git: g}\n  - {id: x, git: h}\n", "duplicate"),
])
def test_malformed_runtimes_are_rejected(tmp_path, runtimes, message):
    body = "runtimes:\n" + runtimes + "models:\n  - {id: a, hf_repo: o/a, category: t}\n"
    with pytest.raises(hfa.CatalogError, match=message):
        hfa.load_catalog(write_config(tmp_path / "m.yaml", body))


def test_example_configs_still_load_with_the_new_schema():
    for path in (EXAMPLE, CURATED):
        cat = hfa.load_catalog(path)
        known = {rt.id for rt in cat.runtimes}
        for m in cat.models:
            for wf in (m.usage.workflows if m.usage else []):
                assert wf.url.startswith("https://"), f"{m.id}: {wf.url}"
        assert len(known) == len(cat.runtimes)


# ------------------------------------------------------- runtime selection
def ids(items):
    return [x.id for x in items]


def test_runtimes_follow_usage_and_profile_tags(tmp_path):
    cat = usage_catalog(tmp_path)
    # named by usage.runtime of a selected model
    assert ids(cat.select_runtimes(cat.select("img"), "img")) == ["comfyui", "comfy-nodes"]
    # llm tag matches llama.cpp; the disabled `old` stays out
    assert ids(cat.select_runtimes(cat.select("llm"), "llm")) == ["llama.cpp"]
    # profile `ids` reach runtimes too
    assert ids(cat.select_runtimes(cat.select("pick"), "pick")) == ["comfy-nodes"]
    # include_disabled brings in disabled runtimes
    assert "old" in ids(cat.select_runtimes(cat.select("everything"), "everything"))
    # without tag rules only the named runtimes remain
    only = cat.select(only=["gguf-a-q8"])
    assert ids(cat.select_runtimes(only, by_tags=False)) == ["llama.cpp"]
    assert ids(cat.select_runtimes(cat.select(only=["plain"]), by_tags=False)) == []


def test_cli_list_shows_runtimes_and_workflows(tmp_path, capsys):
    cfg = str(write_config(tmp_path / "u.yaml", USAGE_CONFIG))
    assert hfa.main(["--config", cfg, "--list", "--profile", "img"]) == 0
    out = capsys.readouterr().out
    assert "runtimes (2):" in out and "comfy-nodes" in out
    assert "workflows (2):" in out and "https://example.org/t2i.json" in out

    assert hfa.main(["--config", cfg, "--list", "--profile", "img", "--no-runtimes"]) == 0
    out = capsys.readouterr().out
    assert "runtimes (" not in out and "comfyui" in out  # named but not archived

    assert hfa.main(["--config", cfg, "--list", "--json", "--only", "gguf-a"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert [rt["id"] for rt in data["runtimes"]] == ["llama.cpp"]
    assert data["models"][0]["usage"]["runtime"] == "llama.cpp"


# -------------------------------------------------------- model cards
class FakeSibling:
    def __init__(self, name, size, sha256=None, blob=None):
        self.rfilename, self.size, self.blob_id = name, size, blob or "b" * 40
        self.lfs = {"sha256": sha256, "size": size} if sha256 else None


class FakeApi:
    def __init__(self, siblings, sha="c" * 40):
        self.siblings, self.sha = siblings, sha

    def model_info(self, repo, revision=None, files_metadata=False):
        assert files_metadata
        return type("Info", (), {"sha": self.sha, "gated": False, "siblings": self.siblings})


SIBLINGS = [
    FakeSibling("README.md", 10, blob="1" * 40),
    FakeSibling("LICENSE.txt", 20, blob="2" * 40),
    FakeSibling("docs/README.md", 5),
    FakeSibling("Q4/model-Q4.gguf", 1000, sha256="a" * 64),
    FakeSibling("Shard_Rewrite/model-Q4.gguf", 7, sha256="d" * 64),
    FakeSibling("Q8/model-Q8.gguf", 2000, sha256="e" * 64),
]


def test_model_cards_are_always_included():
    m = hfa.Model(id="a", hf_repo="o/a", category="t", include=["Q4/*"], exclude=["*.md"])
    names = [s.rfilename for s in SIBLINGS]
    matched, extra = hfa.select_files(m, names)
    assert matched == ["Q4/model-Q4.gguf"]
    assert extra == ["README.md", "LICENSE.txt"]
    assert hfa.select_files(hfa.Model(id="b", hf_repo="o/b", category="t"), names)[1] == []

    st = hfa.probe(m, FakeApi(SIBLINGS))
    assert st.error is None
    assert st.files == [("Q4/model-Q4.gguf", 1000), ("README.md", 10), ("LICENSE.txt", 20)]
    assert st.hashes["Q4/model-Q4.gguf"] == {"sha256": "a" * 64}
    assert st.hashes["README.md"] == {"git_sha1": "1" * 40}

    only_cards = hfa.Model(id="c", hf_repo="o/c", category="t", include=["nothing*"])
    assert hfa.probe(only_cards, FakeApi(SIBLINGS)).error


def test_excluded_cards_are_fetched_by_name_at_the_probed_commit(tmp_path, monkeypatch):
    m = hfa.Model(id="a", hf_repo="o/a", category="t", include=["Q4/*"], revision="main")
    st = hfa.probe(m, FakeApi(SIBLINGS))
    calls = []
    monkeypatch.setattr(hfa, "_run_hf", lambda cmd: calls.append(cmd) or (0, ""))
    res = hfa.download_one(st, tmp_path, "hf", 2, True, threading.Lock())
    assert res.status == "downloaded"
    main_cmd, card_cmd = calls
    assert main_cmd[main_cmd.index("--revision") + 1] == "c" * 40
    assert card_cmd[:5] == ["hf", "download", "o/a", "README.md", "LICENSE.txt"]
    assert card_cmd[card_cmd.index("--revision") + 1] == "c" * 40


# ------------------------------------------------- usage, workflows, index
def fake_fetch_url(payloads):
    def fetch(url, target, expected_size=None, expected_sha256=None):
        if url not in payloads:
            raise hfa.urllib.error.URLError("404")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payloads[url])
        return len(payloads[url]), hashlib.sha256(payloads[url]).hexdigest()
    return fetch


def test_usage_md_and_workflows(tmp_path, monkeypatch):
    cat = usage_catalog(tmp_path)
    monkeypatch.setattr(hfa, "fetch_url",
                        fake_fetch_url({"https://example.org/t2i.json": b'{"a": 1}'}))
    commits = {"flux": "f" * 40, "gguf-a": "a" * 40}
    extras = hfa.write_usage_files([cat.by_id("flux"), cat.by_id("gguf-a"), cat.by_id("plain")],
                                   tmp_path, cat, commits)
    assert set(extras) == {"flux", "gguf-a"}

    flux_dir = tmp_path / "image" / "org" / "flux"
    assert (flux_dir / "_workflows" / "t2i.json").read_bytes() == b'{"a": 1}'
    wfs = {w["name"]: w for w in extras["flux"]["workflows"]}
    assert wfs["t2i.json"]["sha256"] == hashlib.sha256(b'{"a": 1}').hexdigest()
    assert "error" in wfs["edit.json"], "a failed workflow is reported, not fatal"

    usage = (flux_dir / "USAGE.md").read_text()
    assert "https://huggingface.co/org/flux" in usage
    assert "euler, 20 steps" in usage
    assert "`_workflows/t2i.json`" in usage
    assert "Runtime: comfyui, archived under `../../../_runtimes/comfyui/`" in usage
    assert "f" * 40 in usage

    # two entries share org/a-GGUF: one USAGE.md covers both
    shared = (tmp_path / "llm" / "org" / "a-GGUF" / "USAGE.md").read_text()
    assert "## gguf-a\n" in shared and "## gguf-a-q8\n" in shared
    assert "--temp 0.6" in shared and "Q8 for a bigger card." in shared
    assert extras["gguf-a"]["usage"] == "llm/org/a-GGUF/USAGE.md"


def test_index_md_lists_models_and_runtimes(tmp_path):
    cat = usage_catalog(tmp_path)
    flux = cat.by_id("flux")
    st = hfa.RepoStatus(model=flux, exists=True, commit="f" * 40,
                        files=[("model.safetensors", 3_000_000_000)])
    res = hfa.DownloadResult(flux, "downloaded", commit="f" * 40)
    extras = {"flux": {"usage": "image/org/flux/USAGE.md", "workflows": [
        {"name": "t2i.json", "path": "image/org/flux/_workflows/t2i.json", "size": 1},
        {"name": "edit.json", "path": "image/org/flux/_workflows/edit.json", "error": "404"},
    ]}}
    hfa.manifest_append(tmp_path, [res], {"flux": st}, extras)
    failed = hfa.DownloadResult(cat.by_id("plain"), "failed", message="boom")
    hfa.manifest_append(tmp_path, [failed], {})
    rt = cat.runtime_by_id("comfyui")
    hfa.manifest_write(tmp_path, hfa.runtime_records([hfa.RuntimeResult(
        rt, "downloaded", "9" * 40,
        [{"path": "_runtimes/comfyui/comfyui-" + "9" * 40 + ".bundle", "size": 1}])], tmp_path))

    text = hfa.write_index(tmp_path).read_text()
    assert "### image" in text
    assert "[org/flux](<image/org/flux/>)" in text
    assert "`main` @ `" + "f" * 40 + "`" in text
    assert "3.00 GB" in text
    assert "| comfyui |" in text
    assert "[USAGE.md](<image/org/flux/USAGE.md>)" in text
    assert "[t2i.json](<image/org/flux/_workflows/t2i.json>)" in text
    assert "edit.json" not in text, "failed workflows are not listed as archived"
    assert "org/plain" not in text, "failed models are not listed"
    assert "## Runtimes (1)" in text and "`v0.3.50`" in text and "9" * 40 in text


# ------------------------------------------------------------------ check
def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def fabricate_archive(dest: Path) -> dict[str, Path]:
    """A drive with one model (LFS + non-LFS file), a workflow and a runtime."""
    m = hfa.Model(id="tiny", hf_repo="org/tiny", category="text")
    root = m.dest_dir(dest)
    (root / "sub").mkdir(parents=True)
    weights, readme = os.urandom(4096), b"# tiny\n"
    (root / "sub" / "w.gguf").write_bytes(weights)
    (root / "README.md").write_bytes(readme)
    wf = root / "_workflows" / "x.json"
    wf.parent.mkdir()
    wf.write_bytes(b"{}")
    bundle = dest / "_runtimes" / "rt" / "rt-abc.bundle"
    bundle.parent.mkdir(parents=True)
    bundle.write_bytes(b"bundle bytes")

    st = hfa.RepoStatus(model=m, exists=True, commit="c" * 40,
                        files=[("sub/w.gguf", len(weights)), ("README.md", len(readme))],
                        hashes={"sub/w.gguf": {"sha256": hashlib.sha256(weights).hexdigest()},
                                "README.md": {"git_sha1": git_blob_sha1(readme)}})
    extras = {"tiny": {"workflows": [{"name": "x.json", "path": os.path.relpath(wf, dest),
                                      "size": 2, "sha256": hashlib.sha256(b"{}").hexdigest()}]}}
    hfa.manifest_append(dest, [hfa.DownloadResult(m, "downloaded", commit="c" * 40)],
                        {"tiny": st}, extras)
    rt = hfa.Runtime(id="rt", git="https://example.org/rt.git", ref="v1")
    hfa.manifest_write(dest, hfa.runtime_records(
        [hfa.RuntimeResult(rt, "downloaded", "abc", [hfa.artifact_record(bundle, dest)])], dest))
    return {"weights": root / "sub" / "w.gguf", "readme": root / "README.md",
            "workflow": wf, "bundle": bundle}


def flip_one_byte(path: Path) -> None:
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0xFF
    path.write_bytes(bytes(data))


def check(dest, *extra):
    return hfa.main(["--check", "--dest", str(dest), *extra])


def test_git_blob_hash_matches_git(tmp_path):
    if not shutil.which("git"):
        pytest.skip("git not installed")
    f = tmp_path / "f.txt"
    f.write_bytes(b"hello\n")
    expected = subprocess.run(["git", "hash-object", str(f)], capture_output=True,
                              text=True, check=True).stdout.strip()
    assert hfa.hash_file(f, "git_sha1") == expected == git_blob_sha1(b"hello\n")


def test_check_clean_archive(tmp_path, monkeypatch):
    monkeypatch.setattr(hfa, "HfApi", lambda *a, **k: pytest.fail("--check must be offline"))
    fabricate_archive(tmp_path)
    assert check(tmp_path) == 0
    assert check(tmp_path, "--fast") == 0


@pytest.mark.parametrize("victim", ["weights", "readme", "workflow", "bundle"])
def test_check_finds_a_corrupted_file_only_when_hashing(tmp_path, victim, capsys):
    files = fabricate_archive(tmp_path)
    flip_one_byte(files[victim])  # same size, different bytes
    assert check(tmp_path, "--fast") == 0, "--fast compares sizes only"
    assert check(tmp_path, "--json") == 1
    report = json.loads(capsys.readouterr().out)
    [problem] = report["problems"]
    assert problem["path"] == str(files[victim])
    assert problem["problem"].startswith("wrong ")


def test_check_finds_missing_and_truncated_files(tmp_path, capsys):
    files = fabricate_archive(tmp_path)
    files["bundle"].unlink()
    files["weights"].write_bytes(b"short")
    assert check(tmp_path, "--fast", "--json") == 1
    problems = {p["path"]: p["problem"]
                for p in json.loads(capsys.readouterr().out)["problems"]}
    assert problems[str(files["bundle"])] == "missing"
    assert problems[str(files["weights"])].startswith("wrong size")


def test_check_uses_the_latest_good_record_and_flags_old_ones(tmp_path, capsys):
    fabricate_archive(tmp_path)
    m = hfa.Model(id="tiny", hf_repo="org/tiny", category="text")
    hfa.manifest_append(tmp_path, [hfa.DownloadResult(m, "failed", message="x")], {})
    assert check(tmp_path) == 0, "a later failed run does not hide the good record"
    # a record written before 1.1 has no checksums: reported, not fatal
    path = tmp_path / "_hfarchive" / "manifest.jsonl"
    with path.open("a") as fh:
        fh.write(json.dumps({"id": "old", "status": "downloaded", "rel_dest": "x/y/z"}) + "\n")
    assert check(tmp_path, "--json") == 0
    assert json.loads(capsys.readouterr().out)["unverifiable"] == ["old"]


def test_check_needs_a_destination_and_a_manifest(tmp_path):
    assert hfa.main(["--check"]) == 2
    assert check(tmp_path / "absent") == 2
    assert check(tmp_path) == 2  # no manifest
    with pytest.raises(SystemExit):
        hfa.main(["--fast", "--list"])


# ---------------------------------------------------------------- runtimes
def make_upstream(tmp_path: Path, files: dict[str, str] | None = None):
    """A local repo with an annotated tag v1 and one commit after it: (path, git).
    `files` are added to the tagged commit."""
    if not shutil.which("git"):
        pytest.skip("git not installed")
    repo = tmp_path / "upstream"
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}

    def g(*args):
        return subprocess.run(["git", "-C", str(repo), *args], check=True, env=env,
                              capture_output=True, text=True).stdout.strip()

    repo.mkdir()
    g("init", "-q", "-b", "main")
    (repo / "main.c").write_text("int main(void) { return 0; }\n")
    for name, text in (files or {}).items():
        (repo / name).write_text(text)
    g("add", ".")
    g("commit", "-qm", "first")
    g("tag", "-a", "v1", "-m", "release v1")
    (repo / "main.c").write_text("int main(void) { return 1; }\n")
    g("commit", "-qam", "second")
    return repo, g


def test_resolve_remote_ref_peels_tags(tmp_path):
    up, g = make_upstream(tmp_path)
    tagged = g("rev-parse", "v1^{commit}")
    assert hfa.resolve_remote_ref(str(up), "v1") == tagged
    assert hfa.resolve_remote_ref(str(up), "main") == g("rev-parse", "main")
    assert hfa.resolve_remote_ref(str(up), "HEAD") == g("rev-parse", "HEAD")
    assert hfa.resolve_remote_ref(str(up), "nope") is None
    assert hfa.resolve_remote_ref(str(up), "a" * 40) == "a" * 40


def test_runtime_archive_bundle_tarball_assets_and_idempotency(tmp_path, monkeypatch):
    up, g = make_upstream(tmp_path)
    dest = tmp_path / "drive"
    dest.mkdir()
    rt = hfa.Runtime(id="tool", git=str(up), ref="v1")
    asset = b"prebuilt binary"
    monkeypatch.setattr(hfa, "release_assets", lambda r: [{
        "name": "tool-v1-linux.zip", "size": len(asset), "url": "https://example.org/a.zip",
        "sha256": hashlib.sha256(asset).hexdigest()}])
    monkeypatch.setattr(hfa, "fetch_url", fake_fetch_url({"https://example.org/a.zip": asset}))
    rt.release_assets = ["*linux*"]  # set after parse-time validation

    st = hfa.probe_runtime(rt, dest, {})
    tagged = g("rev-parse", "v1^{commit}")
    assert st.commit == tagged and not st.archived and st.asset_bytes == len(asset)
    res = hfa.archive_runtime(st, dest)
    assert res.status == "downloaded", res.message
    d = dest / "_runtimes" / "tool"
    bundle = d / f"tool-{tagged}.bundle"
    assert bundle.is_file() and (d / "tool-v1-src.tar.gz").is_file()
    assert (d / "release-v1" / "tool-v1-linux.zip").read_bytes() == asset
    assert {a["kind"] for a in res.artifacts} == {"git-bundle", "source", "release-asset"}

    # the bundle is a portable, complete copy of the history
    clone = tmp_path / "restored"
    subprocess.run(["git", "clone", "-q", str(bundle), str(clone)], check=True)
    subprocess.run(["git", "-C", str(clone), "checkout", "-q", tagged], check=True)
    assert "return 0" in (clone / "main.c").read_text()

    import tarfile
    with tarfile.open(d / "tool-v1-src.tar.gz") as tar:
        assert "tool-v1/main.c" in tar.getnames()

    hfa.manifest_write(dest, hfa.runtime_records([res], dest))
    assert check(dest) == 0

    # same sha already archived: skipped without touching git
    again = hfa.probe_runtimes([rt], dest)[0]
    assert again.archived
    monkeypatch.setattr(hfa, "git", lambda *a: pytest.fail("must not run git"))
    skipped = hfa.archive_runtime(again, dest)
    assert skipped.status == "up-to-date" and skipped.artifacts == res.artifacts


def test_runtime_errors_are_reported_not_raised(tmp_path, monkeypatch):
    up, g = make_upstream(tmp_path)

    def rate_limited(rt):
        raise hfa.GitHubError("GitHub API rate limit reached.")

    monkeypatch.setattr(hfa, "release_assets", rate_limited)
    rt = hfa.Runtime(id="tool", git=str(up), ref="v1", release_assets=["*"])
    st = hfa.probe_runtime(rt, tmp_path, {})
    assert "rate limit" in st.error
    res = hfa.archive_runtime(st, tmp_path)
    assert res.status == "failed" and "rate limit" in res.message

    bad = hfa.probe_runtime(hfa.Runtime(id="x", git=str(tmp_path / "nowhere")), tmp_path, {})
    assert bad.error and "ls-remote" in bad.error


def test_github_rate_limit_message(monkeypatch):
    import email.message

    headers = email.message.Message()
    headers["X-RateLimit-Remaining"] = "0"

    def limited(url, headers_=None, *a, **k):
        raise hfa.urllib.error.HTTPError(url, 403, "rate limited", headers, None)

    monkeypatch.setattr(hfa, "http_open", limited)
    monkeypatch.setattr(hfa, "github_token", lambda: None)
    with pytest.raises(hfa.GitHubError, match="GITHUB_TOKEN"):
        hfa.github_api("/repos/o/r/releases/tags/v1")


def test_release_assets_filters_by_glob_and_reads_digests(monkeypatch):
    monkeypatch.setattr(hfa, "github_api", lambda path: {"assets": [
        {"name": "llama-b1-bin-ubuntu-x64.zip", "size": 5, "browser_download_url": "u1",
         "digest": "sha256:" + "a" * 64},
        {"name": "llama-b1-bin-win-cuda.zip", "size": 9, "browser_download_url": "u2"},
    ]})
    rt = hfa.Runtime(id="l", git="https://github.com/o/r", ref="b1",
                     release_assets=["*ubuntu-x64*"])
    assert hfa.release_assets(rt) == [
        {"name": "llama-b1-bin-ubuntu-x64.zip", "size": 5, "url": "u1", "sha256": "a" * 64}]
    rt.release_assets = ["*macos*"]
    with pytest.raises(hfa.GitHubError, match="no assets"):
        hfa.release_assets(rt)


# ---------------------------------------------------- full run, all mocked
def test_full_run_writes_usage_index_runtimes_and_passes_check(tmp_path, monkeypatch):
    up, g = make_upstream(tmp_path)
    dest = tmp_path / "drive"
    dest.mkdir()
    body = USAGE_CONFIG.replace("https://github.com/comfyanonymous/ComfyUI", str(up))
    body = body.replace("ref: v0.3.50", "ref: v1")
    cfg = write_config(tmp_path / "u.yaml", body)
    flux = hfa.load_catalog(cfg).by_id("flux")

    weights, readme = os.urandom(2048), b"# flux\n"
    st = hfa.RepoStatus(model=flux, exists=True, commit="f" * 40,
                        files=[("flux.safetensors", len(weights)), ("README.md", len(readme))],
                        hashes={"flux.safetensors": {"sha256": hashlib.sha256(weights).hexdigest()},
                                "README.md": {"git_sha1": git_blob_sha1(readme)}})
    monkeypatch.setattr(hfa, "probe_all", lambda models: [st])
    monkeypatch.setattr(hfa, "resolve_hf_bin", lambda: "hf")

    def fake_hf(cmd):
        root = Path(cmd[cmd.index("--local-dir") + 1])
        (root / "flux.safetensors").write_bytes(weights)
        (root / "README.md").write_bytes(readme)
        return 0, ""

    monkeypatch.setattr(hfa, "_run_hf", fake_hf)
    monkeypatch.setattr(hfa, "fetch_url", fake_fetch_url({
        "https://example.org/t2i.json": b"{}", "https://example.org/edit.json": b"[]"}))

    args = ["--config", str(cfg), "--dest", str(dest), "--allow-home-dest",
            "--min-free-gb", "0", "--only", "flux"]
    assert hfa.main(args) == 0
    assert (dest / "INDEX.md").is_file()
    assert (dest / "image" / "org" / "flux" / "USAGE.md").is_file()
    assert (dest / "image" / "org" / "flux" / "_workflows" / "edit.json").read_bytes() == b"[]"
    assert list((dest / "_runtimes" / "comfyui").glob("comfyui-*.bundle"))
    assert "comfyui" in (dest / "INDEX.md").read_text()
    assert check(dest) == 0

    # a workflow that fails marks the run partial, but the rest still lands
    monkeypatch.setattr(hfa, "fetch_url", fake_fetch_url({"https://example.org/t2i.json": b"{}"}))
    assert hfa.main(args) == 1
    assert (dest / "INDEX.md").is_file()

    flip_one_byte(dest / "image" / "org" / "flux" / "flux.safetensors")
    assert check(dest, "--fast") == 0
    assert check(dest) == 1


# ------------------------------------------------- benchmarks and the guide
BENCH_CONFIG = """\
guide: docs/guide.md
models:
  - id: coder
    hf_repo: org/coder-GGUF
    category: llm
    benchmarks:
      - name: SWE-bench Verified
        score: 70.6
        source: https://example.org/card
        date: 2026-11-13
        note: self-reported, OpenHands harness
      - {name: LiveCodeBench, score: 81, source: https://example.org/lcb}
      - {name: Arena, score: 1156 Elo, source: https://example.org/arena}
      - {name: GPQA Diamond, score: 77.2, source: https://example.org/gpqa}
  - id: plain
    hf_repo: org/plain
    category: text
"""


def bench_catalog(tmp_path, body=BENCH_CONFIG):
    return hfa.load_catalog(write_config(tmp_path / "b.yaml", body))


def test_benchmarks_and_guide_parse(tmp_path):
    cat = bench_catalog(tmp_path)
    assert cat.guide == "docs/guide.md"
    swe, lcb, arena, _ = cat.by_id("coder").benchmarks
    assert (swe.score, swe.date, swe.note) == (70.6, "2026-11-13", "self-reported, OpenHands harness")
    assert (lcb.score, lcb.date) == (81, None)
    assert arena.score == "1156 Elo" and arena.headline == "Arena 1156 Elo"
    assert cat.by_id("plain").benchmarks == []


@pytest.mark.parametrize("snippet, message", [
    ("    benchmarks: {name: x}\n", "must be a list"),
    ("    benchmarks: [{name: x, score: 1}]\n", "missing"),
    ("    benchmarks: [{name: x, score: 1, source: https://e.org, url: u}]\n", "unknown keys"),
    ("    benchmarks: [{name: '', score: 1, source: https://e.org}]\n", "name"),
    ("    benchmarks: [{name: x, score: true, source: https://e.org}]\n", "score"),
    ("    benchmarks: [{name: x, score: [1], source: https://e.org}]\n", "score"),
    ("    benchmarks: [{name: x, score: '" + "9" * 41 + "', source: https://e.org}]\n", "score"),
    ("    benchmarks: [{name: x, score: 1, source: e.org/card}]\n", "http"),
    ("    benchmarks: [{name: x, score: 1, source: https://e.org, date: 3}]\n", "date"),
    ("    benchmarks: [{name: x, score: 1, source: https://e.org, note: [a]}]\n", "note"),
    ("    benchmarks: [nope]\n", "mapping"),
])
def test_malformed_benchmarks_are_rejected(tmp_path, snippet, message):
    body = "models:\n  - id: a\n    hf_repo: o/a\n    category: t\n" + snippet
    with pytest.raises(hfa.CatalogError, match=message):
        hfa.load_catalog(write_config(tmp_path / "m.yaml", body))


@pytest.mark.parametrize("guide, message", [
    ("''", "path or an https URL"),
    ("[a.md]", "path or an https URL"),
    ("ftp://example.org/g.md", "http"),
    ("docs/INDEX.md", "plain file name"),
    ("https://example.org/", "plain file name"),
])
def test_malformed_guide_is_rejected(tmp_path, guide, message):
    body = f"guide: {guide}\nmodels:\n  - {{id: a, hf_repo: o/a, category: t}}\n"
    with pytest.raises(hfa.CatalogError, match=message):
        hfa.load_catalog(write_config(tmp_path / "m.yaml", body))


def test_guide_names_and_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(hfa, "repo_root", lambda: tmp_path / "clone")
    assert hfa.guide_name("https://example.org/a/models.md?raw=1") == "models.md"
    cat = bench_catalog(tmp_path)
    assert hfa.guide_sources(cat) == [
        tmp_path / "docs" / "guide.md", tmp_path / "clone" / "docs" / "guide.md",
        *hfa.repo_raw_urls("docs/guide.md")]
    assert hfa.repo_raw_urls("docs/guide.md")[1] == (
        "https://raw.githubusercontent.com/IpastorSan/hfarchive/main/docs/guide.md")
    # nothing local: the first GitHub URL is what --list reports
    assert hfa.guide_source(cat) == hfa.repo_raw_urls("docs/guide.md")[0]
    # paths that leave the config's directory are not looked up in the repo
    cat.guide = "../guide.md"
    assert hfa.guide_sources(cat) == [(tmp_path / ".." / "guide.md")]
    cat.guide = "https://example.org/g.md"
    assert hfa.guide_sources(cat) == ["https://example.org/g.md"]


def test_usage_md_is_written_for_benchmarks_alone(tmp_path):
    cat = bench_catalog(tmp_path)
    extras = hfa.write_usage_files(cat.models, tmp_path, cat, {"coder": "c" * 40})
    assert set(extras) == {"coder"}
    assert extras["coder"]["workflows"] == []
    usage = (tmp_path / "llm" / "org" / "coder-GGUF" / "USAGE.md").read_text()
    assert "### Benchmarks" in usage
    assert hfa.BENCHMARK_CAVEAT in usage
    assert ("| SWE-bench Verified | 70.6 | <https://example.org/card> | 2026-11-13 | "
            "self-reported, OpenHands harness |") in usage
    assert "| Arena | 1156 Elo | <https://example.org/arena> | - | - |" in usage


def test_usage_md_has_benchmarks_next_to_usage(tmp_path, monkeypatch):
    body = USAGE_CONFIG.replace(
        "  - id: plain\n",
        "    benchmarks:\n"
        "      - {name: Arena, score: 1100 Elo, source: https://example.org/arena}\n"
        "  - id: plain\n")
    cat = hfa.load_catalog(write_config(tmp_path / "u.yaml", body))
    monkeypatch.setattr(hfa, "fetch_url", fake_fetch_url({}))
    hfa.write_usage_files([cat.by_id("flux")], tmp_path, cat, {})
    usage = (tmp_path / "image" / "org" / "flux" / "USAGE.md").read_text()
    assert usage.index("### Workflows") < usage.index("### Benchmarks")
    assert "| Arena | 1100 Elo |" in usage and "euler, 20 steps" in usage


def test_index_md_shows_headline_benchmarks_guide_and_config(tmp_path):
    cat = bench_catalog(tmp_path)
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "guide.md").write_text("# guide\n")
    coder = cat.by_id("coder")
    st = hfa.RepoStatus(model=coder, exists=True, commit="c" * 40, files=[("a.gguf", 1)])
    hfa.manifest_append(tmp_path, [hfa.DownloadResult(coder, "downloaded", commit="c" * 40)],
                        {"coder": st})
    text = hfa.index_markdown(hfa.manifest_read(tmp_path))
    assert "SWE-bench Verified 70.6; LiveCodeBench 81; Arena 1156 Elo |" in text
    assert "GPQA Diamond" not in text, "only the first three are headlines"
    assert "Selection guide" not in text and "config-latest" not in text

    hfa.manifest_write(tmp_path, [hfa.archive_config(cat.path, tmp_path),
                                  hfa.archive_guide(cat, tmp_path)])
    text = hfa.write_index(tmp_path).read_text()
    head = text.split("## Models")[0]
    assert "- Selection guide: [guide.md](<guide.md>)" in head
    assert "[config-latest.yaml](<_hfarchive/config-latest.yaml>)" in head
    assert "[_hfarchive/configs/](<_hfarchive/configs/>)" in head


def test_config_is_archived_with_a_dated_copy_and_a_latest(tmp_path):
    cfg = write_config(tmp_path / "my-models.yaml", MINIMAL)
    dest = tmp_path / "drive"
    rec = hfa.archive_config(cfg, dest)
    stamped, latest = rec["artifacts"]
    assert re.fullmatch(r"_hfarchive/configs/my-models-\d{8}-\d{6}\.yaml", stamped["path"])
    assert latest["path"] == "_hfarchive/config-latest.yaml"
    for art in (stamped, latest):
        assert (dest / art["path"]).read_text() == MINIMAL
        assert art["sha256"] == hashlib.sha256(MINIMAL.encode()).hexdigest()
    assert rec["source"] == str(cfg.resolve())


def test_guide_is_copied_from_the_config_directory_first(tmp_path, monkeypatch):
    clone = tmp_path / "clone"
    (clone / "docs").mkdir(parents=True)
    (clone / "docs" / "guide.md").write_text("from the clone\n")
    monkeypatch.setattr(hfa, "repo_root", lambda: clone)
    monkeypatch.setattr(hfa, "fetch_url", lambda *a, **k: pytest.fail("must not fetch"))
    cat = bench_catalog(tmp_path)
    dest = tmp_path / "drive"

    # not next to the config: the clone's copy
    rec = hfa.archive_guide(cat, dest)
    assert rec["status"] == "downloaded" and rec["source"] == str(clone / "docs" / "guide.md")
    assert (dest / "guide.md").read_text() == "from the clone\n"

    # next to the config: that one wins
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "guide.md").write_text("mine\n")
    rec = hfa.archive_guide(cat, dest)
    assert rec["source"] == str(tmp_path / "docs" / "guide.md")
    assert (dest / "guide.md").read_text() == "mine\n"
    assert rec["sha256"] == hashlib.sha256(b"mine\n").hexdigest() and rec["size"] == 5


def test_guide_falls_back_to_github_then_reports_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(hfa, "repo_root", lambda: tmp_path / "standalone")
    tag_url, main_url = hfa.repo_raw_urls("docs/guide.md")
    monkeypatch.setattr(hfa, "fetch_url", fake_fetch_url({main_url: b"# from main\n"}))
    cat = bench_catalog(tmp_path)
    dest = tmp_path / "drive"
    rec = hfa.archive_guide(cat, dest)
    assert rec["status"] == "downloaded" and rec["source"] == main_url
    assert (dest / "guide.md").read_bytes() == b"# from main\n"

    monkeypatch.setattr(hfa, "fetch_url", fake_fetch_url({}))
    rec = hfa.archive_guide(cat, dest)
    assert rec["status"] == "failed"
    assert tag_url in rec["message"] and main_url in rec["message"]
    assert (dest / "guide.md").read_bytes() == b"# from main\n", "a failure keeps the old copy"

    cat.guide = "https://example.org/x/remote.md"
    monkeypatch.setattr(hfa, "fetch_url", fake_fetch_url({cat.guide: b"remote"}))
    assert hfa.archive_guide(cat, dest)["path"] == "remote.md"
    assert (dest / "remote.md").read_bytes() == b"remote"


def test_init_from_a_clone_keeps_the_guide_resolvable(tmp_path, monkeypatch):
    monkeypatch.setattr(hfa, "fetch_text", lambda url: pytest.fail("must not fetch"))
    target = tmp_path / "work" / "models.yaml"
    assert hfa.main(["--init", "--config", str(target)]) == 0
    cat = hfa.load_catalog(target)
    assert cat.guide == "docs/models-2026-11.md"
    # nothing at ./docs: the clone's copy is used, unchanged config
    assert hfa.guide_source(cat) == PROJECT_ROOT / "docs" / "models-2026-11.md"
    dest = tmp_path / "drive"
    rec = hfa.archive_guide(cat, dest)
    assert rec["status"] == "downloaded"
    assert (dest / "models-2026-11.md").read_bytes() == (
        PROJECT_ROOT / "docs" / "models-2026-11.md").read_bytes()

    # standalone script, no clone: GitHub at this version's tag, then main
    monkeypatch.setattr(hfa, "repo_root", lambda: tmp_path / "standalone")
    assert hfa.guide_source(cat) == hfa.repo_raw_urls("docs/models-2026-11.md")[0]


def test_curated_selection_names_its_guide():
    cat = hfa.load_catalog(CURATED)
    assert cat.guide and (PROJECT_ROOT / cat.guide).is_file()


def test_cli_list_and_dry_run_mention_the_guide_and_benchmarks(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(hfa, "repo_root", lambda: tmp_path / "standalone")
    cfg = str(write_config(tmp_path / "b.yaml", BENCH_CONFIG))
    assert hfa.main(["--config", cfg, "--list"]) == 0
    out = capsys.readouterr().out
    assert "BENCH" in out and re.search(r"coder\s.*\s4\s", out)
    assert "guide: docs/guide.md -> <DEST>/guide.md (from https://raw.githubusercontent" in out

    assert hfa.main(["--config", cfg, "--list", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["guide"]["name"] == "guide.md"
    assert data["models"][0]["benchmarks"][0]["name"] == "SWE-bench Verified"

    cat = hfa.load_catalog(cfg)
    monkeypatch.setattr(hfa, "probe_all", lambda models: [
        hfa.RepoStatus(model=m, exists=True, commit="c" * 40, files=[("a", 1)]) for m in models])
    assert hfa.main(["--config", cfg, "--dry-run", "--json"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan[0]["benchmarks"] == 4
    assert plan[-1] == {"kind": "guide", **hfa.guide_json(cat)}
    assert hfa.main(["--config", cfg, "--dry-run"]) == 0
    captured = capsys.readouterr()
    assert "BENCH" in captured.out
    assert "guide: docs/guide.md" in captured.err


def test_full_run_archives_config_and_guide_and_check_covers_them(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(hfa, "repo_root", lambda: tmp_path / "standalone")
    dest = tmp_path / "drive"
    dest.mkdir()
    cfg = write_config(tmp_path / "b.yaml", BENCH_CONFIG)
    (tmp_path / "docs").mkdir()
    guide = tmp_path / "docs" / "guide.md"
    guide.write_text("# guide v1\n")
    coder = hfa.load_catalog(cfg).by_id("coder")
    weights = os.urandom(1024)
    st = hfa.RepoStatus(model=coder, exists=True, commit="c" * 40,
                        files=[("a.gguf", len(weights))],
                        hashes={"a.gguf": {"sha256": hashlib.sha256(weights).hexdigest()}})
    monkeypatch.setattr(hfa, "probe_all", lambda models: [st])
    monkeypatch.setattr(hfa, "resolve_hf_bin", lambda: "hf")

    def fake_hf(cmd):
        (Path(cmd[cmd.index("--local-dir") + 1]) / "a.gguf").write_bytes(weights)
        return 0, ""

    monkeypatch.setattr(hfa, "_run_hf", fake_hf)
    monkeypatch.setattr(hfa, "fetch_url", fake_fetch_url({}))
    args = ["--config", str(cfg), "--dest", str(dest), "--allow-home-dest",
            "--min-free-gb", "0", "--only", "coder"]
    assert hfa.main(args) == 0
    assert (dest / "guide.md").read_text() == "# guide v1\n"
    assert (dest / "_hfarchive" / "config-latest.yaml").read_text() == BENCH_CONFIG
    assert len(list((dest / "_hfarchive" / "configs").glob("b-*.yaml"))) == 1
    index = (dest / "INDEX.md").read_text()
    assert "[guide.md](<guide.md>)" in index and "SWE-bench Verified 70.6" in index
    assert "### Benchmarks" in (dest / "llm" / "org" / "coder-GGUF" / "USAGE.md").read_text()
    assert check(dest) == 0

    # a changed config next run: the old dated copy stays verifiable, latest follows
    real_strftime = hfa.time.strftime
    monkeypatch.setattr(hfa.time, "strftime", lambda fmt, *a: "20990101-000000"
                        if fmt == "%Y%m%d-%H%M%S" else real_strftime(fmt, *a))
    cfg.write_text(BENCH_CONFIG + "# edited\n")
    assert hfa.main(args) == 0
    assert len(list((dest / "_hfarchive" / "configs").glob("b-*.yaml"))) == 2
    assert check(dest) == 0

    # the guide cannot be found anywhere: partial run, everything else lands
    guide.unlink()
    assert hfa.main(args) == 1
    assert (dest / "guide.md").read_text() == "# guide v1\n"
    assert check(dest) == 0, "the last good guide is still checked"

    capsys.readouterr()
    flip_one_byte(dest / "guide.md")
    flip_one_byte(dest / "_hfarchive" / "config-latest.yaml")
    assert check(dest, "--fast") == 0
    assert check(dest, "--json") == 1
    labels = sorted(p["file"] for p in json.loads(capsys.readouterr().out)["problems"])
    assert labels == ["config: _hfarchive/config-latest.yaml", "guide: guide.md"]
    (dest / "_hfarchive" / "config-latest.yaml").unlink()
    for old in (dest / "_hfarchive" / "configs").glob("b-*.yaml"):
        old.unlink()
    assert check(dest, "--fast", "--json") == 1
    problems = json.loads(capsys.readouterr().out)["problems"]
    assert sum(p["file"].startswith("config: _hfarchive/configs/") for p in problems) == 2


# ------------------------------------------------------------------ wheels
WHEELS_OK = {"python": "3.12", "platforms": ["manylinux_2_28_x86_64", "win_amd64"],
             "requirements": ["requirements.txt"]}


def test_wheels_parse():
    rt = hfa.Runtime.parse({"id": "c", "git": "g", "ref": "v1", "wheels": {
        **WHEELS_OK, "packages": ["torch==2.14.1+cu130"],
        "index_url": "https://download.pytorch.org/whl/cu130",
        "extra_index_urls": ["https://pypi.org/simple"], "start": "python main.py"}})
    assert rt.wheels.python == "3.12" and rt.wheels.start == "python main.py"
    assert rt.wheels_dir(Path("/d")) == Path("/d/_runtimes/c/wheels-v1")
    assert rt.wheels.describe() == ("py3.12: manylinux_2_28_x86_64, win_amd64; "
                                    "1 requirement file; 1 package")
    assert hfa.Runtime.parse({"id": "c", "git": "g"}).wheels is None


@pytest.mark.parametrize("change, message", [
    ({"python": 3.12}, "quoted"),
    ({"python": "py3"}, "quoted"),
    ({"platforms": ["linux"]}, "not supported"),
    ({"platforms": []}, "non-empty"),
    ({"platforms": ["win_amd64", "win_amd64"]}, "duplicates"),
    ({"requirements": ["../x.txt"]}, "relative path"),
    ({"requirements": ["/etc/x.txt"]}, "relative path"),
    ({"requirements": []}, "requirements` or `packages"),
    ({"index_url": "download.pytorch.org"}, "index_url"),
    ({"extra_index_urls": ["ftp://x"]}, "http"),
    ({"colour": "red"}, "unknown keys"),
])
def test_malformed_wheels_are_rejected(change, message):
    with pytest.raises(hfa.CatalogError, match=message):
        hfa.Runtime.parse({"id": "c", "git": "g", "wheels": {**WHEELS_OK, **change}})


def test_platform_tags_map_to_uv():
    assert hfa.uv_platform("manylinux_2_28_x86_64") == "x86_64-manylinux_2_28"
    assert hfa.uv_platform("manylinux_2_31_aarch64") == "aarch64-manylinux_2_31"
    assert hfa.uv_platform("manylinux2014_x86_64") == "x86_64-manylinux2014"
    assert hfa.uv_platform("win_amd64") == "x86_64-pc-windows-msvc"
    assert hfa.uv_platform("macosx_11_0_arm64") == "aarch64-apple-darwin"
    assert hfa.uv_platform("linux_x86_64") is None


def test_wheel_commands():
    w = hfa.Wheels.parse("c", {**WHEELS_OK, "index_url": "https://i/cu130",
                               "extra_index_urls": ["https://pypi.org/simple"]})
    resolve = hfa.build_resolve_command("uv", w, "win_amd64", [Path("r.txt"), Path("p.txt")],
                                        Path("out.txt"))
    assert resolve[:3] == ["uv", "pip", "compile"]
    joined = " ".join(resolve)
    for part in ("--python-version 3.12", "--python-platform x86_64-pc-windows-msvc",
                 "--only-binary :all:", "--index-url https://i/cu130",
                 "--extra-index-url https://pypi.org/simple", "-o out.txt r.txt p.txt"):
        assert part in joined
    pip = ["uv", "run", "--no-project", "--with", "pip", "python", "-m", "pip"]
    download = hfa.build_download_command(pip, w, "win_amd64", Path("lock"), Path("whl"))
    assert download[: len(pip) + 1] == [*pip, "download"]
    joined = " ".join(download)
    for part in ("--no-deps", "--only-binary=:all:", "--python-version 3.12",
                 "--platform win_amd64", "--implementation cp", "-d whl",
                 "--index-url https://i/cu130", "-r lock"):
        assert part in joined


def test_wheel_tools_fall_back_to_a_uv_pip(monkeypatch):
    monkeypatch.setattr(hfa.shutil, "which", lambda name: None)
    assert hfa.wheel_tools()[2] and "uv" in hfa.wheel_tools()[2]
    monkeypatch.setattr(hfa.shutil, "which", lambda name: "/bin/uv")
    monkeypatch.setattr(hfa.importlib.util, "find_spec", lambda name: None)
    assert hfa.wheel_tools() == (
        "/bin/uv", ["/bin/uv", "run", "--no-project", "--with", "pip", "python", "-m", "pip"],
        None)


def test_requirement_specs_follow_includes(tmp_path):
    (tmp_path / "req").mkdir()
    (tmp_path / "req" / "common.txt").write_text("numpy>=1.25  # inline\nclick==8.1.7\n")
    (tmp_path / "requirements.txt").write_text(
        "# a comment\n-r req/common.txt\n--extra-index-url https://x\n\n"
        "torch\npydantic~=2.0\nfoo @ https://x/foo.whl\nbar==1; sys_platform == 'win32'\n")
    specs = hfa.requirement_specs(tmp_path / "requirements.txt")
    assert specs == ["numpy>=1.25", "click==8.1.7", "torch", "pydantic~=2.0",
                     "foo @ https://x/foo.whl", "bar==1; sys_platform == 'win32'"]
    assert [s for s in specs if not hfa.is_pinned(s)] == ["numpy>=1.25", "torch",
                                                          "pydantic~=2.0"]
    assert hfa.wheel_name_version("Torch-2.14.1+cu130-cp312-cp312-win_amd64.whl") == (
        "torch", "2.14.1+cu130")


# A fake index: what the resolver and pip download can find.
AVAILABLE = {"click": "8.1.7", "colorama": "0.4.6", "numpy": "2.5.4"}


def fake_run_tool(calls, fail_download=(), abort_batch=False):
    """uv pip compile and pip download against AVAILABLE; Windows adds colorama,
    as click's `platform_system == "Windows"` marker does."""
    def run(cmd):
        calls.append(cmd)
        if cmd[1:3] == ["pip", "compile"]:
            out = Path(cmd[cmd.index("-o") + 1])
            specs = [s for f in cmd[cmd.index("-o") + 2:] for s in hfa.requirement_specs(Path(f))]
            names = [re.split(r"[=<>~ ;\[]", s)[0].lower() for s in specs]
            missing = [n for n in names if n not in AVAILABLE]
            if missing:
                return 1, f"  × No solution found\n  ╰─▶ Because {missing[0]} was not found\n"
            if "windows" in cmd[cmd.index("--python-platform") + 1]:
                names.append("colorama")
            out.write_text("".join(f"{n}=={AVAILABLE[n]}\n" for n in sorted(set(names))))
            return 0, ""
        assert cmd[:2] == ["pip", "download"], cmd
        d = Path(cmd[cmd.index("-d") + 1])
        pins = hfa.read_pins(Path(cmd[cmd.index("-r") + 1]))
        if abort_batch and fail_download & set(pins):  # real pip stops at the first failure
            return 1, "ERROR: No matching distribution found"
        for name, ver in pins.items():
            if name not in fail_download:
                (d / f"{name}-{ver}-py3-none-any.whl").write_bytes(f"{name} {ver}".encode())
        return (1, "ERROR: connection reset") if fail_download else (0, "")
    return run


def make_python_upstream(tmp_path: Path, requirements: str) -> Path:
    return make_upstream(tmp_path, {"requirements.txt": requirements})[0]


@pytest.fixture
def wheels_env(tmp_path, monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr(hfa, "wheel_tools", lambda: ("uv", ["pip"], None))
    monkeypatch.setattr(hfa, "run_tool", fake_run_tool(calls))
    dest = tmp_path / "drive"
    dest.mkdir()
    return dest, calls


def python_runtime(up: Path, **wheels) -> hfa.Runtime:
    return hfa.Runtime.parse({"id": "app", "git": str(up), "ref": "v1", "wheels": {
        **WHEELS_OK, "packages": ["numpy==2.5.4"], "start": "python main.py", **wheels}})


def test_wheels_layout_lock_install_md_and_check(tmp_path, wheels_env):
    dest, calls = wheels_env
    up = make_python_upstream(tmp_path, "click\n")
    rt = python_runtime(up)
    res = hfa.archive_runtime(hfa.probe_runtime(rt, dest, {}), dest)
    assert res.status == "downloaded", res.message
    root = dest / "_runtimes" / "app" / "wheels-v1"
    linux, win = root / "manylinux_2_28_x86_64", root / "win_amd64"
    assert sorted(p.name for p in linux.iterdir()) == [
        "click-8.1.7-py3-none-any.whl", "numpy-2.5.4-py3-none-any.whl", "requirements.lock.txt"]
    assert (win / "colorama-0.4.6-py3-none-any.whl").is_file()

    # the requirements were read from the archived source at the pinned commit
    compile_inputs = calls[0][calls[0].index("-o") + 2:]
    assert compile_inputs[0].endswith("app-v1/requirements.txt")

    lock = (linux / "requirements.lock.txt").read_text()
    sha = hashlib.sha256(b"click 8.1.7").hexdigest()
    assert f"click==8.1.7 --hash=sha256:{sha}" in lock
    assert "numpy==2.5.4 --hash=sha256:" in lock and "colorama" not in lock

    install = (root / "INSTALL.md").read_text()
    assert ("app-v1/.venv/bin/python -m pip install --no-index --find-links "
            "wheels-v1/manylinux_2_28_x86_64 -r wheels-v1/manylinux_2_28_x86_64/"
            "requirements.lock.txt") in install
    assert "python3.12 -m venv app-v1/.venv" in install
    assert "py -3.12 -m venv app-v1\\.venv" in install
    assert "app-v1\\.venv\\Scripts\\python -m pip install --no-index" in install
    assert "tar -xzf app-v1-src.tar.gz" in install
    assert "## Unpinned requirements" in install and "- `click`" in install
    assert install.rstrip().endswith("python main.py")

    w = res.wheels
    assert res.status == "downloaded" and not res.incomplete
    assert w["status"] == "complete" and w["python"] == "3.12" and w["unpinned"] == ["click"]
    assert w["install"]["path"] == "_runtimes/app/wheels-v1/INSTALL.md"
    assert hfa.wheel_stats(w["platforms"]["win_amd64"]) == (3, 36)
    wheel_arts = [a for a in hfa.wheel_artifacts(w) if a["kind"] == "wheel"]
    assert len(wheel_arts) == 5 and all(len(a["sha256"]) == 64 for a in wheel_arts)
    assert not any(a["kind"].startswith("wheel") for a in res.artifacts)

    hfa.manifest_write(dest, hfa.runtime_records([res], dest))
    assert check(dest) == 0
    flip_one_byte(linux / "click-8.1.7-py3-none-any.whl")
    assert check(dest) == 1

    index = hfa.index_markdown(hfa.manifest_read(dest))
    assert "[INSTALL.md](<_runtimes/app/wheels-v1/INSTALL.md>)" in index
    assert "manylinux_2_28_x86_64 complete" in index
    assert ".whl" not in index  # wheels are reached through INSTALL.md


def test_wheels_are_skipped_when_lock_and_wheels_match(tmp_path, wheels_env, monkeypatch):
    dest, calls = wheels_env
    rt = python_runtime(make_python_upstream(tmp_path, "click==8.1.7\n"))
    first = hfa.archive_runtime(hfa.probe_runtime(rt, dest, {}), dest)
    hfa.manifest_write(dest, hfa.runtime_records([first], dest))
    n = len(calls)

    again = hfa.probe_runtimes([rt], dest)[0]
    assert again.archived
    with monkeypatch.context() as m:
        m.setattr(hfa, "git", lambda *a: pytest.fail("must not run git"))
        second = hfa.archive_runtime(again, dest)
    assert len(calls) == n  # neither resolved nor downloaded again
    assert second.status == "up-to-date" and second.wheels["status"] == "complete"
    assert second.wheels["platforms"] == first.wheels["platforms"]

    # a wheel that went missing brings back that platform only
    (dest / "_runtimes/app/wheels-v1/win_amd64/colorama-0.4.6-py3-none-any.whl").unlink()
    hfa.manifest_write(dest, hfa.runtime_records([second], dest))
    third = hfa.archive_runtime(hfa.probe_runtimes([rt], dest)[0], dest)
    platforms = [c[c.index("--python-platform") + 1] for c in calls[n:] if "compile" in c]
    assert platforms == ["x86_64-pc-windows-msvc"]
    assert third.wheels["status"] == "complete"

    # a changed config is downloaded again
    changed = python_runtime(Path(rt.git), packages=["numpy==2.5.4", "colorama"])
    hfa.manifest_write(dest, hfa.runtime_records([third], dest))
    m = len(calls)
    hfa.archive_runtime(hfa.probe_runtimes([changed], dest)[0], dest)
    assert sum("compile" in c for c in calls[m:]) == 2


def test_unavailable_requirements_make_the_runtime_partial(tmp_path, wheels_env):
    dest, calls = wheels_env
    rt = python_runtime(make_python_upstream(tmp_path, "click\nnot-on-any-index==1.0\n"))
    res = hfa.archive_runtime(hfa.probe_runtime(rt, dest, {}), dest)
    assert res.status == "downloaded" and res.incomplete
    assert res.wheels["status"] == "partial"
    assert "not available as wheels: not-on-any-index==1.0" in res.message
    plat = res.wheels["platforms"]["manylinux_2_28_x86_64"]
    assert plat["status"] == "partial" and hfa.wheel_stats(plat)[0] == 2
    assert plat["unavailable"][0]["error"] == "No solution found | Because not-on-any-index was not found"
    lock = (dest / "_runtimes/app/wheels-v1/manylinux_2_28_x86_64/requirements.lock.txt").read_text()
    assert "# not available as a wheel: not-on-any-index==1.0" in lock
    assert "click==8.1.7" in lock
    install = (dest / res.wheels["install"]["path"]).read_text()
    assert "Not in this wheelhouse" in install and "`not-on-any-index==1.0`" in install

    # a partial runtime is recorded, checked and retried on the next run
    hfa.manifest_write(dest, hfa.runtime_records([res], dest))
    assert check(dest) == 0
    n = len(calls)
    again = hfa.archive_runtime(hfa.probe_runtimes([rt], dest)[0], dest)
    assert again.status == "up-to-date" and again.incomplete and len(calls) > n


def test_failed_downloads_and_missing_requirement_files(tmp_path, wheels_env, monkeypatch):
    dest, calls = wheels_env
    up = make_python_upstream(tmp_path, "click\n")
    monkeypatch.setattr(hfa, "run_tool", fake_run_tool(calls, fail_download={"numpy"}))
    res = hfa.archive_runtime(hfa.probe_runtime(python_runtime(up), dest, {}), dest)
    plat = res.wheels["platforms"]["manylinux_2_28_x86_64"]
    assert res.incomplete and plat["missing"] == ["numpy"]
    assert "connection reset" in plat["message"]

    gone = python_runtime(up, requirements=["nope/requirements.txt"])
    res = hfa.archive_runtime(hfa.probe_runtime(gone, tmp_path / "other", {}), tmp_path / "other")
    assert res.incomplete and res.wheels["status"] == "failed"
    assert "nope/requirements.txt is not in the source at v1" in res.message
    assert "Not archived:" in (tmp_path / "other" / res.wheels["install"]["path"]).read_text()


def test_no_wheels_keeps_what_is_archived(tmp_path, wheels_env):
    dest, calls = wheels_env
    rt = python_runtime(make_python_upstream(tmp_path, "click\n"))
    skipped = hfa.archive_runtime(hfa.probe_runtime(rt, dest, {}), dest, wheels=False)
    assert skipped.wheels is None and not calls
    assert not (dest / "_runtimes/app/wheels-v1").exists()

    first = hfa.archive_runtime(hfa.probe_runtime(rt, dest, {}), dest)
    hfa.manifest_write(dest, hfa.runtime_records([first], dest))
    n = len(calls)
    kept = hfa.archive_runtime(hfa.probe_runtimes([rt], dest)[0], dest, wheels=False)
    assert len(calls) == n and kept.wheels == first.wheels


WHEELS_CONFIG = """\
runtimes:
  - id: comfyui
    git: https://github.com/Comfy-Org/ComfyUI
    ref: v0.45.0
    wheels:
      python: "3.12"
      platforms: [manylinux_2_28_x86_64, win_amd64]
      requirements: [requirements.txt]
      packages: ["torch==2.14.1+cu130"]
models:
  - id: flux
    hf_repo: org/flux
    category: image
    usage: {runtime: comfyui}
"""


def test_cli_list_shows_wheels(tmp_path, capsys):
    cfg = str(write_config(tmp_path / "w.yaml", WHEELS_CONFIG))
    assert hfa.main(["--config", cfg, "--list"]) == 0
    out = capsys.readouterr().out
    assert "WHEELS" in out
    assert "py3.12: manylinux_2_28_x86_64, win_amd64; 1 requirement file; 1 package" in out
    assert hfa.main(["--config", cfg, "--list", "--no-wheels"]) == 0
    assert "skipped (--no-wheels)" in capsys.readouterr().out
    assert hfa.main(["--config", cfg, "--list", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["runtimes"][0]["wheels"]["packages"] == ["torch==2.14.1+cu130"]


def test_curated_python_runtimes_pin_wheels():
    cat = hfa.load_catalog(CURATED)
    with_wheels = {rt.id: rt.wheels for rt in cat.runtimes if rt.wheels}
    assert set(with_wheels) == {"ComfyUI", "ComfyUI-GGUF", "ComfyUI-GGUF-molbal", "vllm",
                                "transformers"}
    assert {w.python for w in with_wheels.values()} == {"3.12"}
    comfy = with_wheels["ComfyUI"]
    assert comfy.platforms == ["manylinux_2_28_x86_64", "win_amd64"]
    assert comfy.requirements == ["requirements.txt"]
    assert any(p.startswith("torch==") and "+cu" in p for p in comfy.packages)
    assert comfy.index_url.startswith("https://download.pytorch.org/whl/cu")
    for rid in ("ComfyUI-GGUF", "ComfyUI-GGUF-molbal"):
        assert with_wheels[rid].requirements == ["requirements.txt"]
    assert all(w.start for w in with_wheels.values())


def test_full_run_with_partial_wheels_exits_1(tmp_path, wheels_env, monkeypatch):
    dest, calls = wheels_env
    up = make_python_upstream(tmp_path, "click\nnot-on-any-index==1.0\n")
    body = (WHEELS_CONFIG.replace("https://github.com/Comfy-Org/ComfyUI", str(up))
            .replace("ref: v0.45.0", "ref: v1")
            .replace('packages: ["torch==2.14.1+cu130"]', 'packages: ["numpy==2.5.4"]'))
    cfg = write_config(tmp_path / "w.yaml", body)
    flux = hfa.load_catalog(cfg).by_id("flux")
    readme = b"# flux\n"
    st = hfa.RepoStatus(model=flux, exists=True, commit="f" * 40,
                        files=[("README.md", len(readme))],
                        hashes={"README.md": {"git_sha1": git_blob_sha1(readme)}})
    monkeypatch.setattr(hfa, "probe_all", lambda models: [st])
    monkeypatch.setattr(hfa, "resolve_hf_bin", lambda: "hf")

    def fake_hf(cmd):
        (Path(cmd[cmd.index("--local-dir") + 1]) / "README.md").write_bytes(readme)
        return 0, ""

    monkeypatch.setattr(hfa, "_run_hf", fake_hf)
    args = ["--config", str(cfg), "--dest", str(dest), "--allow-home-dest", "--min-free-gb", "0"]
    assert hfa.main(args) == 1
    rec = hfa.latest_records(hfa.manifest_read(dest), "runtime")["comfyui"]
    assert rec["status"] == "downloaded" and rec["wheels"]["status"] == "partial"
    assert "INSTALL.md" in (dest / "INDEX.md").read_text()
    assert check(dest) == 0

    n = len(calls)
    assert hfa.main(args + ["--no-wheels"]) == 0
    assert len(calls) == n
    rec = hfa.latest_records(hfa.manifest_read(dest), "runtime")["comfyui"]
    assert rec["wheels"]["status"] == "partial"  # still recorded, still checked
    assert check(dest) == 0


def test_selection_flags_accept_repeats_and_commas():
    assert hfa.csv_list(["a,b", "c", " d , "]) == ["a", "b", "c", "d"]
    assert hfa.csv_list("a,b") == ["a", "b"]
    assert hfa.csv_list([]) == []
    args = hfa.build_parser().parse_args(["--only", "a,b", "--only", "c"])
    assert hfa.csv_list(args.only) == ["a", "b", "c"]


def test_compatible_platforms_cover_older_manylinux():
    tags = hfa.compatible_platforms("manylinux_2_28_x86_64")
    assert tags[0] == "manylinux_2_28_x86_64" and "manylinux_2_17_x86_64" in tags
    assert "manylinux2014_x86_64" in tags and "manylinux_2_29_x86_64" not in tags
    assert hfa.compatible_platforms("win_amd64") == ["win_amd64"]
    cmd = hfa.build_download_command(["pip"], hfa.Wheels.parse("app", WHEELS_OK), "manylinux_2_28_x86_64",
                                     Path("lock"), Path("whl"))
    assert cmd.count("--platform") == len(tags)


def test_one_unavailable_pin_does_not_empty_the_wheelhouse(tmp_path, wheels_env, monkeypatch):
    dest, calls = wheels_env
    up = make_python_upstream(tmp_path, "click\n")
    monkeypatch.setattr(hfa, "run_tool",
                        fake_run_tool(calls, fail_download={"numpy"}, abort_batch=True))
    res = hfa.archive_runtime(hfa.probe_runtime(python_runtime(up), dest, {}), dest)
    plat = res.wheels["platforms"]["manylinux_2_28_x86_64"]
    assert plat["status"] == "partial" and plat["missing"] == ["numpy"]
    names = {a["path"].rsplit("/", 1)[-1] for a in plat["artifacts"]}
    assert "click-8.1.7-py3-none-any.whl" in names
