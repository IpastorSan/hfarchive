"""Tests for the hfarchive script.

Deliberately cheap: the only network transfer is a ~2 KB config.json, so this
never pulls model weights. Run with:

    uv run --with pytest --with huggingface_hub --with pyyaml pytest -q
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import os
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
    assert hfa.main(["--init", "--config", str(target)]) == 0
    cat = hfa.load_catalog(target)
    assert 2 <= len(cat.models) <= 3
    assert all(m.approx_bytes < 10e9 for m in cat.models), "starter entries must be small"
    assert cat.select()

    target.write_text("mine")
    assert hfa.main(["--init", "--config", str(target)]) == 2
    assert target.read_text() == "mine"


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
    assert st.matched_files == 1, f"expected just {TINY_FILE}, got {st.files}"
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
