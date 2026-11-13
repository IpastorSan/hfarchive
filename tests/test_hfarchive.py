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
    assert [rt.id for rt in cat.runtimes] == ["llama.cpp"]


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
def make_upstream(tmp_path: Path):
    """A local repo with an annotated tag v1 and one commit after it: (path, git)."""
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
