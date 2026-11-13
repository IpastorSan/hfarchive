# hfarchive

[![tests](https://github.com/IpastorSan/hfarchive/actions/workflows/test.yml/badge.svg)](https://github.com/IpastorSan/hfarchive/actions/workflows/test.yml)

`hfarchive` copies open-weight models from Hugging Face onto a drive you own,
together with what you need to run them later, and checks that copy for
damage.

## Why

Open-weight models can disappear from Hugging Face. Repos get deleted, gated,
relicensed or renamed, and the Hub itself can be unreachable. A copy of the
weights alone is not enough to run a model years later. You also need a
version of the runtime that supports it, the workflows that wire it up, and
the settings it was meant to run with.

## What it does

From one YAML file, `hfarchive`:

- downloads the weights, only the files you choose (one quant out of the
  dozens a GGUF repo ships), with the official `hf download` CLI;
- always keeps the model card (`README.md`) and `LICENSE` files;
- writes your usage notes for each model to a `USAGE.md` next to the weights,
  and downloads the workflow files you list, such as ComfyUI workflow JSONs;
- archives pinned runtimes (llama.cpp, ComfyUI, custom nodes) as full git
  history bundles, source tarballs and GitHub release builds, and the Python
  dependencies of Python runtimes (PyTorch with CUDA included) as wheels, so
  they install with no network;
- writes an `INDEX.md` at the drive root that you can read without the tool;
- puts each model's benchmarks, the selection's models guide and the config
  used on the drive, so the archive explains what it holds and why;
- records every file's size and hash in a manifest, so `--check` can show
  years later that nothing has rotted.

Before it downloads anything it checks the repos against the Hub, tells you
how much space the selection needs, and refuses runs that would not fit or
that would land on your system disk. Runs resume where they stopped.

It is not a model runner. It does not sync or mirror whole repos. It does not
replace the Hugging Face cache: it writes plain folders you point a runtime
at.

## Hardware tiers

The default selection tags each model with the tiers it runs on in practice:

| Tag | GPU | System RAM |
|---|---|---|
| `tier-8gb` | 8 GB VRAM, possibly older than Ampere | about 128 GB; large MoE models keep their experts in RAM |
| `tier-16gb` | 16 GB VRAM | 64 to 128 GB |
| `tier-24gb` | 24 GB VRAM (a few entries need 32 GB) | 128 GB |

[`docs/models-2026-11.md`](docs/models-2026-11.md) covers each model in the
selection, with benchmarks and a comparison with current closed frontier
models.

## Install

```bash
curl -LO https://raw.githubusercontent.com/IpastorSan/hfarchive/main/hfarchive
chmod +x hfarchive
./hfarchive --help
```

On Windows, in PowerShell:

```powershell
Invoke-WebRequest https://raw.githubusercontent.com/IpastorSan/hfarchive/main/hfarchive -OutFile hfarchive
uv run --script hfarchive --help
```

It is a single Python file. [uv](https://docs.astral.sh/uv/) installs its two
dependencies (`huggingface_hub`, `PyYAML`) on first run. It runs on Linux,
macOS and Windows (see [Platform support](#platform-support)), and needs `git`
if you archive runtimes and `uv` if a runtime has `wheels`. Put the file
anywhere on your `PATH` to call it as `hfarchive`.

Without uv, install the dependencies yourself and run it with Python 3.11+:
`pip install huggingface_hub pyyaml && python3 hfarchive --help`.

## Quickstart

```bash
./hfarchive --init                          # writes ./models.yaml
$EDITOR models.yaml                         # trim it to what you want
./hfarchive --dry-run --dest /mnt/backup    # sizes, target paths; writes nothing
./hfarchive --dest /mnt/backup              # download
./hfarchive --check --dest /mnt/backup      # later: is everything still intact?
```

`--init` writes the default selection: a curated set of open models (LLMs,
image and video models, security, uncensored and decision models) chosen for
the 8, 16 and 24 GB VRAM tiers above, dated November 2026. Its `core` profile
is about 880 GB; everything is about 2.2 TB. Trim it, select a smaller
profile, or start from a small commented file with `--init --minimal`.

Set the destination once instead of passing `--dest` every time:

```bash
export HFARCHIVE_DEST=/mnt/backup
```

`Ctrl-C` is safe. Re-running skips files that are already complete and resumes
partial ones, so a long archive can run over several sessions.

## Platform support

The script runs on Linux, macOS and Windows; CI runs the tests on all three.

On Windows:

- Run it as `uv run --script hfarchive ...`; Windows does not read the
  script's first line.
- Install [Git for Windows](https://git-scm.com/download/win) to archive
  runtimes.
- Windows refuses paths of 260 characters or more unless long paths are
  enabled, and some model folders have long file names. `hfarchive` warns
  before a run when a path is that long. To enable long paths, run this in an
  administrator PowerShell, then open a new terminal:

  ```powershell
  New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force
  ```

A drive written on one OS reads on the others: the manifest stores paths with
`/`, and the files `hfarchive` writes use LF line endings.

What the curated selection archives for each runtime, per OS. "Wheels" means
an offline Python install; "source" means the git bundle and source tarball,
which every runtime gets.

| Runtime | Linux x86_64 | Windows x64 | macOS arm64 (Apple Silicon) |
|---|---|---|---|
| llama.cpp | CUDA build | CUDA build | Metal build |
| ComfyUI + GGUF nodes | wheels (CUDA PyTorch) | portable 7z and wheels (CUDA PyTorch) | wheels (PyPI PyTorch, MPS); fp8 weights load at 16 bits |
| stable-diffusion.cpp | CPU and Vulkan builds | CUDA build | build without Metal; rebuild from source with `-DSD_METAL=ON` |
| Edge0 | source only; build it | Edge0 Chat app | source only; builds for Metal, runs the 35B model only |
| vLLM | wheels (CUDA) | not supported by vLLM (use WSL) | source only (vLLM builds for CPU there) |
| transformers | wheels (CUDA PyTorch) | source only | source only |

The hardware tiers assume an NVIDIA GPU. On a Mac, the GPU and system RAM are
one pool, so a model fits when its files and context fit in unified memory.

## Commands

| Flag | Effect |
|---|---|
| `--init` | Write the curated selection to `--config` (default `./models.yaml`). Never overwrites. |
| `--init --minimal` | Write a small commented starter instead. |
| `--list` | Print the selected models, runtimes and workflows. No network. |
| `--verify-only` | Check every selected repo on the Hub. Downloads nothing. |
| `--dry-run` | Resolve live sizes, matched files, runtime commits, release assets and workflow sizes. Writes nothing. |
| *(none of the above)* | Download. |
| `--refresh-sizes` | Measure each selected repo and write `approx_bytes` back into the config, keeping comments. |
| `--check` | Recompute every archived file's hash and compare it with the manifest. Needs `--dest`, no config, no network. |
| `--check --fast` | Compare sizes only. Takes seconds. |

Add `--json` to `--list`, `--verify-only`, `--dry-run` and `--check` for
machine-readable output.

Without `--config`, `hfarchive` reads `./models.yaml`. If that file is
missing and the script runs from a clone of this repo, it uses
`examples/archive-2026-11.yaml` and says so.

### Selection

```bash
--config path/to/models.yaml       # default: ./models.yaml
--profile coding                   # a profile from the config
--only whisper,qwen3-embedding     # exactly these ids, ignoring the profile
--exclude kimi-k3                  # drop these ids
--enable qwen2.5-coder-7b-gguf     # force in an entry with enabled: false
--category ocr,embeddings          # keep only these categories
--no-runtimes                      # skip runtimes
--no-wheels                        # archive runtimes without their Python wheels
```

Without `--profile`, the config's `default_profile` is used. If it has none,
every entry with `enabled: true` is selected.

## Config reference

```yaml
defaults:
  min_free_gb: 50          # free space that must remain on the destination
  large_repo_gb: 100       # entries this big or bigger need --allow-large
  jobs: 2                  # repos downloaded in parallel
  max_workers_per_repo: 8  # files in parallel within one repo

default_profile: small     # optional
guide: docs/models-2026-11.md  # optional: a Markdown guide to this selection

profiles:
  small:
    description: Things that run on a laptop.
    any_tags: [small]      # entries with at least one of these tags
  gguf-text:
    all_tags: [gguf, text] # entries with all of these tags
  picks:
    ids: [whisper, flux]   # entries by id
  everything:
    all: true              # every entry
    include_disabled: true # also entries with enabled: false

models:
  - id: qwen2.5-coder-7b-gguf
    hf_repo: Qwen/Qwen2.5-Coder-7B-Instruct-GGUF
    revision: main
    include: ["*-q4_k_m.gguf"]
    exclude: []
    category: coding
    tags: [coding, gguf, small]
    license: apache-2.0
    gated: false
    approx_bytes: 4683079206
    enabled: true
    notes: One quant out of the dozen the repo ships.
    benchmarks:
      - name: HumanEval
        score: 88.4
        source: https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct
        date: 2026-11-13       # when it was read
        note: self-reported
    usage:
      runtime: llama.cpp
      notes: |
        llama-cli -m qwen2.5-coder-7b-instruct-q4_k_m.gguf -c 8192 --temp 0.7
      workflows:
        - name: t2i-basic.json
          url: https://raw.githubusercontent.com/<org>/<repo>/<commit>/t2i-basic.json

runtimes:
  - id: llama.cpp
    git: https://github.com/ggml-org/llama.cpp
    ref: b6100
    release_assets: ["*ubuntu-x64*"]
    tags: [llm]
    enabled: true
    notes: CPU build; build from source for CUDA.
  - id: ComfyUI
    git: https://github.com/Comfy-Org/ComfyUI
    ref: v0.45.0
    wheels:
      python: "3.12"
      platforms: [manylinux_2_28_x86_64, win_amd64]
      requirements: [requirements.txt]
      packages: ["torch==2.14.1+cu130", "torchvision==0.29.1+cu130"]
      index_url: https://download.pytorch.org/whl/cu130
      extra_index_urls: [https://pypi.org/simple]
      start: Run `.venv/bin/python main.py` in the unpacked source.
```

Every top-level section is optional except `models`.

| Entry field | Required | Meaning |
|---|:-:|---|
| `id` | yes | Local name, used by `--only`, `--exclude`, `--enable` and `ids:`. Must be unique. |
| `hf_repo` | yes | Hub repo id, `org/name`, exact casing. |
| `category` | yes | Top-level folder on the destination. |
| `revision` | | Branch, tag or commit. Default: `main`. |
| `include` | | Glob list. If set, only matching files are downloaded. |
| `exclude` | | Glob list, applied after `include`. |
| `tags` | | Labels that profiles select on. |
| `license` | | Informational. |
| `gated` | | `true` if the repo needs accepted terms and a token. Informational; the Hub's answer decides. |
| `approx_bytes` | | Size of the matched files. Written by `--refresh-sizes`, shown by `--list`. |
| `enabled` | | `false` keeps the entry out unless a profile has `include_disabled: true` or you pass `--enable`. Default `true`. |
| `notes` | | Free text. |
| `usage.runtime` | | What runs the model. Free text; when it matches a `runtimes[].id`, that runtime is archived with the model. |
| `usage.notes` | | Recommended settings: temperature, sampler and steps, CLI flags. Written to `USAGE.md`. |
| `usage.workflows` | | List of `{name, url}`. Each URL is downloaded to `_workflows/<name>` in the model's folder. Pin URLs to a commit. |
| `benchmarks` | | List of published scores, written to a table in `USAGE.md`. The first three appear in `INDEX.md`, so list the headline ones first. |
| `benchmarks[].name` | yes | Benchmark name, e.g. `SWE-bench Verified`. |
| `benchmarks[].score` | yes | A number, or a short string such as `1156 Elo` (up to 40 characters). |
| `benchmarks[].source` | yes | `http(s)` URL where the number was published. |
| `benchmarks[].date` | | When the number was published or read, e.g. `2026-11-13`. |
| `benchmarks[].note` | | Free text, e.g. `self-reported, OpenHands harness`. |

Published scores are for the full-precision model; the archived quants score
lower, and `USAGE.md` says so.

`guide` (top level) names a Markdown guide to the selection: a path or an
`http(s)` URL. Each download run copies it to the drive root under its own
file name. A path is looked up against the config file's directory, then
against the directory of the `hfarchive` script (a clone of this repo), then
in this repo on GitHub at the running version's tag and then at `main`. A
config written by `--init` therefore finds its guide without edits. If the
guide cannot be found, the run reports it and exits 1, and everything else
still lands.

`README.md` and `LICENSE*` at the repo root are always downloaded, even when
`include` or `exclude` would leave them out.

| Runtime field | Required | Meaning |
|---|:-:|---|
| `id` | yes | Folder name under `_runtimes/`. Must be unique. |
| `git` | yes | Git URL. |
| `ref` | | Tag, branch or commit to archive. Default: the remote's `HEAD`. The resolved commit is recorded. |
| `release_assets` | | Globs over the asset names of the GitHub release tagged `ref`. Needs a `github.com` URL and a tag. |
| `tags` | | Labels that profiles select on. |
| `enabled` | | `false` keeps the runtime out unless the profile has `include_disabled: true`. |
| `notes` | | Free text. |
| `wheels` | | Python dependencies to save as wheels; see below. |
| `wheels.python` | yes | Target Python version, quoted: `"3.12"`. The wheels only install on that version. |
| `wheels.platforms` | yes | pip platform tags, one wheelhouse each: `manylinux_2_28_x86_64` (or another glibc version, `manylinux2014_x86_64`, `*_aarch64`), `win_amd64`, `win_arm64`, `macosx_11_0_arm64` style. |
| `wheels.requirements` | | Requirements files, as paths inside the runtime's source. They are read from the archived source at `ref`, so they match the pinned commit. `-r` includes are followed. |
| `wheels.packages` | | Extra requirement lines, such as a CUDA build of PyTorch. `requirements` or `packages` is needed. Environment markers pick lines per platform, e.g. `"torch==2.14.1+cu130; sys_platform != 'darwin'"` and `"torch==2.14.1; sys_platform == 'darwin'"`. |
| `wheels.index_url` | | Primary package index, e.g. `https://download.pytorch.org/whl/cu130`. Default: PyPI. |
| `wheels.extra_index_urls` | | More indexes. Every index is searched for the best match. |
| `wheels.start` | | How to start the runtime once installed. Written to `INSTALL.md`. |

Unknown keys are rejected, so a typo fails loudly instead of being ignored.

A profile matches an entry if any of its rules match: `all`, `ids`, `all_tags`
or `any_tags`. A profile includes the runtimes its selected models name in
`usage.runtime`, plus the runtimes its rules match. With `--only` or
`--category`, only the runtimes the chosen models name are included.

**Globs** follow the Hub's `fnmatch` rules, where `*` also crosses `/`:
`*.safetensors` matches `transformer/x.safetensors` too. To exclude one file at
the repo root, name it exactly.

**GGUF repos** usually ship every quantization, often hundreds of GB in total.
Pin the one you want with `include`. Unsloth repos also keep stub copies of
the first shard under `Shard_Rewrite/`, so pin sharded quants by folder
(`UD-IQ1_S/*`), not with `*UD-IQ1_S*`.

## Runtimes

Each runtime is archived in `$DEST/_runtimes/<id>/`:

| File | What it is |
|---|---|
| `mirror.git/` | A bare mirror of the upstream branches and tags, updated on each run. |
| `<id>-<sha>.bundle` | A `git bundle` of the whole history, one file. `git clone <id>-<sha>.bundle src && git -C src checkout <sha>` restores it anywhere. |
| `<id>-<ref>-src.tar.gz` | A plain source tree of `ref`, for when git is not at hand. Submodules are not included. |
| `release-<ref>/` | The release assets that match `release_assets`, checked against the size GitHub publishes and, when it publishes one, the sha256. |

A runtime whose `ref` still resolves to the archived commit is skipped. The
GitHub API is used for release assets. `hfarchive` uses `GITHUB_TOKEN` or the
token from `gh auth token` if there is one. Without a token, GitHub allows 60
requests per hour, and a run that hits the limit says so.

### Python wheels

A runtime with `wheels` also gets its Python dependencies, so it installs on
a machine with no network:

| File | What it is |
|---|---|
| `wheels-<ref>/<platform>/*.whl` | Every wheel the runtime needs on that platform, dependencies included. |
| `wheels-<ref>/<platform>/requirements.lock.txt` | The exact set, `name==version --hash=sha256:…` per wheel; pip checks the hashes on install. |
| `wheels-<ref>/INSTALL.md` | The offline install commands for each platform and how to start the runtime. |

For each platform, `uv pip compile` resolves the requirements files and
`packages` for the target Python and platform, then `pip download --no-deps
--only-binary=:all:` fetches one wheel per pin. uv does the resolving because
`pip download --platform` evaluates environment markers for the machine it
runs on: a Linux run would leave out the Windows-only dependencies of a
Windows set. For a macOS tag such as `macosx_14_0_arm64`, the resolve targets
that macOS release, so wheels built for macOS 14 (PyTorch's) qualify. pip
comes from the script's environment if it has one, else from `uv run --with
pip`. Installing needs only Python and pip:

```bash
cd _runtimes/ComfyUI
tar -xzf ComfyUI-v0.45.0-src.tar.gz
python3.12 -m venv ComfyUI-v0.45.0/.venv
ComfyUI-v0.45.0/.venv/bin/python -m pip install --no-index \
  --find-links wheels-v0.45.0/manylinux_2_28_x86_64 \
  -r wheels-v0.45.0/manylinux_2_28_x86_64/requirements.lock.txt
```

Requirement lines that name no exact version are listed in `INSTALL.md`; the
lock pins what they resolved to. A line with no wheel for the platform (only
a source distribution, or not on any index) is reported by name. The rest is
still resolved and downloaded, and the runtime is recorded as `partial` and
the run exits 1. A platform whose lock and wheels are on the drive and match
the manifest is skipped; one that changed config or commit is resolved again,
and wheels no longer in its set are removed. Sizes are known only after the
download: a CUDA PyTorch set is 3–4 GB per platform. `--no-wheels` skips the
wheels and keeps any already archived.

## Checking an archive

Each successful download records, in the manifest, every file's path and size
and the hash the Hub publishes for it: sha256 for LFS files (the weights), the
git blob sha1 for the rest. Runtime files, workflows, the guide and the
config copies get a sha256.

```bash
hfarchive --check --dest /mnt/backup          # rehash everything
hfarchive --check --dest /mnt/backup --fast   # sizes only
```

`--check` reads only the drive. It reports each file that is missing, has the
wrong size or has the wrong hash, and exits 1 if it found any. Large files
show progress while they are hashed. Without `hfarchive`, `sha256sum <file>`
and `git hash-object <file>` compute the same values the manifest holds.

## Guards

Before downloading anything, `hfarchive` refuses to start if:

- the destination is not a directory or is not writable;
- the destination is inside `$HOME` or on the same filesystem as `$HOME`
  (override with `--allow-home-dest`), so a mistyped mount point cannot fill
  your system disk;
- less than `min_free_gb` would remain free (`--min-free-gb`);
- the selection does not fit in the free space minus that headroom
  (`--ignore-space` overrides);
- any single entry is `large_repo_gb` or bigger (`--allow-large`, alias
  `--i-know`; change the threshold with `--large-repo-gb`).

A refused run writes nothing to the destination.

A destination that does not exist yet, such as a drive that is not plugged in,
is not an error: the run becomes a dry run so you can plan ahead.

## Gated models

Some repos require accepting a licence on the model page. Log in once and
accept the terms on each page:

```bash
hf auth login
```

Without a token, gated entries are skipped with a message and the rest of the
run continues. Tokens are never printed or written to logs or the manifest.

## Layout on the drive

```
/mnt/backup/
├── INDEX.md                              what is here, readable without hfarchive
├── models-2026-11.md                     the selection's guide (`guide:`), if set
├── llm/unsloth/Qwen3.8-Flash-Next-GGUF/
│   ├── UD-Q3_K_XL/…                       the weights, as in the repo
│   ├── README.md                          the model card
│   └── USAGE.md                           runtime, settings, workflows, benchmarks
├── image/black-forest-labs/FLUX.2-dev/
│   ├── …
│   ├── USAGE.md
│   └── _workflows/t2i-basic.json
├── _runtimes/
│   ├── llama.cpp/
│   │   ├── mirror.git/
│   │   ├── llama.cpp-<sha>.bundle
│   │   ├── llama.cpp-b6100-src.tar.gz
│   │   └── release-b6100/…
│   └── ComfyUI/
│       ├── …
│       └── wheels-v0.45.0/
│           ├── INSTALL.md                 offline install commands per platform
│           ├── manylinux_2_28_x86_64/     *.whl + requirements.lock.txt
│           └── win_amd64/
└── _hfarchive/
    ├── manifest.jsonl
    ├── config-latest.yaml                 the config of the latest run
    ├── configs/<name>-<YYYYMMDD-HHMMSS>.yaml   the config of every run
    └── logs/
```

Each model lives at `<dest>/<category>/<org>/<repo>/` as a plain snapshot of
the repo, so llama.cpp, ComfyUI or `from_pretrained()` can load it directly.
When several entries pin different files of one repo, they share its folder
and one `USAGE.md` with a section per entry.

`INDEX.md` is rewritten after every download run, even when everything was
already current. It lists every
archived model by category with its repo, revision and commit, size, runtime,
headline benchmarks, `USAGE.md` and workflows, and every runtime with its ref,
commit, files and, for runtimes with wheels, a link to `INSTALL.md`. At the top it links the guide and the config of the latest run.
It covers everything in the manifest, not only the current config.

Every run that downloads copies the config it used into `_hfarchive/`, with
a timestamp, so you can tell later which selection produced the drive.

`manifest.jsonl` is append-only. It gets one line per model and per runtime
per run: id, repo, commit, status, sizes, licence, timestamp, and the file
hashes `--check` uses.

Run logs are written to `${XDG_STATE_HOME:-~/.local/state}/hfarchive/logs/`
and copied to `_hfarchive/logs/` after any run that downloaded something.

## Keeping an archive healthy

- Run `hfarchive --verify-only` on a schedule, for example a weekly cron job.
  It exits 1 when a repo has been removed or renamed upstream, which is when
  your copy becomes the only one.
- Run `hfarchive --check` every few months, and `--check --fast` whenever you
  plug the drive in. Disks lose data silently.
- Keep a second copy on another drive, ideally in another place. `--check`
  tells you a file is damaged; only a second copy gets it back.
- Prefer a filesystem that checksums data, such as btrfs or ZFS. Run
  `btrfs scrub start /mnt/backup` (or `zpool scrub`) periodically; with two
  copies in a btrfs RAID1 it also repairs what it finds.

## Exit codes

| Code | Meaning |
|---:|---|
| 0 | Success, a clean dry run, or a clean `--check` |
| 1 | A model, runtime, workflow or guide download failed, or a runtime's wheels are partial (the rest of the run completes); `--verify-only` found a missing repo; `--refresh-sizes` could not measure a repo; `--check` found a missing or damaged file |
| 2 | Config error, empty selection, bad destination, not enough space, or `--check` without a destination or manifest |
| 3 | Refused: an entry is at or above the large-repo threshold |

## Environment

| Variable | Use |
|---|---|
| `HFARCHIVE_DEST` | Default for `--dest`. |
| `HFARCHIVE_HF_BIN` | Path to an `hf` binary, used when the script's own environment has none. |
| `HF_TOKEN` | Hub token, if you prefer it to `hf auth login`. |
| `GITHUB_TOKEN` | GitHub token for release assets, if you prefer it to `gh auth login`. |

`hfarchive` uses the `hf` CLI from its own environment, then
`$HFARCHIVE_HF_BIN`, then the first `hf` on your `PATH`.

## Examples

[`examples/`](examples/) holds complete configs you can run or copy from:

- [`archive-2026-11.yaml`](examples/archive-2026-11.yaml): the default
  selection `--init` writes, described above.
- [`tweet-2026-09.yaml`](examples/tweet-2026-09.yaml): 52 open-weights models
  from a list shared in September 2026 (text, coding, image, video, speech,
  music, OCR, embeddings), about 5.9 TB, with smaller profiles for GGUF-only,
  coding and 8 GB-VRAM machines.

```bash
hfarchive --config examples/tweet-2026-09.yaml --list --profile tweet-gguf
```

To split an archive across two drives, run twice with different `--category`
lists. Each drive keeps its own manifest and index.

## Troubleshooting

**"could not find the `hf` CLI"**: run the script through uv, or install
`huggingface_hub` and make `hf` available on `PATH` or via `HFARCHIVE_HF_BIN`.

**A repo is missing**: it was renamed or removed upstream. `--verify-only` shows
which; fix `hf_repo` in the config.

**The drive unmounted mid-run**: remount and run again. Complete files are
skipped, partial ones resume.

**Slow transfers**: try `--jobs 1 --max-workers 4`. USB drives are usually the
bottleneck, not the network.

**"GitHub API rate limit reached"**: set `GITHUB_TOKEN` or run `gh auth login`,
or wait an hour.

**`--check` reports a damaged file**: delete it and run the download again
(`--only <id>` for a model). Missing model files, runtime files and workflows
are all fetched again.

## Development

```bash
uv run --no-project --with pytest --with huggingface_hub --with pyyaml pytest -q
```

The tests download one ~2 KB `config.json` and a model card to exercise
`hf download`; they never download model weights. Network tests are skipped
when offline. Runtimes are tested against a local git repository, with GitHub
mocked; wheels with uv and pip mocked.

## License

MIT. See [LICENSE](LICENSE).
