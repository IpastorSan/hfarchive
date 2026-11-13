# hfarchive

Archive a list of Hugging Face models onto a drive and keep it updated.

You describe the models in a YAML file: which repos, which files, which folder
they go in. `hfarchive` checks them against the Hub, tells you how much space
the selection needs, refuses runs that would not fit or that would land on
your system disk, and then downloads with the official `hf download` CLI. Runs
resume where they stopped, and every run is recorded in a manifest on the
drive.

It is a single Python file. [uv](https://docs.astral.sh/uv/) installs its two
dependencies (`huggingface_hub`, `PyYAML`) on first run.

## Install

```bash
curl -LO https://raw.githubusercontent.com/IpastorSan/hfarchive/main/hfarchive
chmod +x hfarchive
./hfarchive --help
```

Requires `uv` and Linux or macOS. Put the file anywhere on your `PATH` if you
want to call it as `hfarchive`.

Without uv, install the dependencies yourself and run it with Python 3.11+:
`pip install huggingface_hub pyyaml && python3 hfarchive --help`.

## Quickstart

```bash
hfarchive --init                          # writes a commented ./models.yaml
hfarchive --list                          # what the config selects (offline)
hfarchive --verify-only                   # do the repos exist on the Hub?
hfarchive --dry-run --dest /mnt/backup    # sizes, target paths; writes nothing
hfarchive --dest /mnt/backup              # download
```

Set the destination once instead of passing `--dest` every time:

```bash
export HFARCHIVE_DEST=/mnt/backup
```

`Ctrl-C` is safe. Re-running skips files that are already complete and resumes
partial ones, so a long archive can run over several sessions.

## Commands

| Flag | Effect |
|---|---|
| `--init` | Write a starter config at `--config` (default `./models.yaml`). Never overwrites. |
| `--list` | Print the selected entries and their recorded sizes. No network. |
| `--verify-only` | Check every selected repo on the Hub. Downloads nothing. |
| `--dry-run` | Resolve live sizes, matched files and target paths. Writes nothing. |
| *(none of the above)* | Download. |
| `--refresh-sizes` | Measure each selected repo and write `approx_bytes` back into the config, keeping comments. |

Add `--json` to `--list`, `--verify-only` and `--dry-run` for machine-readable
output.

### Selection

```bash
--config path/to/models.yaml       # default: ./models.yaml
--profile coding                   # a profile from the config
--only whisper,qwen3-embedding     # exactly these ids, ignoring the profile
--exclude kimi-k3                  # drop these ids
--enable qwen2.5-coder-7b-gguf     # force in an entry with enabled: false
--category ocr,embeddings          # keep only these categories
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
    include: ["*-q4_k_m.gguf", README.md]
    exclude: []
    category: coding
    tags: [coding, gguf, small]
    license: apache-2.0
    gated: false
    approx_bytes: 4683079206
    enabled: true
    notes: One quant out of the dozen the repo ships.
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

Unknown keys are rejected, so a typo fails loudly instead of being ignored.

A profile matches an entry if any of its rules match: `all`, `ids`, `all_tags`
or `any_tags`.

**Globs** follow the Hub's `fnmatch` rules, where `*` also crosses `/`:
`*.safetensors` matches `transformer/x.safetensors` too. To exclude one file at
the repo root, name it exactly.

**GGUF repos** usually ship every quantization, often hundreds of GB in total.
Pin the one you want with `include`.

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
├── text-llm/openai/gpt-oss-120b/
├── coding/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF/
├── embeddings/Qwen/Qwen3-Embedding-0.6B/
└── _hfarchive/
    ├── manifest.jsonl
    └── logs/
```

Each model lives at `<dest>/<category>/<org>/<repo>/` as a plain snapshot of
the repo, so llama.cpp, ComfyUI or `from_pretrained()` can load it directly.

`manifest.jsonl` gets one line per repo per download run: id, repo, commit,
status, matched files and bytes, licence and timestamp. It is append-only.

Run logs are written to `${XDG_STATE_HOME:-~/.local/state}/hfarchive/logs/`
and copied to `_hfarchive/logs/` after any run that downloaded something.

## Exit codes

| Code | Meaning |
|---:|---|
| 0 | Success, or a clean dry run |
| 1 | A download failed, `--verify-only` found a missing repo, or `--refresh-sizes` could not measure a repo |
| 2 | Config error, empty selection, bad destination, or not enough space |
| 3 | Refused: an entry is at or above the large-repo threshold |

`--verify-only` works as a cron job to notice repos that disappear upstream.

## Environment

| Variable | Use |
|---|---|
| `HFARCHIVE_DEST` | Default for `--dest`. |
| `HFARCHIVE_HF_BIN` | Path to an `hf` binary, used when the script's own environment has none. |
| `HF_TOKEN` | Hub token, if you prefer it to `hf auth login`. |

`hfarchive` uses the `hf` CLI from its own environment, then
`$HFARCHIVE_HF_BIN`, then the first `hf` on your `PATH`.

## Examples

[`examples/`](examples/) holds complete configs you can run or copy from:

- [`tweet-2026-09.yaml`](examples/tweet-2026-09.yaml): 52 open-weights models
  from a list shared in September 2026 (text, coding, image, video, speech,
  music, OCR, embeddings), about 5.9 TB, with smaller profiles for GGUF-only,
  coding and 8 GB-VRAM machines.

```bash
hfarchive --config examples/tweet-2026-09.yaml --list --profile tweet-gguf
```

To split an archive across two drives, run twice with different `--category`
lists. Each drive keeps its own manifest.

## Troubleshooting

**"could not find the `hf` CLI"**: run the script through uv, or install
`huggingface_hub` and make `hf` available on `PATH` or via `HFARCHIVE_HF_BIN`.

**A repo is missing**: it was renamed or removed upstream. `--verify-only` shows
which; fix `hf_repo` in the config.

**The drive unmounted mid-run**: remount and run again. Complete files are
skipped, partial ones resume.

**Slow transfers**: try `--jobs 1 --max-workers 4`. USB drives are usually the
bottleneck, not the network.

## Development

```bash
uv run --with pytest --with huggingface_hub --with pyyaml pytest -q
```

The tests download one ~2 KB `config.json` to exercise `hf download`; they never
download model weights. Network tests are skipped when offline.

## License

MIT. See [LICENSE](LICENSE).
