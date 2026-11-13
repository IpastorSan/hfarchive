# local-model-backup

Mirror a curated set of open Hugging Face models onto external hard drives, so
you keep local copies of the open-weights models that matter.

The default catalog is the list from
[Dustin Hollywood's tweet](https://x.com/dustinhollywood/status/2099912741125324917):

> "If I were you all, I would download this list to a hard drive with lots of
> space and keep it updated. These are all on Hugging Face. You never know.."

Every repo id in `models.yaml` was verified against the Hub API, OCR mistakes
fixed, and the sizes are measured — not guessed.

**The default `tweet` profile is ~5.9 TB.** Read [Disk math](#disk-math) before
you start.

---

## Saturday quickstart

```bash
# 0. one-time setup (already done if .venv exists)
python3 -m venv .venv
.venv/bin/pip install huggingface_hub pyyaml

# 1. plug the drive in, then find where it mounted
lsblk -o NAME,SIZE,FSTYPE,MOUNTPOINT
#   -> e.g. /run/media/ignacio/BACKUP1
df -h /run/media/ignacio/BACKUP1      # confirm free space

# 2. see what would happen — writes nothing
.venv/bin/python -m lmb --dry-run --dest /run/media/ignacio/BACKUP1 --profile tweet

# 3. start with the things that actually run on your box (~241 GB)
.venv/bin/python -m lmb --dest /run/media/ignacio/BACKUP1 --profile current-box

# 4. then the big archival pull, if the drive is big enough
.venv/bin/python -m lmb --dest /run/media/ignacio/BACKUP1 --profile tweet --allow-large
```

Set the destination once and skip `--dest` everywhere:

```bash
export LOCAL_MODEL_BACKUP_DEST=/run/media/ignacio/BACKUP1
```

Interrupting with `Ctrl-C` is safe — everything resumes. Re-running is cheap:
complete files are skipped, so you can leave step 4 running overnight and start
it again the next day.

---

## Profiles

| Profile | Models | Size | What it is |
|---|---:|---:|---|
| `tweet` **(default)** | 52 | **5.90 TB** | Every model from the tweet. |
| `tweet-gguf` | 8 | 700.73 GB | Only the GGUF / local-friendly rows. |
| `domestic-coding` | 9 | 125.89 GB | Coding models for a domestic ATX box (16–32 GB VRAM). |
| `current-box` | 29 | 241.10 GB | Fits **today's** box: RTX 2070 Super 8 GB + 128 GB RAM. |
| `all` | 52 | 5.90 TB | Everything marked `enabled` in the YAML. |

```bash
.venv/bin/python -m lmb --list --profile tweet-gguf     # no network needed
.venv/bin/python -m lmb --list --profile tweet --json   # machine-readable
```

### Commands

| Command | Effect |
|---|---|
| `--list` | Print the resolved catalog. Offline. |
| `--verify-only` | Check every repo still exists on the Hub. Downloads nothing. |
| `--dry-run` | Resolve live sizes, matched files and target paths. Writes nothing. |
| *(no mode flag)* | Download. |
| `--refresh-sizes` | Re-measure every repo and write `approx_bytes` back into the YAML. |

### Selection

```bash
--only  qwen2.5-coder-7b-gguf,glm-ocr     # just these ids
--exclude kimi-k3,kimi-k3-gguf            # everything but these
--enable qwen2.5-coder-32b-gguf           # force in a model that is disabled
--category ocr,embeddings                 # keep only these categories
```

`--only` overrides the profile; `--exclude`, `--enable` and `--category` refine it.

---

## Disk math

`tweet`, by category:

| Category | Models | Size |
|---|---:|---:|
| text-llm | 8 | 3.39 TB |
| coding | 4 | 0.73 TB |
| text-llm-gguf | 4 | 0.64 TB |
| video | 6 | 0.60 TB |
| image | 6 | 0.34 TB |
| music | 5 | 0.07 TB |
| embeddings | 6 | 0.04 TB |
| ocr | 4 | 0.03 TB |
| asr | 3 | 0.03 TB |
| small | 3 | 0.02 TB |
| tts | 3 | 0.02 TB |
| **Total** | **52** | **5.90 TB** |

Sizes are decimal GB/TB (1 GB = 10⁹ bytes), the same units drive vendors and the
Hub use — so a "6 TB" drive really does fit the 5.90 TB `tweet` profile, barely.

Four repos dominate the total:

| Model | Size |
|---|---:|
| `kimi-k3` | 1.56 TB |
| `glm-5.3` | 755.66 GB |
| `kimi-k2.7-code` | 595.20 GB |
| `deepseek-v4.1-flash` | 510.31 GB |

Dropping them takes `tweet` from 5.90 TB to ~2.48 TB:

```bash
.venv/bin/python -m lmb --dest "$LOCAL_MODEL_BACKUP_DEST" --profile tweet \
  --exclude kimi-k3,glm-5.3,kimi-k2.7-code,deepseek-v4.1-flash
```

### The large-repo guard

Any single entry of **100 GB or more** is refused unless you pass `--allow-large`
(alias `--i-know`):

```
refusing to download 1 repo(s) of 100.00 GB or more without --allow-large: kimi-k3-gguf
```

`--dry-run` never needs the flag. Change the threshold with `--large-repo-gb`.

### Space guards

The tool refuses to start if the destination is missing, is not a directory,
is not writable, has less than **50 GB** free (`--min-free-gb`), or if the plan
does not fit in the remaining space (`--ignore-space` overrides).

**It also refuses any destination inside `/home`** — that disk only has ~118 GB
free, and this is a backup to external media. `--allow-home-dest` overrides it.

A destination that simply *does not exist yet* is not an error: the run degrades
to a dry run, so you can plan before the drive is plugged in.

---

## Gated models

Six entries need you to accept a licence on the model page first:

`flux2-dev`, `flux2-klein-9b`, `ltx-2.5`, `stable-audio-3-medium`,
`sam-audio-large`, `embeddinggemma-300m`

Without a token they are **skipped with a message**, and the rest still runs:

```
2 gated repo(s) will be skipped (no HF token): stable-audio-3-medium, embeddinggemma-300m
  -> run `hf auth login` and accept each licence on the model page
```

To include them:

```bash
.venv/bin/hf auth login          # older installs: huggingface-cli login
```

Then open each model page and accept the terms. `sam-audio-large` and
`embeddinggemma-300m` use **manual** approval, which can take a while — request
access a few days before you plan to download.

Tokens are never printed or written to the logs or the manifest.

---

## Layout on the drive

```
/run/media/ignacio/BACKUP1/
├── text-llm/openai/gpt-oss-120b/
├── text-llm-gguf/unsloth/Qwen3.8-27B-GGUF/
├── coding/Qwen/Qwen3-Coder-30B-A3B-Instruct/
├── coding-small/TheBloke/deepseek-coder-6.7B-instruct-GGUF/
├── image/  video/  asr/  tts/  music/  ocr/  embeddings/  small/
└── _lmb/
    ├── manifest.jsonl        # what was downloaded, commit hash, bytes, timestamp
    └── logs/                 # copy of each run's log
```

That is `$DEST/<category>/<repo_org>/<repo_name>/`, so a model folder is a
normal local snapshot of the repo — point `llama.cpp`, ComfyUI or
`from_pretrained()` straight at it.

The manifest is append-only JSONL, one line per repo per run:

```json
{"timestamp": "2026-09-16T08:48:30+0200", "id": "tiny-bge", "hf_repo": "BAAI/bge-reranker-v2-m3",
 "commit": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e", "status": "downloaded",
 "files": 1, "bytes_matched": 795, "rel_dest": "embeddings/BAAI/bge-reranker-v2-m3"}
```

Logs are written to `logs/` **in this project** (git-ignored) and copied to the
drive after a run that actually downloaded something.

---

## Two drives

v1 takes a single `--dest`. To split across two drives, run it twice and split
by category — `--category` exists for exactly this:

```bash
# BACKUP1: the text and coding models (~4.8 TB)
.venv/bin/python -m lmb --dest /run/media/ignacio/BACKUP1 --profile tweet \
  --category text-llm,text-llm-gguf,coding,small --allow-large

# BACKUP2: everything visual and audio (~1.1 TB)
.venv/bin/python -m lmb --dest /run/media/ignacio/BACKUP2 --profile tweet \
  --category image,video,asr,tts,music,ocr,embeddings --allow-large
```

Each drive gets its own `_lmb/manifest.jsonl`, so the two stay independently
verifiable. Categories never overlap, so no file is stored twice.

---

## Keeping it updated

```bash
# has anything moved or been pulled from the Hub?
.venv/bin/python -m lmb --verify-only --profile tweet

# re-measure sizes after upstream repos change
.venv/bin/python -m lmb --refresh-sizes --profile all

# pull new revisions (complete files are skipped)
.venv/bin/python -m lmb --dest "$LOCAL_MODEL_BACKUP_DEST" --profile tweet --allow-large
```

`--verify-only` exits non-zero if any repo has disappeared, so it works in cron.

---

## Editing the catalog

Everything lives in `models.yaml` — no script changes needed.

```yaml
  - id: qwen2.5-coder-7b-gguf
    hf_repo: Qwen/Qwen2.5-Coder-7B-Instruct-GGUF
    include:                       # ONLY these files are fetched
      - "*-q4_k_m.gguf"
      - README.md
    category: coding-small         # -> $DEST/coding-small/Qwen/...
    tags: [coding-small, gguf, coding, current-box]
    license: apache-2.0
    gated: false
    approx_bytes: 4683073312
    enabled: false                 # off unless a profile opts in
```

Flip `enabled` to turn a model on or off permanently; use `--only` / `--exclude`
for one-off runs. Add a `revision:` to pin a branch, tag or commit.

> **Glob gotcha:** Hub patterns are `fnmatch`-style and `*` **crosses `/`**.
> `*.safetensors` also matches `transformer/x.safetensors`. To exclude a single
> root-level file, name it exactly (that is why `flux2-dev.safetensors` appears
> verbatim in the catalog).

### Picking a different quant

GGUF repos ship every quant — `unsloth/Kimi-K3-GGUF` alone is ~8 TB. Each GGUF
entry pins **one**. To take a bigger one, edit its `include:`, then re-measure:

```bash
.venv/bin/python -m lmb --refresh-sizes --profile all
```

---

## Catalog decisions worth knowing

**Two repos from the brief do not exist on the Hub.** Both were replaced with the
highest-download GGUF for that model family, as instructed:

| Requested | Substituted with | Why |
|---|---|---|
| `unsloth/DeepSeek-Coder-V2-Lite-Instruct-GGUF` | `bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF` | Requested repo 404s. bartowski's has ~449k downloads, the most of that family. |
| `bartowski/deepseek-coder-6.7b-instruct-GGUF` | `TheBloke/deepseek-coder-6.7B-instruct-GGUF` | Requested repo 404s. TheBloke's has ~42k downloads, the most of that family. |

**Three OCR'd ids needed casing fixes** (the Hub is case-sensitive on redirect):
`google/gemma-4-31b-it` → `gemma-4-31B-it`, `google/gemma-4-26b-A4B-it` →
`gemma-4-26B-A4B-it`, `tencent/Auk` → `tencent/AuK`.

**Everything else on the tweet exists exactly as written**, including the three
flagged as uncertain: `Edge0/Audio8-TTS-Preview-0.6b`, `audio-cpp/Yue2-3B-GGUF`
and `audio-cpp/MiniMax-Music3-GGUF`.

**`kimi-k3` is 1.56 TB, not the ~2.8 TB the tweet quotes.** The catalog uses the
measured number.

**Several repos ship the same weights more than once.** The catalog excludes the
duplicates and says so in each entry's `notes`:

| Model | Full repo | Kept | Dropped |
|---|---:|---:|---|
| `gpt-oss-120b` | 195.8 GB | 65.2 GB | `original/`, `metal/` — the same weights twice more |
| `flux2-dev` | 177.6 GB | 113.2 GB | root single-file copy of `transformer/` |
| `krea-realtime-video` | 114.4 GB | 28.6 GB | training-init checkpoint + root duplicate |
| `minimax-h3` | 498.5 GB | 210.4 GB | `FL2VA/`, `Ref2VA/` task variants |
| `wan2.2-comfyui-repackaged` | 728.7 GB | 117.7 GB | bf16/fp16 twins of the fp8 files |
| `ltx-2.5` | 200.9 GB | 100.5 GB | `dev` and int8 variants of the distilled model |

Drop the `exclude:` lines from any entry to take the full repo.

---

## Hardware notes

Current box: **AMD 7800X3D, 128 GB DDR5, RTX 2070 Super 8 GB.**

The `current-box` profile is what genuinely runs there. It leans on MoE GGUFs,
where a low active-parameter count keeps CPU inference fast:

- `qwen3-coder-30b-a3b-gguf` — 17.7 GB, 30B total but ~3B active
- `deepseek-coder-v2-lite-gguf` — 10.4 GB, 16B total but ~2.4B active
- `qwen2.5-coder-7b-gguf` — 4.7 GB, fits **entirely** in the 8 GB card
- `deepseek-coder-6.7b-gguf` — 4.1 GB, the DeepSeek-family option that fits it

`domestic-coding` targets the next box up (RTX 5080/5090 class, 16–32 GB VRAM)
and adds the 14B/32B Q4 models.

---

## Testing

```bash
.venv/bin/python tests/test_smoke.py        # standalone, no pytest needed
.venv/bin/python -m pytest tests/ -q        # if pytest is installed
```

20 checks covering catalog integrity (all 52 tweet models present and tagged),
profile resolution, glob semantics, the destination guards, and the manifest.

The only network transfer is a **~2 KB `config.json`**, used to exercise the real
`hf download` path end to end — download, skip-if-complete, and re-fetch of a
corrupted file. **The smoke test never downloads model weights.**

---

## How it works

`hf download` from `huggingface_hub` does the transfers, so you get its resume,
integrity checking and Xet acceleration for free. This project is the layer
around it: catalog, profiles, size accounting, safety guards and the manifest.

- **Resume** — partial files continue; complete files are skipped by size+etag.
- **Repair** — a truncated file is detected and re-fetched.
- **Concurrency** — `--jobs` repos at a time (default 2), `--max-workers` files
  within each (default 8). Conservative on purpose: USB drives and spinning rust
  do not like 16 parallel writers.
- **Python 3.11+**, two dependencies: `huggingface_hub` and `PyYAML`.

Faster large-file transfers, optional:

```bash
.venv/bin/pip install hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
```

### Exit codes

| Code | Meaning |
|---:|---|
| 0 | Success, or a clean dry run |
| 1 | Some repos failed to download, or `--verify-only` found a missing repo |
| 2 | Bad destination, not enough space, or a catalog error |
| 3 | Refused: a repo is at or above the large-repo threshold |

---

## Troubleshooting

**"could not find the `hf` CLI"** — install it in the venv
(`.venv/bin/pip install huggingface_hub`) or point `LMB_HF_BIN` at one.

**Drive unmounts mid-run** — downloads fail, nothing is corrupted. Remount and
re-run; it picks up where it stopped.

**A repo 404s** — upstream renamed or pulled it. Run `--verify-only` to see
which, then fix `hf_repo` in the YAML.

**Slow speeds** — try `--jobs 1 --max-workers 4`; USB 3 external drives are
usually the bottleneck, not the network.
