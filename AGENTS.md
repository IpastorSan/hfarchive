# AGENTS.md

Instructions for coding agents working in this repo. Humans: see README.md.

## The repo

- `hfarchive`: the whole tool, one executable Python file run by `uv run --script` (PEP 723 header). There is no package, no `pyproject.toml`; keep it that way.
- `examples/archive-2026-11.yaml`: the curated default selection that `--init` writes.
- `examples/tweet-2026-09.yaml`: an older list, kept as an example.
- `tests/test_hfarchive.py`: loads the script with `SourceFileLoader`.

Checks before calling work done:

```sh
uv run --no-project --with pytest --with huggingface_hub --with pyyaml pytest -q
./hfarchive --help
./hfarchive --config examples/archive-2026-11.yaml --list
./hfarchive --config examples/archive-2026-11.yaml --profile all --verify-only   # hits the Hub, no downloads
```

Always pass `--no-project` to `uv run`; without it uv finds a `pyproject.toml` in a parent directory and rebuilds that project's `.venv`. Never run a real download in a test or a check.

## When a user asks you to research models

The job runs from research to a verified archive on their drive. Do it in this order and don't skip the decision step.

1. **Ask only what you can't find out.** Their GPU, VRAM and system RAM; this you can usually read yourself (`nvidia-smi`, `free -h`). Ask what they will use the models for; most users of this tool are programmers, so default to coding first. Ask the target drive and its free space (`df -h <dest>`), and whether size limits or licences matter to them.
2. **Research** as in "Refreshing the curated selection" below: candidates per category, every repo verified on the Hub, the quant chosen per tier, model cards read, benchmarks gathered with sources, and coding benchmarks first.
3. **Present a decision brief** before writing any config or downloading anything. It must let the user decide without opening another page:
   - their hardware tier and what it means for them;
   - per category, a table with each model: params (total / active), the tier it runs on, quant and size, what it is best at, 3–5 benchmarks with sources (coding benchmarks for the coding and general tables), licence and gating, and the speed estimate;
   - how the best candidates compare with the current closed frontier, on coding especially;
   - a proposed `core` and `extended` set with size totals checked against their free space, and what to drop first if it does not fit;
   - anything unverified, self-reported or surprising (renamed repos, territory-restricted licences, runtimes that only build on new GPUs).

   Put a long brief in a file (`docs/models-YYYY-MM.md` when refreshing the default, or a file in the user's working directory otherwise) and summarise it in chat.
4. **Wait for the user's choice.** They add, drop or swap models. Don't download before they say so: a run is hundreds of GB and hours of transfer.
5. **Write the config** for exactly what they chose, with `usage` notes, workflows and the runtimes it needs, with `wheels` pinned for Python runtimes (step 5 below). Check it with `--verify-only`, then show the `--dry-run` to the user: the totals, what is over `large_repo_gb`, and the free space after the run.
6. **Download everything** they approved:
   ```sh
   ./hfarchive --config <their config> --dest <drive> --profile <profile> [--allow-large]
   ```
   Run it in the background, as it takes hours. If it stops (network, rate limit), run the same command again; downloads resume.
7. **Verify and report.** Run `./hfarchive --check --dest <drive>`. Report what was archived, the total size, any entry that failed and why, and point the user at `<drive>/INDEX.md`. Suggest a second copy of the core set and a periodic `--verify-only` / `--check`.

## Refreshing the curated selection

The default selection goes stale in months. To build a new one (`examples/archive-YYYY-MM.yaml`), research it the way the current one was built. Work from evidence on huggingface.co, not from memory: model names, quant names and sizes change too fast.

### 1. Fix the hardware tiers first

Every recommendation is "runs on tier X". The current tiers:

| Tag | GPU | System RAM | What fits |
|---|---|---|---|
| `tier-8gb` | 8 GB VRAM, possibly pre-Ampere (no bf16/FP8) | ~128 GB | Small dense models fully on GPU; large MoE models with experts offloaded to RAM (`--n-cpu-moe`); image models as GGUF with text encoders in RAM |
| `tier-16gb` | 16 GB | 64–128 GB | Dense ~27–31B at Q3–Q4; mid image/video models at Q4–Q8 |
| `tier-24gb` | 24 GB | 128 GB | Dense ~27–31B at Q4–Q5; large image models (FLUX.2 dev class) at Q4 |

Rules of thumb used for the fit:
- LLM memory needed ≈ GGUF size + 10–20 % for KV cache and context. Dense models are fast only when the whole file fits in VRAM.
- MoE speed scales with **active** parameters, so a 100B+ MoE with ~3–6B active is usable from system RAM. A dense 27B spilling into RAM is not.
- Image/video VRAM need ≈ diffusion-model file + 1–2 GB; text encoders can sit in RAM.
- FP8 needs Ada/Blackwell; NVFP4 needs Blackwell. On older cards prefer GGUF or int8.

### 2. Gather candidates

Cover, at their latest generation:
- **General LLMs** from every major open-weight family (at the time of writing: Qwen, Gemma, DeepSeek, GLM, Kimi, ERNIE, LFM, MiniMax, gpt-oss, Nemotron, Mistral, Nex) plus any newcomer that tops a tier.
- **Coding/agent** models.
- **Security** models (Foundation-Sec and successors), and note honestly whether a general model beats them.
- **Abliterated/uncensored** variants of the best general models. Prefer Heretic-style (lower KL divergence) over older manual abliteration; always keep the original too.
- **Decision models** (option-scoring models such as Clef; check whether a model is actually open-weight or hosted-only).
- **Image** generation and editing, **video**, and abliterated text encoders for them.

Sources, in order of trust: the official org's Hub page and model card; unsloth.ai docs and changelog; the ggml-org, unsloth, bartowski, city96, QuantStack and Comfy-Org repos; ComfyUI's blog and workflow templates; then press and blogs. Treat benchmark claims in announcements as the author's own until someone else reproduces them.

### 3. Verify every repo on the Hub

For each candidate, use the API, not search snippets:

```sh
curl -s "https://huggingface.co/api/models?search=<name>&sort=downloads&limit=20"          # find repos
curl -s "https://huggingface.co/api/models/<org>/<repo>?blobs=true"                        # files, sizes, gated, license, base_model
curl -s "https://huggingface.co/api/models/<org>/<repo>/tree/main?recursive=true"          # exact paths (sharded quants live in folders)
```

Record: the repo exists, `gated`, the license, the base model, the exact file names of the quant you want and their summed size. A redirect means the repo was renamed: use the new id. If a repo is gone, find the closest real replacement and say so in `notes`; never invent a repo or a file name.

### 4. Choose the quant per tier

- Prefer GGUF from unsloth (UD dynamic quants), ggml-org (official) or bartowski.
- Typical picks: Q4_K_M / UD-Q4_K_XL as the default; IQ4_XS or Q3_K_XL to squeeze into a tier; IQ1_S/IQ2 only for huge MoE models you otherwise could not run; Q8_0 as the "keep for a bigger card" copy of small models.
- Include `mmproj` files for vision LLMs.
- Image/video models need their components: text encoder(s), VAE, upscaler. Make each component its own entry and share it between models instead of duplicating it.
- A model whose smallest quant does not fit any tier is either skipped or kept as archive-only (no tier tag, `extended`, a note saying what hardware it needs).

### 5. Write the entries

Follow the schema in the README. Per entry:
- `include` globs pin only the chosen files. Hub globs are fnmatch-style and `*` crosses `/`. Unsloth repos keep stub first-shard copies under `Shard_Rewrite/`, so pin sharded quants by folder (`UD-IQ1_S/*`), never by a bare `*UD-IQ1_S*`.
- `approx_bytes` is the measured size of the matched files (`--refresh-sizes` rewrites it).
- `gated` and `license` from the API.
- Tags: `core` or `extended`; a type (`llm`, `image`, `video`, `security`, `uncensored`, `decision`); one tier tag per tier it runs on.
- `usage.runtime` matching a `runtimes[].id`, and `usage.notes` with the model card's recommended settings (LLMs: temperature, top_p, top_k, min_p, thinking toggles, context; image/video: sampler, scheduler, steps, guidance, resolution, frames/fps), offload flags for the small tier, and which component entries it needs. Mark anything not from a model card as "(estimate)".
- `benchmarks` for every model with published numbers: 3–8 items (`name`, `score`, `source` URL, `date`, `note` for self-reported or the harness), the most relevant first since `INDEX.md` shows the first three. Coding benchmarks come first for LLMs. Use exactly the numbers in the models guide (step 7), and give components (encoders, VAEs, mmproj) none.
- `guide:` at the top level, pointing at the selection's `docs/models-YYYY-MM.md`; it is copied to the drive on every run.
- `usage.workflows` for every image and video model: ComfyUI workflow JSONs, preferably from `Comfy-Org/workflow_templates`, at a pinned commit URL. Check each URL returns valid JSON.

Pin every runtime in `runtimes:` to a release tag (or a commit when there are no releases) found with `gh api`, and check that `release_assets` globs match real assets. Include every custom-node pack the workflows need.

Pin `wheels` for every Python runtime (ComfyUI and its node packs, vLLM, transformers), so it installs offline:
- `python`: one quoted version every pinned runtime supports (read each `requires-python`).
- `requirements`: the runtime's own requirements files at `ref`; read them in a clone at that ref.
- `packages`: the PyTorch build the runtime recommends at `ref` (its README or install docs), pinned with the CUDA local version (`torch==X+cu130`), and `index_url` set to that CUDA index with PyPI as an extra index.
- `platforms`: Linux x86_64 and Windows amd64 where the runtime runs on both. Pick the manylinux version from the wheels the set actually needs (some packages ship only `manylinux_2_31`).
- `start`: how to start it from the unpacked source; node packs install into ComfyUI's environment.
- Check that each set resolves without downloading wheels: `uv pip compile --python-version 3.12 --python-platform x86_64-manylinux_2_28 --only-binary :all: --index-strategy unsafe-best-match [--index-url …] <requirements.txt> <packages file>` (and `x86_64-pc-windows-msvc`). Explain in `notes` anything that cannot be made offline-installable.

### 6. Check and size it

```sh
./hfarchive --config examples/archive-YYYY-MM.yaml --profile all --verify-only
./hfarchive --config examples/archive-YYYY-MM.yaml --profile core --list
./hfarchive --config examples/archive-YYYY-MM.yaml --profile all --list
```

All repos must report OK. Report totals per profile and keep `core` to a size that fits one common drive with room for a second copy (the 2026-11 set: ~880 GB core, ~2.2 TB all). Then point `--init`'s curated-file constant at the new file.

### 7. Write the models guide

Every selection ships with `docs/models-YYYY-MM.md`, linked from the README. It holds:
- The hardware tiers in plain language: typical GPUs and RAM per tier, VRAM vs RAM, dense vs MoE, rough speeds marked as estimates.
- One table per category (general, coding, security, uncensored, decision, image, video). Each row gives the model, its total and active params, the tiers it runs on, the archived quant and size, what it is best at, and 3–5 headline benchmarks. Use one consistent benchmark set per table and "—" where no number exists; never invent one.
- A comparison of the best open models with the current closed frontier (the latest Claude, OpenAI and Google flagships) on the same benchmarks, saying plainly where open models are close and where they lag.
- Caveats: published scores are at full precision, so the archived Q3/Q4/IQ1 quants score lower. Mark self-reported numbers as such, and give each number a source and a date.

Benchmark sources, in order: model cards, official tech reports, Artificial Analysis, LMArena. Prefer the benchmarks most widely reported at the time (in 2026-11: MMLU-Pro, GPQA Diamond, Humanity's Last Exam, AIME, LiveCodeBench, SWE-bench Verified, Terminal-Bench 2.x, τ²-bench, the Artificial Analysis Intelligence Index; arena Elo for image and video).

## Writing style

The README and example comments are public. Plain, concrete sentences; no marketing; no personal names or machines; no text addressed to an AI outside this file.
