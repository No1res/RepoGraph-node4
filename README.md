# RepoGraph Customizations (What We Changed)

This document summarizes all modifications made during our adaptation of RepoGraph for CoderEval-style context building and long-context budget studies.

---

## 1) Graph Construction Changes (`repograph/construct_graph.py`)

### 1.1 Line-level node identity via `idx`
- Extended `Tag` to include a unique integer id:
  - `Tag(idx, rel_fname, fname, line, name, kind, category, info)`
- Graph nodes are keyed by `tag.idx` (not by `tag.name`).
- Node attributes store `name/kind/category/rel_fname/line/info`.
- **Why:** avoids cross-file name collisions (same function name in multiple files previously merged into one node).

### 1.2 Robust `get_tags_raw` (prevent `KeyError` on missing structure entries)
- Replaced hard indexing:
  - `structure_all_funcs[tag_name]`
- With safe lookup + fallback:
  - `info = structure_all_funcs.get(tag_name)`; if missing, fallback to minimal snippet/line.
- **Why:** some repos have functions captured by tree-sitter but absent from the AST-derived structure (previously caused crashes like `KeyError: get_auth`).

### 1.3 Fix root-level `.py` files (e.g., `setup.py`) not producing tags
- Root cause: `create_structure()` stores root files under `structure[repo_name][file]`, while `get_tags_raw()` originally indexed from the top level using `rel_fname` directly.
- Fix: when `len(ref_fname_lst) == 1`, first step into `structure[repo_name]` before indexing the file.
- **Why:** ensures root-level files are parsed and appear in tags/graph; required for embedding-meta → RepoGraph tag mapping (Issue #18 aligned).

### 1.4 Directory filtering in file discovery (reduce noise + speedup)
- Updated `find_src_files()` to skip:
  - `.git`, `__pycache__`, `venv/.venv`, `site-packages`, `dist-packages`, `node_modules`, etc.
- **Why:** prevents third-party/env code from polluting the graph and dramatically reduces runtime and warnings.

### 1.5 Misc correctness fixes
- Ensure `self.num_tags = 0` is initialized and incremented for every yielded tag (including fallback refs).
- Fix namedtuple access patterns (use `tag.name`, not `tag['name']`).
- Align class method list parsing (`info` uses `\n` join → split by `\n`).
- Fix class→method edges to use `idx` ids (not raw names).

---

## 2) Context Dataset Builder (`build_ranked_context_repograph.py`)

Goal: generate a ranked context dataset (`jsonl`) compatible with our inference script:
- Required fields: `"_id"`, `"model_input"`, `"context"` (list of blocks)

### 2.1 Output schema aligned to inference script
Each output line contains:
- `_id`: question id
- `repo_id`, `project`
- `model_input`: `signature + docstring` input from `CEPythonHumanLabel.jsonl`
- `context`: `List[str]` (ranked blocks)
- `depth`, `anchors_mode`, `faiss_topk`
- Optional debug: `faiss_topk_meta`, `faiss_anchor_idxs`
- Optional debug: `context_tokens` if enabled

### 2.2 Strict GroundTruth removal (no leakage)
We remove any context blocks overlapping the GT region:
- Same file (suffix match with `file_path`)
- Line overlap with `[lineno, end_lineno]` (0/1-based tolerance)
- Backstop: contains `def <name>(`
Verified by **TRUE leak check** (same file + `def <name>(`): `true leak count = 0`.

### 2.3 FAISS/Embedding anchors (TopK → RepoGraph BFS expansion)
Added `anchors_mode=faiss_then_name`:
1. Build query embedding using OpenAI-compatible embeddings endpoint:
   - `POST /v1/embeddings` with Qwen3-Embedding-4B
2. Search repo-local FAISS index (`repo_id.flat.ip.faiss`) for topK hits
3. Map `meta.jsonl` rows to RepoGraph `tag.idx` anchors using:
   - file suffix match + `symbol → last segment name` + line overlap (+ text containment fallback)
4. Expand context via RepoGraph graph traversal:
   - BFS on `G` and BFS on `G.reverse()` (incoming callers)
5. If FAISS anchors fail, fallback to name anchors (signature-derived def/ref).

### 2.4 Anchor ordering (no filtering; only reordering)
To reduce “tests/versioneer” dominance without excluding content:
- Reorder topK hits so “good paths” are processed first and “bad paths” later:
  - bad prefixes: `tests/`, `test/`, `unittests/`, `docs/`, `examples/`
  - bad exact: `setup.py`, `versioneer.py`
- **No blocks are removed**, only anchor priority changes.
Effect: bad-path top1 ratio reduced significantly (≈19% → ≈6.6% in our 200k pool).

### 2.5 Multi-stage fallback to increase token coverage
To avoid empty/tiny context (leaf functions) and support long budgets:
1. Same-file defs fallback (GT file)
2. TopK-hit files defs fallback (from FAISS file paths)
3. Same-directory defs fallback (GT file’s folder)
Controlled by:
- `--min_ctx_tokens`
- `--fallback_same_file_defs`
- `--fallback_topk_file_defs`
- `--fallback_same_dir_defs`
This enabled near-200k candidate pools (p99 ≈ 200k tokens in sampled evaluation).

---

## 3) Quality Evaluation Commands / Metrics

We used a one-shot Python3 checker to validate:
- JSON validity
- Coverage: input count vs output count + missing ids
- Context block counts distribution
- TRUE GT leak check (same file + `def <name>(`)
- FAISS diagnostics:
  - empty anchor ratio
  - top1 path distribution + bad-path ratio
- Token distribution (Qwen3-Coder tokenizer, add_special_tokens=False)

---

## 4) Infrastructure Alignment (vLLM)

### 4.1 Embedding service (Qwen3-Embedding-4B)
- Served via vLLM OpenAI-compatible `/v1/embeddings`
- Verified:
  - embedding dimension matches FAISS index `d` and `embeddings.npy` shape
  - vectors are L2-normalized (norm ≈ 1), so IP index behaves like cosine

### 4.2 Generation service (Qwen3-30B-A3B-Instruct-2507)
- Served via vLLM OpenAI-compatible `/v1/chat/completions`
- Recommended for reproducibility:
  - `--generation-config vllm` to avoid HF generation_config overriding defaults
  - `--served-model-name` aligned with client `--model` field

---

## 5) Summary of Key Outcomes
- Stable graph construction (no crashes on missing structure entries)
- Root-level files now included in tags/graph (mapping from embedding meta works)
- FAISS anchors integrated with RepoGraph BFS expansion
- Strict GT removal verified (`true leak count = 0`)
- Candidate pool scaled to long-context budgets (200k-ready)
- Bad-path dominance reduced by anchor reordering (without excluding content)
