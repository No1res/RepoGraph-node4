#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import pickle
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
import faiss
from transformers import AutoTokenizer

# RepoGraph BFS wrapper (uses networkx graph.neighbors internally)
from repograph.graph_searcher import RepoSearcher


# -----------------------------
# IO helpers
# -----------------------------
def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_codereval_index(path: str) -> Dict[str, Dict[str, Any]]:
    obj = json.load(open(path, "r", encoding="utf-8"))
    recs = obj["RECORDS"]
    return {r["_id"]: r for r in recs}


# -----------------------------
# Parsing / indexing helpers
# -----------------------------
def extract_func_name(signature: str) -> Optional[str]:
    m = re.search(r"\bdef\s+([A-Za-z_]\w*)\s*\(", signature)
    return m.group(1) if m else None


def build_indexes(tags: List[Dict[str, Any]]):
    by_idx: Dict[int, Dict[str, Any]] = {}
    def_index: Dict[str, List[int]] = defaultdict(list)
    ref_index: Dict[str, List[int]] = defaultdict(list)
    for t in tags:
        idx = int(t["idx"])
        by_idx[idx] = t
        if t.get("kind") == "def":
            def_index[t.get("name", "")].append(idx)
        elif t.get("kind") == "ref":
            ref_index[t.get("name", "")].append(idx)
    return by_idx, def_index, ref_index


def norm_path(p: str) -> str:
    return (p or "").replace("\\", "/").lstrip("./")


def file_match(tag_rel_fname: str, gt_file_path: str) -> bool:
    # suffix match is most robust across different relpath conventions
    return norm_path(tag_rel_fname).endswith(norm_path(gt_file_path))


def parse_int(x) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


def tag_line_range(tag_line) -> Optional[Tuple[int, int]]:
    # tag_line can be -1, or [a,b]
    if tag_line is None or tag_line == -1:
        return None
    if isinstance(tag_line, (list, tuple)) and len(tag_line) == 2:
        a = parse_int(tag_line[0])
        b = parse_int(tag_line[1])
        if a is None or b is None:
            return None
        if a > b:
            a, b = b, a
        return (a, b)
    return None


def overlap(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return not (a[1] < b[0] or b[1] < a[0])


def is_groundtruth_region(tag: Dict[str, Any], gt: Dict[str, Any]) -> bool:
    """
    Strict GT removal:
    - same file (suffix match)
    - and tag line-range overlaps [gt.lineno, gt.end_lineno] (try both 0/1-based)
    - OR info contains `def <name>(` as a backstop
    """
    if not file_match(tag.get("rel_fname", ""), gt.get("file_path", "")):
        return False

    name = gt.get("name", "")
    info = tag.get("info") or ""
    if f"def {name}(" in info:
        return True

    gt_l0 = parse_int(gt.get("lineno"))
    gt_l1 = parse_int(gt.get("end_lineno"))
    if gt_l0 is None or gt_l1 is None:
        return (tag.get("kind") == "def") and (tag.get("name") == name)

    gt_range = (min(gt_l0, gt_l1), max(gt_l0, gt_l1))
    tr = tag_line_range(tag.get("line"))
    if tr is None:
        return False

    if overlap(tr, gt_range):
        return True
    tr_plus1 = (tr[0] + 1, tr[1] + 1)
    if overlap(tr_plus1, gt_range):
        return True

    return False


# -----------------------------
# RepoGraph traversal
# -----------------------------
def collect_neighbors_bfs(searcher: RepoSearcher, anchor: int, depth: int) -> List[int]:
    # keep RepoSearcher's BFS order
    return searcher.bfs(anchor, depth)


# -----------------------------
# Embedding + FAISS anchors
# -----------------------------
def embed_query(base_url: str, model: str, text: str, timeout: int = 600) -> np.ndarray:
    """
    Must match your embedding precompute script:
      POST /v1/embeddings
      payload: {"model": model, "input": [text]}
      response: {"data": [{"embedding":[...], "index":0}, ...]}
    """
    url = base_url.rstrip("/") + "/v1/embeddings"
    payload = {"model": model, "input": [text]}
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    row = sorted(data["data"], key=lambda x: x["index"])[0]
    v = np.asarray(row["embedding"], dtype=np.float32)
    return v


def l2_normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n > 0:
        return v / n
    return v


def last_symbol_name(symbol: str) -> str:
    # "Server.__init__" -> "__init__"
    return (symbol or "").split(".")[-1]


def load_meta_rows(meta_path: Path) -> List[Dict[str, Any]]:
    rows = []
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def faiss_search_topk(index: faiss.Index, qvec: np.ndarray, topk: int) -> Tuple[List[float], List[int]]:
    q = qvec.reshape(1, -1).astype(np.float32)
    D, I = index.search(q, topk)
    return D[0].tolist(), I[0].tolist()


def map_meta_row_to_tag_idxs(
    meta_row: Dict[str, Any],
    tags: List[Dict[str, Any]],
    gt: Dict[str, Any],
    prefer_kind: str = "def",
) -> List[int]:
    """
    Map embedding snippet (meta row) -> RepoGraph tag.idx (graph node id).

    Strong match:
      - file suffix match
      - name match: tag.name == last(meta.symbol)
      - line overlap (try 0/1-based)
    Fallback:
      - file+name match + text containment (meta.text in tag.info or vice versa)

    We also skip candidates that fall inside the GT region (to avoid anchor=GT-only leaf).
    """
    fp = meta_row.get("file_path", "")
    sym = meta_row.get("symbol", "")
    nm = last_symbol_name(sym)

    sr = parse_int(meta_row.get("start_line"))
    er = parse_int(meta_row.get("end_line"))
    meta_range = None
    if sr is not None and er is not None:
        meta_range = (min(sr, er), max(sr, er))

    text = meta_row.get("text") or ""

    # pass 1: strict (file + name + overlap)
    cand: List[int] = []
    for t in tags:
        if prefer_kind and t.get("kind") != prefer_kind:
            continue
        if not file_match(t.get("rel_fname", ""), fp):
            continue
        if t.get("name") != nm:
            continue

        # skip pygments backfill
        if t.get("line") == -1:
            continue

        # skip GT region candidates as anchors
        if is_groundtruth_region(t, gt):
            continue

        if meta_range is None:
            cand.append(int(t["idx"]))
            continue

        tr = tag_line_range(t.get("line"))
        if tr is None:
            continue

        if overlap(tr, meta_range) or overlap((tr[0] + 1, tr[1] + 1), meta_range):
            cand.append(int(t["idx"]))

    if cand:
        return cand

    # pass 2: text containment fallback (still file+name constrained)
    cand2: List[int] = []
    if text:
        for t in tags:
            if prefer_kind and t.get("kind") != prefer_kind:
                continue
            if not file_match(t.get("rel_fname", ""), fp):
                continue
            if t.get("name") != nm:
                continue
            if t.get("line") == -1:
                continue
            if is_groundtruth_region(t, gt):
                continue
            info = t.get("info") or ""
            if info and (text in info or info in text):
                cand2.append(int(t["idx"]))

    return cand2


# -----------------------------
# Fallback helpers (token coverage)
# -----------------------------
def safe_start_line(tag: Dict[str, Any]) -> int:
    lr = tag_line_range(tag.get("line"))
    if lr is None:
        return 10**9
    return lr[0]


def add_same_file_defs_fallback(
    ctx_blocks: List[str],
    ctx_token_lens: List[int],
    count_tokens,
    tags: List[Dict[str, Any]],
    gt: Dict[str, Any],
    keep_class_blocks: bool,
    min_ctx_tokens: int,
    max_defs: int,
) -> None:
    used = sum(ctx_token_lens) if ctx_token_lens else sum(count_tokens(b) for b in ctx_blocks)
    if used >= min_ctx_tokens:
        return

    gt_file = gt.get("file_path", "")

    cand = []
    for t in tags:
        if t.get("kind") != "def":
            continue
        if (not keep_class_blocks) and t.get("category") == "class":
            continue
        if t.get("line") == -1:
            continue
        if not file_match(t.get("rel_fname", ""), gt_file):
            continue
        if is_groundtruth_region(t, gt):
            continue
        cand.append(t)

    cand.sort(key=lambda x: (norm_path(x.get("rel_fname", "")), safe_start_line(x)))

    added = 0
    for t in cand:
        header = f"{t.get('kind')} {t.get('category')} {t.get('name')} | {t.get('rel_fname')} | {t.get('line')}"
        body = (t.get("info") or "").rstrip()
        block = header + "\n" + body

        tlen = count_tokens(block)
        ctx_blocks.append(block)
        ctx_token_lens.append(tlen)
        used += tlen

        added += 1
        if added >= max_defs:
            break
        if used >= min_ctx_tokens:
            break


def add_defs_from_filepaths(
    ctx_blocks: List[str],
    ctx_token_lens: List[int],
    count_tokens,
    tags: List[Dict[str, Any]],
    gt: Dict[str, Any],
    keep_class_blocks: bool,
    min_ctx_tokens: int,
    max_defs: int,
    file_paths: List[str],
) -> None:
    """
    Add def blocks from specified file_paths (e.g., FAISS topK hit file paths).
    No directory filtering, no block dedup; only removes GT region and line=-1.
    """
    used = sum(ctx_token_lens) if ctx_token_lens else sum(count_tokens(b) for b in ctx_blocks)
    if used >= min_ctx_tokens:
        return

    cand = []
    for fp in file_paths:
        for t in tags:
            if t.get("kind") != "def":
                continue
            if (not keep_class_blocks) and t.get("category") == "class":
                continue
            if t.get("line") == -1:
                continue
            if not file_match(t.get("rel_fname", ""), fp):
                continue
            if is_groundtruth_region(t, gt):
                continue
            cand.append(t)

    cand.sort(key=lambda x: (norm_path(x.get("rel_fname", "")), safe_start_line(x)))

    added = 0
    for t in cand:
        header = f"{t.get('kind')} {t.get('category')} {t.get('name')} | {t.get('rel_fname')} | {t.get('line')}"
        body = (t.get("info") or "").rstrip()
        block = header + "\n" + body

        tlen = count_tokens(block)
        ctx_blocks.append(block)
        ctx_token_lens.append(tlen)
        used += tlen

        added += 1
        if added >= max_defs:
            break
        if used >= min_ctx_tokens:
            break


def add_defs_from_same_dir(
    ctx_blocks: List[str],
    ctx_token_lens: List[int],
    count_tokens,
    tags: List[Dict[str, Any]],
    gt: Dict[str, Any],
    keep_class_blocks: bool,
    min_ctx_tokens: int,
    max_defs: int,
) -> None:
    """
    Add def blocks from the same directory as GT file.
    No directory filtering, no block dedup; only removes GT region and line=-1.
    """
    used = sum(ctx_token_lens) if ctx_token_lens else sum(count_tokens(b) for b in ctx_blocks)
    if used >= min_ctx_tokens:
        return

    gt_fp = norm_path(gt.get("file_path", ""))
    gt_dir = "/".join(gt_fp.split("/")[:-1])
    if not gt_dir:
        return

    cand = []
    for t in tags:
        if t.get("kind") != "def":
            continue
        if (not keep_class_blocks) and t.get("category") == "class":
            continue
        if t.get("line") == -1:
            continue
        rel = norm_path(t.get("rel_fname", ""))
        if not rel.startswith(gt_dir + "/"):
            continue
        if is_groundtruth_region(t, gt):
            continue
        cand.append(t)

    cand.sort(key=lambda x: (norm_path(x.get("rel_fname", "")), safe_start_line(x)))

    added = 0
    for t in cand:
        header = f"{t.get('kind')} {t.get('category')} {t.get('name')} | {t.get('rel_fname')} | {t.get('line')}"
        body = (t.get("info") or "").rstrip()
        block = header + "\n" + body

        tlen = count_tokens(block)
        ctx_blocks.append(block)
        ctx_token_lens.append(tlen)
        used += tlen

        added += 1
        if added >= max_defs:
            break
        if used >= min_ctx_tokens:
            break


# -----------------------------
# Anchor ordering (do not filter out; just reorder)
# -----------------------------
BAD_PREFIXES = ("tests/", "test/", "unittests/", "docs/", "doc/", "examples/")
BAD_EXACT = ("setup.py", "versioneer.py")


def is_bad_path(p: str) -> bool:
    p = norm_path(p)
    if p in BAD_EXACT:
        return True
    return p.startswith(BAD_PREFIXES)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--ce_label_jsonl", required=True, help="CEPythonHumanLabel.jsonl")
    ap.add_argument("--codereval_json", required=True, help="CoderEval4Python.json")
    ap.add_argument("--graphs_dir", required=True, help="dir containing <repo_id>.pkl and tags_<repo_id>.json")
    ap.add_argument("--out_jsonl", required=True, help="output dataset jsonl for your inference script")

    ap.add_argument("--tokenizer_path", default="/workspace/models/Qwen3-Coder-30B-A3B-Instruct",
                    help="Qwen3 tokenizer local path")
    ap.add_argument("--depth", type=int, default=2, help="BFS depth (k-hop)")
    ap.add_argument("--max_ctx_blocks", type=int, default=20000, help="cap number of context blocks per sample")
    ap.add_argument("--keep_class_blocks", action="store_true",
                    help="keep class category blocks (default: drop class blocks because they are method-name lists)")
    ap.add_argument("--write_ctx_token_lens", action="store_true",
                    help="also write context_tokens(list[int]) for debugging")

    # token coverage knobs
    ap.add_argument("--min_ctx_tokens", type=int, default=30000,
                    help="Try to reach at least this many tokens of context blocks by adding fallbacks.")
    ap.add_argument("--fallback_same_file_defs", type=int, default=500,
                    help="Max def blocks to add from the same file as GT.")
    ap.add_argument("--fallback_topk_file_defs", type=int, default=2000,
                    help="Max def blocks to add from FAISS topK hit file_paths.")
    ap.add_argument("--fallback_same_dir_defs", type=int, default=5000,
                    help="Max def blocks to add from same directory as GT file.")

    # repo_id mapping
    ap.add_argument("--repo_id_style", choices=["slash_to_triple_dash", "basename"], default="slash_to_triple_dash",
                    help="how to map CoderEval 'project' to repo_id file names")

    # anchors mode
    ap.add_argument("--anchors_mode", choices=["name", "faiss", "faiss_then_name"], default="faiss_then_name",
                    help="Anchor selection: name (signature) or faiss (embedding topK) or faiss_then_name (fallback).")

    # embeddings + faiss
    ap.add_argument("--emb_dir", default="embeddings_out",
                    help="Directory that contains <repo_id>/meta.jsonl")
    ap.add_argument("--faiss_dir", default="faiss_indexes_flat",
                    help="Directory that contains <repo_id>.flat.ip.faiss")
    ap.add_argument("--faiss_topk", type=int, default=5)

    ap.add_argument("--embed_base_url", default="http://127.0.0.1:7269")
    ap.add_argument("--embed_model", default="", help="Model tag/path for /v1/embeddings (must match precompute).")
    ap.add_argument("--embed_timeout", type=int, default=600)
    ap.add_argument("--l2_normalize_query", action="store_true", default=True,
                    help="L2 normalize query embedding before FAISS search (your vectors are ~unit-norm).")

    # debug
    ap.add_argument("--write_faiss_debug", action="store_true",
                    help="Write faiss_topk_meta and faiss_anchor_idxs fields for debugging.")
    ap.add_argument("--cache_repos", type=int, default=8,
                    help="Max number of repos to cache in memory (graphs/tags/meta/faiss).")

    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True, use_fast=True)

    def count_tokens(text: str) -> int:
        return len(tok.encode(text, add_special_tokens=False))

    gt_by_id = load_codereval_index(args.codereval_json)
    graphs_dir = Path(args.graphs_dir)

    emb_dir = Path(args.emb_dir)
    faiss_dir = Path(args.faiss_dir)

    # Simple in-memory caches keyed by repo_id
    cache: Dict[str, Dict[str, Any]] = {}
    cache_order: List[str] = []

    def cache_get(repo_id: str) -> Optional[Dict[str, Any]]:
        return cache.get(repo_id)

    def cache_put(repo_id: str, obj: Dict[str, Any]) -> None:
        nonlocal cache_order
        if repo_id in cache:
            return
        cache[repo_id] = obj
        cache_order.append(repo_id)
        if len(cache_order) > args.cache_repos:
            old = cache_order.pop(0)
            cache.pop(old, None)

    with open(args.out_jsonl, "w", encoding="utf-8") as fw:
        for ex in read_jsonl(args.ce_label_jsonl):
            qid = str(ex.get("question_id", ""))
            gt = gt_by_id.get(qid)
            if gt is None:
                continue

            project = gt.get("project", "")
            if args.repo_id_style == "slash_to_triple_dash":
                repo_id = project.replace("/", "---")
            else:
                repo_id = Path(project).name

            # Load / cache repo assets
            cached = cache_get(repo_id)
            if cached is None:
                graph_path = graphs_dir / f"{repo_id}.pkl"
                tags_path = graphs_dir / f"tags_{repo_id}.json"
                if not graph_path.exists() or not tags_path.exists():
                    continue

                G = pickle.load(open(graph_path, "rb"))
                tags = json.load(open(tags_path, "r", encoding="utf-8"))
                by_idx, def_index, ref_index = build_indexes(tags)

                # Optional FAISS assets
                meta_rows = None
                index = None
                meta_path = emb_dir / repo_id / "meta.jsonl"
                index_path = faiss_dir / f"{repo_id}.flat.ip.faiss"
                if meta_path.exists() and index_path.exists():
                    meta_rows = load_meta_rows(meta_path)
                    index = faiss.read_index(str(index_path))

                cached = {
                    "G": G,
                    "tags": tags,
                    "by_idx": by_idx,
                    "def_index": def_index,
                    "ref_index": ref_index,
                    "meta_rows": meta_rows,
                    "faiss_index": index,
                }
                cache_put(repo_id, cached)

            G = cached["G"]
            tags = cached["tags"]
            by_idx = cached["by_idx"]
            def_index = cached["def_index"]
            ref_index = cached["ref_index"]
            meta_rows = cached.get("meta_rows")
            faiss_index = cached.get("faiss_index")

            signature = ex.get("signature") or ""
            model_input = ex.get("input") or ""
            func_name = extract_func_name(signature) or extract_func_name(model_input)
            search_term = func_name or ""

            anchors: List[int] = []
            faiss_debug_meta: List[Dict[str, Any]] = []
            faiss_anchor_idxs: List[int] = []
            topk_file_paths: List[str] = []

            # -------------------------
            # Anchor selection
            # -------------------------
            if args.anchors_mode in ("faiss", "faiss_then_name"):
                if args.embed_model and faiss_index is not None and meta_rows is not None:
                    qtext = model_input  # exactly CE input (signature + docstring)
                    qvec = embed_query(args.embed_base_url, args.embed_model, qtext, timeout=args.embed_timeout)
                    if args.l2_normalize_query:
                        qvec = l2_normalize(qvec)

                    # dimension check
                    if getattr(faiss_index, "d", None) is not None and faiss_index.d == qvec.shape[0]:
                        scores, ids = faiss_search_topk(faiss_index, qvec, args.faiss_topk)

                        hits = []
                        for score, mid in zip(scores, ids):
                            if mid < 0 or mid >= len(meta_rows):
                                continue
                            mr = meta_rows[mid]
                            fp0 = mr.get("file_path", "")
                            hits.append((float(score), int(mid), fp0, mr))

                        # Reorder: good paths first, bad paths later (no filtering)
                        good = [h for h in hits if not is_bad_path(h[2])]
                        bad = [h for h in hits if is_bad_path(h[2])]
                        ordered = good + bad

                        mapped: List[int] = []
                        for score, mid, fp0, mr in ordered:
                            topk_file_paths.append(fp0)

                            if args.write_faiss_debug:
                                faiss_debug_meta.append({
                                    "score": float(score),
                                    "file_path": mr.get("file_path"),
                                    "symbol": mr.get("symbol"),
                                    "snippet_type": mr.get("snippet_type"),
                                    "start_line": mr.get("start_line"),
                                    "end_line": mr.get("end_line"),
                                    "sha1": mr.get("sha1"),
                                })

                            idxs = map_meta_row_to_tag_idxs(mr, tags, gt, prefer_kind="def")
                            if not idxs:
                                idxs = map_meta_row_to_tag_idxs(mr, tags, gt, prefer_kind="")
                            mapped.extend(idxs)

                        # dedup preserve order
                        seen_a = set()
                        for a in mapped:
                            if a not in seen_a:
                                anchors.append(a)
                                seen_a.add(a)

                        if args.write_faiss_debug:
                            faiss_anchor_idxs = anchors[:]

            if (not anchors) and args.anchors_mode in ("name", "faiss_then_name"):
                if func_name:
                    anchors = def_index.get(func_name, [])
                    if not anchors:
                        anchors = ref_index.get(func_name, [])
                # else: keep empty

            if not anchors:
                # keep “only existing methods”: cannot proceed
                continue

            # -------------------------
            # RepoGraph traversal: BFS on G and G.reverse
            # -------------------------
            searcher_fwd = RepoSearcher(G)
            searcher_bwd = RepoSearcher(G.reverse(copy=False))

            visited_order: List[int] = []
            seen = set()

            # start with anchors
            for a in anchors:
                if a not in seen:
                    visited_order.append(a)
                    seen.add(a)

            for a in anchors:
                for nid in collect_neighbors_bfs(searcher_fwd, a, args.depth):
                    if nid not in seen:
                        visited_order.append(nid)
                        seen.add(nid)
                for nid in collect_neighbors_bfs(searcher_bwd, a, args.depth):
                    if nid not in seen:
                        visited_order.append(nid)
                        seen.add(nid)

            # -------------------------
            # Build ranked context blocks
            # -------------------------
            ctx_blocks: List[str] = []
            ctx_token_lens: List[int] = []

            for nid in visited_order:
                t = by_idx.get(nid)
                if not t:
                    continue

                # skip pygments backfill
                if t.get("line") == -1:
                    continue

                if (not args.keep_class_blocks) and t.get("category") == "class":
                    continue

                # remove GT region blocks
                if is_groundtruth_region(t, gt):
                    continue

                header = f"{t.get('kind')} {t.get('category')} {t.get('name')} | {t.get('rel_fname')} | {t.get('line')}"
                body = (t.get("info") or "").rstrip()
                block = header + "\n" + body

                ctx_blocks.append(block)
                if args.write_ctx_token_lens:
                    ctx_token_lens.append(count_tokens(block))

                if len(ctx_blocks) >= args.max_ctx_blocks:
                    break

            # -------------------------
            # Coverage boosting (no filtering; just add more defs)
            # -------------------------
            cur_tokens = sum(ctx_token_lens) if args.write_ctx_token_lens else sum(count_tokens(b) for b in ctx_blocks)

            if cur_tokens < args.min_ctx_tokens:
                add_same_file_defs_fallback(
                    ctx_blocks=ctx_blocks,
                    ctx_token_lens=ctx_token_lens if args.write_ctx_token_lens else [],
                    count_tokens=count_tokens,
                    tags=tags,
                    gt=gt,
                    keep_class_blocks=args.keep_class_blocks,
                    min_ctx_tokens=args.min_ctx_tokens,
                    max_defs=args.fallback_same_file_defs,
                )
                cur_tokens = sum(ctx_token_lens) if args.write_ctx_token_lens else sum(count_tokens(b) for b in ctx_blocks)

            if cur_tokens < args.min_ctx_tokens and topk_file_paths:
                add_defs_from_filepaths(
                    ctx_blocks=ctx_blocks,
                    ctx_token_lens=ctx_token_lens if args.write_ctx_token_lens else [],
                    count_tokens=count_tokens,
                    tags=tags,
                    gt=gt,
                    keep_class_blocks=args.keep_class_blocks,
                    min_ctx_tokens=args.min_ctx_tokens,
                    max_defs=args.fallback_topk_file_defs,
                    file_paths=topk_file_paths,
                )
                cur_tokens = sum(ctx_token_lens) if args.write_ctx_token_lens else sum(count_tokens(b) for b in ctx_blocks)

            if cur_tokens < args.min_ctx_tokens:
                add_defs_from_same_dir(
                    ctx_blocks=ctx_blocks,
                    ctx_token_lens=ctx_token_lens if args.write_ctx_token_lens else [],
                    count_tokens=count_tokens,
                    tags=tags,
                    gt=gt,
                    keep_class_blocks=args.keep_class_blocks,
                    min_ctx_tokens=args.min_ctx_tokens,
                    max_defs=args.fallback_same_dir_defs,
                )

            out = {
                "_id": qid,  # aligns with your inference script
                "repo_id": repo_id,
                "project": project,
                "model_input": model_input,
                "context": ctx_blocks,
                "search_term": search_term,
                "depth": args.depth,
                "anchors_mode": args.anchors_mode,
                "faiss_topk": args.faiss_topk if args.anchors_mode != "name" else 0,
            }

            if args.write_ctx_token_lens:
                out["context_tokens"] = ctx_token_lens

            if args.write_faiss_debug:
                out["faiss_topk_meta"] = faiss_debug_meta
                out["faiss_anchor_idxs"] = faiss_anchor_idxs

            fw.write(json.dumps(out, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()