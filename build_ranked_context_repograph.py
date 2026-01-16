#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import pickle
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from transformers import AutoTokenizer

# 使用你给的 RepoGraph 现有 BFS 方法思想：图 + bfs
from repograph.graph_searcher import RepoSearcher


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
    return p.replace("\\", "/").lstrip("./")


def file_match(tag_rel_fname: str, gt_file_path: str) -> bool:
    # repo 内相对路径可能带 repo_name 前缀/不同层级，做后缀匹配最稳
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
    更严格的 GT 去除：
    - 同文件（file_path 后缀匹配）
    - 并且 tag 的行区间与 [gt.lineno, gt.end_lineno] 重叠（考虑 0/1-based 两种可能）
    - 或者 info 里包含 def <name>( 也视为 GT 块（兜底）
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
        # 没行号就只能按 file+def+name 去除（弱一些）
        return (tag.get("kind") == "def") and (tag.get("name") == name)

    gt_range = (min(gt_l0, gt_l1), max(gt_l0, gt_l1))
    tr = tag_line_range(tag.get("line"))
    if tr is None:
        return False

    # 你的 tag.line 可能是 0-based 或 1-based；两种都试一次
    if overlap(tr, gt_range):
        return True
    tr_plus1 = (tr[0] + 1, tr[1] + 1)
    if overlap(tr_plus1, gt_range):
        return True

    return False


def collect_neighbors_bfs(searcher: RepoSearcher, anchor: int, depth: int) -> List[int]:
    # 直接复用 RepoSearcher.bfs 的语义（返回 visited 列表）
    # 这里保留顺序：bfs 返回的 visited 是按层次扩展加入的
    return searcher.bfs(anchor, depth)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--ce_label_jsonl", required=True, help="CEPythonHumanLabel.jsonl")
    ap.add_argument("--codereval_json", required=True, help="CoderEval4Python.json")
    ap.add_argument("--graphs_dir", required=True, help="dir containing <repo_id>.pkl and tags_<repo_id>.json")
    ap.add_argument("--out_jsonl", required=True, help="output dataset jsonl for your inference script")

    ap.add_argument("--tokenizer_path", default="/workspace/models/Qwen3-Coder-30B-A3B-Instruct",
                    help="Qwen3 tokenizer local path")
    ap.add_argument("--depth", type=int, default=2, help="BFS depth (k-hop)")
    ap.add_argument("--max_ctx_blocks", type=int, default=5000, help="cap number of context blocks per sample")
    ap.add_argument("--keep_class_blocks", action="store_true",
                    help="keep class category blocks (default: drop class blocks because they are method-name lists)")
    ap.add_argument("--write_ctx_token_lens", action="store_true",
                    help="also write context_tokens(list[int]) for debugging")

    # repo_id mapping
    ap.add_argument("--repo_id_style", choices=["slash_to_triple_dash", "basename"], default="slash_to_triple_dash",
                    help="how to map CoderEval 'project' to repo_id file names")

    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True, use_fast=True)

    def count_tokens(text: str) -> int:
        return len(tok.encode(text, add_special_tokens=False))

    gt_by_id = load_codereval_index(args.codereval_json)
    graphs_dir = Path(args.graphs_dir)

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

            graph_path = graphs_dir / f"{repo_id}.pkl"
            tags_path = graphs_dir / f"tags_{repo_id}.json"
            if not graph_path.exists() or not tags_path.exists():
                continue

            G = pickle.load(open(graph_path, "rb"))
            tags = json.load(open(tags_path, "r", encoding="utf-8"))

            by_idx, def_index, ref_index = build_indexes(tags)

            signature = ex.get("signature") or ""
            model_input = ex.get("input") or ""
            func_name = extract_func_name(signature) or extract_func_name(model_input)
            if not func_name:
                continue

            anchors = def_index.get(func_name, [])
            if not anchors:
                # def 找不到就退化到 ref anchors（仍然是 RepoGraph 图节点）
                anchors = ref_index.get(func_name, [])
            if not anchors:
                # 完全找不到就只能跳过（保持“只用 RepoGraph 方法”，不引入别的检索）
                continue

            # RepoGraph 检索：BFS on G and BFS on G.reverse
            searcher_fwd = RepoSearcher(G)
            searcher_bwd = RepoSearcher(G.reverse(copy=False))

            visited_order: List[int] = []
            seen = set()

            # 先放 anchors（稳定顺序）
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

            ctx_blocks: List[str] = []
            ctx_token_lens: List[int] = []

            for nid in visited_order:
                t = by_idx.get(nid)
                if not t:
                    continue

                # 过滤 pygments backfill 这类 line=-1
                if t.get("line") == -1:
                    continue

                # 默认丢掉 class blocks：你现在 class 的 info 是 method name 列表，不是代码
                if (not args.keep_class_blocks) and t.get("category") == "class":
                    continue

                # 去除 ground truth 范围内的任何 tag（def/ref）以防泄漏
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

            out = {
                "_id": qid,  # ✅ 对齐你的推理脚本：rec.get("_id")
                "repo_id": repo_id,
                "project": project,
                "model_input": model_input,  # ✅ 对齐你的推理脚本：rec.get("model_input")
                "context": ctx_blocks,        # ✅ List[str]，由你的脚本按 budget 顺序截取
                "search_term": func_name,
                "depth": args.depth,
            }
            if args.write_ctx_token_lens:
                out["context_tokens"] = ctx_token_lens

            fw.write(json.dumps(out, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
