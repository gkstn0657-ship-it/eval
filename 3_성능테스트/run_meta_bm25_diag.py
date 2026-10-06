# -*- coding: utf-8 -*-
"""메타데이터 BM25 2차: 원인 분석 실험 (계획서: 메타데이터BM25_실험계획.md 의 '2차').

기관명을 BM25 에 넣어도 기관 필터를 대신하지 못하는 원인을 단계별로 쪼개 검증한다.
기존 run_eval.py, run_meta_bm25.py 는 수정하지 않고 가져다 쓴다.

사용: python run_meta_bm25_diag.py --set dev|employee [--sub20] [--corpus L2P_A|L2_A]
"""
import os, sys, json, time
from pathlib import Path
from collections import defaultdict

ARGS = sys.argv[1:]
def _arg(n, d=None): return ARGS[ARGS.index(n) + 1] if n in ARGS else d
SET = _arg("--set", "dev"); SUB20 = "--sub20" in ARGS; CORPUS = _arg("--corpus", "L2P_A")
sys.argv = [sys.argv[0], "--corpus", CORPUS] + (["--sub20"] if SUB20 else [])  # run_meta_bm25 의 load_corpus 설정용
os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import run_meta_bm25 as M   # variants, MetaIndex, load_corpus 재사용
R = M.R
sys.stdout.reconfigure(encoding="utf-8")
TOK = R.TOKEN_RE

def org_vocab(c):
    """색인 쪽 기관명 토큰 집합 (사전 없음: 규칙 변형의 토큰)."""
    v = set()
    for o in set(c.orgs.tolist()):
        for x in M.variants(o): v.update(TOK.findall(x.lower()))
    return v

def bm25_query(q, vocab, k):
    toks = TOK.findall(q.lower()); org_t = [t for t in toks if t in vocab]
    return toks + org_t * (k - 1), org_t

def pipeline(c, mi, vocab, q, k=1, use_dense=True, ce_org=False):
    """기관 필터 없음. BM25(기관명 포함 색인, 기관 토큰 k배) [+ dense] → RRF → CE. 후보 풀과 최종 순위를 함께 돌려준다."""
    toks, _ = bm25_query(q, vocab, k)
    b = mi.concat_bm25.get_scores(toks)
    b_rank = {int(i): r for r, i in enumerate(np.argsort(-b)[:R.CANDIDATE_K])}
    rrf = defaultdict(float)
    for i, r in b_rank.items(): rrf[i] += 1 / (R.RRF_K + r)
    if use_dense:
        qv = R.st_model("BAAI/bge-m3").encode([q], normalize_embeddings=True, convert_to_numpy=True)[0]
        d = c.bge @ qv
        for r, i in enumerate(np.argsort(-d)[:R.CANDIDATE_K]): rrf[int(i)] += 1 / (R.RRF_K + r)
    cand = sorted(rrf, key=lambda i: (-rrf[i], i))
    pool, rest = cand[:R.RERANK_TOP_N], cand[R.RERANK_TOP_N:]
    txt = (lambda i: f"[기관] {c.rows[i]['org']}\n{c.texts[i]}") if ce_org else (lambda i: c.texts[i])
    ce = R.ce_model().predict([(q, txt(i)) for i in pool], batch_size=16, show_progress_bar=False) if pool else []
    cm = {i: float(s) for i, s in zip(pool, ce)}
    final = sorted(pool, key=lambda i: (-cm[i], -rrf[i], i)) + rest
    return list(pool), final

def main():
    t0 = time.time()
    gold = [json.loads(l) for l in (R.ROOT / "1_데이터셋" / "06_eval" / f"goldenset_{SET}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    gold = [q for q in gold if q["type"] != "Q9"]
    c = M.load_corpus(); mi = M.MetaIndex(c); vocab = org_vocab(c)
    tag = f"{'org20' if SUB20 else 'org292'}_{SET}_{CORPUS}"
    out = HERE / "results_meta_bm25_diag" / tag; out.mkdir(parents=True, exist_ok=True)
    print(f"[{tag}] 청크 {len(c.rows)}, 기관 {len(set(c.orgs.tolist()))}, 채점 {len(gold)}", flush=True)
    E = {"E0_nofilter": None,
         "E1_bm25_org": dict(k=1), "E2a_org_x3": dict(k=3), "E2b_org_x10": dict(k=10),
         "E3_bm25_only": dict(k=1, use_dense=False), "E4_ce_org": dict(k=1, ce_org=True),
         "E5_all": dict(k=10, use_dense=False, ce_org=True), "B1_prefilter": None}
    rows, diag = [], []
    for name, kw in E.items():
        for q in gold:
            ctx = q.get("context_org"); qt = q["query"] + (f" {ctx}" if ctx else "")
            gid = set(q["gold_ids"]); gorg = {g.split("|")[0] for g in gid}
            pool = None
            if name == "E0_nofilter": final, _ = R.run_hybrid_nofilter(c, qt)
            elif name == "B1_prefilter":
                R._CTX["org"] = ctx; final, _ = R.run_hybrid_prefilter(c, q["query"]); R._CTX["org"] = None
            else: pool, final = pipeline(c, mi, vocab, qt, **kw)
            arts = R.to_articles(final, c.rows); m = R.metrics(arts, q["gold_ids"])
            rec = {"cond": name, "qid": q["qid"], "type": q["type"], **m,
                   "org_hit5": sum(a.split("|")[0] in gorg for a in arts[:5]) / max(1, len(arts[:5]))}
            if pool is not None:
                rec["pool_gold"] = float(any(c.rows[i]["article_id"] in gid for i in pool))
                rec["pool_org"] = float(np.mean([c.orgs[i] in gorg for i in pool]))
            rows.append(rec)
        print(f"  {name} 완료 ({time.time()-t0:.0f}s)", flush=True)

    # H1·H2 진단 (E1 색인, 가중 없음)
    for q in gold:
        ctx = q.get("context_org"); qt = q["query"] + (f" {ctx}" if ctx else "")
        toks, org_t = bm25_query(qt, vocab, 1)
        gid = set(q["gold_ids"]); gorg = {g.split("|")[0] for g in gid}
        full = mi.concat_bm25.get_scores(toks)
        orgpart = mi.concat_bm25.get_scores(org_t) if org_t else np.zeros(len(full))
        gi = [i for i, r in enumerate(c.rows) if r["article_id"] in gid]
        if not gi: continue
        g = max(gi, key=lambda i: full[i])
        diag.append({"qid": q["qid"], "org_tokens": org_t, "org_lexical_match": bool(org_t),
                     "gold_org_token_share": float(orgpart[g] / full[g]) if full[g] > 0 else 0.0,
                     "other_org_above_gold": int(sum(1 for i in range(len(full)) if full[i] > full[g] and c.orgs[i] not in gorg)),
                     "gold_bm25_rank": int((full > full[g]).sum()) + 1})
    json.dump({"rows": rows, "diag": diag}, open(out / "per_query.json", "w", encoding="utf-8"), ensure_ascii=False)

    def mean(v): v = [x for x in v if x is not None]; return sum(v) / len(v) if v else None
    def ser(n): return [r["ndcg5"] for r in sorted([r for r in rows if r["cond"] == n], key=lambda r: r["qid"])]
    def boot(a, b, n=2000):
        rng = np.random.default_rng(42); a, b = np.array(a), np.array(b)
        d = [(a[s] - b[s]).mean() for s in (rng.integers(0, len(a), len(a)) for _ in range(n))]
        return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    summ = {"tag": tag, "chunks": len(c.rows), "n": len(gold), "conditions": [], "comparisons": [], "diag": {}}
    for n in E:
        rs = [r for r in rows if r["cond"] == n]
        summ["conditions"].append({"cond": n, **{k: mean([r.get(k) for r in rs]) for k in ("ndcg5", "recall5", "mrr", "org_hit5", "pool_gold", "pool_org")}})
    for a, b in [("E1_bm25_org", "E0_nofilter"), ("E2b_org_x10", "E1_bm25_org"), ("E3_bm25_only", "E1_bm25_org"), ("E4_ce_org", "E1_bm25_org"),
                 ("E5_all", "E0_nofilter"), ("E5_all", "B1_prefilter"), ("E4_ce_org", "B1_prefilter")]:
        lo, hi = boot(ser(a), ser(b))
        summ["comparisons"].append({"label": f"{a} vs {b}", "delta": mean(ser(a)) - mean(ser(b)), "ci95": [lo, hi], "significant": lo > 0 or hi < 0})
    summ["diag"] = {"org_lexical_match_rate": mean([float(d["org_lexical_match"]) for d in diag]),
                    "gold_org_token_share_median": float(np.median([d["gold_org_token_share"] for d in diag if d["org_lexical_match"]] or [0])),
                    "other_org_above_gold_median": float(np.median([d["other_org_above_gold"] for d in diag])),
                    "gold_bm25_rank_median": float(np.median([d["gold_bm25_rank"] for d in diag])),
                    "n_diag": len(diag)}
    json.dump(summ, open(out / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n=== {tag} ===")
    f = lambda x: "  –  " if x is None else f"{x:.3f}"
    print("조건              nDCG@5  R@5    기관적중5  후보:정답포함  후보:정답기관")
    for x in summ["conditions"]:
        print(f"{x['cond']:17} {f(x['ndcg5'])}  {f(x['recall5'])}  {f(x['org_hit5'])}     {f(x['pool_gold'])}        {f(x['pool_org'])}")
    for x in summ["comparisons"]:
        print(f"{x['label']:28} Δ {x['delta']:+.3f} CI [{x['ci95'][0]:+.3f}, {x['ci95'][1]:+.3f}] {'유의' if x['significant'] else ''}")
    print("진단:", summ["diag"])

if __name__ == "__main__":
    main()
