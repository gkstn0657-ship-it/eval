# -*- coding: utf-8 -*-
"""1006 실험: 세트별 결과(results_chunk_<세트>[_expand]/per_query.json)를 모아 합산 비교한다.
쌍대 부트스트랩은 (세트, 문항) 단위로 짝을 맞춘다. 계획서 1006_청킹비교_실험계획.md 의 5단계.
사용: python chunk_pool.py [--expand]
"""
import json, sys
from pathlib import Path
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
SETS = ["dev", "employee", "kgs_faq", "matrix"]
BASE, VARIANTS, PIPE = "L2P_A", ["CH_FIX_A", "CH_SEM_A", "CH_PARA_A"], "hybrid_prefilter"
suffix = "_expand" if "--expand" in sys.argv else ""
np.random.seed(42)

scores = {}  # (corpus) -> {(set, qid): {metric: v}}
for st in SETS:
    f = HERE / f"results_chunk_{st}{suffix}" / "per_query.json"
    if not f.exists(): sys.exit(f"없음: {f}")
    for r in json.load(open(f, encoding="utf-8")):
        if r["pipeline"] != PIPE or r["type"] == "Q9": continue
        scores.setdefault(r["corpus"], {})[(st, r["qid"])] = {k: r[k] for k in ("ndcg5", "recall5", "mrr")}

def boot(a, b, n=2000):
    a, b = np.array(a), np.array(b); d = []
    for _ in range(n):
        s = np.random.randint(0, len(a), len(a)); d.append((a[s] - b[s]).mean())
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

keys = sorted(scores[BASE])
out = {"scoring": "expand" if suffix else "main_article", "n": len(keys), "conditions": [], "comparisons": [], "by_set": []}
for c in [BASE] + VARIANTS:
    out["conditions"].append({"corpus": c, **{m: float(np.mean([scores[c][k][m] for k in keys])) for m in ("ndcg5", "recall5", "mrr")}})
for v in VARIANTS:
    a = [scores[v][k]["ndcg5"] for k in keys]; b = [scores[BASE][k]["ndcg5"] for k in keys]
    lo, hi = boot(a, b)
    out["comparisons"].append({"a": v, "b": BASE, "delta_ndcg5": float(np.mean(a) - np.mean(b)), "ci95": [lo, hi], "significant": lo > 0 or hi < 0})
    for st in SETS:
        ks = [k for k in keys if k[0] == st]
        a = [scores[v][k]["ndcg5"] for k in ks]; b = [scores[BASE][k]["ndcg5"] for k in ks]; lo, hi = boot(a, b)
        out["by_set"].append({"set": st, "a": v, "n": len(ks), "delta_ndcg5": float(np.mean(a) - np.mean(b)), "ci95": [lo, hi], "significant": lo > 0 or hi < 0})

(HERE / f"results_chunk_pooled{suffix}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"=== 합산 n={out['n']} ({out['scoring']}) ===")
for c in out["conditions"]: print(f"{c['corpus']:10} nDCG@5 {c['ndcg5']:.3f}  R@5 {c['recall5']:.3f}  MRR {c['mrr']:.3f}")
for c in out["comparisons"]: print(f"{c['a']} vs {c['b']}: Δ {c['delta_ndcg5']:+.3f} CI [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}] {'유의' if c['significant'] else ''}")
print("--- 세트별 ---")
for c in out["by_set"]: print(f"{c['set']:9} {c['a']:10} n={c['n']:3} Δ {c['delta_ndcg5']:+.3f} CI [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}] {'유의' if c['significant'] else ''}")
