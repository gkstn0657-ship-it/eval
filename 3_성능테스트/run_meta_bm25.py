# -*- coding: utf-8 -*-
"""메타데이터 BM25 실험 (계획서: 메타데이터BM25_실험계획.md).

기관 필터 없이, 기관 메타데이터를 BM25 의 검색 대상에 넣어 필터의 역할을 대신할 수 있는지 측정한다.
기존 run_eval.py 는 수정하지 않고 공통 함수(코퍼스 로딩·채점·모델)만 가져다 쓴다.

조건
  B0 hybrid_nofilter  : 기존. dense + BM25(본문) → RRF → CE. 필터 없음
  B1 hybrid_prefilter : 기존. 별칭 사전 기관 추출 → 기관 조문으로 먼저 자름 → dense + BM25 → RRF → CE
  M1 meta_concat      : BM25 색인 텍스트 앞에 "[기관] 기관명 + 규칙 변형". dense·CE 는 본문 그대로
M1 은 기관 필터와 별칭 사전을 쓰지 않는다.

사용: python run_meta_bm25.py --set dev|employee [--sub20]
"""
import os, sys, json, re, math, time
from pathlib import Path
from collections import defaultdict, Counter

# ---- 인자 (run_eval 을 import 하기 전에 떼어 둔다: run_eval 은 import 시 sys.argv 를 읽는다)
ARGS = sys.argv[1:]
def _arg(name, default=None):
    return ARGS[ARGS.index(name) + 1] if name in ARGS else default
SET = _arg("--set", "dev"); SUB20 = "--sub20" in ARGS
sys.argv = [sys.argv[0]]
os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))  # run_eval 기본값(E:\hf_cache)이 없는 PC 대응

import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_eval as R  # 읽기 전용 재사용
from rank_bm25 import BM25Okapi
sys.stdout.reconfigure(encoding="utf-8")
np.random.seed(42)

TOKEN_RE = R.TOKEN_RE
GOLD = R.ROOT / "1_데이터셋" / "06_eval" / f"goldenset_{SET}.jsonl"
ORIG20 = sorted(set(R.ORG_ALIAS.values()) - {"한국지역난방공사"})  # 기존 실험의 20개 기관

# ---------------------------------------------------------------- 메타데이터 색인
def variants(org):
    """기관명의 규칙 변형. 사전 없이 접두·접미 법인 표기와 '한국' 만 뗀다."""
    v = {org}
    s = re.sub(r"^\((주|사|재)\)\s*", "", org)
    s = re.sub(r"\(주\)$", "", s)
    s = re.sub(r"^주식회사\s*", "", s); s = re.sub(r"\s*주식회사$", "", s).strip()
    v.add(s)
    if s.startswith("한국") and len(s) > 4: v.add(s[2:])
    return sorted(x for x in v if x)

class MetaIndex:
    def __init__(self, c):
        self.org_list = sorted(set(c.orgs.tolist()))
        self.concat_bm25 = BM25Okapi([TOKEN_RE.findall(("[기관] " + " ".join(variants(r["org"])) + " " + r["text"]).lower()) for r in c.rows])

# ---------------------------------------------------------------- 파이프라인
def _fuse(c, q, b_scores, ce_text):
    qv = R.st_model("BAAI/bge-m3").encode([q], normalize_embeddings=True, convert_to_numpy=True)[0]
    d = c.bge @ qv
    d_rank = {int(i): r for r, i in enumerate(np.argsort(-d)[:R.CANDIDATE_K])}
    b_rank = {int(i): r for r, i in enumerate(np.argsort(-b_scores)[:R.CANDIDATE_K])}
    rrf = defaultdict(float)
    for i, r in d_rank.items(): rrf[i] += 1 / (R.RRF_K + r)
    for i, r in b_rank.items(): rrf[i] += 1 / (R.RRF_K + r)
    cand = sorted(rrf, key=lambda i: (-rrf[i], i))
    pool, rest = cand[:R.RERANK_TOP_N], cand[R.RERANK_TOP_N:]
    ce = R.ce_model().predict([(q, ce_text(i)) for i in pool], batch_size=16, show_progress_bar=False) if pool else []
    ce_map = {i: float(s) for i, s in zip(pool, ce)}
    pool.sort(key=lambda i: (-ce_map[i], -rrf[i], i))
    return pool + rest, {"top1_ce": ce_map[pool[0]] if pool else None}

def run_m1(c, mi, q): return _fuse(c, q, mi.concat_bm25.get_scores(TOKEN_RE.findall(q.lower())), lambda i: c.texts[i])

# ---------------------------------------------------------------- 실행
def load_corpus():
    c = R.Corpus("L2P_A")
    if SUB20:
        keep = [k for k, o in enumerate(c.orgs) if o in ORIG20]
        missing = set(ORIG20) - set(c.orgs.tolist())
        assert not missing, f"20개 기관 중 코퍼스에 없는 기관: {missing}"
        c.rows = [c.rows[k] for k in keep]; c.texts = [c.texts[k] for k in keep]; c.orgs = c.orgs[keep]
        c.org_names = sorted(set(c.orgs.tolist()))
    return c

def main():
    t0 = time.time()
    gold = [json.loads(l) for l in GOLD.read_text(encoding="utf-8").splitlines() if l.strip()]
    c = load_corpus(); mi = MetaIndex(c)
    tag = f"{'org20' if SUB20 else 'org292'}_{SET}"
    out = HERE / "results_meta_bm25" / tag; out.mkdir(parents=True, exist_ok=True)
    print(f"[{tag}] 청크 {len(c.rows)}, 기관 {len(mi.org_list)}, 문항 {len(gold)}", flush=True)
    conds = {"B0_hybrid_nofilter": None, "B1_hybrid_prefilter": None, "M1_meta_concat": run_m1}
    per_query = []
    for name, fn in conds.items():
        for q in gold:
            ctx = q.get("context_org")
            q_text = q["query"] + (f" {ctx}" if ctx else "")  # employee: 소속 기관을 텍스트로 전달 (B0·M 공통)
            if name.startswith("B1"):
                R._CTX["org"] = ctx; idx, info = R.run_hybrid_prefilter(c, q["query"]); R._CTX["org"] = None
            elif name.startswith("B0"):
                idx, info = R.run_hybrid_nofilter(c, q_text)
            else:
                idx, info = fn(c, mi, q_text)
            arts = R.to_articles(idx, c.rows)
            rec = {"cond": name, "qid": q["qid"], "type": q["type"], "gold": q["gold_ids"], "top5": arts[:5],
                   "top_org": info.get("top_org"), "top1_ce": info.get("top1_ce")}
            if q["type"] != "Q9":
                rec.update(R.metrics(arts, q["gold_ids"]))
                gorg = {g.split("|")[0] for g in q["gold_ids"]}
                rec["org_hit5"] = sum(a.split("|")[0] in gorg for a in arts[:5]) / max(1, len(arts[:5]))
            per_query.append(rec)
        print(f"  {name} 완료 ({time.time() - t0:.0f}s)", flush=True)
    json.dump(per_query, open(out / "per_query.json", "w", encoding="utf-8"), ensure_ascii=False)

    def rows_of(n): return sorted([r for r in per_query if r["cond"] == n and r["type"] != "Q9"], key=lambda r: r["qid"])
    def mean(v): return sum(v) / len(v) if v else None
    def boot(a, b, n=2000):
        rng = np.random.default_rng(42); a, b = np.array(a), np.array(b)
        diffs = [(a[s] - b[s]).mean() for s in (rng.integers(0, len(a), len(a)) for _ in range(n))]
        return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
    summ = {"tag": tag, "chunks": len(c.rows), "orgs": len(mi.org_list), "n_scored": len(rows_of(next(iter(conds)))),
            "conditions": [], "by_type": [], "comparisons": []}
    for n in conds:
        rs = rows_of(n)
        summ["conditions"].append({"cond": n, **{k: mean([r[k] for r in rs]) for k in ("ndcg5", "recall5", "mrr", "org_hit5")}})
        for t in sorted({r["type"] for r in rs}, key=lambda x: int(x[1:])):
            tr = [r for r in rs if r["type"] == t]
            summ["by_type"].append({"cond": n, "type": t, "n": len(tr), "recall5": mean([r["recall5"] for r in tr]), "ndcg5": mean([r["ndcg5"] for r in tr])})
    for m in [k for k in conds if k.startswith("M")]:
        for base in [k for k in conds if k.startswith("B")]:
            a, b = [r["ndcg5"] for r in rows_of(m)], [r["ndcg5"] for r in rows_of(base)]
            lo, hi = boot(a, b)
            summ["comparisons"].append({"label": f"{m} vs {base}", "delta": mean(a) - mean(b), "ci95": [lo, hi], "significant": lo > 0 or hi < 0})
    json.dump(summ, open(out / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n=== {tag} (채점 {summ['n_scored']}문항) ===")
    for x in summ["conditions"]:
        print(f"{x['cond']:22} nDCG@5 {x['ndcg5']:.3f}  R@5 {x['recall5']:.3f}  MRR {x['mrr']:.3f}  기관적중 {x['org_hit5']:.2f}")
    for x in summ["comparisons"]:
        print(f"{x['label']:42} Δ {x['delta']:+.3f} CI [{x['ci95'][0]:+.3f}, {x['ci95'][1]:+.3f}] {'유의' if x['significant'] else ''}")

if __name__ == "__main__":
    main()
