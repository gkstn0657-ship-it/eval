# -*- coding: utf-8 -*-
"""골든셋(dev)으로 두 파이프라인 구조를 6개 코퍼스(L0/L1/L2 × A/B)에서 측정한다.

파이프라인 (원 레포의 구성요소·상수를 그대로 사용)
- vanilla : chat_rag. jhgan/ko-sbert-nli 임베딩, 정규화 내적, 단일 벡터 검색, SIMILARITY_THRESHOLD 0.5
- hybrid  : hybrid-search-eval. BAAI/bge-m3 dense(top 30) + BM25(top 30, 토크나이저 [0-9A-Za-z가-힣]+)
            → RRF(k=60) → 기관 하드필터(원 레포의 지역·연차 필터에 대응, 결과 5개 미만이면 완화)
            → bge-reranker-v2-m3(max_length 384) 상위 30 재랭킹
베이스라인: random, bm25 단독, dense(bge-m3) 단독
지표: Recall@5, MRR, nDCG@5 (조문 ID 단위, 조각 #k 와 중복 접미 @n 은 본문 첫 조문만 정답). Q9 는 거절률로 별도 집계.
정책 B 는 같은 기관·같은 조번호의 다른 개정판도 정답으로 인정(gold_ids_any_revision).
"""
import json, re, sys, hashlib, random, time, math
from pathlib import Path
from collections import defaultdict
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
GOLD = ROOT / "1_데이터셋" / "06_eval" / ("goldenset_holdout_sealed.jsonl" if "--holdout" in sys.argv else "goldenset_dev.jsonl")
OUT = HERE / ("results_holdout" if "--holdout" in sys.argv else "results"); OUT.mkdir(exist_ok=True)
CACHE = HERE / "cache"; CACHE.mkdir(exist_ok=True)
CORPORA = ["L0_A", "L0_B", "L1_A", "L1_B", "L2_A", "L2_B"]
TOP_K, RRF_K, CANDIDATE_K, RERANK_TOP_N, MIN_RESULTS, SIM_THRESHOLD, CE_MAX_LEN = 10, 60, 30, 30, 5, 0.5, 384
TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣]+")
random.seed(42); np.random.seed(42)

ORG_ALIAS = {  # 질의 속 기관 표현 → 코퍼스 org. 긴 표현부터 매칭
    "인천국제공항공사": "인천국제공항공사", "인천공항": "인천국제공항공사", "한국공항공사": "한국공항공사", "공항공사": "한국공항공사",
    "제주국제자유도시개발센터": "제주국제자유도시개발센터", "제주개발센터": "제주국제자유도시개발센터", "JDC": "제주국제자유도시개발센터",
    "주택도시보증공사": "주택도시보증공사", "HUG": "주택도시보증공사", "한국토지주택공사": "한국토지주택공사", "토지주택공사": "한국토지주택공사", "LH": "한국토지주택공사",
    "한국가스기술공사": "(주)한국가스기술공사", "가스기술공사": "(주)한국가스기술공사", "한국가스공사": "한국가스공사", "가스공사": "한국가스공사",
    "한국전력기술": "한국전력기술주식회사", "전력기술": "한국전력기술주식회사", "한전KDN": "한전KDN", "KDN": "한전KDN",
    "한국광해광업공단": "한국광해광업공단", "광해광업공단": "한국광해광업공단", "광해": "한국광해광업공단",
    "한국석유공사": "한국석유공사", "석유공사": "한국석유공사", "한국도로공사": "한국도로공사", "도로공사": "한국도로공사",
    "한국철도공사": "한국철도공사", "철도공사": "한국철도공사", "한국수자원공사": "한국수자원공사", "수자원공사": "한국수자원공사",
    "한국마사회": "한국마사회", "마사회": "한국마사회", "한국부동산원": "한국부동산원", "부동산원": "한국부동산원",
    "해양환경공단": "해양환경공단", "강원랜드": "(주)강원랜드", "그랜드코리아레저": "그랜드코리아레저(주)", "GKL": "그랜드코리아레저(주)",
    "에스알": "주식회사 에스알", "SR": "주식회사 에스알",
}
def detect_org(q):
    for k in sorted(ORG_ALIAS, key=len, reverse=True):
        if k in q: return ORG_ALIAS[k]
    return None

# ---------------------------------------------------------------- 모델 (지연 로딩)
_models = {}
def st_model(name):
    if name not in _models:
        from sentence_transformers import SentenceTransformer
        _models[name] = SentenceTransformer(name, device="cuda")
    return _models[name]
def ce_model():
    if "ce" not in _models:
        from sentence_transformers import CrossEncoder
        _models["ce"] = CrossEncoder("BAAI/bge-reranker-v2-m3", device="cuda", max_length=CE_MAX_LEN)
    return _models["ce"]

def embed_cached(model_name, texts, tag):
    """텍스트 해시 단위 캐시. 같은 조문이 A/B 코퍼스에 겹치므로 재계산을 피한다."""
    f = CACHE / f"{tag}.npz"
    cache = dict(np.load(f, allow_pickle=True)["d"].item()) if f.exists() else {}
    keys = [hashlib.sha1(t.encode()).hexdigest() for t in texts]
    todo = [t for t, k in zip(texts, keys) if k not in cache]
    if todo:
        vecs = st_model(model_name).encode(todo, batch_size=32, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
        for t, v in zip(todo, vecs): cache[hashlib.sha1(t.encode()).hexdigest()] = v.astype(np.float32)
        np.savez_compressed(f, d=np.array(cache, dtype=object))
    return np.stack([cache[k] for k in keys])

# ---------------------------------------------------------------- 지표
def to_articles(ranked_chunk_idx, rows):
    seen, out = set(), []
    for i in ranked_chunk_idx:
        a = rows[i]["article_id"]
        if a not in seen: seen.add(a); out.append(a)
    return out
def metrics(ranked_articles, gold):
    gold = set(gold); hits = [1 if a in gold else 0 for a in ranked_articles[:TOP_K]]
    first = next((i for i, h in enumerate(hits) if h), None)
    dcg = sum(h / math.log2(i + 2) for i, h in enumerate(hits[:5]))
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(gold), 5)))
    return {"recall5": float(any(hits[:5])), "recall10": float(any(hits)), "mrr": 1 / (first + 1) if first is not None else 0.0,
            "ndcg5": dcg / idcg if idcg else 0.0, "first_rank": first + 1 if first is not None else None}

# ---------------------------------------------------------------- 검색기
class Corpus:
    def __init__(self, name):
        self.name = name
        self.rows = [json.loads(l) for l in (HERE / "corpora" / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()]
        self.texts = [r["text"] for r in self.rows]
        self.orgs = np.array([r["org"] for r in self.rows])
        self._bm25 = None; self._sbert = None; self._bge = None
    @property
    def sbert(self):
        if self._sbert is None: self._sbert = embed_cached("jhgan/ko-sbert-nli", self.texts, "ko-sbert-nli")
        return self._sbert
    @property
    def bge(self):
        if self._bge is None: self._bge = embed_cached("BAAI/bge-m3", self.texts, "bge-m3")
        return self._bge
    @property
    def bm25(self):
        if self._bm25 is None:
            from rank_bm25 import BM25Okapi
            self._bm25 = BM25Okapi([TOKEN_RE.findall(t.lower()) for t in self.texts])
        return self._bm25

def run_vanilla(c: Corpus, q):
    qv = st_model("jhgan/ko-sbert-nli").encode([q], normalize_embeddings=True, convert_to_numpy=True)[0]
    s = c.sbert @ qv; idx = np.argsort(-s)[:TOP_K * 3]
    return list(idx), {"top1_score": float(s[idx[0]]), "abstain": bool(s[idx[0]] < SIM_THRESHOLD)}

def run_vanilla_bge(c: Corpus, q):
    """Vanilla 와 같은 경로(단일 벡터, 정규화 내적, 임계값 0.5). 임베딩 모델만 bge-m3 로 교체. 사전등록 후 추가된 조건(v2)."""
    qv = st_model("BAAI/bge-m3").encode([q], normalize_embeddings=True, convert_to_numpy=True)[0]
    s = c.bge @ qv; idx = np.argsort(-s)[:TOP_K * 3]
    return list(idx), {"top1_score": float(s[idx[0]]), "abstain": bool(s[idx[0]] < SIM_THRESHOLD)}

def run_dense(c: Corpus, q):
    qv = st_model("BAAI/bge-m3").encode([q], normalize_embeddings=True, convert_to_numpy=True)[0]
    s = c.bge @ qv; idx = np.argsort(-s)[:TOP_K * 3]
    return list(idx), {"top1_score": float(s[idx[0]])}

def run_bm25(c: Corpus, q):
    s = c.bm25.get_scores(TOKEN_RE.findall(q.lower())); idx = np.argsort(-s)[:TOP_K * 3]
    return list(idx), {"top1_score": float(s[idx[0]])}

def run_hybrid(c: Corpus, q):
    qv = st_model("BAAI/bge-m3").encode([q], normalize_embeddings=True, convert_to_numpy=True)[0]
    d = c.bge @ qv; b = c.bm25.get_scores(TOKEN_RE.findall(q.lower()))
    d_rank = {int(i): r for r, i in enumerate(np.argsort(-d)[:CANDIDATE_K])}
    b_rank = {int(i): r for r, i in enumerate(np.argsort(-b)[:CANDIDATE_K])}
    rrf = defaultdict(float)
    for i, r in d_rank.items(): rrf[i] += 1 / (RRF_K + r)
    for i, r in b_rank.items(): rrf[i] += 1 / (RRF_K + r)
    org = detect_org(q)
    cand = sorted(rrf, key=lambda i: (-rrf[i], i))
    exact = [i for i in cand if org is None or c.orgs[i] == org]
    relaxed = [i for i in cand if i not in exact]
    if org is not None and len(exact) < MIN_RESULTS: tiers = exact + relaxed  # 완화
    else: tiers = exact + relaxed
    pool, rest = tiers[:RERANK_TOP_N], tiers[RERANK_TOP_N:]
    ce = ce_model().predict([(q, c.texts[i]) for i in pool], batch_size=16, show_progress_bar=False) if pool else []
    ce_map = {i: float(s) for i, s in zip(pool, ce)}
    pool.sort(key=lambda i: (not (org is None or c.orgs[i] == org), -ce_map[i], -rrf[i], i))
    top = pool + rest
    return top, {"org_filter": org, "exact_in_pool": sum(1 for i in pool if org is None or c.orgs[i] == org),
                 "top1_ce": ce_map[top[0]] if top else None, "abstain": bool(top and ce_map.get(top[0], 0) < 0)}

def run_random(c: Corpus, q):
    idx = list(np.random.permutation(len(c.rows))[:TOP_K * 3]); return idx, {}

PIPELINES = {"vanilla": run_vanilla, "vanilla_bge": run_vanilla_bge, "hybrid": run_hybrid, "baseline_dense": run_dense, "baseline_bm25": run_bm25, "baseline_random": run_random}

# ---------------------------------------------------------------- 실행
gold = [json.loads(l) for l in GOLD.read_text(encoding="utf-8").splitlines() if l.strip()]
per_query, t0 = [], time.time()
for cname in CORPORA:
    c = Corpus(cname); policy = cname[-1]
    for pname, fn in PIPELINES.items():
        for q in gold:
            idx, info = fn(c, q["query"])
            arts = to_articles(idx, c.rows)
            g = q["gold_ids_any_revision"] if policy == "B" else q["gold_ids"]
            m = metrics(arts, g) if q["type"] != "Q9" else {}
            per_query.append({"corpus": cname, "pipeline": pname, "qid": q["qid"], "type": q["type"], "gold": g,
                              "top5": arts[:5], **m, **info})
        print(f"{cname} {pname} done ({time.time()-t0:.0f}s)")
json.dump(per_query, open(OUT / "per_query.json", "w", encoding="utf-8"), ensure_ascii=False)

# 집계 + 쌍대 부트스트랩
def agg(rows, key):
    v = [r[key] for r in rows if key in r and r[key] is not None]; return sum(v) / len(v) if v else None
def boot_ci(a, b, n=2000):
    a, b = np.array(a), np.array(b); diffs = []
    for _ in range(n):
        s = np.random.randint(0, len(a), len(a)); diffs.append((a[s] - b[s]).mean())
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
summary = {"n_queries": len(gold), "n_scored": sum(1 for q in gold if q["type"] != "Q9"), "conditions": [], "by_type": [], "q9": [], "comparisons": []}
for cname in CORPORA:
    for pname in PIPELINES:
        rows = [r for r in per_query if r["corpus"] == cname and r["pipeline"] == pname and r["type"] != "Q9"]
        summary["conditions"].append({"corpus": cname, "pipeline": pname, "recall5": agg(rows, "recall5"), "recall10": agg(rows, "recall10"), "mrr": agg(rows, "mrr"), "ndcg5": agg(rows, "ndcg5")})
        for t in sorted({r["type"] for r in rows}, key=lambda x: int(x[1:])):
            tr = [r for r in rows if r["type"] == t]
            summary["by_type"].append({"corpus": cname, "pipeline": pname, "type": t, "n": len(tr), "recall5": agg(tr, "recall5"), "mrr": agg(tr, "mrr"), "ndcg5": agg(tr, "ndcg5")})
        q9 = [r for r in per_query if r["corpus"] == cname and r["pipeline"] == pname and r["type"] == "Q9" and "abstain" in r]
        if q9: summary["q9"].append({"corpus": cname, "pipeline": pname, "n": len(q9), "abstain_rate": sum(r["abstain"] for r in q9) / len(q9)})
def series(cname, pname, key="ndcg5"):
    return [r[key] for r in sorted(per_query, key=lambda r: r["qid"]) if r["corpus"] == cname and r["pipeline"] == pname and r["type"] != "Q9"]
pairs = [("hybrid vs vanilla (L2_A)", ("L2_A", "hybrid"), ("L2_A", "vanilla")),
         ("vanilla_bge vs vanilla (L2_A)", ("L2_A", "vanilla_bge"), ("L2_A", "vanilla")),
         ("vanilla_bge vs vanilla (L0_A)", ("L0_A", "vanilla_bge"), ("L0_A", "vanilla")),
         ("hybrid vs vanilla_bge (L2_A)", ("L2_A", "hybrid"), ("L2_A", "vanilla_bge")),
         ("L2 vs L0, vanilla_bge (A)", ("L2_A", "vanilla_bge"), ("L0_A", "vanilla_bge")),
         ("hybrid vs vanilla (L0_A)", ("L0_A", "hybrid"), ("L0_A", "vanilla")),
         ("L2 vs L0, vanilla (A)", ("L2_A", "vanilla"), ("L0_A", "vanilla")),
         ("L2 vs L0, hybrid (A)", ("L2_A", "hybrid"), ("L0_A", "hybrid")),
         ("L1 vs L0, hybrid (A)", ("L1_A", "hybrid"), ("L0_A", "hybrid")),
         ("B vs A, hybrid (L2)", ("L2_B", "hybrid"), ("L2_A", "hybrid")),
         ("B vs A, vanilla (L2)", ("L2_B", "vanilla"), ("L2_A", "vanilla")),
         ("hybrid vs dense-only (L2_A)", ("L2_A", "hybrid"), ("L2_A", "baseline_dense")),
         ("hybrid vs bm25-only (L2_A)", ("L2_A", "hybrid"), ("L2_A", "baseline_bm25"))]
for label, a, b in pairs:
    sa, sb = series(*a), series(*b)
    lo, hi = boot_ci(sa, sb)
    summary["comparisons"].append({"label": label, "a": a, "b": b, "delta_ndcg5": sum(sa) / len(sa) - sum(sb) / len(sb), "ci95": [lo, hi], "significant": lo > 0 or hi < 0})
json.dump(summary, open(OUT / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("\n=== 조건별 (Q9 제외, n=%d) ===" % summary["n_scored"])
for c in summary["conditions"]: print(f"{c['corpus']} {c['pipeline']:16} R@5 {c['recall5']:.3f}  MRR {c['mrr']:.3f}  nDCG@5 {c['ndcg5']:.3f}")
print("\n=== 쌍대 비교 (nDCG@5, 95% CI) ===")
for c in summary["comparisons"]: print(f"{c['label']:32} Δ {c['delta_ndcg5']:+.3f}  CI [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}] {'유의' if c['significant'] else ''}")
print("\n=== Q9 거절률 ===")
for c in summary["q9"]: print(f"{c['corpus']} {c['pipeline']:16} {c['abstain_rate']:.2f}")
