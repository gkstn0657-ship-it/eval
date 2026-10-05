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
import os; os.environ.setdefault("HF_HOME", r"E:\hf_cache"); os.environ.setdefault("HF_HUB_OFFLINE", "1")  # 환경변수 없는 셸에서 C드라이브로 재다운로드되는 것 방지
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
GOLD = ROOT / "1_데이터셋" / "06_eval" / ("goldenset_holdout_sealed.jsonl" if "--holdout" in sys.argv else "goldenset_dev.jsonl")
if "--gold" in sys.argv: GOLD = Path(sys.argv[sys.argv.index("--gold") + 1])  # v8: 외부 골든셋(새 문항) 지정
OUT_SUFFIX = ("_" + sys.argv[sys.argv.index("--tag") + 1]) if "--tag" in sys.argv else ""
OUT = HERE / (("results_holdout" if "--holdout" in sys.argv else "results") + OUT_SUFFIX); OUT.mkdir(exist_ok=True)
CACHE = HERE / "cache"; CACHE.mkdir(exist_ok=True)
CORPORA = ["L0_A", "L0_B", "L1_A", "L1_B", "L2_A", "L2_B"]
if "--corpora" in sys.argv: CORPORA = sys.argv[sys.argv.index("--corpora") + 1].split(",")
TOP_K, RRF_K, CANDIDATE_K, RERANK_TOP_N, MIN_RESULTS, SIM_THRESHOLD, CE_MAX_LEN = 10, 60, 30, 30, 5, 0.5, 384
TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣]+")
def _device():
    import torch; return "cuda" if torch.cuda.is_available() else "cpu"
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
    "한국지역난방공사": "한국지역난방공사", "지역난방공사": "한국지역난방공사", "지역난방": "한국지역난방공사", "난방공사": "한국지역난방공사",  # 외부 검증 기관(2026-10-04)
}
def strip_org(q):
    """질의에서 기관 표현을 제거(v8). 기관은 필터가 맡으므로 점수 계산에는 넣지 않는다."""
    for k in sorted(ORG_ALIAS, key=len, reverse=True):
        if k in q: return re.sub(r"\s+", " ", q.replace(k, " ")).strip()
    return q
sys.path.insert(0, str(ROOT / "5_가드레일")); import guardrails as GR  # 거절 게이트·입력 분류 (2026-10-05)
_CTX = {"org": None}  # 사내 챗봇 상황: 질의 레코드에 context_org 가 있으면 질의 문자열 대신 이 값을 기관으로 쓴다 (employee 세트)
def detect_org(q):
    if _CTX["org"] is not None: return _CTX["org"]
    for k in sorted(ORG_ALIAS, key=len, reverse=True):
        if k in q: return ORG_ALIAS[k]
    return None

# ---------------------------------------------------------------- 모델 (지연 로딩)
_models = {}
def st_model(name):
    if name not in _models:
        from sentence_transformers import SentenceTransformer
        _models[name] = SentenceTransformer(name, device="cpu")  # VRAM 3GB: 임베딩은 CPU(코퍼스는 캐시), CE 만 GPU
    return _models[name]
def ce_model():
    if "ce" not in _models:
        from sentence_transformers import CrossEncoder
        _models["ce"] = CrossEncoder("BAAI/bge-reranker-v2-m3", device=_device(), max_length=CE_MAX_LEN)
        if _device() == "cuda": _models["ce"].model.half()
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
EXPAND = "--expand" in sys.argv  # 1006 실험 민감도 분석: 조 경계를 넘는 청크가 포함한 조문 전체(article_ids)를 펼쳐 채점
def to_articles(ranked_chunk_idx, rows):
    seen, out = set(), []
    for i in ranked_chunk_idx:
        for a in (rows[i].get("article_ids") or [rows[i]["article_id"]]) if EXPAND else [rows[i]["article_id"]]:
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
        self.org_names = sorted(set(self.orgs.tolist()))
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

def run_hybrid_nobm25(c: Corpus, q):
    """하이브리드에서 BM25 축만 제거. bge-m3 dense top 30 → 기관 필터 → CE 재랭킹. 사전등록 후 추가(v4)."""
    qv = st_model("BAAI/bge-m3").encode([q], normalize_embeddings=True, convert_to_numpy=True)[0]
    d = c.bge @ qv
    cand = [int(i) for i in np.argsort(-d)[:CANDIDATE_K]]
    org = detect_org(q)
    exact = [i for i in cand if org is None or c.orgs[i] == org]; relaxed = [i for i in cand if i not in exact]
    tiers = exact + relaxed
    pool, rest = tiers[:RERANK_TOP_N], tiers[RERANK_TOP_N:]
    ce = ce_model().predict([(q, c.texts[i]) for i in pool], batch_size=16, show_progress_bar=False) if pool else []
    ce_map = {i: float(s) for i, s in zip(pool, ce)}
    pool.sort(key=lambda i: (not (org is None or c.orgs[i] == org), -ce_map[i], -float(d[i]), i))
    top = pool + rest
    return top, {"org_filter": org, "top1_ce": ce_map[top[0]] if top else None, "abstain": bool(top and ce_map.get(top[0], 0) < 0)}

def _vanilla_filter(c: Corpus, q, model_name, emb):
    """Vanilla 단일 벡터 검색 + 하이브리드와 같은 기관 필터(상위 30 후보를 정확 매치 우선, 5개 미만이면 완화). 재랭킹 없음. v5."""
    qv = st_model(model_name).encode([q], normalize_embeddings=True, convert_to_numpy=True)[0]
    s = emb @ qv
    cand = [int(i) for i in np.argsort(-s)[:CANDIDATE_K]]
    org = detect_org(q)
    exact = [i for i in cand if org is None or c.orgs[i] == org]; relaxed = [i for i in cand if i not in exact]
    top = exact + relaxed
    return top, {"org_filter": org, "top1_score": float(s[top[0]]), "abstain": bool(s[top[0]] < SIM_THRESHOLD)}

def run_vanilla_filter(c: Corpus, q): return _vanilla_filter(c, q, "jhgan/ko-sbert-nli", c.sbert)
def run_vanilla_bge_filter(c: Corpus, q): return _vanilla_filter(c, q, "BAAI/bge-m3", c.bge)

def run_hybrid_nofilter(c: Corpus, q):
    """하이브리드에서 기관 필터만 제거. dense + BM25 → RRF → CE 재랭킹(RRF 상위 30). v6."""
    qv = st_model("BAAI/bge-m3").encode([q], normalize_embeddings=True, convert_to_numpy=True)[0]
    d = c.bge @ qv; b = c.bm25.get_scores(TOKEN_RE.findall(q.lower()))
    d_rank = {int(i): r for r, i in enumerate(np.argsort(-d)[:CANDIDATE_K])}
    b_rank = {int(i): r for r, i in enumerate(np.argsort(-b)[:CANDIDATE_K])}
    rrf = defaultdict(float)
    for i, r in d_rank.items(): rrf[i] += 1 / (RRF_K + r)
    for i, r in b_rank.items(): rrf[i] += 1 / (RRF_K + r)
    cand = sorted(rrf, key=lambda i: (-rrf[i], i))
    pool, rest = cand[:RERANK_TOP_N], cand[RERANK_TOP_N:]
    ce = ce_model().predict([(q, c.texts[i]) for i in pool], batch_size=16, show_progress_bar=False) if pool else []
    ce_map = {i: float(s) for i, s in zip(pool, ce)}
    pool.sort(key=lambda i: (-ce_map[i], -rrf[i], i))
    top = pool + rest
    return top, {"org_filter": None, "top1_ce": ce_map[top[0]] if top else None, "abstain": bool(top and ce_map.get(top[0], 0) < 0)}

def _hybrid_core(c: Corpus, q_score, org, diverse=False):
    qv = st_model("BAAI/bge-m3").encode([q_score], normalize_embeddings=True, convert_to_numpy=True)[0]
    d = c.bge @ qv; b = c.bm25.get_scores(TOKEN_RE.findall(q_score.lower()))
    d_rank = {int(i): r for r, i in enumerate(np.argsort(-d)[:CANDIDATE_K])}
    b_rank = {int(i): r for r, i in enumerate(np.argsort(-b)[:CANDIDATE_K])}
    rrf = defaultdict(float)
    for i, r in d_rank.items(): rrf[i] += 1 / (RRF_K + r)
    for i, r in b_rank.items(): rrf[i] += 1 / (RRF_K + r)
    cand = sorted(rrf, key=lambda i: (-rrf[i], i))
    exact = [i for i in cand if org is None or c.orgs[i] == org]; relaxed = [i for i in cand if i not in exact]
    pool, rest = (exact + relaxed)[:RERANK_TOP_N], (exact + relaxed)[RERANK_TOP_N:]
    ce = ce_model().predict([(q_score, c.texts[i]) for i in pool], batch_size=16, show_progress_bar=False) if pool else []
    ce_map = {i: float(s) for i, s in zip(pool, ce)}
    pool.sort(key=lambda i: (not (org is None or c.orgs[i] == org), -ce_map[i], -rrf[i], i))
    if diverse and org is None:  # 기관 비교 질의: 기관당 1개를 먼저 배치
        seen, first, later = set(), [], []
        for i in pool:
            (later if c.orgs[i] in seen else first).append(i); seen.add(c.orgs[i])
        pool = first + later
    top = pool + rest
    return top, {"org_filter": org, "exact_in_pool": sum(1 for i in pool if org is None or c.orgs[i] == org),
                 "top1_ce": ce_map[top[0]] if top else None, "abstain": bool(top and ce_map.get(top[0], 0) < 0)}
def run_hybrid_qstrip(c: Corpus, q):
    """v8 수정 1: 기관명을 질의에서 떼고 dense·BM25·CE 점수를 계산. 필터는 그대로."""
    org = detect_org(q); return _hybrid_core(c, strip_org(q) if org else q, org)
def run_hybrid_qstrip_diverse(c: Corpus, q):
    """v8 수정 1+2: 기관명 제거 + 기관 미지정 질의는 기관당 1개 우선 배치."""
    org = detect_org(q); return _hybrid_core(c, strip_org(q) if org else q, org, diverse=True)
def run_hybrid_prefilter(c: Corpus, q):
    """v9: 기관 필터를 후보 생성 앞으로. 기관이 감지되면 그 기관 청크만 대상으로 dense·BM25 → RRF → CE. 질의 기관명 제거(v8) 포함.
    기관 미감지 질의는 hybrid_qstrip 과 동일 경로."""
    org = detect_org(q)
    if org is None: return _hybrid_core(c, q, None)
    q_score = strip_org(q)
    mask = c.orgs == org; idx = np.flatnonzero(mask)
    qv = st_model("BAAI/bge-m3").encode([q_score], normalize_embeddings=True, convert_to_numpy=True)[0]
    d = np.where(mask, c.bge @ qv, -np.inf)
    b = np.full(len(c.rows), -np.inf); b[idx] = c.bm25.get_batch_scores(TOKEN_RE.findall(q_score.lower()), idx.tolist())  # 해당 기관 문서만 BM25 계산
    d_rank = {int(i): r for r, i in enumerate(np.argsort(-d)[:CANDIDATE_K]) if mask[i]}
    b_rank = {int(i): r for r, i in enumerate(np.argsort(-b)[:CANDIDATE_K]) if mask[i]}
    rrf = defaultdict(float)
    for i, r in d_rank.items(): rrf[i] += 1 / (RRF_K + r)
    for i, r in b_rank.items(): rrf[i] += 1 / (RRF_K + r)
    cand = sorted(rrf, key=lambda i: (-rrf[i], i))
    pool, rest = cand[:RERANK_TOP_N], cand[RERANK_TOP_N:]
    ce = ce_model().predict([(q_score, c.texts[i]) for i in pool], batch_size=16, show_progress_bar=False) if pool else []
    ce_map = {i: float(s) for i, s in zip(pool, ce)}
    pool.sort(key=lambda i: (-ce_map[i], -rrf[i], i))
    top = pool + rest
    return top, {"org_filter": org, "exact_in_pool": len(pool), "top1_ce": ce_map[top[0]] if top else None, "abstain": bool(top and ce_map.get(top[0], 0) < 0)}
_SYN = None
def expand_query(q):
    """v10: synonyms_seed(사람 작성, R-14)의 일상어 표현이 질의에 있으면 규정어를 덧붙인다. 토큰(2자 그대로, 3자+는 끝 글자 제거)이 모두 포함될 때 매칭."""
    global _SYN
    if _SYN is None:
        _SYN = [json.loads(l) for l in (ROOT / "1_데이터셋" / "05_clean" / "L2" / "synonyms_seed.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    add = []
    for e in _SYN:
        for phrase in e["everyday"]:
            toks = [t if len(t) == 2 else t[:-1] for t in re.findall(r"[가-힣]{2,}", phrase)]
            if toks and all(t in q for t in toks):
                add += [t for t in e["terms"] if t not in add]; break
    return (q + " " + " ".join(add)).strip() if add else q
def _prefilter_core(c: Corpus, q_dense, q_bm25, q_ce, org):
    mask = c.orgs == org
    qv = st_model("BAAI/bge-m3").encode([q_dense], normalize_embeddings=True, convert_to_numpy=True)[0]
    idx = np.flatnonzero(mask)
    d = np.where(mask, c.bge @ qv, -np.inf)
    b = np.full(len(c.rows), -np.inf); b[idx] = c.bm25.get_batch_scores(TOKEN_RE.findall(q_bm25.lower()), idx.tolist())  # 해당 기관 문서만 BM25 계산 (22k → 수십 개, 속도 최적화)
    d_rank = {int(i): r for r, i in enumerate(np.argsort(-d)[:CANDIDATE_K]) if mask[i]}
    b_rank = {int(i): r for r, i in enumerate(np.argsort(-b)[:CANDIDATE_K]) if mask[i]}
    rrf = defaultdict(float)
    for i, r in d_rank.items(): rrf[i] += 1 / (RRF_K + r)
    for i, r in b_rank.items(): rrf[i] += 1 / (RRF_K + r)
    cand = sorted(rrf, key=lambda i: (-rrf[i], i)); pool, rest = cand[:RERANK_TOP_N], cand[RERANK_TOP_N:]
    ce = ce_model().predict([(q_ce, c.texts[i]) for i in pool], batch_size=16, show_progress_bar=False) if pool else []
    ce_map = {i: float(s) for i, s in zip(pool, ce)}
    pool.sort(key=lambda i: (-ce_map[i], -rrf[i], i)); top = pool + rest
    return top, {"org_filter": org, "exact_in_pool": len(pool), "top1_ce": ce_map[top[0]] if top else None, "abstain": bool(top and ce_map.get(top[0], 0) < 0)}
def run_hybrid_prefilter_qexp(c: Corpus, q):
    """v10: pre-filter + 질의 확장(dense·BM25·CE 모두)."""
    org = detect_org(q)
    if org is None: return _hybrid_core(c, expand_query(q), None)
    qe = expand_query(strip_org(q)); return _prefilter_core(c, qe, qe, qe, org)
def run_hybrid_prefilter_qexp_bm25ce(c: Corpus, q):
    """v10: pre-filter + 질의 확장(BM25·CE 만, dense 는 원 질의)."""
    org = detect_org(q)
    if org is None: return _hybrid_core(c, q, None)
    q0 = strip_org(q); qe = expand_query(q0); return _prefilter_core(c, q0, qe, qe, org)
def run_random(c: Corpus, q):
    idx = list(np.random.permutation(len(c.rows))[:TOP_K * 3]); return idx, {}

PIPELINES = {"vanilla": run_vanilla, "vanilla_bge": run_vanilla_bge, "vanilla_filter": run_vanilla_filter, "vanilla_bge_filter": run_vanilla_bge_filter, "hybrid": run_hybrid, "hybrid_nobm25": run_hybrid_nobm25, "hybrid_nofilter": run_hybrid_nofilter, "hybrid_qstrip": run_hybrid_qstrip, "hybrid_qstrip_diverse": run_hybrid_qstrip_diverse, "hybrid_prefilter": run_hybrid_prefilter, "hybrid_prefilter_qexp": run_hybrid_prefilter_qexp, "hybrid_prefilter_qexp_bm25ce": run_hybrid_prefilter_qexp_bm25ce, "baseline_dense": run_dense, "baseline_bm25": run_bm25, "baseline_random": run_random}

# ---------------------------------------------------------------- 실행
def main():
    if "--pipelines" in sys.argv:
        keep = sys.argv[sys.argv.index("--pipelines") + 1].split(","); [PIPELINES.pop(k) for k in list(PIPELINES) if k not in keep]
    gold = [json.loads(l) for l in GOLD.read_text(encoding="utf-8").splitlines() if l.strip()]
    per_query, t0 = [], time.time()
    for cname in CORPORA:
        c = Corpus(cname); policy = cname[-1]
        for pname, fn in PIPELINES.items():
            for q in gold:
                _CTX["org"] = q.get("context_org")
                idx, info = fn(c, q["query"])
                arts = to_articles(idx, c.rows)
                gd = GR.decide(q["query"], info.get("top1_ce"), c.org_names)
                info["abstain"] = gd["action"] in ("refuse", "refuse_scope", "refuse_multi"); info["gate"] = gd["gate"]; info["gate_action"] = gd["action"]; info["input_label"] = gd["input"]["label"]
                g = q["gold_ids_any_revision"] if policy == "B" else q["gold_ids"]
                m = metrics(arts, g) if q["type"] != "Q9" else {}
                per_query.append({"corpus": cname, "pipeline": pname, "qid": q["qid"], "type": q["type"], "rank": q.get("rank"), "gold": g,
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
    pairs = [("1006 CH_FIX vs L2P (prefilter)", ("CH_FIX_A", "hybrid_prefilter"), ("L2P_A", "hybrid_prefilter")),
         ("1006 CH_SEM vs L2P (prefilter)", ("CH_SEM_A", "hybrid_prefilter"), ("L2P_A", "hybrid_prefilter")),
         ("1006 CH_PARA vs L2P (prefilter)", ("CH_PARA_A", "hybrid_prefilter"), ("L2P_A", "hybrid_prefilter")),
         ("hybrid vs vanilla (L2_A)", ("L2_A", "hybrid"), ("L2_A", "vanilla")),
         ("v8 qstrip vs hybrid (L2P_A)", ("L2P_A", "hybrid_qstrip"), ("L2P_A", "hybrid")),
         ("v8 qstrip+diverse vs hybrid (L2P_A)", ("L2P_A", "hybrid_qstrip_diverse"), ("L2P_A", "hybrid")),
         ("v8 qstrip+diverse vs qstrip (L2P_A)", ("L2P_A", "hybrid_qstrip_diverse"), ("L2P_A", "hybrid_qstrip")),
         ("v9 prefilter vs qstrip (L2P_A)", ("L2P_A", "hybrid_prefilter"), ("L2P_A", "hybrid_qstrip")),
         ("v9 prefilter vs hybrid (L2P_A)", ("L2P_A", "hybrid_prefilter"), ("L2P_A", "hybrid")),
         ("v10 qexp vs prefilter (L2P_A)", ("L2P_A", "hybrid_prefilter_qexp"), ("L2P_A", "hybrid_prefilter")),
         ("v10 qexp_bm25ce vs prefilter (L2P_A)", ("L2P_A", "hybrid_prefilter_qexp_bm25ce"), ("L2P_A", "hybrid_prefilter")),
         ("L2P vs L2, hybrid (A)", ("L2P_A", "hybrid"), ("L2_A", "hybrid")),
         ("L2P vs L2, vanilla (A)", ("L2P_A", "vanilla"), ("L2_A", "vanilla")),
         ("L2P vs L2, vanilla_bge_filter (A)", ("L2P_A", "vanilla_bge_filter"), ("L2_A", "vanilla_bge_filter")),
             ("vanilla_bge vs vanilla (L2_A)", ("L2_A", "vanilla_bge"), ("L2_A", "vanilla")),
         ("vanilla_filter vs vanilla (L2_A)", ("L2_A", "vanilla_filter"), ("L2_A", "vanilla")),
         ("vanilla_bge_filter vs vanilla_bge (L2_A)", ("L2_A", "vanilla_bge_filter"), ("L2_A", "vanilla_bge")),
         ("hybrid_nobm25 vs vanilla_bge_filter (L2_A)", ("L2_A", "hybrid_nobm25"), ("L2_A", "vanilla_bge_filter")),
         ("hybrid vs vanilla_filter (L2_A)", ("L2_A", "hybrid"), ("L2_A", "vanilla_filter")),
         ("hybrid vs hybrid_nofilter (L2_A)", ("L2_A", "hybrid"), ("L2_A", "hybrid_nofilter")),
         ("hybrid vs hybrid_nofilter (L0_A)", ("L0_A", "hybrid"), ("L0_A", "hybrid_nofilter")),
         ("hybrid_nofilter vs vanilla_bge (L2_A)", ("L2_A", "hybrid_nofilter"), ("L2_A", "vanilla_bge")),
         ("hybrid vs hybrid_nobm25 (L2_A)", ("L2_A", "hybrid"), ("L2_A", "hybrid_nobm25")),
         ("hybrid vs hybrid_nobm25 (L0_A)", ("L0_A", "hybrid"), ("L0_A", "hybrid_nobm25")),
         ("hybrid_nobm25 vs vanilla_bge (L2_A)", ("L2_A", "hybrid_nobm25"), ("L2_A", "vanilla_bge")),
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
        if a[0] not in CORPORA or b[0] not in CORPORA or a[1] not in PIPELINES or b[1] not in PIPELINES: continue
        sa, sb = series(*a), series(*b)
        if not sa or not sb: continue
        lo, hi = boot_ci(sa, sb)
        summary["comparisons"].append({"label": label, "a": a, "b": b, "delta_ndcg5": sum(sa) / len(sa) - sum(sb) / len(sb), "ci95": [lo, hi], "significant": lo > 0 or hi < 0})
    json.dump(summary, open(OUT / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n=== 조건별 (Q9 제외, n=%d) ===" % summary["n_scored"])
    for c in summary["conditions"]: print(f"{c['corpus']} {c['pipeline']:16} R@5 {c['recall5']:.3f}  MRR {c['mrr']:.3f}  nDCG@5 {c['ndcg5']:.3f}")
    print("\n=== 쌍대 비교 (nDCG@5, 95% CI) ===")
    for c in summary["comparisons"]: print(f"{c['label']:32} Δ {c['delta_ndcg5']:+.3f}  CI [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}] {'유의' if c['significant'] else ''}")
    print("\n=== Q9 거절률 ===")
    for c in summary["q9"]: print(f"{c['corpus']} {c['pipeline']:16} {c['abstain_rate']:.2f}")


if __name__ == "__main__":
    main()