# -*- coding: utf-8 -*-
"""로컬 대시보드 서버. 대시보드(dashboard_local.html)와 함께 세 파이프라인에 질의를 던져 보는 /search API 를 제공한다.

실행:  python 4_결과분석/serve.py   →  http://127.0.0.1:8765
파이프라인 구현은 3_성능테스트/run_eval.py 의 것을 그대로 쓴다(측정에 쓴 코드와 동일).
"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "3_성능테스트"))
import run_eval as E  # noqa: E402
from fastapi import FastAPI, Query  # noqa: E402
from fastapi.responses import HTMLResponse, JSONResponse  # noqa: E402
import uvicorn  # noqa: E402

app = FastAPI(title="인사규정 RAG 검색 비교")
_corpora = {}
def corpus(name):
    if name not in _corpora: _corpora[name] = E.Corpus(name)
    return _corpora[name]
PIPES = {
    "vanilla": ("Vanilla RAG", "chat_rag · ko-sbert-nli 임베딩 → 단일 벡터 검색 → 유사도 0.5 미만이면 거절", E.run_vanilla),
    "vanilla_bge": ("Vanilla RAG, 임베딩만 bge-m3 (참고)", "chat_rag 구조 그대로, 임베딩 모델만 교체", E.run_vanilla_bge),
    "hybrid": ("하이브리드 + CE", "hybrid-search-eval · bge-m3 dense + BM25 → RRF → 기관 필터 → bge-reranker 재랭킹", E.run_hybrid),
}

@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse((HERE / "dashboard_local.html").read_text(encoding="utf-8"), headers={"Cache-Control": "no-store"})

@app.get("/search")
def search(q: str = Query(..., min_length=1), corpus_name: str = Query("L2_A", alias="corpus"), k: int = Query(5, ge=1, le=10), pipelines: str = Query("vanilla,hybrid")):
    c = corpus(corpus_name); out = {"query": q, "corpus": corpus_name, "org_detected": E.detect_org(q), "pipelines": {}}
    wanted = [x for x in pipelines.split(",") if x in PIPES] or list(PIPES)
    for key in wanted:
        label, desc, fn = PIPES[key]
        idx, info = fn(c, q)
        arts, hits = [], []
        for i in idx:
            r = c.rows[i]
            if r["article_id"] in arts: continue
            arts.append(r["article_id"])
            hits.append({"rank": len(arts), "article_id": r["article_id"], "org": r["org"], "article_no": r["article_no"], "title": r.get("article_title", ""),
                         "revision_date": r["revision_date"], "text": r["text"][:400]})
            if len(hits) >= k: break
        out["pipelines"][key] = {"label": label, "desc": desc, "hits": hits, "info": {kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in info.items()}}
    return JSONResponse(out)

@app.get("/corpora")
def corpora(): return E.CORPORA

if __name__ == "__main__":
    print("모델을 처음 호출할 때 로딩 시간이 몇 초 걸립니다. http://127.0.0.1:8765")
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="warning")
