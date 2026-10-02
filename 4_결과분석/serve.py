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
import urllib.request  # noqa: E402

OLLAMA = "http://127.0.0.1:11434"; LLM_MODEL = "qwen2.5:7b-instruct"
SYSTEM_PROMPT = """너는 회사 내부 규정안내를 도와주는 사내 문서 도우미야.

[참고 문서]의 내용을 바탕으로 사용자의 질문에 답변해. 문서에 없는 내용은 지어내지 말고 모르거나 없다고 대답해.
문서를 기반으로 답변을 생성할 때 그 답변과 관련된 문서의 내용을 직접 확인해서 사용자가 검증해볼 수 있도록 내용의 위치를 같이 출력해.

출력방식은 간결하고 보기 좋게 깔끔하게 정리해서 출력해.
문서의 검증 위치는 참고 문서의 [출처: ...] 에 적힌 기관명과 조번호를 그대로 써서 아래 예시 형식으로 작성해.
예시: " 한국가스공사 인사규정 제31조(정년) "

너가 생성한 답변이 문서 내용과 위배되거나 없는 내용을 임의로 지어내었는지 반복적으로 재검증하고 틀린 부분을 수정해서 최종 답변해줘.
[참고문서]
{context}"""

def retrieve(c, key, q, k):
    label, desc, fn = PIPES[key]; idx, info = fn(c, q)
    arts, hits = [], []
    for i in idx:
        r = c.rows[i]
        if r["article_id"] in arts: continue
        arts.append(r["article_id"])
        hits.append({"rank": len(arts), "article_id": r["article_id"], "org": r["org"], "article_no": r["article_no"], "title": r.get("article_title", ""),
                     "revision_date": r["revision_date"], "text": r["text"]})
        if len(hits) >= k: break
    return label, desc, hits, info

def ollama_chat(system, user):
    body = json.dumps({"model": LLM_MODEL, "stream": False, "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 600},
                       "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}).encode("utf-8")
    req = urllib.request.Request(OLLAMA + "/api/chat", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r: return json.loads(r.read().decode("utf-8"))["message"]["content"]

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
        label, desc, hits, info = retrieve(c, key, q, k)
        for h in hits: h["text"] = h["text"][:400]
        out["pipelines"][key] = {"label": label, "desc": desc, "hits": hits, "info": {kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in info.items()}}
    return JSONResponse(out)

@app.get("/answer")
def answer(q: str = Query(..., min_length=1), pipeline: str = Query("hybrid"), corpus_name: str = Query("L2_A", alias="corpus"), k: int = Query(5, ge=1, le=8)):
    """검색된 조문을 컨텍스트로 로컬 LLM(Ollama qwen2.5:7b-instruct)이 답한다. chat_rag 의 프롬프트와 동일, 출처 형식만 조문 단위."""
    if pipeline not in PIPES: pipeline = "hybrid"
    c = corpus(corpus_name); label, desc, hits, info = retrieve(c, pipeline, q, k)
    abstain = bool(info.get("abstain")) and pipeline != "hybrid"  # chat_rag: 임계값 미달이면 참고 문서 없이 답하게 함
    if abstain or not hits: context = "(검색결과 없음)"
    else: context = "\n\n".join(f"[출처: {h['org']} 인사규정 {h['article_no']}{'(' + h['title'] + ')' if h['title'] else ''} | 개정 {h['revision_date']}]\n{h['text'][:1500]}" for h in hits)
    try: ans = ollama_chat(SYSTEM_PROMPT.format(context=context), q)
    except Exception as e: return JSONResponse({"error": f"Ollama 호출 실패: {e}"}, status_code=502)
    return JSONResponse({"pipeline": label, "model": LLM_MODEL, "abstain": abstain, "answer": ans,
                         "sources": [f"{h['org']} 인사규정 {h['article_no']}" for h in hits] if not abstain else []})

@app.get("/corpora")
def corpora(): return E.CORPORA

if __name__ == "__main__":
    print("모델을 처음 호출할 때 로딩 시간이 몇 초 걸립니다. http://127.0.0.1:8765")
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="warning")
