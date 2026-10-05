# -*- coding: utf-8 -*-
"""HITL 비교 창. 사용자가 질문을 넣으면 왼쪽에 내 파이프라인(hybrid_prefilter)의 검색 근거(+Ollama 가 있으면 답변),
오른쪽에 Fable 5.1 의 직접 답변(채팅에서 받아 붙여넣기)을 두고 사람이 판정한다. 판정은 hitl_log.jsonl 에 쌓인다.

실행: E:\\conda_envs\\rag_gpu\\python.exe 4_결과분석/hitl_serve.py  →  http://127.0.0.1:8766
"""
import sys, json, time, os, urllib.request
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "3_성능테스트"))
os.environ.setdefault("HF_HOME", r"E:\hf_cache"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
import run_eval as E  # noqa: E402
sys.path.insert(0, str(ROOT / "5_가드레일")); import guardrails as GR  # noqa: E402
from fastapi import FastAPI, Query, Body  # noqa: E402
from fastapi.responses import HTMLResponse, JSONResponse  # noqa: E402
import uvicorn  # noqa: E402

CORPUS = "L2P_A"; LOG = ROOT / "3_성능테스트" / "hitl_log.jsonl"
OLLAMA = "http://127.0.0.1:11434"; LLM_MODEL = "qwen2.5:7b-instruct"
HAIKU = "claude-haiku-4-5"  # 파이프라인 답변 생성 모델. ANTHROPIC_API_KEY 환경변수 또는 ROOT/.anthropic_key 파일(한 줄)에서 키를 읽는다.
_key_file = ROOT / ".anthropic_key"
if not os.environ.get("ANTHROPIC_API_KEY") and _key_file.exists(): os.environ["ANTHROPIC_API_KEY"] = _key_file.read_text(encoding="utf-8").strip()
HF_MODEL = os.environ.get("HF_MODEL", "Qwen/Qwen2.5-72B-Instruct")  # HF Inference API 모델. 답변평가에 쓴 qwen2.5:7b-instruct 는 라우터에 없어(2026-10-05) 같은 계열 72B 가 기본. HF_TOKEN 필요.
def hf_ok(): return bool(os.environ.get("HF_TOKEN"))
def hf_chat(system, user):
    body = json.dumps({"model": HF_MODEL, "temperature": 0, "max_tokens": 600,
                       "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}).encode("utf-8")
    req = urllib.request.Request("https://router.huggingface.co/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + os.environ["HF_TOKEN"]})
    with urllib.request.urlopen(req, timeout=300) as r: return json.loads(r.read().decode("utf-8"))["choices"][0]["message"]["content"]
_anthropic = {}
def haiku_ok(): return bool(os.environ.get("ANTHROPIC_API_KEY"))
def haiku_chat(system, user):
    import anthropic
    if "c" not in _anthropic: _anthropic["c"] = anthropic.Anthropic()
    r = _anthropic["c"].messages.create(model=HAIKU, max_tokens=1500, system=system, messages=[{"role": "user", "content": user}])
    return "".join(b.text for b in r.content if b.type == "text")
SYSTEM_PROMPT = """너는 회사 내부 규정안내를 도와주는 사내 문서 도우미야. [참고 문서]의 내용만 근거로 답하고, 문서에 없는 내용은 없다고 말해.
답 끝에 근거 조문을 "기관명 인사규정 제N조(제목)" 형식으로 적어.
[참고문서]
{context}"""
app = FastAPI(title="HITL 비교")
_c = {}
def corpus():
    if "c" not in _c: _c["c"] = E.Corpus(CORPUS)
    return _c["c"]
def ollama_ok():
    try:
        with urllib.request.urlopen(OLLAMA + "/api/tags", timeout=2) as r: return any(m["name"].startswith(LLM_MODEL.split(":")[0]) for m in json.loads(r.read())["models"])
    except Exception: return False
def ollama_chat(system, user):
    body = json.dumps({"model": LLM_MODEL, "stream": False, "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 600},
                       "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}).encode("utf-8")
    req = urllib.request.Request(OLLAMA + "/api/chat", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r: return json.loads(r.read().decode("utf-8"))["message"]["content"]

@app.get("/", response_class=HTMLResponse)
def index(): return HTMLResponse((HERE / "hitl.html").read_text(encoding="utf-8"), headers={"Cache-Control": "no-store"})

@app.get("/orgs")
def orgs():
    c = corpus(); return sorted(set(c.orgs.tolist()))

@app.get("/ask")
def ask(q: str = Query(..., min_length=1), org: str = Query(""), k: int = Query(5, ge=1, le=10)):
    c = corpus(); t0 = time.time()
    E._CTX["org"] = org or None
    idx, info = E.run_hybrid_prefilter(c, q)
    E._CTX["org"] = None
    arts, hits = [], []
    for i in idx:
        r = c.rows[i]
        if r["article_id"] in arts: continue
        arts.append(r["article_id"])
        hits.append({"rank": len(arts), "org": r["org"], "article_no": r["article_no"], "title": r.get("article_title", ""), "revision_date": r["revision_date"], "text": r["text"]})
        if len(hits) >= k: break
    out = {"q": q, "org": org or E.detect_org(q), "hits": hits, "info": {kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in info.items()}, "sec": round(time.time() - t0, 2), "answer": None, "model": None}
    gd = GR.decide(q, info.get("top1_ce"), c.org_names)
    out["gate"] = {"action": gd["action"], "gate": gd["gate"], "input": gd["input"]["label"], "message": gd["message"], "disclaimer": GR.MSG["disclaimer"]}
    if gd["action"] in ("refuse_scope", "refuse_multi"): hits = []; out["hits"] = []
    if hits and gd["action"] == "answer":
        ctx = "\n\n".join(f"[출처: {h['org']} 인사규정 {h['article_no']}({h['title']}) | 개정 {h['revision_date']}]\n{h['text'][:1500]}" for h in hits)
        if hf_ok():
            try: out["answer"] = hf_chat(SYSTEM_PROMPT.format(context=ctx), q); out["model"] = HF_MODEL
            except Exception as e: out["answer"] = f"(HF API 호출 실패: {type(e).__name__}: {e})"
        elif haiku_ok():
            try: out["answer"] = haiku_chat(SYSTEM_PROMPT.format(context=ctx), q); out["model"] = HAIKU
            except Exception as e: out["answer"] = f"(Haiku 호출 실패: {type(e).__name__}: {e})"
        elif ollama_ok():
            try: out["answer"] = ollama_chat(SYSTEM_PROMPT.format(context=ctx), q); out["model"] = LLM_MODEL
            except Exception as e: out["answer"] = f"(Ollama 호출 실패: {e})"
        if out["answer"] and out["model"]:
            v = GR.verify_answer(out["answer"], [h["text"] for h in hits], [h["article_no"] for h in hits]); out["gate"]["verify"] = v
            if not v["ok"]: out["answer"] = None; out["gate"]["message"] = "생성 답변이 근거 조문과 맞지 않아 원문만 표시합니다. " + str(v)
    with open(ROOT / "3_성능테스트" / "hitl_questions.jsonl", "a", encoding="utf-8") as f:  # 채팅의 Fable 이 읽어갈 질문 기록
        f.write(json.dumps({"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "q": q, "org": out["org"], "hits": [{k: h[k] for k in ("rank", "org", "article_no", "title", "revision_date", "text")} for h in hits]}, ensure_ascii=False) + "\n")
    return JSONResponse(out)

ANS = ROOT / "3_성능테스트" / "hitl_answers.jsonl"  # 채팅의 Fable 이 써 넣는 답변 (q, pipeline_answer, fable_answer)
@app.get("/answers")
def answers(q: str = Query("")):
    if not ANS.exists(): return {}
    recs = [json.loads(l) for l in open(ANS, encoding="utf-8") if l.strip()]
    recs = [r for r in recs if r.get("q") == q] if q else recs
    return recs[-1] if recs else {}

@app.post("/judge")
def judge(rec: dict = Body(...)):
    rec["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG, "a", encoding="utf-8") as f: f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return {"ok": True, "n": sum(1 for _ in open(LOG, encoding="utf-8"))}

@app.get("/log")
def log():
    if not LOG.exists(): return []
    return [json.loads(l) for l in open(LOG, encoding="utf-8") if l.strip()]

if __name__ == "__main__":
    print("첫 질의 때 모델 로딩 20~30초. http://127.0.0.1:8766  | 답변 생성:", HAIKU if haiku_ok() else ("Ollama " + LLM_MODEL if ollama_ok() else "없음(검색 근거만). ANTHROPIC_API_KEY 또는 .anthropic_key 필요"))
    uvicorn.run(app, host="127.0.0.1", port=8766, log_level="warning")
