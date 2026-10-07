# -*- coding: utf-8 -*-
"""배포용 /ask (가드레일 v3). hitl_serve 의 검색(기관 인식 + hybrid_prefilter)은 그대로 쓰고, 답변 생성 대신
가드레일 v3 하네스(5_가드레일/v2/guard.py)를 붙인다. 답변 본문은 조문 원문과 고정 문구로만 만들어진다.

구성(설계용 평가로 확정, 가드레일_v3_결과.md): 입력 규칙 + 적용 대상 규칙(G2) + 근거 추출 + 원문 검증. 범위 판정(G1)은 끔.
근거 추출 LLM: HF Inference API(HF_TOKEN, 기본 Qwen2.5-72B) → 없으면 Ollama(qwen2.5:7b) → 둘 다 없으면 검색 근거만.
"""
import os, json, re, time, urllib.request
from pathlib import Path
from fastapi import Query
from fastapi.responses import JSONResponse
import hitl_serve as H
from guard import Guard, OllamaLLM, ANSWER, REFUSE_SCOPE, REFUSE_NO_EVIDENCE, REFUSE_ABUSE, REFUSE_SUBJECT, ASK_ORG, NO_COMPARE, DISCLAIMER

ROOT = Path(__file__).resolve().parent
GD = json.load(open(ROOT / "guard_data.json", encoding="utf-8"))  # 기관별 위임 규정 목록, 적용 범위 조항
HF_MODEL = os.environ.get("HF_MODEL", "Qwen/Qwen2.5-72B-Instruct")

def _parse_json_block(s):
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s)
    i, j = s.find("{"), s.rfind("}")
    return s[i:j + 1] if i >= 0 and j > i else s

class HFLLM:
    """HF 라우터(OpenAI 호환) JSON 모드. 실패하면 모드 없이 한 번 더 시도. 반환은 JSON 문자열(파싱 실패는 하네스가 거부로 처리)."""
    def __init__(self, model): self.model = model
    def __call__(self, system, user, schema=None):
        for fmt in ({"type": "json_object"}, None):
            body = {"model": self.model, "temperature": 0, "max_tokens": 1200,
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
            if fmt: body["response_format"] = fmt
            req = urllib.request.Request("https://router.huggingface.co/v1/chat/completions", data=json.dumps(body).encode("utf-8"),
                                         headers={"Content-Type": "application/json", "Authorization": "Bearer " + os.environ["HF_TOKEN"]})
            try:
                with urllib.request.urlopen(req, timeout=300) as r:
                    return _parse_json_block(json.loads(r.read().decode("utf-8"))["choices"][0]["message"]["content"])
            except urllib.error.HTTPError as e:
                if fmt and e.code == 400: continue
                raise
        raise RuntimeError("HF chat failed")

def pick_llm():
    if H.hf_ok(): return HFLLM(HF_MODEL), HF_MODEL
    if H.ollama_ok(): return OllamaLLM(H.LLM_MODEL), H.LLM_MODEL
    return None, None

ACTION = {ANSWER: "answer", REFUSE_SCOPE: "refuse_scope", REFUSE_NO_EVIDENCE: "refuse", REFUSE_SUBJECT: "refuse_subject",
          REFUSE_ABUSE: "refuse_abuse", ASK_ORG: "ask_org", NO_COMPARE: "refuse_multi"}

def ask_v3(q: str = Query(..., min_length=1), org: str = Query(""), k: int = Query(5, ge=1, le=10)):
    c = H.corpus(); t0 = time.time()
    target, org_note = H.resolve_org(q, org, c.org_names)
    H.E._CTX["org"] = target
    idx, info = H.E.run_hybrid_prefilter(c, q)
    H.E._CTX["org"] = None
    arts, hits = [], []
    for i in idx:
        r = c.rows[i]
        if r["article_id"] in arts: continue
        arts.append(r["article_id"])
        hits.append({"rank": len(arts), "org": r["org"], "article_no": r["article_no"], "title": r.get("article_title", ""), "revision_date": r["revision_date"], "text": r["text"]})
        if len(hits) >= k: break
    out = {"q": q, "org": target, "org_note": org_note, "hits": hits, "info": {kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in info.items()},
           "sec": None, "answer": None, "model": None, "guard": "v3"}
    ment = H.mentioned_orgs(q, set(c.org_names))
    llm, model = pick_llm()
    if llm is None:
        out["gate"] = {"action": "answer", "message": None, "disclaimer": DISCLAIMER, "note": "답변 생성 모델이 없어 검색 근거만 표시"}
        out["sec"] = round(time.time() - t0, 2); return JSONResponse(out)
    # 하네스 입력: 조문 본문을 합친 기사 단위(평가와 같은 형식)
    articles = [{"org": h["org"], "label": f"{h['article_no']}({h['title']})", "revision_date": h["revision_date"] or "", "text": h["text"]} for h in hits]
    g = Guard(llm, scope_classifier=None, subject_check=True, relaxed=False)
    try:
        resp = g.respond(q, target if len(ment) < 2 else None, articles, GD["delegated"].get(target, []), compare_ok=False,
                         scope_article_text=GD["scope_article"].get(target, ""))
    except Exception as e:  # 생성 모델 호출 실패(한도 초과 402 등): 답변 없이 검색 근거만. 가드레일 원칙상 추측 답변은 내지 않는다
        out["gate"] = {"action": "answer", "message": None, "disclaimer": DISCLAIMER, "note": f"답변 생성 모델을 쓸 수 없어 검색 근거만 표시합니다 ({type(e).__name__})"}
        out["sec"] = round(time.time() - t0, 2); return JSONResponse(out)
    act = ACTION[resp.decision]
    out["answer"] = resp.text; out["model"] = f"guard-v3 / {model}"
    out["gate"] = {"action": act, "message": None if act == "answer" else resp.text, "disclaimer": DISCLAIMER,
                   "quotes": [{"article": x["article"], "text": x["text"]} for x in resp.quotes], "candidates": resp.candidates,
                   "trace": {kk: v for kk, v in resp.trace.items() if kk in ("input", "verified", "rejected", "relinked", "subject", "final_check", "llm_parse_error")}}
    if act != "answer": out["answer"] = None  # 거절 문안은 gate.message 로, 화면은 근거 조문(후보)을 그대로 보여 준다
    out["sec"] = round(time.time() - t0, 2)
    try:
        with open(ROOT / "guard_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "q": q, "org": target, "action": act, "n_quotes": len(resp.quotes), "sec": out["sec"]}, ensure_ascii=False) + "\n")
    except Exception: pass
    return JSONResponse(out)
