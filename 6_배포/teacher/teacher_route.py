# -*- coding: utf-8 -*-
"""교원·교직원 규정 챗봇 API (티타임용). 검색은 run_eval 의 bge-m3 + BM25 → RRF → 재랭커를 그대로 쓰되,
기관 필터 대신 문서(doc)·직종(audience)·설립(applies_to) 마스크로 후보를 좁힌다. 답변은 가드레일 v3
(5_가드레일/v2/guard.py)가 조문 원문 문장만 골라 조립한다. run_eval.py·hitl_serve.py·guard.py 는 수정하지 않는다.

경로
  GET /ask?q=&doc=&audience=&applies_to=&k=   X-RAG-KEY 헤더 필요(RAG_KEY 설정 시)
  GET /orgs                                     문서명 목록 (티타임의 '문서 선택' 자동완성)
  GET /health                                   기동·모델 준비 여부
근거 추출 LLM 우선순위: ANTHROPIC_API_KEY(claude-haiku-4-5) → HF_TOKEN(Qwen2.5-72B) → 없으면 검색 근거만.
"""
import os, re, json, time
from collections import defaultdict
from pathlib import Path
import numpy as np
from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse
import run_eval as E
import hitl_serve as H
from guard import Guard, ANSWER, REFUSE_SCOPE, REFUSE_NO_EVIDENCE, REFUSE_ABUSE, REFUSE_SUBJECT, ASK_ORG, NO_COMPARE

ROOT = Path(__file__).resolve().parent
RAG_KEY = os.environ.get("RAG_KEY", "")
HF_MODEL = os.environ.get("HF_MODEL", "Qwen/Qwen2.5-72B-Instruct")
HAIKU = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5")
AUDIENCES = ("교원", "행정직", "공무직")
APPLIES = ("공립", "사립")
DISCLAIMER = "※ 법령·규정 원문을 그대로 보여드린 참고 정보입니다. 실제 적용 여부는 학교 행정실 또는 교육청 담당 부서에 확인해 주세요."
ACTION = {ANSWER: "answer", REFUSE_SCOPE: "refuse_scope", REFUSE_NO_EVIDENCE: "refuse", REFUSE_SUBJECT: "refuse_subject",
          REFUSE_ABUSE: "refuse_abuse", ASK_ORG: "ask_org", NO_COMPARE: "refuse_multi"}
TOKEN_RE = E.TOKEN_RE
CORPUS_NAME = "TEACH_A"

api = FastAPI(title="교원 규정 RAG")

_corpus = {}
def corpus():
    """교원 코퍼스. 공기업 데모(hitl_serve.corpus, L2P_A)와 한 Space 에 같이 올라가므로 따로 든다."""
    if "c" not in _corpus: _corpus["c"] = E.Corpus(CORPUS_NAME)
    return _corpus["c"]

# ---------------------------------------------------------------- 근거 추출 LLM (JSON 문자열 반환)
def _json_block(s):
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s.strip())
    i, j = s.find("{"), s.rfind("}")
    return s[i:j + 1] if i >= 0 and j > i else s

class AnthropicLLM:
    def __init__(self, model): self.model = model; self._c = None
    def __call__(self, system, user, schema=None):
        import anthropic
        if self._c is None: self._c = anthropic.Anthropic()
        r = self._c.messages.create(model=self.model, max_tokens=1200, temperature=0, system=system, messages=[{"role": "user", "content": user}])
        return _json_block("".join(b.text for b in r.content if b.type == "text"))

class HFLLM:
    def __init__(self, model): self.model = model
    def __call__(self, system, user, schema=None):
        import urllib.request
        for fmt in ({"type": "json_object"}, None):
            body = {"model": self.model, "temperature": 0, "max_tokens": 1200,
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
            if fmt: body["response_format"] = fmt
            req = urllib.request.Request("https://router.huggingface.co/v1/chat/completions", data=json.dumps(body).encode("utf-8"),
                                         headers={"Content-Type": "application/json", "Authorization": "Bearer " + os.environ["HF_TOKEN"]})
            try:
                with urllib.request.urlopen(req, timeout=300) as r:
                    return _json_block(json.loads(r.read().decode("utf-8"))["choices"][0]["message"]["content"])
            except urllib.error.HTTPError as e:
                if fmt and e.code == 400: continue
                raise
        raise RuntimeError("HF chat failed")

def pick_llm():
    if os.environ.get("ANTHROPIC_API_KEY"): return AnthropicLLM(HAIKU), HAIKU
    if os.environ.get("HF_TOKEN"): return HFLLM(HF_MODEL), HF_MODEL
    return None, None

# ---------------------------------------------------------------- 검색 (마스크 기반 hybrid prefilter)
_meta = {}
def meta(c):
    """코퍼스 행의 audience / applies_to 를 numpy 배열로 (한 번만)."""
    if "a" not in _meta:
        _meta["a"] = np.array([r.get("audience") or "전체" for r in c.rows])
        _meta["p"] = np.array([r.get("applies_to") or "전체" for r in c.rows])
    return _meta["a"], _meta["p"]

def build_mask(c, doc, audience, applies_to):
    aud, app = meta(c)
    mask = np.ones(len(c.rows), dtype=bool)
    if doc: mask &= c.orgs == doc
    if audience in AUDIENCES: mask &= (aud == "전체") | (aud == audience)
    if applies_to in APPLIES: mask &= (app == "전체") | (app == applies_to)
    return mask

def search_masked(c, q, mask):
    """run_eval._prefilter_core 와 같은 절차를 임의 마스크로. 반환: (청크 index 순위, info)"""
    q_score = H.strip_org_names(q, H.mentioned_orgs(q, set(c.org_names))) or q
    qv = E.st_model("BAAI/bge-m3").encode([q_score], normalize_embeddings=True, convert_to_numpy=True)[0]
    idx = np.flatnonzero(mask)
    if len(idx) == 0: return [], {"doc_filter": None, "pool": 0, "top1_ce": None, "abstain": True}
    d = np.where(mask, c.bge @ qv, -np.inf)
    b = np.full(len(c.rows), -np.inf); b[idx] = c.bm25.get_batch_scores(TOKEN_RE.findall(q_score.lower()), idx.tolist())
    d_rank = {int(i): r for r, i in enumerate(np.argsort(-d)[:E.CANDIDATE_K]) if mask[i]}
    b_rank = {int(i): r for r, i in enumerate(np.argsort(-b)[:E.CANDIDATE_K]) if mask[i]}
    rrf = defaultdict(float)
    for i, r in d_rank.items(): rrf[i] += 1 / (E.RRF_K + r)
    for i, r in b_rank.items(): rrf[i] += 1 / (E.RRF_K + r)
    cand = sorted(rrf, key=lambda i: (-rrf[i], i)); pool, rest = cand[:E.RERANK_TOP_N], cand[E.RERANK_TOP_N:]
    ce = E.ce_model().predict([(q_score, c.texts[i]) for i in pool], batch_size=16, show_progress_bar=False) if pool else []
    ce_map = {i: float(s) for i, s in zip(pool, ce)}
    pool.sort(key=lambda i: (-ce_map[i], -rrf[i], i)); top = pool + rest
    return top, {"pool": len(pool), "top1_ce": ce_map[top[0]] if top else None, "abstain": bool(top and ce_map.get(top[0], 0) < 0)}

ANSWER_SYSTEM = """너는 고등학교 교원과 행정직원의 규정 질문을 돕는 안내 도우미야. 아래 [참고 조문]만 근거로 2~4문장으로 쉽게 답해.
규칙: 조문에 없는 내용은 쓰지 말고, 없으면 "조문에서 확인되지 않습니다"라고 말해. 숫자·기간·나이는 조문에 있는 그대로 써.
답 끝에 근거를 "(근거: 법령명 제N조)" 형식으로 적어. 존댓말로 쓴다.
[참고 조문]
{context}"""

def gen_answer(q, hits):
    """기존 데모 방식: LLM 이 조문을 근거로 자연어 답을 쓰고, 숫자·조문번호가 근거에 있는지 검증한다(G3). 실패하면 None."""
    import guardrails as GR
    ctx = "\n\n".join(f"[{h['org']} {h['article_no']}({h['title']}) | 시행 {h['revision_date']}]\n{h['text'][:1500]}" for h in hits)
    system = ANSWER_SYSTEM.format(context=ctx)
    if os.environ.get("ANTHROPIC_API_KEY"): text = H.haiku_chat(system, q)
    elif os.environ.get("HF_TOKEN"): text = H.hf_chat(system, q)
    else: return None, None
    v = GR.verify_answer(text, [h["text"] for h in hits], [h["article_no"] for h in hits])
    return (text.strip() if v["ok"] else None), v

def polish(text):
    """guard.py 의 고정 문구는 공공기관 인사규정 기준이라(수정하지 않는다) 교원 규정 표현으로 바꾸고, 면책 문구는 gate.disclaimer 로만 보낸다."""
    text = re.sub(r"^.*인사규정에서 찾은 관련 조문입니다\.?\s*\n+", "", text)
    text = re.sub(r"\n*※[^\n]*$", "", text.rstrip())
    return text.replace(" 인사규정 ", " ").strip()

# ---------------------------------------------------------------- 라우트
def _auth(request: Request):
    return not RAG_KEY or request.headers.get("x-rag-key", "") == RAG_KEY

@api.get("/health")
def health():
    return {"ok": True, "corpus": "TEACH_A", "chunks": len(corpus().rows), "llm": pick_llm()[1]}

@api.get("/orgs")
def orgs(request: Request):
    if not _auth(request): return JSONResponse({"error": "unauthorized"}, status_code=401)
    c = corpus()
    seen, out = set(), []
    for r in c.rows:
        if r["org"] in seen: continue
        seen.add(r["org"]); out.append(r["org"])
    return out

@api.get("/ask")
def ask(request: Request, q: str = Query(..., min_length=1, max_length=300), doc: str = Query(""), org: str = Query(""),
        audience: str = Query(""), applies_to: str = Query(""), k: int = Query(5, ge=1, le=10)):
    if not _auth(request): return JSONResponse({"error": "unauthorized"}, status_code=401)
    c = corpus(); t0 = time.time()
    doc = doc or org  # 티타임 구버전 호환: org 파라미터도 받는다
    target, doc_note = H.resolve_org(q, doc, c.org_names)   # 질문에 문서명이 있으면 그쪽을 따른다
    mask = build_mask(c, target, audience, applies_to)
    timing = {}
    idx, info = search_masked(c, q, mask); timing["search"] = round(time.time() - t0, 2)
    arts, hits = [], []
    for i in idx:
        r = c.rows[i]
        if r["article_id"] in arts: continue
        arts.append(r["article_id"])
        hits.append({"rank": len(arts), "org": r["org"], "article_no": r["article_no"], "title": r.get("article_title", ""),
                     "revision_date": r["revision_date"], "text": r["text"], "layer": r.get("layer"), "audience": r.get("audience"),
                     "applies_to": r.get("applies_to"), "source_url": r.get("source_url")})
        if len(hits) >= k: break
    out = {"q": q, "org": target, "org_note": doc_note, "audience": audience or None, "applies_to": applies_to or None,
           "hits": hits, "info": {kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in info.items()},
           "sec": None, "answer": None, "model": None, "guard": "v3"}
    llm, model = pick_llm()
    if llm is None or not hits:
        out["gate"] = {"action": "answer" if hits else "refuse", "message": None if hits else "관련 조문을 찾지 못했습니다.",
                       "disclaimer": DISCLAIMER, "note": None if hits else None}
        if hits and llm is None: out["gate"]["note"] = "답변 생성 모델이 없어 검색 근거만 표시"
        out["sec"] = round(time.time() - t0, 2); return JSONResponse(out)
    ment = H.mentioned_orgs(q, set(c.org_names))
    # 라벨에 법령명을 넣어 답변의 "■ 제44조(휴직)" 줄이 어느 법인지 보이게 한다
    articles = [{"org": h["org"], "label": f"{h['org']} {h['article_no']}({h['title']})", "revision_date": h["revision_date"] or "", "text": h["text"]} for h in hits]
    g = Guard(llm, scope_classifier=None, subject_check=False, relaxed=False)
    t1 = time.time()
    try:
        # 가드레일은 기관이 비면 되묻는다(R4). 문서를 지정하지 않은 질문은 코퍼스 전체를 '교원·교직원 규정' 한 묶음으로 본다.
        resp = g.respond(q, (target or "교원·교직원 규정") if len(ment) < 2 else None, articles, (), compare_ok=False, scope_article_text="")
    except Exception as e:  # 모델 호출 실패(한도 등): 추측 답변 대신 근거만
        out["gate"] = {"action": "answer", "message": None, "disclaimer": DISCLAIMER, "note": f"답변 생성 모델을 쓸 수 없어 검색 근거만 표시합니다 ({type(e).__name__})"}
        out["sec"] = round(time.time() - t0, 2); return JSONResponse(out)
    timing["guard_llm"] = round(time.time() - t1, 2)
    act = ACTION[resp.decision]
    out["model"] = f"guard-v3 / {model}"
    out["gate"] = {"action": act, "message": None if act == "answer" else resp.text, "disclaimer": DISCLAIMER,
                   "quotes": [{"article": x["article"], "text": x["text"]} for x in resp.quotes], "candidates": resp.candidates,
                   "trace": {kk: v for kk, v in resp.trace.items() if kk in ("input", "verified", "rejected", "relinked", "final_check", "llm_parse_error")}}
    if act == "answer":
        # 자연어 답(검증 통과 시) 을 본문으로, 가드레일이 검증한 원문 인용은 gate.quotes 로 함께 보낸다. 검증 실패면 원문 인용만.
        natural, verify = None, None
        t2 = time.time()
        try: natural, verify = gen_answer(q, hits)
        except Exception as e: out["gate"]["gen_error"] = type(e).__name__
        timing["gen_llm"] = round(time.time() - t2, 2)
        out["gate"]["verify"] = verify
        out["answer"] = natural or polish(resp.text)
        out["gate"]["answer_mode"] = "generated" if natural else "quotes"
    else:
        out["answer"] = None
    out["sec"] = round(time.time() - t0, 2); out["timing"] = timing
    try:
        with open(ROOT / "guard_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "q": q, "doc": target, "audience": audience, "action": act,
                                "n_quotes": len(resp.quotes), "sec": out["sec"]}, ensure_ascii=False) + "\n")
    except Exception: pass
    return JSONResponse(out)
