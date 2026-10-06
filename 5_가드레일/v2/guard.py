# -*- coding: utf-8 -*-
"""가드레일 v2 하네스 (정책: ../가드레일_정책_v2.md).

정해진 순서로만 동작한다. 각 단계의 입력과 출력이 고정돼 있고, 답변 본문은 검증된 조문 원문으로만 만든다.

  1. 입력 판정 (규칙)    : R6 악용 → R4 기관 불명·비교 질문
  2. 근거 추출 (LLM)     : 답을 쓰지 않는다. 조문에서 답이 되는 문장을 글자 그대로 골라 JSON 으로만 낸다
  3. 근거 검증 (규칙)    : 고른 문장이 해당 조문 본문에 문자 그대로 있는지 확인. 없으면 버린다
  4. 처리 결정 (규칙)    : 검증된 문장이 있으면 답변, 없으면 R1(범위 밖) 또는 R2(근거 없음) 거절
  5. 응답 조립 (템플릿)  : 고정 문구 + 검증된 원문만. 생성 문장 없음
  6. 최종 검사 (규칙)    : 조립된 응답을 다시 검사. 실패하면 R2 거절로 바꾼다 (fail-closed)

근거 없는 답변은 5·6 단계 때문에 구조적으로 나올 수 없다. 거절률·오거절률은 2단계 LLM 의 추출 품질에 달려 있어 측정으로 확인한다.
"""
import json, re, time, unicodedata, urllib.request
from dataclasses import dataclass, field, asdict

# ---------------------------------------------------------------- 결정 유형
ANSWER = "ANSWER"                    # 검증된 근거로 답변
REFUSE_SCOPE = "REFUSE_SCOPE"        # R1 범위 밖 (다른 규정 소관·무관)
REFUSE_NO_EVIDENCE = "REFUSE_NO_EVIDENCE"  # R2 조문에서 답을 찾지 못함
REFUSE_ABUSE = "REFUSE_ABUSE"        # R6 지시 무력화 시도
ASK_ORG = "ASK_ORG"                  # R4 기관 불명 → 되묻기
NO_COMPARE = "NO_COMPARE"            # R4 기관 비교 질문 → 미지원 안내
REFUSALS = {REFUSE_SCOPE, REFUSE_NO_EVIDENCE, REFUSE_ABUSE}

DISCLAIMER = "※ 인사규정 원문을 그대로 보여드린 참고 정보입니다. 실제 적용 여부는 인사부서에 확인해 주세요."
PII_WARNING = "※ 질문에 개인정보(주민등록번호·전화번호)가 포함되어 있습니다. 개인정보는 입력하지 마세요."
PERSONAL_NOTE = "※ 특정인의 결과(징계 수위, 면직 여부 등)는 규정만으로 단정할 수 없습니다. 아래 조문을 참고하고 인사부서에 확인해 주세요."
PARTIAL_NOTE = "※ 질문 중 아래 조문에 없는 부분은 이 인사규정에 명시되어 있지 않습니다."

# ---------------------------------------------------------------- 1단계 규칙
INJECTION = [r"(이전|앞의|위의|기존|모든)\s*(지시|명령|규칙|프롬프트|설정)\S*\s*.{0,6}(무시|잊|따르지)",
             r"시스템\s*프롬프트", r"ignore\s+(all\s+|the\s+)?(previous|prior|above)", r"jailbreak|탈옥",
             r"(관리자|개발자|디버그)\s*모드", r"너(는|의)\s*(역할|규칙|지시).{0,8}(바꿔|무시|해제|잊)"]
PII = [r"\d{6}\s*-\s*[1-4]\d{6}", r"01[016789][-\s]?\d{3,4}[-\s]?\d{4}"]
COMPARE = [r"기관(들)?\s*(중|마다|별로|끼리|간)", r"(다른|여러)\s*(기관|공사|공단|회사)", r"비교해", r"기관별"]
def _any(pats, s): return any(re.search(p, s, re.I) for p in pats)

def norm(s):
    s = unicodedata.normalize("NFC", s or "")
    s = s.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    return re.sub(r"\s+", " ", s).strip()

# ---------------------------------------------------------------- 2단계 LLM
EVIDENCE_SCHEMA = {
    "type": "object",
    "properties": {
        "scope": {"type": "string", "enum": ["인사규정", "다른규정", "무관"]},
        "other_rule": {"type": "string"},
        "personal_judgment": {"type": "boolean"},
        "fully_answered": {"type": "boolean"},
        "quotes": {"type": "array", "items": {"type": "object", "properties": {"source": {"type": "integer"}, "text": {"type": "string"}}, "required": ["source", "text"]}},
    },
    "required": ["scope", "other_rule", "personal_judgment", "fully_answered", "quotes"],
}
EVIDENCE_SYSTEM = """너는 공공기관 인사규정의 근거 추출기다. 답변을 쓰지 않는다. JSON 만 출력한다.

[조문]에서 질문에 대한 답이 되는 문장을 찾아 글자 그대로 복사한다.

규칙
1. quotes[].text 는 [조문] 본문을 한 글자도 바꾸지 않고 그대로 복사한다. 요약, 의역, 말줄임(...)은 금지다. 한 문장 또는 한 항(①, ② 등) 단위로 복사한다.
2. quotes[].source 는 그 문장이 있는 [조문] 번호다.
3. 질문에 직접 답이 되는 문장만 넣는다. 주제가 비슷할 뿐 답이 아니면 넣지 않는다.
4. 답이 되는 문장이 없으면 quotes 는 빈 배열 [] 이다.
5. scope: 질문 주제가 인사 사항(채용, 수습, 임용, 승진, 전보, 휴직, 복직, 징계 절차, 직위해제, 퇴직, 정년, 인사위원회 등)이면 "인사규정".
   보수·수당·급여, 연차·휴가·근무시간, 여비·출장, 복리후생, 교육훈련 세부 등 [위임 규정 목록]에 있거나 다른 규정이 정하는 주제면 "다른규정".
   직장 규정과 관계없는 질문이면 "무관".
6. other_rule: scope 가 "다른규정"이면 [위임 규정 목록]에서 가장 맞는 규정 이름을 그대로 쓴다. 목록에 없으면 짐작되는 규정 이름. 그 외에는 "".
7. personal_judgment: 특정인의 결과(징계 수위, 해고·면직 여부, 승진 여부 등)를 단정해 달라는 질문이면 true.
8. fully_answered: 복사한 문장들로 질문에 빠짐없이 답이 되면 true, 일부만 답이 되면 false. quotes 가 비면 false."""

def build_user_prompt(query, org, delegated, articles):
    deleg = "\n".join(f"- {d}" for d in delegated) or "- (없음)"
    arts = "\n\n".join(f"[{i}] {a['org']} 인사규정 {a['label']}\n{a['text']}" for i, a in enumerate(articles, 1))
    return f"[질문]\n{query}\n\n[질문자 소속 기관]\n{org}\n\n[위임 규정 목록]\n{deleg}\n\n[조문]\n{arts}"

class OllamaLLM:
    def __init__(self, model, host="http://127.0.0.1:11434", num_ctx=8192):
        self.model, self.host, self.num_ctx = model, host, num_ctx
    def __call__(self, system, user):
        body = {"model": self.model, "stream": False, "format": EVIDENCE_SCHEMA, "keep_alive": "30m",
                "options": {"temperature": 0, "num_ctx": self.num_ctx, "num_predict": 1200},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        req = urllib.request.Request(self.host + "/api/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=1800) as r:
            return json.loads(r.read().decode())["message"]["content"]

# ---------------------------------------------------------------- 결과 객체
@dataclass
class Response:
    decision: str
    text: str
    quotes: list = field(default_factory=list)       # [{"source": i, "article": label, "text": 원문}]
    candidates: list = field(default_factory=list)   # R2 거절 시 후보 조항
    other_rule: str = ""
    flags: dict = field(default_factory=dict)
    trace: dict = field(default_factory=dict)        # 단계별 판정 기록 (디버깅·감사용)
    def to_dict(self): return asdict(self)

# ---------------------------------------------------------------- 5단계 템플릿
def render_answer(org, articles, quotes, flags):
    lines = []
    if flags.get("pii"): lines.append(PII_WARNING)
    if flags.get("personal_judgment"): lines.append(PERSONAL_NOTE)
    lines.append(f"{org} 인사규정에서 찾은 관련 조문입니다.")
    for q in quotes:
        a = articles[q["source"] - 1]
        lines.append(f"■ {a['label']} · {a['revision_date']} 개정\n“{q['text']}”")
    if not flags.get("fully_answered", True): lines.append(PARTIAL_NOTE)
    lines.append(DISCLAIMER)
    return "\n\n".join(lines)

def render_refusal(decision, org="", other_rule="", candidates=(), flags=None):
    flags = flags or {}
    head = [PII_WARNING] if flags.get("pii") else []
    if decision == REFUSE_SCOPE:
        body = (f"이 내용은 {org} 인사규정이 아니라 {other_rule} 소관입니다. 해당 규정이나 인사부서에 확인해 주세요."
                if other_rule else f"이 내용은 {org} 인사규정에서 다루지 않는 사항입니다. 관련 규정이나 담당 부서에 확인해 주세요.")
    elif decision == REFUSE_NO_EVIDENCE:
        body = f"{org} 인사규정에서 질문에 대한 답을 찾지 못했습니다."
        if candidates: body += "\n혹시 다음 조항을 찾으시나요?\n" + "\n".join(f"- {c}" for c in candidates)
    elif decision == REFUSE_ABUSE:
        body = "이 요청은 처리할 수 없습니다. 인사규정에 대해 질문해 주세요."
    elif decision == ASK_ORG:
        body = "어느 기관의 인사규정을 찾을까요? 기관 이름을 함께 알려 주세요."
    elif decision == NO_COMPARE:
        body = "여러 기관을 비교하는 질문은 아직 지원하지 않습니다. 기관을 하나씩 지정해 질문해 주세요."
    else:
        raise ValueError(decision)
    return "\n\n".join(head + [body])

# ---------------------------------------------------------------- 6단계 최종 검사
def verify_grounded(resp, org, articles):
    """답변이 '고정 문구 + 검증된 원문'으로만 이뤄졌는지 확인한다. 거절 응답은 근거 주장이 없으므로 통과."""
    if resp.decision != ANSWER: return True
    if not resp.quotes: return False
    for q in resp.quotes:
        if not (1 <= q["source"] <= len(articles)): return False
        if norm(q["text"]) not in norm(articles[q["source"] - 1]["text"]): return False
    return resp.text == render_answer(org, articles, resp.quotes, resp.flags)

# ---------------------------------------------------------------- 하네스
class Guard:
    MIN_QUOTE = 8          # 이보다 짧은 인용은 근거로 인정하지 않는다 (조사·단어 조각 방지)
    MAX_QUOTES = 4
    N_CANDIDATES = 3

    def __init__(self, llm):
        self.llm = llm

    def respond(self, query, org, articles, delegated=(), compare_ok=False):
        """query: 질문, org: 확정된 기관(없으면 None), articles: 검색된 조문 [{org,label,revision_date,text}], delegated: 위임 규정 목록."""
        t0 = time.time(); trace = {}
        flags = {"pii": _any(PII, query)}
        # 1단계: 입력 규칙
        if _any(INJECTION, query):
            trace["input"] = "R6"; return self._done(Response(REFUSE_ABUSE, render_refusal(REFUSE_ABUSE, flags=flags), flags=flags, trace=trace), t0)
        if not org:
            d = NO_COMPARE if (_any(COMPARE, query) and not compare_ok) else ASK_ORG
            trace["input"] = "R4"; return self._done(Response(d, render_refusal(d, flags=flags), flags=flags, trace=trace), t0)
        trace["input"] = "ok"
        candidates = [f"{a['label']}" for a in articles[: self.N_CANDIDATES]]
        if not articles:
            trace["evidence"] = "no_articles"
            return self._done(Response(REFUSE_NO_EVIDENCE, render_refusal(REFUSE_NO_EVIDENCE, org, flags=flags), flags=flags, trace=trace), t0)
        # 2단계: LLM 근거 추출
        raw = self.llm(EVIDENCE_SYSTEM, build_user_prompt(query, org, list(delegated), articles))
        try:
            ev = json.loads(raw); assert isinstance(ev, dict)
        except Exception:
            ev = {"scope": "인사규정", "other_rule": "", "personal_judgment": False, "fully_answered": False, "quotes": []}
            trace["llm_parse_error"] = raw[:200]
        trace["llm"] = ev
        # 3단계: 근거 검증
        verified, rejected, seen = [], [], set()
        for q in ev.get("quotes") or []:
            src, txt = q.get("source"), norm(q.get("text", ""))
            ok = isinstance(src, int) and 1 <= src <= len(articles) and len(txt) >= self.MIN_QUOTE and txt in norm(articles[src - 1]["text"])
            if ok and txt not in seen:
                seen.add(txt); verified.append({"source": src, "article": articles[src - 1]["label"], "text": txt})
            elif not ok: rejected.append(q)
        verified = verified[: self.MAX_QUOTES]
        trace["verified"], trace["rejected"] = len(verified), len(rejected)
        flags.update({"personal_judgment": bool(ev.get("personal_judgment")), "fully_answered": bool(ev.get("fully_answered")) if verified else False})
        # 4단계: 처리 결정
        if verified:
            resp = Response(ANSWER, render_answer(org, articles, verified, flags), quotes=verified, flags=flags, trace=trace)
        elif ev.get("scope") in ("다른규정", "무관"):
            rule = ev.get("other_rule", "").strip() if ev.get("scope") == "다른규정" else ""
            resp = Response(REFUSE_SCOPE, render_refusal(REFUSE_SCOPE, org, rule, flags=flags), other_rule=rule, flags=flags, trace=trace)
        else:
            resp = Response(REFUSE_NO_EVIDENCE, render_refusal(REFUSE_NO_EVIDENCE, org, candidates=candidates, flags=flags), candidates=candidates, flags=flags, trace=trace)
        # 6단계: 최종 검사 (fail-closed)
        if not verify_grounded(resp, org, articles):
            trace["final_check"] = "failed→R2"
            resp = Response(REFUSE_NO_EVIDENCE, render_refusal(REFUSE_NO_EVIDENCE, org, candidates=candidates, flags=flags), candidates=candidates, flags=flags, trace=trace)
        else:
            trace["final_check"] = "pass"
        return self._done(resp, t0)

    @staticmethod
    def _done(resp, t0):
        resp.trace["sec"] = round(time.time() - t0, 2); return resp
