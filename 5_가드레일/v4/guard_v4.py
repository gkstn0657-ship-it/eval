# -*- coding: utf-8 -*-
"""가드레일 v4 하네스 (아스트라 실행 지시 2026-10-12, 계획 ../가드레일_v4_계획.md).

원문 인용 방식은 v2 를 그대로 쓴다(추출 프롬프트·원문 검증). v2 에서 가져온 것: norm, EVIDENCE_SYSTEM, build_user_prompt, OllamaLLM,
INJECTION/COMPARE 규칙, SUBJECT_RE. v2 파일은 수정하지 않고 여기서 감싼다.

순서
  0. 개인정보 마스킹        질문의 식별자를 검색·모델·로그 전에 치환                        규칙
  1. 입력 규칙              직접 인젝션 → 거부, 기관 없음 → 되묻기/비교 미지원                규칙
  2. 근거 코드 검사         기관 일치, 출처 ID 존재, 버전 존재, 삭제 조항, 문서 내 지시        규칙 (탈락 조문은 근거에서 제외)
  3. 적용 대상              포함 확인 / 배제 확인 / 미확인 / 미발동                            규칙 (배제 → 거부, 미확인 → 보류)
  4. 근거 추출              7B 가 조문 문장을 글자 그대로 고름. (선택) 14B 캐스케이드 1회       LLM
  5. 원문 검증              문자열 대조 + 출처 재연결                                          규칙
  6. (선택) 답변성 판정     M4 가 조문 단위로 충분/일부/관련만/모순                            판별기 (이번 구현에는 훅만)
  7. 응답 조립·최종 검사    원문 카드 + 고정 문구. 고정 문구에는 사실 주장이 없다               규칙 (fail-closed)
  장애: 모델 호출 예외·타임아웃 → FAIL_SAFE (답변 없음, 코드 검사를 통과한 원문만 표시)
요청 격리: 모듈 전역 상태 없음. 모든 입력은 respond() 인자.
"""
import sys, re, json, time, html
from pathlib import Path
from dataclasses import dataclass, field, asdict
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "v2"))
from guard import (norm, EVIDENCE_SYSTEM, build_user_prompt, OllamaLLM, INJECTION, COMPARE, SUBJECT_RE, _any,  # noqa: F401
                   ANSWER, REFUSE_SUBJECT, REFUSE_NO_EVIDENCE, REFUSE_ABUSE, ASK_ORG, NO_COMPARE)

# ---------------------------------------------------------------- 결정 유형 (v2 + 신규)
HOLD_SUBJECT = "HOLD_SUBJECT"      # 적용 대상 미확인 → 답변 보류, 원문만
PARTIAL = "PARTIAL"                # 부분 답변 (M4 '일부' 또는 추출기 fully_answered=false)
FAIL_SAFE = "FAIL_SAFE"            # 장애: 미검증 답변 비공개
REFUSALS = {REFUSE_SUBJECT, REFUSE_NO_EVIDENCE, REFUSE_ABUSE}
HOLDS = {HOLD_SUBJECT, FAIL_SAFE, ASK_ORG, NO_COMPARE}
ANSWERS = {ANSWER, PARTIAL}

# ---------------------------------------------------------------- 고정 문구 (사실 주장 금지. 원문이거나 "확인하지 못했다"는 문장만)
PHRASES = {
    "disclaimer": "※ 인사규정 수집본의 원문을 그대로 보여드린 참고 정보입니다. 실제 적용 여부는 인사부서에 확인해 주세요.",
    "version": "수집본 기준: {org} 인사규정 {date} 개정본",            # '최신'이라고 적지 않는다
    "pii": "※ 질문에 포함된 개인정보는 가린 뒤 처리했습니다. 개인정보는 입력하지 마세요.",
    "personal": "※ 특정인의 결과(징계 수위, 면직 여부 등)는 규정 원문만으로 단정할 수 없습니다. 아래 원문을 참고하고 인사부서에 확인해 주세요.",
    "partial": "※ 질문 중 위 원문에서 확인되지 않은 부분은 검색된 자료로는 확인하지 못했습니다.",
    "not_found": "검색된 자료에서는 질문에 대한 답을 확인하지 못했습니다.",
    "candidates": "검색된 조항은 다음과 같습니다. 필요하면 원문을 확인해 주세요.",
    "subject_excluded": "적용 범위 조항에 '{subj}'에 적용하지 않는다는 취지의 문구가 있습니다. 아래 원문을 확인해 주세요.",
    "subject_unknown": "검색된 자료의 적용 범위 조항에서 '{subj}'이(가) 포함되는지 확인하지 못했습니다. 답변 대신 관련 원문만 보여드립니다.",
    "subject_no_scope": "검색된 자료에 적용 범위 조항이 없어 '{subj}'이(가) 적용 대상인지 확인하지 못했습니다. 답변 대신 관련 원문만 보여드립니다.",
    "deleted": "다음 조항은 수집본에 삭제로 표기되어 있어 근거로 쓰지 않았습니다: {items}",
    "dropped": "다음 검색 결과는 코드 검사({reasons})에서 제외했습니다: {items}",
    "delegation": "관련 조문에 다음 위임 문구가 있습니다. 어느 규정 소관인지는 이 문구와 인사부서 확인으로 판단해 주세요.",
    "failsafe": "답변 생성 단계에서 오류가 발생해 답변을 만들지 않았습니다. 검색된 원문만 보여드립니다.",
    "abuse": "이 요청은 처리할 수 없습니다. 인사규정에 대해 질문해 주세요.",
    "ask_org": "어느 기관의 인사규정을 찾을까요? 기관 이름을 함께 알려 주세요.",
    "no_compare": "여러 기관을 비교하는 질문은 지원하지 않습니다. 기관을 하나씩 지정해 질문해 주세요.",
}
CLAIM_WORDS = re.compile(r"(소관입니다|적용됩니다|직원에게만|최신|별도 규정에 따릅니다|명시되어 있지 않습니다)")  # 고정 문구 전수 점검용
assert not any(CLAIM_WORDS.search(v) for v in PHRASES.values()), "고정 문구에 사실 주장이 들어 있다"

# ---------------------------------------------------------------- 0. 개인정보
PII_PATTERNS = [(re.compile(r"\d{6}\s*-\s*[1-4]\d{6}"), "[주민등록번호]"), (re.compile(r"01[016789][-\s.]?\d{3,4}[-\s.]?\d{4}"), "[전화번호]"),
                (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[이메일]"), (re.compile(r"(사번|사원번호)\s*[:：]?\s*[A-Za-z0-9-]{4,}"), "[사번]")]
def mask_pii(text):
    found = []
    for pat, tag in PII_PATTERNS:
        if pat.search(text): found.append(tag); text = pat.sub(tag, text)
    return text, found

# ---------------------------------------------------------------- 2. 근거 코드 검사
DELETED_RE = re.compile(r"^\s*(\[[^\]]*\]\s*)?제\s*\d+\s*조(의\s*\d+)?\s*(\([^)]*\))?\s*(삭제|삭\s*제)\s*(\(|<|\[|$)")
DOC_INJECTION = INJECTION + [r"(답변|응답|출력)에\s*.{0,12}(포함|추가|출력)\s*(하라|하세요|할 것|해라)", r"(이 문서|이 조문|아래 내용)를?\s*(읽는|보는)\s*(AI|모델|어시스턴트|챗봇)",
                             r"(assistant|system)\s*:\s*", r"<\s*/?\s*(script|system|instruction)"]
def is_deleted_article(text):
    """조문 전체가 삭제 표기인가 (항 일부 삭제는 아님). '제N조(제목) 삭제 (날짜)' 뒤에 실체 문장이 없어야 한다."""
    t = norm(text)
    if not DELETED_RE.match(t): return False
    rest = DELETED_RE.sub("", t); rest = re.sub(r"^[^)]*\)\s*", "", rest) if rest.startswith("(") else rest
    rest = re.sub(r"[\d.\s()<>\[\]개정신설]+", "", rest)
    return len(rest) < 6
def doc_injection(text): return _any(DOC_INJECTION, text)
def check_evidence(selected_org, articles, known_ids=None):
    """통과한 조문과 탈락한 조문(사유)을 돌려준다. 통과 조문의 revision_date 가 섞여 있으면 version_mixed 를 표시한다."""
    ok, dropped = [], []
    for a in articles:
        reasons = []
        if selected_org and a.get("org") != selected_org: reasons.append("org_mismatch")
        if known_ids is not None and a.get("id") not in known_ids: reasons.append("unknown_id")
        if not a.get("revision_date"): reasons.append("no_version")
        if is_deleted_article(a.get("text", "")): reasons.append("deleted")
        if doc_injection(a.get("text", "")): reasons.append("doc_injection")
        (dropped if reasons else ok).append({**a, "reasons": reasons} if reasons else a)
    dates = {a.get("revision_date") for a in ok}
    return ok, dropped, {"version_mixed": len(dates) > 1, "versions": sorted(dates)}

# ---------------------------------------------------------------- 3. 적용 대상 (세 상태)
def subject_status(query, scope_article_text=""):
    """('none'|'covered'|'excluded'|'unknown', 대상 명사). 명시적 긍정만 covered, 명시적 부정만 excluded, 둘 다거나 단서(다만 등)가 붙으면 unknown."""
    m = SUBJECT_RE.search(query)
    if not m: return "none", None
    subj = re.sub(r"\s", "", m.group(0)); key = "임원" if "임원" in subj else subj[:2]
    t = norm(scope_article_text)
    if not t or key not in t: return "unknown", subj
    pos = re.search(key + r".{0,20}(포함|적용|준용)(?!\s*(하지|되지|대상에서|대상이\s*아니))", t)
    neg = re.search(key + r".{0,20}((적용|포함|준용)(하지|되지)\s*(않|아니)|제외|적용\s*대상이\s*아니)", t)
    exc = re.search(key + r".{0,80}(다만|단,|단서|그러하지\s*아니|경우에는\s*그러하지)", t)
    if pos and neg: return "unknown", subj
    if neg: return ("unknown" if exc else "excluded"), subj
    if pos: return "covered", subj
    return "unknown", subj

# ---------------------------------------------------------------- 4·5. 추출과 검증 (v2 와 같은 기준)
MIN_QUOTE, MAX_QUOTES, N_CANDIDATES = 8, 4, 3
def extract(llm, query, org, delegated, articles):
    """llm(system, user) → JSON 문자열. 반환 (ev dict, parse_error 문자열 또는 None). 호출 예외는 밖으로 던진다(FAIL_SAFE 처리)."""
    raw = llm(EVIDENCE_SYSTEM, build_user_prompt(query, org, list(delegated), articles))
    try:
        ev = json.loads(raw); assert isinstance(ev, dict); return ev, None
    except Exception:
        return {"scope": "", "other_rule": "", "personal_judgment": False, "fully_answered": False, "quotes": []}, (raw or "")[:200]
def verify_quotes(ev, articles):
    verified, rejected, seen, relinked = [], [], set(), 0
    normed = [norm(a["text"]) for a in articles]
    for q in ev.get("quotes") or []:
        src, txt = q.get("source"), norm(str(q.get("text", "")))
        if len(txt) < MIN_QUOTE: rejected.append(q); continue
        if isinstance(src, int) and 1 <= src <= len(articles) and txt in normed[src - 1]: ok_src = src
        else:
            hits = [i + 1 for i, a in enumerate(normed) if txt in a]; ok_src = hits[0] if len(hits) == 1 else None
            if ok_src: relinked += 1
        if ok_src and txt not in seen: seen.add(txt); verified.append({"source": ok_src, "article": articles[ok_src - 1]["label"], "text": txt})
        elif not ok_src: rejected.append(q)
    return verified[:MAX_QUOTES], rejected, relinked

DELEGATION_RE = re.compile(r"[^.。]*?([가-힣A-Za-z·\s]{2,14}(규정|규칙|지침|요령|세칙))[^.。]*?(따른다|정한다|의한다|정하는 바)[^.。]*[.。]?")
def delegation_sentences(articles):
    """조문 원문 안의 위임 문장(규정명을 실제로 적은 문장)만 돌려준다. 규정명을 짐작해 만들지 않는다."""
    out = []
    for a in articles:
        for m in DELEGATION_RE.finditer(norm(a["text"])):
            s = m.group(0).strip()
            if 10 <= len(s) <= 160 and s not in [x["text"] for x in out]: out.append({"article": a["label"], "text": s})
    return out[:3]

# ---------------------------------------------------------------- 7. 응답 조립
@dataclass
class Response:
    decision: str
    text: str
    quotes: list = field(default_factory=list)
    candidates: list = field(default_factory=list)
    flags: dict = field(default_factory=dict)
    trace: dict = field(default_factory=dict)
    def to_dict(self): return asdict(self)
    def html(self): return "<pre>" + html.escape(self.text) + "</pre>"   # 안전한 표시: 원문·질문의 태그를 실행하지 않는다

def _card(a, body=None): return f"■ {a['org']} 인사규정 {a['label']} · {a.get('revision_date') or '개정일 미상'} 개정본\n“{body if body is not None else norm(a['text'])}”"
def _notes(flags):
    n = []
    if flags.get("pii"): n.append(PHRASES["pii"])
    if flags.get("personal_judgment"): n.append(PHRASES["personal"])
    return n
def _dropped_lines(dropped):
    lines = []
    dl = [d for d in dropped if "deleted" in d["reasons"]]
    if dl: lines.append(PHRASES["deleted"].format(items=", ".join(d["label"] for d in dl)))
    ot = [d for d in dropped if "deleted" not in d["reasons"]]
    if ot: lines.append(PHRASES["dropped"].format(reasons=", ".join(sorted({r for d in ot for r in d["reasons"]})), items=", ".join(d["label"] for d in ot)))
    return lines
def render(decision, org, articles, quotes=(), flags=None, dropped=(), scope_article_text="", full_cards=False, delegations=()):
    flags = flags or {}; L = _notes(flags)
    if decision in ANSWERS:
        L.append(PHRASES["version"].format(org=org, date=articles[quotes[0]["source"] - 1].get("revision_date") or "개정일 미상"))
        if full_cards:  # M4 가 조문 전체를 검증한 경우: 조문 전체 원문 카드
            for i in sorted({q["source"] for q in quotes}): L.append(_card(articles[i - 1]))
        else:
            for q in quotes: L.append(_card(articles[q["source"] - 1], q["text"]))
        if decision == PARTIAL: L.append(PHRASES["partial"])
    elif decision == REFUSE_NO_EVIDENCE:
        L.append(PHRASES["not_found"])
        if delegations: L.append(PHRASES["delegation"]); L += [f"- {d['article']}: “{d['text']}”" for d in delegations]
        if articles: L.append(PHRASES["candidates"]); L += [f"- {a['label']}" for a in articles[:N_CANDIDATES]]
    elif decision == REFUSE_SUBJECT:
        L.append(PHRASES["subject_excluded"].format(subj=flags.get("subject", ""))); L.append(f"“{norm(scope_article_text)}”")
    elif decision == HOLD_SUBJECT:
        L.append((PHRASES["subject_unknown"] if scope_article_text else PHRASES["subject_no_scope"]).format(subj=flags.get("subject", "")))
        if scope_article_text: L.append(f"“{norm(scope_article_text)}”")
        for a in articles[:N_CANDIDATES]: L.append(_card(a))
    elif decision == FAIL_SAFE:
        L.append(PHRASES["failsafe"]); L += [_card(a) for a in articles[:N_CANDIDATES]]
    elif decision == REFUSE_ABUSE: L.append(PHRASES["abuse"])
    elif decision == ASK_ORG: L.append(PHRASES["ask_org"])
    elif decision == NO_COMPARE: L.append(PHRASES["no_compare"])
    else: raise ValueError(decision)
    L += _dropped_lines(dropped)
    if decision not in (REFUSE_ABUSE, ASK_ORG, NO_COMPARE): L.append(PHRASES["disclaimer"])
    return "\n\n".join(L)

def verify_grounded(resp, org, articles, dropped=(), scope_article_text="", full_cards=False, delegations=()):
    """답변 응답이 '고정 문구 + 검증된 원문' 그대로인지 다시 확인한다. 답변이 아닌 응답은 근거 주장이 없으므로 통과."""
    if resp.decision not in ANSWERS: return True
    if not resp.quotes: return False
    for q in resp.quotes:
        if not (1 <= q["source"] <= len(articles)) or norm(q["text"]) not in norm(articles[q["source"] - 1]["text"]): return False
    return resp.text == render(resp.decision, org, articles, resp.quotes, resp.flags, dropped, scope_article_text, full_cards, delegations)

# ---------------------------------------------------------------- 하네스
class GuardV4:
    def __init__(self, llm7, llm14=None, cascade=False, m4=None, subject_rule=True, code_checks=True):
        """llm7/llm14: (system, user) → JSON 문자열. cascade: 14B 1회 재시도. m4: (query, article_text) → '충분'|'일부'|'관련만'|'모순' (없으면 끔)."""
        self.llm7, self.llm14, self.cascade, self.m4, self.subject_rule, self.code_checks = llm7, llm14, cascade, m4, subject_rule, code_checks

    def respond(self, query, org, articles, delegated=(), scope_article_text="", known_ids=None, compare_ok=False):
        t0 = time.time(); tr = {"calls": 0}
        q_masked, pii = mask_pii(query); flags = {"pii": bool(pii), "pii_types": pii}; tr["query_masked"] = q_masked
        # 1. 입력 규칙
        if _any(INJECTION, q_masked):
            tr["input"] = "abuse"; return self._done(Response(REFUSE_ABUSE, render(REFUSE_ABUSE, org, [], flags=flags), flags=flags, trace=tr), t0)
        if not org:
            d = NO_COMPARE if (_any(COMPARE, q_masked) and not compare_ok) else ASK_ORG
            tr["input"] = d; return self._done(Response(d, render(d, org, [], flags=flags), flags=flags, trace=tr), t0)
        tr["input"] = "ok"
        # 2. 코드 검사
        if self.code_checks:
            ok, dropped, ver = check_evidence(org, articles, known_ids); tr["code_checks"] = {"ok": len(ok), "dropped": [(d["label"], d["reasons"]) for d in dropped], **ver}
        else: ok, dropped, ver = list(articles), [], {}
        # 3. 적용 대상
        if self.subject_rule:
            st, subj = subject_status(q_masked, scope_article_text); tr["subject"] = {"state": st, "subject": subj}
            if st == "excluded":
                flags["subject"] = subj; return self._done(Response(REFUSE_SUBJECT, render(REFUSE_SUBJECT, org, ok, flags=flags, dropped=dropped, scope_article_text=scope_article_text), candidates=[a["label"] for a in ok[:N_CANDIDATES]], flags=flags, trace=tr), t0)
            if st == "unknown":
                flags["subject"] = subj; return self._done(Response(HOLD_SUBJECT, render(HOLD_SUBJECT, org, ok, flags=flags, dropped=dropped, scope_article_text=scope_article_text), candidates=[a["label"] for a in ok[:N_CANDIDATES]], flags=flags, trace=tr), t0)
        deleg = delegation_sentences(ok)
        if not ok:
            tr["evidence"] = "no_articles"; return self._done(Response(REFUSE_NO_EVIDENCE, render(REFUSE_NO_EVIDENCE, org, ok, flags=flags, dropped=dropped, delegations=deleg), flags=flags, trace=tr), t0)
        # 4·5. 추출 + 검증 (+ 캐스케이드)
        try:
            ev, perr = extract(self.llm7, q_masked, org, delegated, ok); tr["calls"] += 1; tr["llm7"] = {"raw_quotes": len(ev.get("quotes") or []), "parse_error": perr}
            verified, rejected, relinked = verify_quotes(ev, ok); tr["llm7"].update({"verified": len(verified), "rejected": len(rejected), "relinked": relinked})
            used = "7B"
            if self.cascade and self.llm14 is not None and not verified:   # 조건: 원시 인용 0개 또는 검증 전부 탈락. 1회만
                ev14, perr14 = extract(self.llm14, q_masked, org, delegated, ok); tr["calls"] += 1
                v14, r14, rl14 = verify_quotes(ev14, ok); tr["llm14"] = {"raw_quotes": len(ev14.get("quotes") or []), "parse_error": perr14, "verified": len(v14), "rejected": len(r14), "relinked": rl14}
                if v14: ev, verified, used = ev14, v14, "14B"
            tr["model_used"] = used if verified else None
        except Exception as e:
            tr["error"] = f"{type(e).__name__}: {str(e)[:120]}"
            return self._done(Response(FAIL_SAFE, render(FAIL_SAFE, org, ok, flags=flags, dropped=dropped), candidates=[a["label"] for a in ok[:N_CANDIDATES]], flags=flags, trace=tr), t0)
        flags["personal_judgment"] = bool(ev.get("personal_judgment"))
        # 6. 답변성 (M4 훅): 조문 단위. 모순 → 통과 금지, 충분 → 조문 전체 카드, 일부만 → 부분 답변
        full_cards = False
        if verified and self.m4 is not None:
            labels = {}
            for i in sorted({q["source"] for q in verified}): labels[i] = self.m4(q_masked, ok[i - 1]["text"])
            tr["m4"] = labels
            if "모순" in labels.values(): verified = []
            elif "충분" in labels.values(): verified = [q for q in verified if labels[q["source"]] in ("충분", "일부")]; full_cards = True; tr["m4_decision"] = ANSWER
            elif "일부" in labels.values(): verified = [q for q in verified if labels[q["source"]] == "일부"]; tr["m4_decision"] = PARTIAL
            else: verified = []
        # 7. 결정·조립·최종 검사
        if verified:
            dec = tr.get("m4_decision") or (ANSWER if ev.get("fully_answered") else PARTIAL)
            resp = Response(dec, render(dec, org, ok, verified, flags, dropped, scope_article_text, full_cards), quotes=verified, flags=flags, trace=tr)
            if not verify_grounded(resp, org, ok, dropped, scope_article_text, full_cards):
                tr["final_check"] = "failed"; resp = Response(REFUSE_NO_EVIDENCE, render(REFUSE_NO_EVIDENCE, org, ok, flags=flags, dropped=dropped, delegations=deleg), candidates=[a["label"] for a in ok[:N_CANDIDATES]], flags=flags, trace=tr)
            else: tr["final_check"] = "pass"
        else:
            resp = Response(REFUSE_NO_EVIDENCE, render(REFUSE_NO_EVIDENCE, org, ok, flags=flags, dropped=dropped, delegations=deleg), candidates=[a["label"] for a in ok[:N_CANDIDATES]], flags=flags, trace=tr)
        return self._done(resp, t0)

    @staticmethod
    def _done(resp, t0):
        resp.trace["sec"] = round(time.time() - t0, 3); return resp

def log_record(query, resp):
    """앱 로그용. 원문 질문이 아니라 마스킹된 질문만 남긴다."""
    return {"q": resp.trace.get("query_masked", mask_pii(query)[0]), "decision": resp.decision, "calls": resp.trace.get("calls"), "sec": resp.trace.get("sec")}
