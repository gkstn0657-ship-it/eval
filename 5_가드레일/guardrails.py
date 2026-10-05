# -*- coding: utf-8 -*-
"""가드레일: 입력 분류(G1), 검색 신뢰도 게이트(G2), 출력 검증(G3). 정책은 정책.md. run_eval.py 와 hitl_serve.py 가 import 한다."""
import re

OUT_OF_SCOPE = {
    "연차": r"연차|연가", "성과급": r"성과급|성과상여", "여비": r"여비|출장비|일비", "퇴직금": r"퇴직금|퇴직급여|연금",
    "육아시간": r"육아시간|단축근무", "회식비": r"회식비", "복리후생": r"복지포인트|복리후생", "급여일": r"급여일|월급날|급여 ?지급일",
}
HIGH_RISK = {
    "징계": r"징계|파면|해임|강등|정직|감봉", "해고": r"해고|직권면직|당연퇴직", "성비위": r"성희롱|성폭력|성비위|성범죄",
}
CE_LO, CE_HI = 0.05, 0.30
_ORG_PREFIX = re.compile(r"^\((주|재|사)\)")

MSG = {
    "out_of_scope": "이 내용은 인사규정이 아니라 보수·복무·복리후생 규정 소관입니다. 해당 규정이나 인사부서에 확인해 주세요.",
    "high_risk": "민감한 사안이라 요약 답변 대신 관련 조문 원문을 보여드립니다. 적용 여부는 인사부서에 확인해 주세요.",
    "multi_org": "기관 간 비교는 지원하지 않습니다. 기관을 하나씩 지정해 질문해 주세요.",
    "reject": "관련 조항을 찾지 못했습니다. 혹시 다음 조항 중 하나인가요?",
    "gray": "검색 신뢰도가 낮습니다. 근거 조문을 꼭 확인하세요.",
    "disclaimer": "이 답변은 인사규정 원문을 바탕으로 자동 생성한 참고 정보입니다. 최종 적용 여부는 인사부서에 확인해 주세요.",
}

def _org_keys(org_names):
    out = {}
    for o in org_names or []:
        base = _ORG_PREFIX.sub("", o)
        if len(base) >= 4: out[base] = o
    return out

def classify_input(q, org_names=None):
    """G1. {'label': ok|out_of_scope|high_risk|multi_org, 'matched': [...]}. 우선순위 multi_org > out_of_scope > high_risk."""
    found = {o for k, o in _org_keys(org_names).items() if k in q}
    if len(found) >= 2: return {"label": "multi_org", "matched": sorted(found)}
    for name, pat in OUT_OF_SCOPE.items():
        if re.search(pat, q): return {"label": "out_of_scope", "matched": [name]}
    for name, pat in HIGH_RISK.items():
        if re.search(pat, q): return {"label": "high_risk", "matched": [name]}
    return {"label": "ok", "matched": []}

def retrieval_gate(top1_ce, lo=CE_LO, hi=CE_HI):
    """G2 1단계. pass | gray | reject"""
    if top1_ce is None or top1_ce < lo: return "reject"
    if top1_ce < hi: return "gray"
    return "pass"

_NUM = re.compile(r"\d+(?:\.\d+)?\s*(?:세|년|개월|월|일|회|퍼센트|%|분의\s*\d+|배수|만원|시간)")
_ART = re.compile(r"제\s*\d+\s*조(?:의\s*\d+)?")
def _norm(s): return re.sub(r"\s+", "", s or "")

def verify_answer(answer, context_texts, allowed_article_nos):
    """G3. 조문 인용과 숫자가 근거에 있는지. {'ok', 'bad_citations', 'missing_numbers'}. 숫자가 전부 근거에 없을 때만 폐기."""
    ctx = _norm(" ".join(context_texts or []))
    allowed = {_norm(a) for a in (allowed_article_nos or [])}
    bad_cit = sorted({_norm(a) for a in _ART.findall(answer or "") if _norm(a) not in allowed})
    nums = {_norm(n) for n in _NUM.findall(answer or "")}
    missing = sorted(n for n in nums if n not in ctx)
    ok = not bad_cit and not (nums and len(missing) == len(nums))
    return {"ok": ok, "bad_citations": bad_cit, "missing_numbers": missing}

def decide(q, top1_ce, org_names=None):
    """G1+G2 종합. {'action': answer|raw_only|refuse|refuse_multi|refuse_scope, 'input', 'gate', 'message'}"""
    inp = classify_input(q, org_names)
    if inp["label"] == "multi_org": return {"action": "refuse_multi", "input": inp, "gate": None, "message": MSG["multi_org"]}
    if inp["label"] == "out_of_scope": return {"action": "refuse_scope", "input": inp, "gate": None, "message": MSG["out_of_scope"]}
    gate = retrieval_gate(top1_ce)
    if gate == "reject": return {"action": "refuse", "input": inp, "gate": gate, "message": MSG["reject"]}
    if inp["label"] == "high_risk": return {"action": "raw_only", "input": inp, "gate": gate, "message": MSG["high_risk"]}
    return {"action": "answer", "input": inp, "gate": gate, "message": MSG["gray"] if gate == "gray" else None}
