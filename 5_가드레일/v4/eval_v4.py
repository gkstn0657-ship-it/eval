# -*- coding: utf-8 -*-
"""v4 개발 평가 (설계용 104, 개발 자료). 구성:
  B       = 7B 원문 인용 + 코드 검사 + 적용 대상 세 상태 (캐스케이드 없음)
  D_cache = B + 14B 캐스케이드. 14B 출력은 v2 캐시 재사용 → '잠재 성능'이지 실측이 아니다. 지연은 미실측
LLM 출력 캐시: results/llm_cache_v4_<model>.json (키 = sha(system+user)). v2 캐시(qid 키)는 프롬프트가 같을 때만 씨앗으로 쓴다.
사용: python eval_v4.py [--config B|D_cache|all] [--no-llm]   (--no-llm: 캐시에 없으면 FAIL_SAFE 로 기록, 모델 호출 안 함)"""
import json, sys, time, collections, csv
import data_v4 as D
from guard_v4 import (GuardV4, OllamaLLM, EVIDENCE_SYSTEM, build_user_prompt, ANSWER, PARTIAL, ANSWERS, REFUSALS, HOLDS, HOLD_SUBJECT, FAIL_SAFE, REFUSE_SUBJECT, REFUSE_NO_EVIDENCE)
sys.stdout.reconfigure(encoding="utf-8")
A = sys.argv[1:]; CONFIG = A[A.index("--config") + 1] if "--config" in A else "all"; NO_LLM = "--no-llm" in A
OUT = D.HERE / "results"; OUT.mkdir(exist_ok=True)

class CachedLLM:
    def __init__(self, model, v2_cache_name):
        self.model = model; self.key = model.replace(":", "_").replace("/", "_"); self.f = OUT / f"llm_cache_v4_{self.key}.json"
        self.cache = json.load(open(self.f, encoding="utf-8")) if self.f.exists() else {}
        v2f = D.V2 / "results" / f"llm_cache_{v2_cache_name}.json"; self.v2 = json.load(open(v2f, encoding="utf-8")) if v2f.exists() else {}
        self.base = OllamaLLM(model); self.misses = 0
    def seed(self, qid, query, org, arts):  # v2 와 프롬프트가 같으면(조문 5개 전부 통과) v2 출력을 재사용
        k = D.sha(EVIDENCE_SYSTEM + build_user_prompt(query, org, D.DELEG.get(org, []), arts))
        if k not in self.cache and qid in self.v2: self.cache[k] = self.v2[qid]
    def __call__(self, system, user):
        k = D.sha(system + user)
        if k not in self.cache:
            if NO_LLM: raise RuntimeError("cache miss (--no-llm)")
            self.misses += 1; self.cache[k] = self.base(system, user); json.dump(self.cache, open(self.f, "w", encoding="utf-8"), ensure_ascii=False)
        return self.cache[k]

llm7 = CachedLLM("qwen2.5:7b-instruct", "qwen2.5_7b-instruct"); llm14 = CachedLLM("q14-q4", "q14-q4")
CONFIGS = {"B": dict(llm7=llm7, cascade=False), "D_cache": dict(llm7=llm7, llm14=llm14, cascade=True)}
def rate(n, d): return {"n": n, "d": d, "rate": (round(n / d, 4) if d else None)}

def run(name):
    g = GuardV4(**CONFIGS[name]); rows = []
    for qid in D.DEV:
        q = D.GOLD[qid]; org = q["context_org"]; arts = D.top5(qid); llm7.seed(qid, q["query"], org, arts); llm14.seed(qid, q["query"], org, arts)
        r = g.respond(q["query"], org, arts, D.DELEG.get(org, []), scope_article_text=D.SCOPE_ART.get(org, ""), known_ids=D.KNOWN_IDS)
        rows.append({"qid": qid, "half": D.half_of(qid), "label": D.LABELS[qid]["label"], "style": q["style"], "query": q["query"], "decision": r.decision,
                     "n_quotes": len(r.quotes), "quotes": r.quotes, "model_used": r.trace.get("model_used"), "calls": r.trace.get("calls"),
                     "code_checks": r.trace.get("code_checks"), "subject": r.trace.get("subject"), "final_check": r.trace.get("final_check"), "error": r.trace.get("error"), "text": r.text})
    L = lambda lab: [x for x in rows if x["label"] == lab]
    neg = L("NA") + L("OUT"); pos = L("R"); ans = [x for x in rows if x["decision"] in ANSWERS]
    S = {"config": name, "split": "dev104 (개발 자료: 선택용 56 + 점검용 48)", "code_rev": D.git_rev(), "llm_cache_misses": {"7B": llm7.misses, "14B": llm14.misses},
         "refusal_recall": rate(sum(x["decision"] in REFUSALS for x in neg), len(neg)),
         "not_answered_recall(refuse+hold)": rate(sum(x["decision"] not in ANSWERS for x in neg), len(neg)),
         "false_refusal(R→refuse)": rate(sum(x["decision"] in REFUSALS for x in pos), len(pos)),
         "R_not_fully_answered(R→refuse/hold/partial)": rate(sum(x["decision"] != ANSWER for x in pos), len(pos)),
         "R_partial": rate(sum(x["decision"] == PARTIAL for x in pos), len(pos)),
         "ungrounded_answers(final_check failed)": sum(x["final_check"] == "failed" for x in rows),
         "fail_safe": sum(x["decision"] == FAIL_SAFE for x in rows), "hold_subject": sum(x["decision"] == HOLD_SUBJECT for x in rows),
         "n_answers": len(ans), "n_14B_calls": sum((x["calls"] or 0) >= 2 for x in rows),
         "code_check_drops": dict(collections.Counter(r for x in rows for _, rs in (x["code_checks"] or {}).get("dropped", []) for r in rs)),
         "by_label": {l: dict(collections.Counter(x["decision"] for x in L(l))) for l in ("R", "W", "NA", "OUT")},
         "by_half": {h: {"refusal_recall": rate(sum(x["decision"] in REFUSALS for x in neg if x["half"] == h), sum(x["half"] == h for x in neg)),
                         "false_refusal": rate(sum(x["decision"] in REFUSALS for x in pos if x["half"] == h), sum(x["half"] == h for x in pos))} for h in ("select", "check")},
         "by_style_false_refusal": {s: rate(sum(x["decision"] in REFUSALS for x in pos if x["style"] == s), sum(x["style"] == s for x in pos)) for s in sorted({x["style"] for x in pos})},
         "latency": "미실측 (캐시 재계산)", "human_review": "미완료 (의미상 오답·정상 답변 손실은 사람 검수 후 계산)"}
    json.dump(rows, open(OUT / f"rows_{name}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(S, open(OUT / f"summary_{name}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in S.items() if k not in ("by_style_false_refusal",)}, ensure_ascii=False, indent=1)); return rows

res = {n: run(n) for n in (CONFIGS if CONFIG == "all" else [CONFIG])}
# 문항별 비교표 (자동 열만. 정답 여부·오류 유형은 사람 검수 후 채운다)
if CONFIG == "all":
    fa = OUT / "baseline_A.jsonl"; A_rows = {json.loads(l)["qid"]: json.loads(l) for l in open(fa, encoding="utf-8")} if fa.exists() else {}
    with open(OUT / "compare_table.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f); w.writerow(["qid", "half", "label", "style", "query", "A_decision", "A_gen_sec", "B_decision", "B_quotes", "D_decision", "D_model", "D_calls", "B_final_check", "human_correct_A", "human_correct_B", "human_correct_D", "error_type"])
        for b, d in zip(res["B"], res["D_cache"]):
            a = A_rows.get(b["qid"]); w.writerow([b["qid"], b["half"], b["label"], b["style"], b["query"], "ANSWER(free)" if a else "미생성", a["gen_seconds"] if a else "", b["decision"], b["n_quotes"], d["decision"], d["model_used"], d["calls"], b["final_check"], "", "", "", ""])
    print("compare_table.csv 작성 (사람 검수 열은 비어 있음)")
