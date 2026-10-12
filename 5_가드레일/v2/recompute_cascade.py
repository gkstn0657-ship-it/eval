# -*- coding: utf-8 -*-
"""캐스케이드(7B→14B)+적용 대상 규칙을 같은 라벨·같은 규칙으로 캐시에서 재계산한다. 새 LLM 호출 없음.
입력: results/rows_dev_g2only.json (7B+규칙), results/rows_dev_q14-q4.json (14B, 규칙 없음)
출력: results/summary_cascade_recompute.json + 표 출력. 사용: python recompute_cascade.py"""
import json, sys
from pathlib import Path
R = Path(__file__).resolve().parent / "results"
G2 = json.load(open(R / "rows_dev_g2only.json", encoding="utf-8"))
R14 = {r["qid"]: r for r in json.load(open(R / "rows_dev_q14-q4.json", encoding="utf-8"))}
subj = {r["qid"] for r in G2 if r["decision"] == "REFUSE_SUBJECT"}          # 규칙은 질문+적용 범위 조항만 보므로 모델 무관
def stats(dec):
    neg = [q for q in dec if dec[q][0] in ("OUT", "NA")]; pos = [q for q in dec if dec[q][0] == "R"]
    return {"refusal_rate": sum(dec[q][1] != "ANSWER" for q in neg) / len(neg), "n_neg": len(neg),
            "false_refusal_rate": sum(dec[q][1] != "ANSWER" for q in pos) / len(pos), "n_pos": len(pos)}
v3 = {r["qid"]: (r["label"], r["decision"]) for r in G2}
d14 = {q: (R14[q]["label"], "REFUSE_SUBJECT" if q in subj else R14[q]["decision"]) for q in R14}
casc = {q: (v3[q][0], v3[q][1] if v3[q][1] in ("ANSWER", "REFUSE_SUBJECT") else d14[q][1]) for q in v3}
out = {"7B+rule(v3)": stats(v3), "14B+rule": stats(d14), "cascade_7B_to_14B+rule": stats(casc),
       "n_14B_calls": sum(v3[q][1] not in ("ANSWER", "REFUSE_SUBJECT") for q in v3), "n_total": len(v3),
       "note": "14B 호출 조건 = 7B 가 ANSWER 도 REFUSE_SUBJECT 도 아닐 때. 캐시 재계산이라 지연은 없음."}
json.dump(out, open(R / "summary_cascade_recompute.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
for k in list(out)[:3]: s = out[k]; print(f"{k:28s} 거부율 {s['refusal_rate']:.0%} ({s['n_neg']})  오거부 {s['false_refusal_rate']:.0%} ({s['n_pos']})")
print("14B 호출", out["n_14B_calls"], "/", out["n_total"])
