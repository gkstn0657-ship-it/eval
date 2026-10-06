# -*- coding: utf-8 -*-
"""가드레일 오프라인 평가. 기존 per_query.json(top1_ce)과 골든셋(query)만으로 G1·G2 판정을 내고 집계한다. GPU 불필요."""
import json, sys, collections
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(HERE)); import guardrails as G
ORGS = sorted({json.loads(l)["org"] for l in open(ROOT / "3_성능테스트/corpora/L2P_A.jsonl", encoding="utf-8")})
SETS = {"가스FAQ": ("results_kgs_faq", "goldenset_kgs_faq.jsonl"), "매트릭스": ("results_matrix", "goldenset_matrix.jsonl"),
        "음성": ("results_negative", "goldenset_negative.jsonl"), "기존 employee": ("results_full_employee", "goldenset_employee.jsonl"),
        "기존 kps": ("results_full_kps", "goldenset_kps.jsonl"), "기존 kaia": ("results_full_kaia", "goldenset_kaia.jsonl"), "기존 knps": ("results_full_knps", "goldenset_knps.jsonl")}
REFUSE = ("refuse", "refuse_scope", "refuse_multi")
out = {}
for name, (rdir, gfile) in SETS.items():
    pq = [r for r in json.load(open(ROOT / "3_성능테스트" / rdir / "per_query.json", encoding="utf-8")) if r["pipeline"] == "hybrid_prefilter"]
    gold = {json.loads(l)["qid"]: json.loads(l) for l in open(ROOT / "1_데이터셋/06_eval" / gfile, encoding="utf-8")}
    rows = []
    for r in pq:
        g = gold[r["qid"]]; d = G.decide(g["query"], r.get("top1_ce"), ORGS)
        rows.append({"qid": r["qid"], "type": r["type"], "answerable": bool(g.get("gold_ids")), "action": d["action"], "gate": d["gate"],
                     "input": d["input"]["label"], "ndcg5": r.get("ndcg5"), "q": g["query"]})
    neg = [x for x in rows if not x["answerable"]]; pos = [x for x in rows if x["answerable"]]
    s = {"n": len(rows), "neg_n": len(neg), "neg_refuse_rate": (sum(x["action"] in REFUSE for x in neg) / len(neg)) if neg else None,
         "neg_by_reason": dict(collections.Counter(x["action"] for x in neg)), "pos_n": len(pos),
         "pos_false_refuse_rate": (sum(x["action"] in REFUSE for x in pos) / len(pos)) if pos else None,
         "pos_gray_rate": (sum(x["gate"] == "gray" for x in pos) / len(pos)) if pos else None,
         "pos_raw_only_rate": (sum(x["action"] == "raw_only" for x in pos) / len(pos)) if pos else None,
         "pos_false_refuse_by_type": {}, "pos_false_refused": [(x["qid"], x["type"], round(x["ndcg5"] or 0, 2), x["action"], x["q"][:40]) for x in pos if x["action"] in REFUSE]}
    for t in sorted({x["type"] for x in pos}, key=lambda t: int(t[1:])):
        tr = [x for x in pos if x["type"] == t]; s["pos_false_refuse_by_type"][t] = (sum(x["action"] in REFUSE for x in tr), len(tr))
    out[name] = s
json.dump(out, open(HERE / "가드레일_평가.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
for name, s in out.items():
    print(f"\n== {name} (n={s['n']}) ==")
    if s["neg_n"]: print(f"  정답 없음 {s['neg_n']}건: 거절률 {s['neg_refuse_rate']:.0%}  사유 {s['neg_by_reason']}")
    if s["pos_n"]:
        print(f"  정답 있음 {s['pos_n']}건: 오거절 {s['pos_false_refuse_rate']:.1%}, 회색(확신 낮음) {s['pos_gray_rate']:.0%}, 원문만 표시(고위험) {s['pos_raw_only_rate']:.0%}")
        bt = {t: f"{a}/{b}" for t, (a, b) in s["pos_false_refuse_by_type"].items() if a}
        if bt: print("  유형별 오거절(건/전체):", bt)
        for x in s["pos_false_refused"]: print("   -", *x)
