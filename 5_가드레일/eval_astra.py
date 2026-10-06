# -*- coding: utf-8 -*-
"""아스트라 질문 200건: 1단계 게이트(키워드·점수) → 회색/거절 후보만 로컬 LLM 2차 판정 → 스타일별 집계.
정답 라벨이 없으므로 재는 것은 거절률과 점수 분포뿐이다. 사용: python eval_astra.py [--summary]"""
import json, sys, time, collections
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(HERE)); import guardrails as G
CACHE = HERE / "astra_stage2_cache.json"
ORGS = sorted({json.loads(l)["org"] for l in open(ROOT / "3_성능테스트/corpora/L2P_A.jsonl", encoding="utf-8")})
ART = {}
for l in open(ROOT / "3_성능테스트/corpora/L2P_A.jsonl", encoding="utf-8"):
    r = json.loads(l); ART.setdefault((r["org"], r["revision_date"], r["article_no"]), []).append(r)
def article(aid):
    o, d, n = aid.split("|")[:3]; d = None if d == "None" else d; rs = ART[(o, d, n)]
    return f"{n}({rs[0].get('article_title') or ''})", " ".join(x["text"] for x in rs)
pq = {r["qid"]: r for r in json.load(open(ROOT / "3_성능테스트/results_astra/per_query.json", encoding="utf-8")) if r["pipeline"] == "hybrid_prefilter"}
gold = {json.loads(l)["qid"]: json.loads(l) for l in open(ROOT / "1_데이터셋/06_eval/goldenset_astra.jsonl", encoding="utf-8")}
items = []
for qid, g in gold.items():
    r = pq[qid]; d = G.decide(g["query"], r.get("top1_ce"), ORGS)
    items.append({"qid": qid, "q": g["query"], "style": g["style"], "persona": g["persona"], "topic": g["topic"], "top1_ce": r.get("top1_ce"),
                  "top5": r["top5"], "input": d["input"]["label"], "gate": d["gate"], "action1": d["action"]})
cache = json.load(open(CACHE, encoding="utf-8")) if CACHE.exists() else {}
need = [it for it in items if it["action1"] in ("refuse", "answer", "raw_only") and it["gate"] in ("gray", "reject")]
if "--summary" not in sys.argv:
    import llm_judge as J
    t0 = time.time()
    for i, it in enumerate(need):
        if it["qid"] in cache: continue
        ok, raw = J.judge_relevance(it["q"], [article(a) for a in it["top5"][:3]])
        cache[it["qid"]] = {"judge_yes": ok, "raw": raw}
        json.dump(cache, open(CACHE, "w", encoding="utf-8"), ensure_ascii=False)
        print(f"[{i+1}/{len(need)}] {time.time()-t0:.0f}s {it['qid']} {it['style']} gate={it['gate']} -> {raw} | {it['q'][:30]}", flush=True)
for it in items:
    a = it["action1"]
    if it["qid"] in cache:
        a = "answer" if cache[it["qid"]]["judge_yes"] else "refuse"
        if a == "answer" and it["input"] == "high_risk": a = "raw_only"
    it["action_final"] = a
REF = ("refuse", "refuse_scope", "refuse_multi")
out = {"n": len(items), "stage2_n": len(need), "stage2_done": sum(it["qid"] in cache for it in need), "by_style": {}, "by_input": dict(collections.Counter(it["input"] for it in items)),
       "gate_dist": dict(collections.Counter(it["gate"] for it in items if it["gate"])), "ce_by_style": {}}
for st in sorted({it["style"] for it in items}):
    xs = [it for it in items if it["style"] == st]
    out["by_style"][st] = {"n": len(xs), "refuse_stage1": sum(it["action1"] in REF for it in xs), "refuse_final": sum(it["action_final"] in REF for it in xs),
                           "raw_only_final": sum(it["action_final"] == "raw_only" for it in xs), "answer_final": sum(it["action_final"] == "answer" for it in xs)}
    ces = sorted(it["top1_ce"] for it in xs if it["top1_ce"] is not None)
    if ces: out["ce_by_style"][st] = {"min": ces[0], "median": ces[len(ces)//2], "max": ces[-1], "below_0.05": sum(c < 0.05 for c in ces), "gray_0.05_0.30": sum(0.05 <= c < 0.30 for c in ces)}
out["items"] = items
json.dump(out, open(HERE / "astra_결과.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"\n질문 {out['n']}건 | 입력 분류 {out['by_input']} | 점수 게이트 {out['gate_dist']} | 2차 판정 {out['stage2_done']}/{out['stage2_n']}")
print(f"\n{'스타일':<6}{'n':>4}{'1단계 거절':>10}{'최종 거절':>9}{'원문만':>7}{'답변':>6}   점수 중앙값 / <0.05 / 회색")
for st, s in out["by_style"].items():
    c = out["ce_by_style"].get(st, {})
    print(f"{st:<6}{s['n']:>4}{s['refuse_stage1']:>10}{s['refuse_final']:>9}{s['raw_only_final']:>7}{s['answer_final']:>6}   {c.get('median',0):.3f} / {c.get('below_0.05','-')} / {c.get('gray_0.05_0.30','-')}")
