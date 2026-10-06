# -*- coding: utf-8 -*-
"""G2 2단계 오프라인 평가. 1·2단계(키워드·점수)에서 gray/reject 로 걸린 항목만 로컬 LLM 으로 재판정한다. GPU 미사용.
사용: python eval_stage2.py        (판정 실행, 결과는 stage2_cache.json 에 건별 저장·재개 가능)
      python eval_stage2.py --summary  (집계만)"""
import json, sys, time, collections
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(HERE)); import guardrails as G, llm_judge as J
CACHE = HERE / "stage2_cache.json"
ORGS = sorted({json.loads(l)["org"] for l in open(ROOT / "3_성능테스트/corpora/L2P_A.jsonl", encoding="utf-8")})
ART = {}
for l in open(ROOT / "3_성능테스트/corpora/L2P_A.jsonl", encoding="utf-8"):
    r = json.loads(l); ART.setdefault((r["org"], r["revision_date"], r["article_no"]), []).append(r)
def article(aid):
    o, d, n = aid.split("|")[:3]; d = None if d == "None" else d; rs = ART[(o, d, n.split("#")[0].split("@")[0])]
    return f"{n}({rs[0].get('article_title') or ''})", " ".join(x["text"] for x in rs)
SETS = {"가스FAQ": ("results_kgs_faq", "goldenset_kgs_faq.jsonl"), "매트릭스": ("results_matrix", "goldenset_matrix.jsonl"), "음성": ("results_negative", "goldenset_negative.jsonl"),
        "employee": ("results_full_employee", "goldenset_employee.jsonl"), "kps": ("results_full_kps", "goldenset_kps.jsonl"),
        "kaia": ("results_full_kaia", "goldenset_kaia.jsonl"), "knps": ("results_full_knps", "goldenset_knps.jsonl")}
items = []
for name, (rd, gf) in SETS.items():
    pq = [r for r in json.load(open(ROOT / f"3_성능테스트/{rd}/per_query.json", encoding="utf-8")) if r["pipeline"] == "hybrid_prefilter"]
    gold = {json.loads(l)["qid"]: json.loads(l) for l in open(ROOT / f"1_데이터셋/06_eval/{gf}", encoding="utf-8")}
    for r in pq:
        g = gold[r["qid"]]; d = G.decide(g["query"], r.get("top1_ce"), ORGS)
        if d["action"] in ("refuse", "answer", "raw_only") and d["gate"] in ("gray", "reject"):
            items.append({"key": f"{name}:{r['qid']}", "set": name, "q": g["query"], "answerable": bool(g.get("gold_ids")), "gate": d["gate"],
                          "ndcg5": r.get("ndcg5"), "top3": r["top5"][:3]})
cache = json.load(open(CACHE, encoding="utf-8")) if CACHE.exists() else {}
if "--summary" not in sys.argv:
    t0 = time.time()
    for i, it in enumerate(items):
        if it["key"] in cache: continue
        ok, raw = J.judge_relevance(it["q"], [article(a) for a in it["top3"]])
        cache[it["key"]] = {"judge_yes": ok, "raw": raw}
        json.dump(cache, open(CACHE, "w", encoding="utf-8"), ensure_ascii=False)
        print(f"[{i+1}/{len(items)}] {time.time()-t0:.0f}s {it['key']} gate={it['gate']} answerable={it['answerable']} -> {raw}", flush=True)
done = [it for it in items if it["key"] in cache]
print(f"\n판정 완료 {len(done)}/{len(items)}")
for gate in ("gray", "reject"):
    for ans in (True, False):
        xs = [it for it in done if it["gate"] == gate and it["answerable"] == ans]
        if xs:
            yes = sum(cache[it["key"]]["judge_yes"] for it in xs)
            print(f"  {gate:6s} 정답{'있음' if ans else '없음'} n={len(xs):2d}: 판정 '예' {yes}, '아니오' {len(xs)-yes}")
# 최종 정책: reject→(예면 구제, 아니오면 거절), gray→(예면 통과, 아니오면 거절)
neg = [it for it in done if not it["answerable"]]; pos = [it for it in done if it["answerable"]]
print(f"\n2차 판정 적용 후 (판정 대상 {len(done)}건 한정)")
print(f"  정답 없음 {len(neg)}건 중 거절 유지 {sum(not cache[it['key']]['judge_yes'] for it in neg)}건")
print(f"  정답 있음 {len(pos)}건 중 거절 {sum(not cache[it['key']]['judge_yes'] for it in pos)}건 (1단계 reject 단독이었다면 {sum(it['gate']=='reject' for it in pos)}건)")
