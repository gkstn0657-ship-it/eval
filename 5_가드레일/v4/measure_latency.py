# -*- coding: utf-8 -*-
"""캐스케이드 실측 지연 (아스트라 지시 5절). 캐시를 쓰지 않고 실제로 7B(+14B)를 호출해 요청별 벽시계 시간을 잰다. 모델 로딩·교체 비용이 포함된다.
실행 조건: 정확도상 채택 가능성이 확인된 뒤(사람 검수 후)에만 돌린다. GPU 작업은 한 번에 하나.
조건 구분: cold(첫 요청, 모델 미로딩) / warm(반복). 7B 는 상주(keep_alive), 14B 는 필요할 때 로드되며 8GB GPU 에서 둘이 같이 안 올라가 교체가 일어난다.
출력: results/latency_<config>.json (p50/p95, 호출 수, cold/warm 분리)
사용: python measure_latency.py --config B|D [--limit N]"""
import json, sys, time, statistics
import data_v4 as D
from guard_v4 import GuardV4, OllamaLLM
sys.stdout.reconfigure(encoding="utf-8")
A = sys.argv[1:]; CONFIG = A[A.index("--config") + 1] if "--config" in A else "D"; LIMIT = int(A[A.index("--limit") + 1]) if "--limit" in A else 0
g = GuardV4(OllamaLLM("qwen2.5:7b-instruct"), OllamaLLM("q14-q4") if CONFIG == "D" else None, cascade=(CONFIG == "D"))
qids = D.DEV[: LIMIT or None]; recs = []
for i, qid in enumerate(qids, 1):
    q = D.GOLD[qid]; org = q["context_org"]; arts = D.top5(qid); t0 = time.time()
    r = g.respond(q["query"], org, arts, D.DELEG.get(org, []), scope_article_text=D.SCOPE_ART.get(org, ""), known_ids=D.KNOWN_IDS)
    recs.append({"qid": qid, "sec": round(time.time() - t0, 2), "calls": r.trace.get("calls", 0), "decision": r.decision, "model_used": r.trace.get("model_used"), "cold": i == 1})
    print(f"[{i}/{len(qids)}] {qid} {recs[-1]['sec']}s calls={recs[-1]['calls']} {r.decision}", flush=True)
warm = [x["sec"] for x in recs if not x["cold"]]; allsec = [x["sec"] for x in recs]
def p(v, q): return round(statistics.quantiles(v, n=100)[q - 1], 2) if len(v) >= 2 else (v[0] if v else None)
out = {"config": CONFIG, "n": len(recs), "code_rev": D.git_rev(), "cold_first_request_sec": recs[0]["sec"] if recs else None,
       "warm": {"p50": p(warm, 50), "p95": p(warm, 95), "mean": round(sum(warm) / max(1, len(warm)), 2)}, "all": {"p50": p(allsec, 50), "p95": p(allsec, 95)},
       "calls_per_request": round(sum(x["calls"] for x in recs) / max(1, len(recs)), 2), "n_two_calls": sum(x["calls"] >= 2 for x in recs),
       "by_calls_p50": {k: p([x["sec"] for x in recs if x["calls"] == k], 50) for k in (1, 2)}, "note": "모델 교체 비용 포함. 7B 상주, 14B 는 필요 시 로드.", "rows": recs}
json.dump(out, open(D.HERE / "results" / f"latency_{CONFIG}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "rows"}, ensure_ascii=False, indent=1))
