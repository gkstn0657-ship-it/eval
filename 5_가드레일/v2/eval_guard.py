# -*- coding: utf-8 -*-
"""가드레일 v2 평가 실행기. 외부 질문 200건(사람 라벨)을 설계용(dev)·봉인용(sealed)으로 나눠 정책 지표를 잰다.

라벨: R 답해야 함, W 답이 있는데 검색 실패, NA 규정에 답 없음, OUT 범위 밖
지표 (정책 5절)
  거절 재현율 = (NA+OUT 중 거절) / (NA+OUT)          목표 95% 이상
  오거절률    = (R 중 거절·되묻기) / R                  목표 0%
  근거 없는 답변 = ANSWER 중 최종 검사 실패 건수         목표 0건 (구조적으로 0 이어야 함)
  W: 오거절로 세지 않고, 거절 시 후보 3개에 정답 조항이 있었는지 따로 잰다
검색은 다시 돌리지 않는다. 기존 측정(results_astra, hybrid_prefilter)의 상위 5개 조문을 그대로 쓴다.

사용: python eval_guard.py [--model qwen2.5:7b-instruct] [--limit N]
      python eval_guard.py --sealed --confirm-open   (봉인셋 1회 개봉. 개봉 기록이 남는다)
"""
import json, sys, time, glob, random, collections, re
from pathlib import Path
from guard import Guard, OllamaLLM, verify_grounded, ANSWER, REFUSALS, ASK_ORG, NO_COMPARE
sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent.parent
A = sys.argv[1:]
def arg(n, d=None): return A[A.index(n) + 1] if n in A else d
MODEL = arg("--model", "qwen2.5:7b-instruct"); LIMIT = int(arg("--limit", "0")); SEALED = "--sealed" in A
OUT = HERE / "results"; OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------- 데이터
gold = {json.loads(l)["qid"]: json.loads(l) for l in open(ROOT / "1_데이터셋/06_eval/goldenset_astra.jsonl", encoding="utf-8") if l.strip()}
labels = json.load(open(ROOT / "5_가드레일/astra_manual_labels.json", encoding="utf-8"))
pq = {r["qid"]: r for r in json.load(open(ROOT / "3_성능테스트/results_astra/per_query.json", encoding="utf-8")) if r["pipeline"] == "hybrid_prefilter"}
chunks = collections.defaultdict(list)
for l in open(ROOT / "3_성능테스트/corpora/L2P_A.jsonl", encoding="utf-8"):
    r = json.loads(l); chunks[r["article_id"]].append(r)
def article(aid):
    rs = chunks[aid]; r0 = rs[0]
    return {"org": r0["org"], "label": f"{r0['article_no']}({r0.get('article_title') or ''})", "revision_date": r0["revision_date"] or "",
            "text": " ".join(x["text"] for x in rs), "id": aid}
DELEG = {}
for f in glob.glob(str(ROOT / "1_데이터셋/05_clean/L2/*/*/*.meta.json")):
    m = json.load(open(f, encoding="utf-8"))
    if m.get("is_latest"): DELEG[m["org"]] = sorted({d.strip() for d in m.get("delegated_to", []) if "이 규정" not in d and re.fullmatch(r"[가-힣A-Za-z·\s]{2,14}(규정|규칙|지침|요령|세칙)", d.strip())})  # 규정 이름만

# ---------------------------------------------------------------- 분할 (한 번 만들면 고정)
SPLIT = HERE / "split.json"
if not SPLIT.exists():
    rng = random.Random(2026); strata = collections.defaultdict(list)
    for qid in sorted(gold): strata[(labels[qid]["label"], gold[qid]["style"])].append(qid)
    split = {"dev": [], "sealed": []}
    for k in sorted(strata):
        ids = strata[k][:]; rng.shuffle(ids)
        for i, q in enumerate(ids): split["dev" if i % 2 == 0 else "sealed"].append(q)
    split["rule"] = "라벨×말투 층화, 층마다 섞어 번갈아 배정, seed 2026"
    json.dump(split, open(SPLIT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
split = json.load(open(SPLIT, encoding="utf-8"))
if SEALED:
    if "--confirm-open" not in A: sys.exit("봉인셋은 --confirm-open 을 함께 줘야 연다.")
    log = HERE / "sealed_open.log"
    with open(log, "a", encoding="utf-8") as f: f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} 개봉 model={MODEL}\n")
qids = split["sealed" if SEALED else "dev"][: LIMIT or None]

# ---------------------------------------------------------------- 실행 (LLM 출력은 캐시)
cache_f = OUT / f"llm_cache_{MODEL.replace(':', '_')}.json"
cache = json.load(open(cache_f, encoding="utf-8")) if cache_f.exists() else {}
base = OllamaLLM(MODEL)
class Cached:
    qid = None
    def __call__(self, s, u):
        if self.qid not in cache:
            cache[self.qid] = base(s, u); json.dump(cache, open(cache_f, "w", encoding="utf-8"), ensure_ascii=False)
        return cache[self.qid]
llm = Cached(); g = Guard(llm); rows = []; t0 = time.time()
for i, qid in enumerate(qids, 1):
    q = gold[qid]; org = q["context_org"]; arts = [article(a) for a in pq[qid]["top5"]]
    llm.qid = qid
    r = g.respond(q["query"], org, arts, DELEG.get(org, []))
    grounded = verify_grounded(r, org, arts)
    lab = labels[qid]["label"]; note = labels[qid].get("note", "")
    m = re.search(r"제\s*\d+\s*조(?:의\s*\d+)?", note or ""); want = m.group(0).replace(" ", "") if m else None
    rows.append({"qid": qid, "label": lab, "style": q["style"], "query": q["query"], "decision": r.decision, "grounded": grounded,
                 "n_quotes": len(r.quotes), "other_rule": r.other_rule, "candidates": r.candidates,
                 "w_candidate_hit": (any(c.startswith(want) for c in r.candidates) if (lab == "W" and want and r.candidates) else None),
                 "retrieved_hit": (any(a["label"].startswith(want) for a in arts) if (lab == "W" and want) else None),
                 "sec": r.trace.get("sec"), "trace": r.trace, "text": r.text})
    print(f"[{i}/{len(qids)}] {qid} {lab:3} → {r.decision:18} quotes={len(r.quotes)} ({time.time()-t0:.0f}s) | {q['query'][:30]}", flush=True)

# ---------------------------------------------------------------- 지표
def rate(n, d): return None if d == 0 else round(n / d, 4)
must_refuse = [x for x in rows if x["label"] in ("NA", "OUT")]; must_answer = [x for x in rows if x["label"] == "R"]; w = [x for x in rows if x["label"] == "W"]
answers = [x for x in rows if x["decision"] == ANSWER]
S = {"split": "sealed" if SEALED else "dev", "model": MODEL, "n": len(rows),
     "refusal_recall": rate(sum(x["decision"] in REFUSALS for x in must_refuse), len(must_refuse)),
     "false_refusal": rate(sum(x["decision"] in REFUSALS | {ASK_ORG, NO_COMPARE} for x in must_answer), len(must_answer)),
     "ungrounded_answers": sum(not x["grounded"] for x in answers),
     "n_answers": len(answers), "n_must_refuse": len(must_refuse), "n_must_answer": len(must_answer),
     "W": {"n": len(w), "refused": sum(x["decision"] in REFUSALS for x in w), "candidate_hit_when_refused": sum(bool(x["w_candidate_hit"]) for x in w if x["decision"] in REFUSALS)},
     "by_label": {l: dict(collections.Counter(x["decision"] for x in rows if x["label"] == l)) for l in ("R", "W", "NA", "OUT")},
     "by_style_false_refusal": {s: rate(sum(x["decision"] in REFUSALS for x in must_answer if x["style"] == s), sum(1 for x in must_answer if x["style"] == s)) for s in sorted({x["style"] for x in must_answer})},
     "leaked": [{"qid": x["qid"], "label": x["label"], "query": x["query"]} for x in must_refuse if x["decision"] == ANSWER],
     "false_refused": [{"qid": x["qid"], "query": x["query"], "decision": x["decision"]} for x in must_answer if x["decision"] != ANSWER],
     "sec_mean": round(sum(x["sec"] or 0 for x in rows) / max(1, len(rows)), 2),
     "targets": {"refusal_recall": ">=0.95", "false_refusal": "0", "ungrounded_answers": "0"}}
tag = f"{S['split']}_{MODEL.replace(':', '_')}" + (f"_n{LIMIT}" if LIMIT else "")
json.dump(rows, open(OUT / f"rows_{tag}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
json.dump(S, open(OUT / f"summary_{tag}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in S.items() if k not in ("leaked", "false_refused")}, ensure_ascii=False, indent=1))
