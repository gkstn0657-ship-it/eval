# -*- coding: utf-8 -*-
"""생성 모델 체급 상향 + 양자화 실험 (계획서: 생성모델_양자화_실험계획.md).

기존 run_gen_eval.py 의 시스템 프롬프트·채점 루브릭·자동 지표를 그대로 가져다 쓰고, 생성 모델만 바꿔 비교한다.
기존 파일은 수정하지 않는다.

단계
  --contexts            : dev + employee 88문항의 검색 결과(hybrid_prefilter, L2P_A 상위 5개 조문)를 한 번만 만들어 저장. 모든 모델이 같은 입력을 받는다
  --gen <ollama모델> <ID> : 저장된 입력으로 답변 생성 (속도 기록)
  --judge <ollama모델> <채점자ID> : 모든 생성 결과를 채점 (어느 모델의 답인지 채점자에게 알리지 않음)
  --summary             : 집계, 쌍대 부트스트랩
"""
import os, sys, json, time, urllib.request
from pathlib import Path
ARGS = sys.argv[1:]
sys.argv = [sys.argv[0]]  # run_gen_eval / run_eval 은 import 시 sys.argv 를 읽는다
os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
import numpy as np
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import run_gen_eval as G   # SYSTEM_PROMPT, JUDGE_PROMPT, build_context, auto_checks 재사용
E = G.E
sys.stdout.reconfigure(encoding="utf-8")
OUT = HERE / "results_gen_quant"; OUT.mkdir(exist_ok=True)
OLLAMA = "http://127.0.0.1:11434"
SETS = [("dev", ROOT / "1_데이터셋/06_eval/goldenset_dev.jsonl"), ("employee", ROOT / "1_데이터셋/06_eval/goldenset_employee.jsonl")]

def chat(model, system, user, num_predict=600, fmt=None):
    opts = {"temperature": 0, "num_ctx": 4096, "num_predict": num_predict}
    if os.environ.get("GEN_NUM_GPU"): opts["num_gpu"] = int(os.environ["GEN_NUM_GPU"])  # GPU 에 올릴 층 수 강제 (Ollama 기본 추정이 보수적일 때). 계산 결과는 같고 속도만 바뀐다
    body = {"model": model, "stream": False, "keep_alive": "30m", "options": opts,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    if fmt: body["format"] = fmt
    req = urllib.request.Request(OLLAMA + "/api/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r: return json.loads(r.read().decode())

def jl(p): return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()] if Path(p).exists() else []

def contexts():
    c = E.Corpus("L2P_A"); out = []
    for st, f in SETS:
        for q in jl(f):
            E._CTX["org"] = q.get("context_org")
            idx, info = E.run_hybrid_prefilter(c, q["query"])
            E._CTX["org"] = None
            arts, hits = [], []
            for i in idx:
                a = c.rows[i]["article_id"]
                if a in arts: continue
                arts.append(a); hits.append(i)
                if len(hits) >= 5: break
            out.append({"set": st, "qid": q["qid"], "type": q["type"], "query": q["query"], "context_org": q.get("context_org"),
                        "gold_ids": q["gold_ids"], "hint": q["answer_hint"], "retrieved": arts,
                        "retrieval_hit5": bool(set(arts) & set(q["gold_ids"])), "context": G.build_context(c, hits, False)})
    with open(OUT / "contexts.jsonl", "w", encoding="utf-8") as f:
        for r in out: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("contexts", len(out), "검색 적중", sum(r["retrieval_hit5"] for r in out if r["type"] != "Q9"), "/", sum(r["type"] != "Q9" for r in out))

def gen(model, mid):
    ctx = jl(OUT / "contexts.jsonl"); path = OUT / f"answers_{mid}.jsonl"
    done = {(r["set"], r["qid"]) for r in jl(path)}; f = open(path, "a", encoding="utf-8"); t0 = time.time()
    for r in ctx:
        if (r["set"], r["qid"]) in done: continue
        user = r["query"] + (f"\n(질문자 소속: {r['context_org']})" if r.get("context_org") else "")
        t1 = time.time(); res = chat(model, G.SYSTEM_PROMPT.format(context=r["context"], delegation=""), user)
        ec, ed = res.get("eval_count", 0), res.get("eval_duration", 0) / 1e9
        rec = {k: r[k] for k in ("set", "qid", "type", "query", "gold_ids", "hint", "retrieval_hit5")}
        rec.update({"model": mid, "answer": res["message"]["content"], "sec": round(time.time() - t1, 1), "tokens": ec, "tok_per_s": round(ec / ed, 1) if ed else None})
        f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
        print(f"{mid} {r['set']}:{r['qid']} {rec['sec']}s {rec['tok_per_s']}tok/s ({time.time()-t0:.0f}s)", flush=True)
    f.close()

def judge(model, jid):
    idx = {json.loads(l)["id"]: json.loads(l) for l in (ROOT / "1_데이터셋/05_clean/L2/index_B_all_revisions.jsonl").read_text(encoding="utf-8").splitlines()}
    path = OUT / f"judge_{jid}.jsonl"; done = {(r["model"], r["set"], r["qid"]) for r in jl(path)}; f = open(path, "a", encoding="utf-8")
    recs = [r for p in sorted(OUT.glob("answers_*.jsonl")) for r in jl(p)]
    order = np.random.default_rng(7).permutation(len(recs))  # 모델 순서가 섞이도록 무작위 순서로 채점
    for k in order:
        r = recs[k]
        if (r["model"], r["set"], r["qid"]) in done: continue
        gold_text = "\n\n".join(f"[{g}]\n{idx[g]['text'][:1200]}" for g in r["gold_ids"] if g in idx) or "(범위 외 질의: 정답 조문 없음. 기대 답 참조)"
        a = G.auto_checks(r, gold_text)
        raw = chat(model, "너는 엄격한 채점자다. JSON 만 출력한다.", G.JUDGE_PROMPT.format(query=r["query"], hint=r["hint"], gold_text=gold_text, answer=r["answer"]), num_predict=300, fmt="json")["message"]["content"]
        try: j = json.loads(raw)
        except Exception: j = {"accuracy": None, "reason": "parse_error: " + raw[:120]}
        tot = sum(v for kk, v in j.items() if kk in ("accuracy", "completeness", "citation", "concise") and isinstance(v, (int, float))) if j.get("accuracy") is not None else None
        f.write(json.dumps({"judge": jid, "model": r["model"], "set": r["set"], "qid": r["qid"], "type": r["type"], "retrieval_hit5": r["retrieval_hit5"], **a, "j": j, "total": tot}, ensure_ascii=False) + "\n"); f.flush()
        print(f"[{jid}] {r['model']} {r['set']}:{r['qid']} total={tot}", flush=True)
    f.close()

def summary():
    def mean(v): v = [x for x in v if x is not None]; return float(np.mean(v)) if v else None
    def boot(a, b, n=2000):
        rng = np.random.default_rng(42); a, b = np.array(a), np.array(b)
        d = [(a[s] - b[s]).mean() for s in (rng.integers(0, len(a), len(a)) for _ in range(n))]
        return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    A = {p.stem[len("answers_"):]: jl(p) for p in sorted(OUT.glob("answers_*.jsonl"))}
    J = {p.stem[len("judge_"):]: jl(p) for p in sorted(OUT.glob("judge_*.jsonl"))}
    S = {"models": {}, "judges": list(J), "comparisons": []}
    for m, rs in A.items():
        sc = [r for r in rs if r["type"] != "Q9"]
        S["models"][m] = {"n": len(rs), "sec_mean": mean([r["sec"] for r in rs]), "tok_per_s": mean([r["tok_per_s"] for r in rs])}
        for jid, js in J.items():
            jm = [x for x in js if x["model"] == m]; js_sc = [x for x in jm if x["type"] != "Q9"]; q9 = [x for x in jm if x["type"] == "Q9"]
            S["models"][m][jid] = {"total": mean([x["total"] for x in js_sc]), "accuracy": mean([x["j"].get("accuracy") for x in js_sc]),
                                   "pass80": mean([1.0 if (x["total"] or 0) >= 80 else 0.0 for x in js_sc if x["total"] is not None]),
                                   "hallucination": mean([1.0 if x["j"].get("hallucination") else 0.0 for x in js_sc if x["j"].get("hallucination") is not None])}
        jany = next(iter(J.values()), [])
        jm = [x for x in jany if x["model"] == m]
        S["models"][m]["auto"] = {"citation_match": mean([1.0 if x["citation_match"] else 0.0 for x in jm if x["type"] != "Q9" and x["citation_match"] is not None]),
                                  "hint_numbers": mean([x["hint_numbers_covered"] for x in jm if x["type"] != "Q9"]),
                                  "q9_refusal": mean([1.0 if x["refusal"] else 0.0 for x in jm if x["type"] == "Q9"])}
    pairs = [("Q14-4", "Q7-4"), ("Q14-4i", "Q7-4"), ("Q14-8", "Q7-4"), ("Q14-4", "Q14-8"), ("Q14-4i", "Q14-4")]
    for a, b in pairs:
        if a not in A or b not in A: continue
        for jid, js in J.items():
            ta = {(x["set"], x["qid"]): x["total"] for x in js if x["model"] == a and x["type"] != "Q9"}
            tb = {(x["set"], x["qid"]): x["total"] for x in js if x["model"] == b and x["type"] != "Q9"}
            ks = sorted(k for k in ta if k in tb and ta[k] is not None and tb[k] is not None)
            if not ks: continue
            xa, xb = [ta[k] for k in ks], [tb[k] for k in ks]; lo, hi = boot(xa, xb)
            S["comparisons"].append({"a": a, "b": b, "judge": jid, "n": len(ks), "delta": float(np.mean(xa) - np.mean(xb)), "ci95": [lo, hi], "significant": lo > 0 or hi < 0})
        # 자동 지표(출처 일치) 차이: 채점자와 무관
        js = next(iter(J.values()), [])
        ca = {(x["set"], x["qid"]): float(bool(x["citation_match"])) for x in js if x["model"] == a and x["type"] != "Q9" and x["citation_match"] is not None}
        cb = {(x["set"], x["qid"]): float(bool(x["citation_match"])) for x in js if x["model"] == b and x["type"] != "Q9" and x["citation_match"] is not None}
        ks = sorted(k for k in ca if k in cb)
        if ks:
            lo, hi = boot([ca[k] for k in ks], [cb[k] for k in ks])
            S["comparisons"].append({"a": a, "b": b, "judge": "auto_citation", "n": len(ks), "delta": float(np.mean([ca[k] for k in ks]) - np.mean([cb[k] for k in ks])), "ci95": [lo, hi], "significant": lo > 0 or hi < 0})
    json.dump(S, open(OUT / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(S, ensure_ascii=False, indent=1))

if __name__ == "__main__":
    if "--contexts" in ARGS: contexts()
    if "--gen" in ARGS: i = ARGS.index("--gen"); gen(ARGS[i + 1], ARGS[i + 2])
    if "--judge" in ARGS: i = ARGS.index("--judge"); judge(ARGS[i + 1], ARGS[i + 2])
    if "--summary" in ARGS: summary()
