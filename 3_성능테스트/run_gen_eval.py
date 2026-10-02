# -*- coding: utf-8 -*-
"""답변 생성 품질 평가 (골든셋 dev 44문항, L2 최신판 코퍼스).

단계
  --gen   : vanilla / hybrid 가 찾은 상위 5개 조문을 컨텍스트로 로컬 LLM(Ollama qwen2.5:7b-instruct)이 답변 생성. chat_rag 프롬프트.
  --judge : 답변을 채점. (a) 자동 지표: 출처 조문 일치, 범위 외 거절, 기대 답 숫자 포함. (b) LLM 판정(동일 모델, 루브릭 100점: 정확성 40 / 완전성 25 / 출처정확도 20 / 간결성 15, chat_rag 실험과 동일).
  --summary : 집계와 쌍대 부트스트랩.
산출: results/gen_answers.jsonl, results/gen_judgments.jsonl, results/gen_summary.json
"""
import json, re, sys, time, urllib.request
from pathlib import Path
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(HERE)); import run_eval as E  # noqa: E402
OUT = HERE / "results"; CORPUS = "L2_A"; PIPES = ["vanilla", "hybrid"]
OLLAMA, MODEL = "http://127.0.0.1:11434", "qwen2.5:7b-instruct"
GOLD = [json.loads(l) for l in (ROOT / "1_데이터셋/06_eval/goldenset_dev.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

SYSTEM_PROMPT = """너는 회사 내부 규정안내를 도와주는 사내 문서 도우미야.

[참고 문서]의 내용을 바탕으로 사용자의 질문에 답변해. 문서에 없는 내용은 지어내지 말고 모르거나 없다고 대답해.
문서를 기반으로 답변을 생성할 때 그 답변과 관련된 문서의 내용을 직접 확인해서 사용자가 검증해볼 수 있도록 내용의 위치를 같이 출력해.

출력방식은 간결하고 보기 좋게 깔끔하게 정리해서 출력해.
문서의 검증 위치는 참고 문서의 [출처: ...] 에 적힌 기관명과 조번호를 그대로 써서 아래 예시 형식으로 작성해.
예시: " 한국가스공사 인사규정 제31조(정년) "

너가 생성한 답변이 문서 내용과 위배되거나 없는 내용을 임의로 지어내었는지 반복적으로 재검증하고 틀린 부분을 수정해서 최종 답변해줘.
[참고문서]
{context}"""

JUDGE_PROMPT = """너는 사내 규정 챗봇의 답변을 채점하는 심사자다. 아래 정보를 보고 100점 만점으로 채점한다. 반드시 JSON 만 출력한다.

[질문] {query}
[기대 답(사람이 적은 요지)] {hint}
[정답 조문 원문]
{gold_text}
[챗봇 답변]
{answer}

채점 기준
- accuracy (0~40): 답변 내용이 정답 조문과 일치하는가. 숫자·기간·조건이 틀리면 크게 감점. 조문에 없는 내용을 사실처럼 말하면 0~10.
- completeness (0~25): 기대 답의 핵심 요소를 빠짐없이 담았는가.
- citation (0~20): 출처로 적은 기관명·조번호가 정답 조문과 일치하는가. 출처가 없거나 틀리면 0~5.
- concise (0~15): 불필요한 내용 없이 간결한가.
- hallucination (true/false): 정답 조문에 없는 사실·숫자·출처를 만들어 냈는가.
- 질문이 규정 범위 밖(기대 답에 "없음"이라 적힘)이면 "규정에 없다/확인할 수 없다"고 답해야 만점이고, 그럴듯하게 답하면 accuracy 0, hallucination true.

출력 형식: {{"accuracy": 정수, "completeness": 정수, "citation": 정수, "concise": 정수, "hallucination": true|false, "reason": "한 문장"}}"""

def ollama(system, user, num_predict=600, fmt=None):
    body = {"model": MODEL, "stream": False, "options": {"temperature": 0, "num_ctx": 4096, "num_predict": num_predict},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    if fmt: body["format"] = fmt
    req = urllib.request.Request(OLLAMA + "/api/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r: return json.loads(r.read().decode())["message"]["content"]

def build_context(c, hits, abstain):
    if abstain or not hits: return "(검색결과 없음)"
    return "\n\n".join(f"[출처: {c.rows[i]['org']} 인사규정 {c.rows[i]['article_no']}{'(' + c.rows[i]['article_title'] + ')' if c.rows[i].get('article_title') else ''} | 개정 {c.rows[i]['revision_date']}]\n{c.rows[i]['text'][:1500]}" for i in hits)

def gen():
    c = E.Corpus(CORPUS); rows = []; t0 = time.time()
    done = {(r["qid"], r["pipeline"]) for r in map(json.loads, (OUT / "gen_answers.jsonl").read_text(encoding="utf-8").splitlines())} if (OUT / "gen_answers.jsonl").exists() else set()
    f = open(OUT / "gen_answers.jsonl", "a", encoding="utf-8")
    for q in GOLD:
        for p in PIPES:
            if (q["qid"], p) in done: continue
            idx, info = E.PIPELINES[p](c, q["query"])
            arts, hits = [], []
            for i in idx:
                a = c.rows[i]["article_id"]
                if a in arts: continue
                arts.append(a); hits.append(i)
                if len(hits) >= 5: break
            abstain = bool(info.get("abstain")) and p == "vanilla"
            ctx = build_context(c, hits, abstain)
            t1 = time.time(); ans = ollama(SYSTEM_PROMPT.format(context=ctx), q["query"])
            rec = {"qid": q["qid"], "type": q["type"], "pipeline": p, "query": q["query"], "gold_ids": q["gold_ids"], "hint": q["answer_hint"],
                   "retrieved": [c.rows[i]["article_id"] for i in hits], "retrieval_hit5": bool(set(arts[:5]) & set(q["gold_ids"])), "abstain_retrieval": abstain,
                   "answer": ans, "gen_seconds": round(time.time() - t1, 1)}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
            print(f"{q['qid']} {p:8} {rec['gen_seconds']:5.1f}s  hit5={rec['retrieval_hit5']}  ({time.time()-t0:.0f}s)")
    f.close()

ART = re.compile(r"제\s*\d+\s*조(?:의\s*\d+)?")
NUM = re.compile(r"\d+")
def auto_checks(rec, gold_text):
    ans = rec["answer"]; cited = {re.sub(r"\s", "", m) for m in ART.findall(ans)}
    gold_nos = {g.split("|")[2] for g in rec["gold_ids"]}; gold_orgs = {g.split("|")[0] for g in rec["gold_ids"]}
    cite_ok = bool(cited & gold_nos) and any(o.replace("(주)", "").replace("주식회사 ", "")[:4] in ans for o in gold_orgs) if rec["gold_ids"] else None
    refuse = bool(re.search(r"없습니다|없다|확인할 수 없|규정에 (명시|나와|포함)되어 있지|해당 (내용|규정)이 없|찾을 수 없|알 수 없", ans))
    hint_nums = set(NUM.findall(rec["hint"])) - {"1", "2", "3"} if rec["type"] in ("Q3", "Q2", "Q6", "Q7", "Q10") else set()
    num_ok = (len(hint_nums & set(NUM.findall(ans))) / len(hint_nums)) if hint_nums else None
    return {"cited_articles": sorted(cited), "citation_match": cite_ok, "refusal": refuse, "hint_numbers_covered": num_ok}

def judge():
    idx = {json.loads(l)["id"]: json.loads(l) for l in (ROOT / "1_데이터셋/05_clean/L2/index_B_all_revisions.jsonl").read_text(encoding="utf-8").splitlines()}
    recs = [json.loads(l) for l in (OUT / "gen_answers.jsonl").read_text(encoding="utf-8").splitlines()]
    done = {(r["qid"], r["pipeline"]) for r in map(json.loads, (OUT / "gen_judgments.jsonl").read_text(encoding="utf-8").splitlines())} if (OUT / "gen_judgments.jsonl").exists() else set()
    f = open(OUT / "gen_judgments.jsonl", "a", encoding="utf-8")
    for r in recs:
        if (r["qid"], r["pipeline"]) in done: continue
        gold_text = "\n\n".join(f"[{g}]\n{idx[g]['text'][:1200]}" for g in r["gold_ids"] if g in idx) or "(범위 외 질의: 정답 조문 없음. 기대 답 참조)"
        a = auto_checks(r, gold_text)
        raw = ollama("너는 엄격한 채점자다. JSON 만 출력한다.", JUDGE_PROMPT.format(query=r["query"], hint=r["hint"], gold_text=gold_text, answer=r["answer"]), num_predict=300, fmt="json")
        try: j = json.loads(raw)
        except Exception: j = {"accuracy": None, "completeness": None, "citation": None, "concise": None, "hallucination": None, "reason": "parse_error: " + raw[:120]}
        total = sum(v for k, v in j.items() if k in ("accuracy", "completeness", "citation", "concise") and isinstance(v, (int, float)))
        out = {**{k: r[k] for k in ("qid", "type", "pipeline", "retrieval_hit5", "abstain_retrieval")}, **a, "judge": j, "total": total if j.get("accuracy") is not None else None}
        f.write(json.dumps(out, ensure_ascii=False) + "\n"); f.flush(); print(r["qid"], r["pipeline"], "total", out["total"], "halluc", j.get("hallucination"), "cite", a["citation_match"])
    f.close()

def summary():
    J = [json.loads(l) for l in (OUT / "gen_judgments.jsonl").read_text(encoding="utf-8").splitlines()]
    def mean(v): v = [x for x in v if x is not None]; return float(np.mean(v)) if v else None
    S = {"n": len(GOLD), "model": MODEL, "corpus": CORPUS, "pipelines": {}, "by_type": [], "comparison": {}}
    for p in PIPES:
        R = [j for j in J if j["pipeline"] == p]; scored = [j for j in R if j["type"] != "Q9"]; q9 = [j for j in R if j["type"] == "Q9"]
        S["pipelines"][p] = {"total_mean": mean([j["total"] for j in scored]), "total_std": float(np.std([j["total"] for j in scored if j["total"] is not None])),
            "accuracy": mean([j["judge"].get("accuracy") for j in scored]), "completeness": mean([j["judge"].get("completeness") for j in scored]),
            "citation": mean([j["judge"].get("citation") for j in scored]), "concise": mean([j["judge"].get("concise") for j in scored]),
            "hallucination_rate": mean([1.0 if j["judge"].get("hallucination") else 0.0 for j in scored if j["judge"].get("hallucination") is not None]),
            "citation_match_rate": mean([1.0 if j["citation_match"] else 0.0 for j in scored if j["citation_match"] is not None]),
            "hint_numbers_covered": mean([j["hint_numbers_covered"] for j in scored]),
            "pass80_rate": mean([1.0 if (j["total"] or 0) >= 80 else 0.0 for j in scored]),
            "q9_refusal_rate": mean([1.0 if j["refusal"] else 0.0 for j in q9]), "q9_hallucination_rate": mean([1.0 if j["judge"].get("hallucination") else 0.0 for j in q9]),
            "answer_when_retrieval_missed": {"n": sum(1 for j in scored if not j["retrieval_hit5"]), "hallucination_rate": mean([1.0 if j["judge"].get("hallucination") else 0.0 for j in scored if not j["retrieval_hit5"]])}}
        for t in sorted({j["type"] for j in R}, key=lambda x: int(x[1:])):
            T = [j for j in R if j["type"] == t]
            S["by_type"].append({"pipeline": p, "type": t, "n": len(T), "total_mean": mean([j["total"] for j in T]), "hallucination_rate": mean([1.0 if j["judge"].get("hallucination") else 0.0 for j in T]), "citation_match_rate": mean([1.0 if j["citation_match"] else 0.0 for j in T if j["citation_match"] is not None])})
    a = {j["qid"]: j["total"] for j in J if j["pipeline"] == "hybrid" and j["type"] != "Q9"}; b = {j["qid"]: j["total"] for j in J if j["pipeline"] == "vanilla" and j["type"] != "Q9"}
    ks = [k for k in a if k in b and a[k] is not None and b[k] is not None]; A = np.array([a[k] for k in ks]); B = np.array([b[k] for k in ks]); np.random.seed(42)
    d = [(A[s] - B[s]).mean() for s in (np.random.randint(0, len(ks), len(ks)) for _ in range(2000))]
    S["comparison"] = {"label": "hybrid vs vanilla, 총점", "delta": float(A.mean() - B.mean()), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], "n": len(ks)}
    json.dump(S, open(OUT / "gen_summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for p, v in S["pipelines"].items(): print(p, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in v.items()})
    print(S["comparison"])

if __name__ == "__main__":
    if "--gen" in sys.argv: gen()
    if "--judge" in sys.argv: judge()
    if "--summary" in sys.argv: summary()
