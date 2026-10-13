# -*- coding: utf-8 -*-
"""기준선 A: 가드레일 없는 RAG. 설계용 104건 전체에 7B 자유 생성 답변을 만든다 (아스트라 지시 3절).
프롬프트·생성 설정은 3_성능테스트/run_gen_eval.py 의 SYSTEM_PROMPT·DELEG_TMPL·ollama() 를 그대로 복사했다(그 파일은 수정하지 않는다).
검색 근거는 results_astra hybrid_prefilter 상위 5개 조문(가드레일 B~D 와 동일). 출력: results/baseline_A.jsonl, results/baseline_A_config.json
사용: python baseline_a.py   (이미 있는 qid 는 건너뛴다)"""
import json, time, sys, urllib.request
from pathlib import Path
import data_v4 as D
sys.stdout.reconfigure(encoding="utf-8")
MODEL = "qwen2.5:7b-instruct"; OLLAMA = "http://127.0.0.1:11434"; OPTS = {"temperature": 0, "num_ctx": 4096, "num_predict": 600}
SYSTEM_PROMPT = """너는 회사 내부 규정안내를 도와주는 사내 문서 도우미야.

[참고 문서]의 내용을 바탕으로 사용자의 질문에 답변해. 문서에 없는 내용은 지어내지 말고 모르거나 없다고 대답해.
문서를 기반으로 답변을 생성할 때 그 답변과 관련된 문서의 내용을 직접 확인해서 사용자가 검증해볼 수 있도록 내용의 위치를 같이 출력해.

출력방식은 간결하고 보기 좋게 깔끔하게 정리해서 출력해.
문서의 검증 위치는 참고 문서의 [출처: ...] 에 적힌 기관명과 조번호를 그대로 써서 아래 예시 형식으로 작성해.
예시: " 한국가스공사 인사규정 제31조(정년) "

너가 생성한 답변이 문서 내용과 위배되거나 없는 내용을 임의로 지어내었는지 반복적으로 재검증하고 틀린 부분을 수정해서 최종 답변해줘.
{delegation}[참고문서]
{context}"""
DELEG_TMPL = """[이 기관 인사규정이 다른 규정에 위임한 사항]
{items}
질문이 위 항목처럼 인사규정 밖의 사항(보수·복지·여비·교육·징계 세부 기준 등)이면, 인사규정에는 없다고 답하고 어느 규정 소관인지 위 목록에서 찾아 안내해.
"""
def build_context(arts): return "\n\n".join(f"[출처: {a['org']} 인사규정 {a['article_no']}{'(' + a['title'] + ')' if a['title'] else ''} | 개정 {a['revision_date']}]\n{a['text'][:1500]}" for a in arts)
def ollama(system, user):
    body = {"model": MODEL, "stream": False, "options": OPTS, "keep_alive": "30m", "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    req = urllib.request.Request(OLLAMA + "/api/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r: return json.loads(r.read().decode())["message"]["content"]

OUT = D.HERE / "results"; OUT.mkdir(exist_ok=True); f_out = OUT / "baseline_A.jsonl"
done = {json.loads(l)["qid"] for l in open(f_out, encoding="utf-8")} if f_out.exists() else set()
json.dump({"model": MODEL, "options": OPTS, "prompt_sha": D.sha(SYSTEM_PROMPT + DELEG_TMPL), "retrieval": "results_astra hybrid_prefilter top5 (L2P_A)", "corpus": "3_성능테스트/corpora/L2P_A.jsonl",
           "code_rev": D.git_rev(), "n": len(D.DEV), "started": time.strftime("%Y-%m-%d %H:%M:%S")}, open(OUT / "baseline_A_config.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
t0 = time.time()
with open(f_out, "a", encoding="utf-8") as f:
    for i, qid in enumerate(D.DEV, 1):
        if qid in done: continue
        q = D.GOLD[qid]; org = q["context_org"]; arts = D.top5(qid)
        items = D.DELEG.get(org, []); deleg = DELEG_TMPL.format(items="\n".join("- " + x for x in items)) if items else ""
        t1 = time.time(); ans = ollama(SYSTEM_PROMPT.format(context=build_context(arts), delegation=deleg), q["query"]); sec = round(time.time() - t1, 1)
        f.write(json.dumps({"qid": qid, "label": D.LABELS[qid]["label"], "query": q["query"], "org": org, "retrieved": [a["id"] for a in arts], "delegation_used": bool(deleg),
                            "answer": ans, "gen_seconds": sec, "model": MODEL}, ensure_ascii=False) + "\n"); f.flush()
        print(f"[{i}/{len(D.DEV)}] {qid} {sec}s ({time.time()-t0:.0f}s)", flush=True)
print("done", len(D.DEV))
