# -*- coding: utf-8 -*-
"""사람 검수 자료 생성 (아스트라 지시 3·4절). AI 판정은 넣지 않는다. 시스템 이름은 가리고 무작위 식별자를 쓴다.
  python make_review.py na27       → review/NA27_재검수.md (+ .jsonl)         설계용 NA 27건
  python make_review.py responses  → review/responses_blind.md (+ .jsonl)      A·B·D 최종 응답, 가림. 대응표는 review/_private_map_responses.json (검수자는 열지 않는다)
  python make_review.py pairs      → review/m4_pairs_blind.md (+ .jsonl)       서로 다른 질문군 100개 × (정답 조문 후보, 비정답 상위 조문 후보). 대응표 review/_private_map_pairs.json
봉인용 96건은 어느 모드에서도 읽지 않는다."""
import json, sys, random, collections, re
from pathlib import Path
import data_v4 as D
sys.stdout.reconfigure(encoding="utf-8")
R = D.HERE / "review"; R.mkdir(exist_ok=True); RES = D.HERE / "results"
MODE = sys.argv[1] if len(sys.argv) > 1 else "all"
rng = random.Random(20261013)
def rid(prefix, n=6): return prefix + "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(n))

def na27():
    qids = [q for q in D.DEV if D.LABELS[q]["label"] == "NA"]
    items, md = [], ["# 설계용 NA 27건 재검수 자료 (2026-10-13)", "", "현재 라벨은 전부 NA(규정에 답 없음). 각 문항에 대해 아래 다섯 중 하나를 적고 근거 문장을 남겨 주세요. 라벨을 바꾸면 이전 값·새 값·근거·검수자·날짜를 `NA27_결과.jsonl` 에 기록합니다.",
                  "", "- `NA` 유지: 검색된 조문 어디에도 답이 없다", "- `R`: 검색된 조문에 답이 있다 (라벨 오류) → 근거 조문·문장", "- `OUT`: 인사규정이 아닌 다른 규정 사항", "- `SUBJ`: 질문 대상(임원·퇴직자 등)이 적용 대상 밖이라 답이 없다", "- `AMB`: 상황이 애매해 사람도 판단 못 함", ""]
    for i, q in enumerate(qids, 1):
        g = D.GOLD[q]; arts = D.top5(q)
        items.append({"qid": q, "query": g["query"], "org": g["context_org"], "style": g["style"], "current_label": "NA", "note": D.LABELS[q].get("note", ""), "retrieved": [a["id"] for a in arts],
                      "new_label": "", "evidence": "", "reviewer": "", "date": ""})
        md += [f"## {i}. {q} · {g['style']}", f"**질문**: {g['query']}", f"기존 메모: {D.LABELS[q].get('note', '') or '(없음)'}", "", "검색 조문 (hybrid_prefilter 상위 5):"]
        md += [f"- **{a['label']}** ({a['revision_date']}): {a['text'][:600]}{'…' if len(a['text']) > 600 else ''}" for a in arts]
        md += ["", "새 라벨: ______   근거 문장: ______________________", ""]
    (R / "NA27_재검수.md").write_text("\n".join(md), encoding="utf-8")
    with open(R / "NA27_재검수.jsonl", "w", encoding="utf-8") as f: [f.write(json.dumps(x, ensure_ascii=False) + "\n") for x in items]
    print("NA27:", len(items))

def responses():
    A = {json.loads(l)["qid"]: json.loads(l) for l in open(RES / "baseline_A.jsonl", encoding="utf-8")}
    B = {x["qid"]: x for x in json.load(open(RES / "rows_B.json", encoding="utf-8"))}; Dc = {x["qid"]: x for x in json.load(open(RES / "rows_D_cache.json", encoding="utf-8"))}
    items, priv = [], {}
    for q in D.DEV:
        texts = {"A": A[q]["answer"], "B": B[q]["text"], "D": Dc[q]["text"]}
        uniq = {}
        for sysname, t in texts.items(): uniq.setdefault(t, []).append(sysname)   # B 와 D 가 같은 응답이면 한 번만 검수
        for t, systems in uniq.items():
            k = rid("R"); priv[k] = {"qid": q, "systems": systems}
            items.append({"rid": k, "qid": q, "query": D.GOLD[q]["query"], "org": D.GOLD[q]["context_org"], "response": t})
    rng.shuffle(items)
    md = ["# 최종 응답 사람 검수 (가림, 2026-10-13)", "", f"항목 {len(items)}개. 어느 시스템의 응답인지 표시하지 않았습니다(같은 질문에 응답이 2~3개). 라벨·검색 결과·정답도 넣지 않았습니다. 각 응답에 대해 아래를 적어 주세요.",
          "", "1. 필요한 내용을 모두 답했는가 (예/일부/아니오/답변 아님)", "2. 적용 대상·기간·조건·예외가 맞는가 (예/아니오/해당 없음)", "3. 출처가 해당 주장을 지지하는가 (예/아니오/출처 없음)",
          "4. 부분 답변이라면 미확인 부분을 분명히 밝혔는가 (예/아니오/해당 없음)", "5. 근거 없는 추가 설명이나 단정이 있는가 (있음/없음) → 있으면 그 문장", "6. 종합: 정답 / 부분 정답 / 오답 / 보류·거절(적절) / 보류·거절(부적절)", "",
          "안내·거절 문구에 사실 주장이 있으면 5번에 적어 주세요. 규정 원문은 수집본 기준이므로 현행 규정과 다를 수 있습니다.", ""]
    for i, x in enumerate(items, 1):
        md += [f"## {i}. {x['rid']}", f"**기관**: {x['org']}  **질문**: {x['query']}", "", "```", x["response"], "```", "", "1.___ 2.___ 3.___ 4.___ 5.___ 6.___", ""]
    (R / "responses_blind.md").write_text("\n".join(md), encoding="utf-8")
    with open(R / "responses_blind.jsonl", "w", encoding="utf-8") as f: [f.write(json.dumps({**x, "q1": "", "q2": "", "q3": "", "q4": "", "q5": "", "q6": "", "reviewer": "", "date": ""}, ensure_ascii=False) + "\n") for x in items]
    json.dump(priv, open(R / "_private_map_responses.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("responses:", len(items), "(질문 104 × 시스템 3, 동일 응답 합침)")

SETS = {"dev": "results_chunk_dev", "employee": "results_chunk_employee", "kgs_faq": "results_chunk_kgs_faq", "matrix": "results_chunk_matrix", "kaia": "results_full_kaia",
        "knps": "results_full_knps", "kps": "results_full_kps", "kdhc": "results_v9_kdhc", "new22": "results_v9_new22"}
def pairs():
    cands, astra_arts = [], {a for q in D.DEV for a in D.PQ[q]["top5"]}
    for s, rd in SETS.items():
        gold = {json.loads(l)["qid"]: json.loads(l) for l in open(D.ROOT / f"1_데이터셋/06_eval/goldenset_{s}.jsonl", encoding="utf-8") if l.strip()}
        pq = {r["qid"]: r for r in json.load(open(D.ROOT / f"3_성능테스트/{rd}/per_query.json", encoding="utf-8")) if r.get("pipeline") == "hybrid_prefilter" and r.get("corpus") == "L2P_A"}
        for qid, g in gold.items():
            if qid not in pq or not g.get("gold_ids"): continue
            golds = [a for a in g["gold_ids"] if a in D.CHUNKS]
            negs = [a for a in pq[qid]["top5"] if a not in set(g["gold_ids"]) | set(g.get("gold_ids_any_revision", [])) and a in D.CHUNKS]
            if not golds or not negs: continue
            if D.CHUNKS[golds[0]][0]["org"] == "한국가스안전공사": continue          # astra(가스안전공사) 와 기관이 겹치는 질문은 제외
            cands.append({"set": s, "qid": qid, "query": g["query"], "org": D.CHUNKS[golds[0]][0]["org"], "gold": golds[0], "neg": negs[0], "group": golds[0]})
    by_group = collections.defaultdict(list)
    for c in cands: by_group[c["group"]].append(c)
    groups = sorted(by_group); rng.shuffle(groups)
    chosen, per_set = [], collections.Counter()
    for gk in groups:                                   # 질문군(정답 조문)당 1문항, 세트별 상한으로 한 세트 쏠림 방지
        c = rng.choice(by_group[gk])
        if per_set[c["set"]] >= 25: continue
        chosen.append(c); per_set[c["set"]] += 1
        if len(chosen) == 100: break
    items, priv, neg_dup = [], {}, collections.Counter(c["neg"] for c in chosen)
    for c in chosen:
        for role, aid in (("gold", c["gold"]), ("neg", c["neg"])):
            a = D.article(aid); k = rid("P")
            priv[k] = {"qid": c["qid"], "set": c["set"], "role": role, "article_id": aid, "group": c["group"], "neg_dup_count": neg_dup[aid] if role == "neg" else None, "in_astra_top5": aid in astra_arts}
            items.append({"pid": k, "query": c["query"], "org": a["org"], "article": a["label"], "revision_date": a["revision_date"], "text": a["text"]})
    rng.shuffle(items)
    md = ["# M4 답변성 라벨 자료 (가림, 2026-10-13)", "", f"쌍 {len(items)}개 = 서로 다른 질문군 {len(chosen)}개 × 조문 2개. 어느 조문이 정답 후보인지 표시하지 않았습니다. 조문 1건 전체를 보고 아래 중 하나를 적어 주세요.", "",
          "- `충분`: 이 조문만으로 질문에 필요한 내용을 모두 답할 수 있다", "- `일부`: 일부만 답한다 (비고에 나머지가 있을 조문 번호를 아는 경우 적기)", "- `관련만`: 주제는 관련 있으나 질문에 답하지 않는다", "- `모순`: 조문 내용이 질문의 전제·주장과 충돌한다 → 비고에 무엇과 충돌하는지 명시. 질문의 전제를 부정하는 정확한 답은 모순이 아니라 충분이다", "",
          f"세트 분포: {dict(per_set)}. 같은 비정답 조문이 여러 질문에 쓰인 수: {sum(1 for v in neg_dup.values() if v > 1)}. astra 검색 결과와 겹치는 조문: {sum(p['in_astra_top5'] for p in priv.values())}.", ""]
    for i, x in enumerate(items, 1):
        md += [f"## {i}. {x['pid']}", f"**질문**: {x['query']}", f"**조문**: {x['org']} 인사규정 {x['article']} ({x['revision_date']})", "", "```", x["text"], "```", "", "라벨: ______  비고: ______", ""]
    (R / "m4_pairs_blind.md").write_text("\n".join(md), encoding="utf-8")
    with open(R / "m4_pairs_blind.jsonl", "w", encoding="utf-8") as f: [f.write(json.dumps({**x, "label": "", "note": "", "reviewer": "", "date": ""}, ensure_ascii=False) + "\n") for x in items]
    json.dump(priv, open(R / "_private_map_pairs.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("pairs:", len(items), "groups:", len(chosen), "per_set:", dict(per_set), "neg_dup:", sum(1 for v in neg_dup.values() if v > 1), "astra_overlap:", sum(p["in_astra_top5"] for p in priv.values()))

if MODE in ("na27", "all"): na27()
if MODE in ("pairs", "all"): pairs()
if MODE in ("responses", "all"): responses()
