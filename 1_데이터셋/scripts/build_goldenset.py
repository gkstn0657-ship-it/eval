# -*- coding: utf-8 -*-
"""06_eval/queries_draft.jsonl 을 검증하고 dev / holdout 으로 분할한다.

검증
- gold_ids 가 05_clean/L2 조문 인덱스(B 전체)에 실제로 존재하는지
- Q11(일상어)은 질의 단어가 정답 조문 본문에 0~1개만 포함되는지 (2개 이상이면 경고)
- 유형별 개수, 기관 분포
분할
- 유형별 층화. qid sha1 마지막 자리 짝수 → dev, 홀수 → holdout. 유형 내 균형(반반)이 깨지면 해시 순으로 보정.
산출
- 06_eval/goldenset_dev.jsonl, 06_eval/goldenset_holdout_sealed.jsonl, 06_eval/goldenset_report.md
"""
import json, re, hashlib, sys, collections, itertools
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
EV = ROOT / "06_eval"
idx = {}
for l in (ROOT / "05_clean/L2/index_B_all_revisions.jsonl").read_text(encoding="utf-8").splitlines():
    a = json.loads(l); idx[a["id"]] = a
by_org_art = collections.defaultdict(list)  # 기관|조번호 -> ids (정책 B 에서 개정판 무시 매칭용)
for k, a in idx.items(): by_org_art[f"{a['org']}|{a['article_no']}"].append(k)

qs = [json.loads(l) for l in (EV / "queries_draft.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
errors, warns = [], []
STOP = set("은 는 이 가 을 를 의 에 에서 으로 로 과 와 도 만 까지 부터 어떻게 뭐야 뭐지 알려줘 궁금해 있어 있나 있을까 하는 하면 할 수 되나 되면 돼 해 몇 언제 누가 어디 왜 어떤 무슨 경우 때 것 거 좀 나 내 우리 회사 기관 규정 조 항 호".split())
def words(s): return [w for w in re.findall(r"[가-힣A-Za-z0-9]+", s) if w not in STOP and len(w) > 1]

for q in qs:
    for k in ("qid", "type", "persona", "query", "gold_ids", "answer_hint"):
        if k not in q: errors.append(f"{q.get('qid')}: 필드 누락 {k}")
    if q["type"] != "Q9" and not q["gold_ids"]: errors.append(f"{q['qid']}: 라벨 없음")
    for g in q["gold_ids"]:
        if g not in idx: errors.append(f"{q['qid']}: 존재하지 않는 조문 ID {g}")
    if q["type"] == "Q11" and q["gold_ids"]:
        text = " ".join(idx[g]["text"] for g in q["gold_ids"] if g in idx)
        hit = [w for w in words(q["query"]) if w in text]
        q["_overlap"] = hit
        if len(hit) >= 2: warns.append(f"{q['qid']}: Q11 단어 겹침 {len(hit)}개 {hit} → Q1 재분류 또는 질의 수정 검토")
    # 정책 B 매칭용: 같은 기관·조번호의 다른 개정판 ID 를 함께 기록
    q["gold_ids_any_revision"] = sorted({x for g in q["gold_ids"] if g in idx for x in by_org_art[f"{idx[g]['org']}|{idx[g]['article_no']}"]})

if errors:
    print("ERRORS"); print("\n".join(errors)); sys.exit(1)

# 분할
dev, hold = [], []
for t, group in itertools.groupby(sorted(qs, key=lambda q: (q["type"], q["qid"])), key=lambda q: q["type"]):
    group = list(group)
    scored = sorted(group, key=lambda q: hashlib.sha1(q["qid"].encode()).hexdigest())
    half = len(scored) // 2
    for i, q in enumerate(scored):
        q["split"] = "dev" if i < half or (len(scored) % 2 and i == len(scored) - 1 and int(hashlib.sha1(q["qid"].encode()).hexdigest()[-1], 16) % 2 == 0) else "holdout"
        (dev if q["split"] == "dev" else hold).append(q)

def dump(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for q in rows: f.write(json.dumps({k: v for k, v in q.items() if not k.startswith("_")}, ensure_ascii=False) + "\n")
dump(EV / "goldenset_dev.jsonl", dev); dump(EV / "goldenset_holdout_sealed.jsonl", hold)

types = collections.Counter(q["type"] for q in qs); orgs = collections.Counter(idx[q["gold_ids"][0]]["org"] if q["gold_ids"] else q.get("gold_org", "-") for q in qs)
rep = ["# 골든셋 구성 보고", "", f"총 {len(qs)}개 질의, dev {len(dev)} / holdout {len(hold)}", "",
       "| 유형 | 개수 | dev | holdout |", "|---|---|---|---|"]
for t in sorted(types): rep.append(f"| {t} | {types[t]} | {sum(1 for q in dev if q['type']==t)} | {sum(1 for q in hold if q['type']==t)} |")
rep += ["", "## 기관 분포", "", "| 기관 | 질의 수 |", "|---|---|"] + [f"| {o} | {n} |" for o, n in orgs.most_common()]
rep += ["", "## 경고", ""] + ([f"- {w}" for w in warns] or ["- 없음"])
rep += ["", "## 파일 해시 (사전등록용)", ""]
for fn in ("queries_draft.jsonl", "goldenset_dev.jsonl", "goldenset_holdout_sealed.jsonl"):
    rep.append(f"- {fn}: sha256 {hashlib.sha256((EV / fn).read_bytes()).hexdigest()}")
(EV / "goldenset_report.md").write_text("\n".join(rep), encoding="utf-8")
print("\n".join(rep))
