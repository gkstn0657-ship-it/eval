# -*- coding: utf-8 -*-
"""v4 평가 공용 로더. 검색은 다시 돌리지 않는다: results_astra(hybrid_prefilter, L2P_A) 상위 5개 조문 고정. 봉인용(split.json 의 sealed)은 절대 읽지 않는다."""
import json, re, glob, collections, hashlib, subprocess
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent.parent; V2 = HERE.parent / "v2"
GOLD = {json.loads(l)["qid"]: json.loads(l) for l in open(ROOT / "1_데이터셋/06_eval/goldenset_astra.jsonl", encoding="utf-8") if l.strip()}
LABELS = json.load(open(ROOT / "5_가드레일/astra_manual_labels.json", encoding="utf-8"))
PQ = {r["qid"]: r for r in json.load(open(ROOT / "3_성능테스트/results_astra/per_query.json", encoding="utf-8")) if r["pipeline"] == "hybrid_prefilter"}
CHUNKS = collections.defaultdict(list)
for l in open(ROOT / "3_성능테스트/corpora/L2P_A.jsonl", encoding="utf-8"):
    r = json.loads(l); CHUNKS[r["article_id"]].append(r)
KNOWN_IDS = set(CHUNKS)
def article(aid):
    rs = CHUNKS[aid]; r0 = rs[0]
    return {"id": aid, "org": r0["org"], "label": f"{r0['article_no']}({r0.get('article_title') or ''})", "article_no": r0["article_no"], "title": r0.get("article_title") or "",
            "revision_date": r0["revision_date"] or "", "text": " ".join(x["text"] for x in rs)}
DELEG, SCOPE_ART = {}, {}
for f in glob.glob(str(ROOT / "1_데이터셋/05_clean/L2/*/*/*.meta.json")):
    m = json.load(open(f, encoding="utf-8"))
    if m.get("is_latest"): DELEG[m["org"]] = sorted({d.strip() for d in m.get("delegated_to", []) if "이 규정" not in d and re.fullmatch(r"[가-힣A-Za-z·\s]{2,14}(규정|규칙|지침|요령|세칙)", d.strip())})
for l in open(ROOT / "1_데이터셋/05_clean/L2/index_A_latest_only.jsonl", encoding="utf-8"):
    r = json.loads(l)
    if re.search(r"적용\s*범위|적용\s*대상", r.get("article_title") or "") and r["org"] not in SCOPE_ART: SCOPE_ART[r["org"]] = r["text"]
SPLIT = json.load(open(V2 / "split.json", encoding="utf-8")); SPLIT2 = json.load(open(V2 / "split2.json", encoding="utf-8"))
DEV = SPLIT["dev"]                      # 설계용 104 (선택용 56 + 점검용 48). 전부 개발 자료
def half_of(qid): return "select" if qid in SPLIT2["select"] else ("check" if qid in SPLIT2["check"] else "?")
def top5(qid): return [article(a) for a in PQ[qid]["top5"]]
def git_rev():
    try: return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception: return "?"
def sha(s): return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]
assert not any(q in SPLIT["sealed"] for q in DEV)
