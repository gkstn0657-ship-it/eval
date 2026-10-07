# -*- coding: utf-8 -*-
"""가드레일 v3 신호 S2: 질문 임베딩이 '인사규정 장 제목'과 '위임 규정명' 중 어느 쪽에 가까운가 (bge-m3, CPU).
선택용 52문항에서 OUT vs R 의 AUC(부트스트랩 95% 구간)와 일상어 질문만의 AUC 를 낸다. 채택 기준: 전체 하한 ≥ 0.80, 일상어 ≥ 0.75.
사용: python measure_s2.py [--half select|check|all]
"""
import os, sys, json, glob, re, collections
from pathlib import Path
import numpy as np
A = sys.argv[1:]; HALF = A[A.index("--half") + 1] if "--half" in A else "select"
sys.argv = [sys.argv[0]]
os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface")); os.environ.setdefault("HF_HUB_OFFLINE", "1")
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "3_성능테스트")); import run_eval as R
sys.stdout.reconfigure(encoding="utf-8")
gold = {json.loads(l)["qid"]: json.loads(l) for l in open(ROOT / "1_데이터셋/06_eval/goldenset_astra.jsonl", encoding="utf-8") if l.strip()}
labels = json.load(open(ROOT / "5_가드레일/astra_manual_labels.json", encoding="utf-8"))
qids = json.load(open(HERE / "split2.json", encoding="utf-8"))[HALF] if HALF in ("select", "check") else json.load(open(HERE / "split.json", encoding="utf-8"))["dev"]
DELEG, SECTIONS = {}, collections.defaultdict(list)
for f in glob.glob(str(ROOT / "1_데이터셋/05_clean/L2/*/*/*.meta.json")):
    m = json.load(open(f, encoding="utf-8"))
    if m.get("is_latest"): DELEG[m["org"]] = sorted({d.strip() for d in m.get("delegated_to", []) if "이 규정" not in d and re.fullmatch(r"[가-힣A-Za-z·\s]{2,14}(규정|규칙|지침|요령|세칙)", d.strip())})
for l in open(ROOT / "1_데이터셋/05_clean/L2/index_A_latest_only.jsonl", encoding="utf-8"):
    r = json.loads(l); sp = (r.get("section_path") or "").strip()
    if sp and sp not in SECTIONS[r["org"]]: SECTIONS[r["org"]].append(sp)
# 위임 규정명이 기관에 없으면 일반 목록으로 보강 (질문 쪽 비교 대상이 비면 신호가 정의되지 않으므로)
GENERIC_OTHER = ["보수규정", "복무규정", "여비규정", "복리후생규정", "교육훈련규정", "퇴직금규정", "상벌규정"]
m = R.st_model("BAAI/bge-m3")
def emb(texts): return m.encode(texts, normalize_embeddings=True, convert_to_numpy=True, batch_size=16)
scores = {}
for qid in qids:
    q = gold[qid]; org = q["context_org"]
    sec = SECTIONS.get(org) or ["총칙", "임용", "승진", "전보", "휴직", "징계", "퇴직", "인사위원회"]
    oth = sorted(set((DELEG.get(org) or []) + GENERIC_OTHER))
    v = emb([q["query"]])[0]; es = emb(sec); eo = emb(oth)
    scores[qid] = float((es @ v).max() - (eo @ v).max())   # 양수면 인사규정 쪽에 가깝다
def auc(pos, neg):
    if not pos or not neg: return None
    return float(np.mean([[1.0 if p > n else 0.5 if p == n else 0.0 for n in neg] for p in pos]))
def boot_auc(pos, neg, n=2000):
    rng = np.random.default_rng(42); pos, neg = np.array(pos), np.array(neg); vals = []
    for _ in range(n):
        p = pos[rng.integers(0, len(pos), len(pos))]; q = neg[rng.integers(0, len(neg), len(neg))]
        vals.append(np.mean((p[:, None] > q[None, :]) + 0.5 * (p[:, None] == q[None, :])))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
def report(name, cond):
    R_ = [scores[q] for q in qids if labels[q]["label"] == "R" and cond(q)]
    O_ = [scores[q] for q in qids if labels[q]["label"] == "OUT" and cond(q)]
    N_ = [scores[q] for q in qids if labels[q]["label"] == "NA" and cond(q)]
    a = auc(R_, O_); ci = boot_auc(R_, O_) if R_ and O_ else (None, None)
    print(f"{name}: R {len(R_)} vs OUT {len(O_)} AUC {a if a is None else round(a,3)} CI [{ci[0] if ci[0] is None else round(ci[0],3)}, {ci[1] if ci[1] is None else round(ci[1],3)}] | R vs NA AUC {auc(R_, N_) if not (R_ and N_) else round(auc(R_, N_),3)}")
    return a, ci
out = {"half": HALF}
out["all"] = report("전체", lambda q: True)
out["casual"] = report("일상어만", lambda q: gold[q]["style"] == "일상어")
out["scores"] = scores
json.dump(out, open(HERE / "results" / f"s2_{HALF}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
lo = out["all"][1][0]; ca = out["casual"][0]
print("판정:", "채택" if (lo is not None and lo >= 0.80 and ca is not None and ca >= 0.75) else "폐기", "(기준: 전체 하한 ≥ 0.80, 일상어 ≥ 0.75)")
