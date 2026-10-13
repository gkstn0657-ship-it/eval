# -*- coding: utf-8 -*-
"""M4 1단계: 사람 라벨 200쌍에서 현행 재랭커(bge-reranker-v2-m3)가 '답함'(충분+일부) vs '안 함'(관련만+모순)을 얼마나 가르는지 잰다.
실행 조건: review/m4_pairs_blind.jsonl 의 label 열이 사람 손으로 채워져 있어야 한다. 비어 있으면 멈춘다 (AI 라벨 대체 금지).
출력: results/m4_rerank_auc.json  (AUC + 부트스트랩 95% 구간, 질문군 단위 재표집, 라벨 분포, 이진화 가능 여부)
판정 기준(아스트라 지시 4절·회신 3절): AUC 하한 ≥ 0.80 이면 임계값 방식으로 간다. 미만이면 질문군 단위 교차검증 미세조정을 시험한다. AUC 만으로 채택하지 않는다.
사용: python m4_rerank_auc.py [--model BAAI/bge-reranker-v2-m3]"""
import json, sys, random, collections
import numpy as np
import data_v4 as D
sys.stdout.reconfigure(encoding="utf-8")
A = sys.argv[1:]; MODEL = A[A.index("--model") + 1] if "--model" in A else "BAAI/bge-reranker-v2-m3"
R = D.HERE / "review"; items = [json.loads(l) for l in open(R / "m4_pairs_blind.jsonl", encoding="utf-8")]
priv = json.load(open(R / "_private_map_pairs.json", encoding="utf-8"))
unl = [x for x in items if not x.get("label")]
if unl: sys.exit(f"라벨 미입력 {len(unl)}/{len(items)}. 사람 검수 후 실행한다.")
POS, NEG = {"충분", "일부"}, {"관련만", "모순"}
bad = [x for x in items if x["label"] not in POS | NEG]
if bad: sys.exit(f"알 수 없는 라벨: {[(x['pid'], x['label']) for x in bad][:5]}")
dist = collections.Counter(x["label"] for x in items); role_dist = collections.Counter((priv[x["pid"]]["role"], x["label"]) for x in items)

from FlagEmbedding import FlagReranker
rr = FlagReranker(MODEL, use_fp16=True)
scores = rr.compute_score([[x["query"], x["text"]] for x in items], normalize=True)
y = np.array([1 if x["label"] in POS else 0 for x in items]); s = np.array(scores, dtype=float); grp = np.array([priv[x["pid"]]["group"] for x in items])
def auc(y, s):
    p, n = s[y == 1], s[y == 0]
    if len(p) == 0 or len(n) == 0: return None
    return float((p[:, None] > n[None, :]).mean() + 0.5 * (p[:, None] == n[None, :]).mean())
rng = random.Random(2026); groups = sorted(set(grp)); boots = []
for _ in range(2000):                                   # 질문군 단위 재표집 (같은 질문의 두 쌍은 함께 뽑힌다)
    pick = [rng.choice(groups) for _ in groups]; idx = np.concatenate([np.where(grp == g)[0] for g in pick])
    a = auc(y[idx], s[idx]); boots.append(a if a is not None else np.nan)
lo, hi = np.nanpercentile(boots, [2.5, 97.5])
out = {"model": MODEL, "n_pairs": len(items), "n_groups": len(groups), "label_dist": dict(dist), "role_x_label": {f"{r}|{l}": c for (r, l), c in role_dist.items()},
       "auc": auc(y, s), "auc_ci95_group_bootstrap": [float(lo), float(hi)],
       "binarization": {"contradiction_rate": dist["모순"] / len(items), "partial_among_gold": role_dist[("gold", "일부")] / max(1, sum(c for (r, _), c in role_dist.items() if r == "gold")),
                        "note": "일부는 비율과 무관하게 부분 답변(아스트라 결정). 모순은 통과 금지."},
       "threshold_candidates": {str(t): {"tpr": float((s[y == 1] >= t).mean()), "fpr": float((s[y == 0] >= t).mean())} for t in (0.1, 0.2, 0.3, 0.5, 0.7)},
       "verdict": "threshold" if lo >= 0.80 else "finetune_cv", "rule": "AUC 하한 ≥ 0.80 → 임계값, 미만 → 질문군 단위 교차검증 미세조정. 최종 채택은 전체 가드레일 지표로."}
json.dump(out, open(D.HERE / "results" / "m4_rerank_auc.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
