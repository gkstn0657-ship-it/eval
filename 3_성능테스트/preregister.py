# -*- coding: utf-8 -*-
"""측정 전에 골든셋·코퍼스 해시와 비교 조건·지표·게이트를 사전등록.md 로 고정한다."""
import hashlib, json, sys, datetime
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
def h(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]
ev = ROOT / "1_데이터셋" / "06_eval"
stats = json.load(open(HERE / "corpora" / "corpora_stats.json", encoding="utf-8"))
lines = [f"# 사전등록 ({datetime.date.today()})", "", "측정 전 확정. 측정 후 변경하지 않는다. 변경이 필요하면 새 버전으로 다시 등록한다.", "",
 "## 골든셋", "", f"- goldenset_dev.jsonl: sha256 {h(ev/'goldenset_dev.jsonl')}  (44문항)", f"- goldenset_holdout_sealed.jsonl: sha256 {h(ev/'goldenset_holdout_sealed.jsonl')}  (44문항, 봉인. dev 측정·분석이 끝난 뒤 1회만 개봉)", "",
 "## 코퍼스", "", "| 코퍼스 | 조문 | 청크 | sha256 |", "|---|---|---|---|"]
for s in stats: lines.append(f"| {s['corpus']} | {s['articles']} | {s['chunks']} | {h(HERE/'corpora'/(s['corpus']+'.jsonl'))} |")
lines += ["", "청킹: 조 단위. 1,200자 초과 조문은 항(①②…) 경계에서 분할, 조 제목 접두어. 같은 문서 안 중복 조번호(부칙 제1조 등)는 두 번째부터 @n 접미를 붙여 본문 조문만 정답으로 인정.", "",
 "## 비교 조건", "", "- 파이프라인: vanilla(chat_rag: ko-sbert-nli 단일 벡터, 임계값 0.5) / hybrid(hybrid-search-eval: bge-m3 + BM25 → RRF k=60 → 기관 필터 → bge-reranker-v2-m3 상위 30)",
 "- 정제 강도: L0 파싱만 / L1 노이즈 제거 / L2 부칙·개정표기 분리", "- 개정판 정책: A 최신판만 / B 전체 개정판(같은 기관·조번호의 다른 개정판도 정답 인정)",
 "- 베이스라인: random, bm25 단독, dense(bge-m3) 단독", "- 총 6 코퍼스 × 5 검색기 = 30 조건", "",
 "## 지표", "", "- Recall@5, Recall@10, MRR, nDCG@5. 조문 ID 단위, 조각(#k)은 조문으로 합침, Q9(범위 외) 제외 n=40.", "- Q9: 거절률(vanilla 상위 점수 < 0.5, hybrid 재랭커 점수 < 0)을 별도 집계.", "",
 "## 결정 규칙", "", "- 주 지표 nDCG@5. 쌍대 부트스트랩 2,000회 95% CI 가 0을 포함하지 않으면 유의.", "- 주 비교: hybrid vs vanilla(L2_A), L2 vs L0(파이프라인별), B vs A(L2), hybrid vs 단독 축.", "",
 "## 게이트", "", "- random 베이스라인 nDCG@5 < 0.05 (코퍼스 크기 대비 타당성)", "- 모든 조건에서 검색 실패(예외) 0건", "- 유형별 n ≥ 3 (dev 기준, 유형당 4문항)", "",
 "## 변경 금지 선언", "", "이 문서 커밋 이후 골든셋·코퍼스·지표·결정 규칙을 바꾸지 않는다."]
(HERE / "사전등록.md").write_text("\n".join(lines), encoding="utf-8"); print("\n".join(lines[:12]))
