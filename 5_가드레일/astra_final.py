# -*- coding: utf-8 -*-
"""아스트라 200건 최종 집계: 수동 라벨(R/W/NA/OUT) × 게이트 최종 판정. eval_astra.py 완료 후 실행."""
import json, sys, collections
sys.stdout.reconfigure(encoding="utf-8")
out = json.load(open("astra_결과.json", encoding="utf-8")); lab = json.load(open("astra_manual_labels.json", encoding="utf-8"))
REF = ("refuse", "refuse_scope", "refuse_multi")
rows = out["items"]
for it in rows: it["label"] = lab[it["qid"]]["label"]
def pct(a, b): return f"{a}/{b} ({a/b:.0%})" if b else "-"
print(f"2차 판정 {out['stage2_done']}/{out['stage2_n']}\n")
print("[검색 품질, 수동 판정] 규정에 답이 있는 109건 중 상위3 안에 답 조문:", pct(sum(it['label']=='R' for it in rows), sum(it['label'] in ('R','W') for it in rows)))
print("  놓친 27건의 정답 조문 분포:", dict(collections.Counter(lab[it['qid']]['note'].split()[0] for it in rows if it['label']=='W')))
print("\n[게이트 결과] 라벨 × 최종 처리")
tab = collections.defaultdict(collections.Counter)
for it in rows: tab[it["label"]][("거절" if it["action_final"] in REF else ("원문만" if it["action_final"]=="raw_only" else "답변"))] += 1
for L, name in [("OUT","범위 밖(보수복지·무관) 40"),("NA","규정에 답 없음 51"),("R","답 있고 검색 성공 82"),("W","답 있는데 검색 실패 27")]:
    print(f"  {name:<24}", dict(tab[L]))
neg = [it for it in rows if it["label"] in ("OUT","NA")]; pos = [it for it in rows if it["label"]=="R"]; w = [it for it in rows if it["label"]=="W"]
print(f"\n정답 없음(OUT+NA) {len(neg)}건 거절률: 1단계 {pct(sum(it['action1'] in REF for it in neg), len(neg))} → 최종 {pct(sum(it['action_final'] in REF for it in neg), len(neg))}")
print(f"검색 성공(R) {len(pos)}건 오거절률: 1단계 {pct(sum(it['action1'] in REF for it in pos), len(pos))} → 최종 {pct(sum(it['action_final'] in REF for it in pos), len(pos))}")
print(f"검색 실패(W) {len(w)}건 중 거절(잘못된 답 차단): 최종 {pct(sum(it['action_final'] in REF for it in w), len(w))}")
print("\n[스타일별 최종 오거절 (R 라벨만)]")
for st in sorted({it['style'] for it in pos}):
    xs=[it for it in pos if it['style']==st]; print(f"  {st:<6}", pct(sum(it['action_final'] in REF for it in xs), len(xs)))
print("\n[R인데 최종 거절된 질문]")
for it in pos:
    if it["action_final"] in REF: print(f"  {it['qid']} ce={it['top1_ce']:.3f} {it['q'][:40]}")
