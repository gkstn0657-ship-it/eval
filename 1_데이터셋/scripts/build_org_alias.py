"""292개 기관 별칭 사전 자동 생성 (2026-10-14).

입력: 3_성능테스트/corpora/L2P_A.jsonl 의 org 필드 (292개 정식 기관명)
출력: 1_데이터셋/05_clean/L2/org_alias.json   {별칭: 정식 기관명}
      1_데이터셋/05_clean/L2/org_alias_report.md   충돌·제외 목록 (사람 검수용)

규칙: 정식 기관명에서 (1) 법인 표기 제거 (2) 공백·가운뎃점 제거 (3) 접두 "한국" 제거
      (4) 접미 "공사/공단/진흥원/연구원/…" 제거. 생성 별칭은 4자 이상만 채택.
      한 별칭이 두 기관 이상에 걸리거나 다른 기관 정식명에 포함되면 제외하고 보고서에 적는다.
      run_eval.py 의 손으로 쓴 ORG_ALIAS 가 항상 우선한다 (이 파일은 보완용).
"""
import json, re, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "3_성능테스트" / "corpora" / "L2P_A.jsonl"
OUT = ROOT / "1_데이터셋" / "05_clean" / "L2" / "org_alias.json"
REPORT = ROOT / "1_데이터셋" / "05_clean" / "L2" / "org_alias_report.md"

LEGAL = re.compile(r"^\((주|재|사)\)\s*|^(재단법인|사단법인|주식회사)\s*|\s*(\(주\)|주식회사)$")
SUFFIXES = ["진흥원", "연구원", "평가원", "연수원", "개발원", "정보원", "공단", "공사", "재단", "협회", "센터", "사업단", "위원회", "진흥회", "기금", "공제회", "학교", "병원", "연구소", "공제회"]
MIN_LEN = 4

def base_name(org):
    return LEGAL.sub("", org).strip()

def variants(org):
    """정식 기관명 하나에서 별칭 후보를 만든다. (별칭, 규칙) 목록."""
    b = base_name(org)
    out = [(b, "법인 표기 제거")]
    compact = re.sub(r"[\s·ㆍ]", "", b)
    if compact != b: out.append((compact, "공백·가운뎃점 제거"))
    for s, rule in list(out):
        if s.startswith("한국") and len(s) - 2 >= MIN_LEN:
            out.append((s[2:], "접두 한국 제거"))
    for s, rule in list(out):
        for suf in SUFFIXES:
            if s.endswith(suf):
                core = s[:-len(suf)]
                if len(core) >= MIN_LEN: out.append((core, f"접미 {suf} 제거"))
                break
    seen, uniq = set(), []
    for s, rule in out:
        s = s.strip()
        if len(s) >= MIN_LEN and s not in seen:  # 정식명 자체도 포함 (detect_org 가 292곳 전부를 잡도록)
            seen.add(s); uniq.append((s, rule))
    return uniq

def main():
    orgs = sorted({json.loads(l)["org"] for l in CORPUS.open(encoding="utf-8")})
    sys.path.insert(0, str(ROOT / "3_성능테스트"))
    cand = defaultdict(set); rule_of = {}
    for o in orgs:
        for s, rule in variants(o):
            cand[s].add(o); rule_of[s] = rule
    # 손으로 쓴 사전 (우선)
    manual = {}
    src = (ROOT / "3_성능테스트" / "run_eval.py").read_text(encoding="utf-8")
    m = re.search(r"ORG_ALIAS = \{(.*?)\n\}", src, re.S)
    for k, v in re.findall(r'"([^"]+)":\s*"([^"]+)"', m.group(1)): manual[k] = v
    alias, collisions, dropped = {}, [], []
    for s, os_ in sorted(cand.items()):
        if s in manual: continue
        if len(os_) > 1: collisions.append((s, sorted(os_))); continue
        alias[s] = next(iter(os_))
    # 포함 관계 점검: 별칭이 다른 기관 정식명의 부분 문자열이면 제외 ("과학기술"·"산업기술"처럼 일반 명사라 기관 없는 질의에서 오탐)
    contained = []
    for s, o in list(alias.items()):
        if s == base_name(o): continue  # 정식명 자체는 긴 표현 우선 매칭으로 안전하므로 유지
        hosts = [p for p in orgs if p != o and s in base_name(p)]
        if hosts: contained.append((s, o, hosts)); del alias[s]
    merged = dict(alias); merged.update(manual)
    OUT.write_text(json.dumps(merged, ensure_ascii=False, indent=1), encoding="utf-8")
    covered = {v for v in merged.values()}
    lines = ["# 기관 별칭 사전 자동 생성 보고 (2026-10-14)", "",
             f"- 기관 {len(orgs)}곳, 별칭 {len(merged)}개 (손으로 쓴 {len(manual)}개 + 자동 {len(alias)}개). 별칭이 하나도 없는 기관: {len(set(orgs) - covered)}곳",
             f"- 충돌로 제외한 별칭 {len(collisions)}개, 다른 기관명에 포함되어 제외한 별칭 {len(contained)}개", "",
             "## 충돌로 제외 (한 별칭이 두 기관 이상)", "", "| 별칭 | 걸리는 기관 |", "|---|---|"]
    lines += [f"| {s} | {', '.join(os_)} |" for s, os_ in collisions]
    lines += ["", "## 다른 기관 정식명에 포함되어 제외한 별칭", "", "| 별칭 | 대상 기관 | 포함하는 다른 기관 |", "|---|---|---|"]
    lines += [f"| {s} | {o} | {', '.join(h)} |" for s, o, h in contained]
    lines += ["", "## 별칭 없는 기관", ""] + [f"- {o}" for o in sorted(set(orgs) - covered)]
    lines += ["", "## 규칙별 생성 수", ""]
    cnt = defaultdict(int)
    for s in alias: cnt[rule_of[s]] += 1
    lines += [f"- {r}: {n}" for r, n in sorted(cnt.items(), key=lambda x: -x[1])]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"orgs {len(orgs)} aliases {len(merged)} auto {len(alias)} collisions {len(collisions)} contained {len(contained)} uncovered {len(set(orgs)-covered)}")

if __name__ == "__main__":
    main()
