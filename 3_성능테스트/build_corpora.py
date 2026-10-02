# -*- coding: utf-8 -*-
"""05_clean 의 조문 파일을 두 파이프라인이 공통으로 읽는 코퍼스로 변환한다.

조건: 정제 강도 L0/L1/L2 × 개정판 정책 A(최신판만)/B(전체) = 6개 코퍼스.
청킹 결정 (2026-10-01): 조 단위를 기본 단위로 한다. 조문이 MAX_CHARS 를 넘으면 항(①②…) 경계에서 나누고
각 조각 앞에 '제N조(제목)' 접두어를 붙인다. 채점은 조문 ID 단위(조각 접미 #k 제거)로 한다.
산출: corpora/{level}_{policy}.jsonl  (chunk_id, article_id, org, revision_date, is_latest, text)
"""
import json, re, sys
from pathlib import Path
from collections import Counter
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "1_데이터셋" / "05_clean"
OUT = Path(__file__).resolve().parent / "corpora"; OUT.mkdir(exist_ok=True)
MAX_CHARS = 1200

def split_article(u):
    text = u["text"]
    if len(text) <= MAX_CHARS: return [text]
    head = re.match(r"제\d+조(?:의\d+)?\s*(?:[\(（][^)）]*[\)）])?", text)
    title = head.group(0) if head else u["article_no"]
    parts = re.split(r"(?=[①-⑮])", text[len(title):] if head else text)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) <= 1:  # 항 표시가 없으면 고정 길이로
        body = text[len(title):] if head else text
        parts = [body[i:i + MAX_CHARS] for i in range(0, len(body), MAX_CHARS)]
    # 항 하나가 MAX_CHARS 를 넘으면 고정 길이로 더 나눈다 (L0 부칙 덩어리 대응)
    parts = [p[i:i + MAX_CHARS] for p in parts for i in range(0, len(p), MAX_CHARS)]
    chunks, cur = [], ""
    for p in parts:
        if cur and len(cur) + len(p) > MAX_CHARS: chunks.append(cur); cur = p
        else: cur = (cur + " " + p).strip()
    if cur: chunks.append(cur)
    return [f"{title} {c}" if not c.startswith(title) else c for c in chunks]

DEF_TITLE = re.compile(r"정의|용어")
def split_definition(u):
    """개선 2: 정의 조항을 호(1. 2. …) 단위로 나눈다. 각 조각 앞에 '제N조(제목)' 접두어."""
    text = u["text"]
    head = re.match(r"제\d+조(?:의\d+)?\s*(?:[\(（][^)）]*[\)）])?", text)
    title = head.group(0) if head else u["article_no"]
    body = text[len(title):] if head else text
    parts = [p.strip() for p in re.split(r"(?=(?:^|\s)\d{1,2}(?:의\d+)?\.\s)", body) if p.strip()]
    if len(parts) <= 2: return [text]
    lead = parts[0] if not re.match(r"\d{1,2}(?:의\d+)?\.\s", parts[0]) else ""
    items = parts[1:] if lead else parts
    return [f"{title} {lead} {it}".strip() for it in items]

def improve_text(u):
    """개선 1: L2 에서 분리한 개정 표기를 조문 앞에 접두어로 되돌린다. 날짜를 모아 정렬하고 최근 8개만, 신설·삭제는 종류를 붙인다."""
    marks = u.get("revision_marks") or []
    if not marks: return u["text"]
    dated = {}
    for m in marks:
        for d in m.get("dates", []):
            dated.setdefault(d, set()).add(m["type"])
    if not dated: return u["text"]
    def key(d):
        n = re.findall(r"\d+", d); return tuple(int(x) for x in n[:3]) + (0,) * (3 - len(n[:3]))
    items = sorted(dated, key=key)[-8:]
    tags = [d + ("" if dated[d] == {"개정"} else " " + "/".join(sorted(t for t in dated[d] if t != "개정"))) for d in items]
    return f"[개정이력 {', '.join(tags)}] " + u["text"]

stats = []
for level in ("L0", "L1", "L2", "L2P"):
    units = []
    src_level = "L2" if level == "L2P" else level
    for f in sorted((CLEAN / src_level).rglob("*.articles.jsonl")):
        for l in f.read_text(encoding="utf-8").splitlines():
            if l.strip(): units.append(json.loads(l))
    for policy in ("A", "B"):
        sel = [u for u in units if policy == "B" or u["is_latest"]]
        # 같은 문서 안에서 조번호가 중복되면(L0 의 부칙 '제1조' 등) 두 번째부터 '@2' 접미를 붙여
        # 골든셋 정답 ID 와 섞이지 않게 한다. 본문 조문이 먼저 나오므로 첫 번째가 본문이다.
        seen, deduped = Counter(), []
        for u in sel:
            seen[u["id"]] += 1
            deduped.append({**u, "id": f"{u['id']}@{seen[u['id']]}"} if seen[u["id"]] > 1 else u)
        sel = deduped
        rows, nsplit, ndup = [], 0, sum(v - 1 for v in seen.values())
        for u in sel:
            if level == "L2P":
                u = {**u, "text": improve_text(u)}
                pieces = split_definition(u) if u.get("article_title") and DEF_TITLE.search(u["article_title"]) else split_article(u)
            else:
                pieces = split_article(u)
            if len(pieces) > 1: nsplit += 1
            for k, t in enumerate(pieces):
                rows.append({"chunk_id": u["id"] + (f"#{k}" if len(pieces) > 1 else ""), "article_id": u["id"],
                             "org": u["org"], "revision_date": u["revision_date"], "is_latest": u["is_latest"],
                             "article_no": u["article_no"], "article_title": u.get("article_title", ""), "text": t})
        p = OUT / f"{level}_{policy}.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        lens = [len(r["text"]) for r in rows]
        stats.append({"corpus": f"{level}_{policy}", "articles": len(sel), "chunks": len(rows), "split_articles": nsplit,
                      "dup_article_ids": ndup, "chars": sum(lens), "max_chunk": max(lens), "orgs": len({r['org'] for r in rows})})
        print(stats[-1])
json.dump(stats, open(OUT / "corpora_stats.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
