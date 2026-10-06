# -*- coding: utf-8 -*-
"""1006 실험: 청킹 방식 비교용 코퍼스 3종을 만든다. 계획서 1006_청킹비교_실험계획.md.

조건 (모두 L2 최신판 + L2P 개정이력 접두어 텍스트를 원천으로 공유, C0 = 기존 corpora/L2P_A.jsonl)
- C1 CH_FIX_A  : 기관별로 조문을 순서대로 이어 붙인 뒤 1,200자 창, 겹침 200자. 조 경계 무시.
- C2 CH_SEM_A  : 같은 이어 붙인 텍스트에 chat_rag semantic_chunk 알고리즘(ko-sbert-nli, buffer 1, 85 백분위, 500자 초과 시 500/50 고정 분할).
                 chat_rag 원본은 문장 200개 초과 시 예외를 던져 기관 문서 전체에 못 쓰므로, 그 상한만 없앴다.
- C3 CH_PARA_A : 모든 조문을 항(①~⑮) 경계로 분할. 각 조각 앞에 개정이력 접두어와 '제N조(제목)'을 붙인다.

조 경계를 넘는 청크(C1, C2)는 글자 수가 가장 많은 조문을 article_id 로, 포함된 조문 전체를 등장 순서대로 article_ids 로 둔다.
사용: python build_chunk_variants.py [--only fix,sem,para] [--orgs N] [--device cpu|cuda] [--out DIR]
"""
import argparse, bisect, json, re, sys, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent

# 기존 코퍼스(C0)와 같은 원천 텍스트를 쓰기 위해 build_corpora.py 의 함수 정의만 가져온다(파일 생성 루프 앞까지).
_ns = {"__file__": str(HERE / "build_corpora.py")}
exec((HERE / "build_corpora.py").read_text(encoding="utf-8").split("stats = []")[0], _ns)
CLEAN, improve_text = _ns["CLEAN"], _ns["improve_text"]

FIX_SIZE, FIX_OVERLAP = 1200, 200
SEM_MAX, SEM_OVERLAP, SEM_PCT, SEM_BUF = 500, 50, 85.0, 1
SEP = "\n\n"  # 조문 사이 구분. 의미 단위 분할의 문장 분리 규칙(빈 줄)과 맞춘다.
TITLE_RE = re.compile(r"제\d+조(?:의\d+)?\s*(?:[\(（][^)）]*[\)）])?")
REV_RE = re.compile(r"^\[개정이력[^\]]*\]\s*")


def load_units():
    """build_corpora.py 의 L2P_A 와 같은 순서·같은 ID·같은 텍스트로 최신판 조문을 읽는다."""
    units = []
    for f in sorted((CLEAN / "L2").rglob("*.articles.jsonl")):
        for l in f.read_text(encoding="utf-8").splitlines():
            if l.strip(): units.append(json.loads(l))
    sel = [u for u in units if u["is_latest"]]
    seen, out = Counter(), []
    for u in sel:
        seen[u["id"]] += 1
        u = {**u, "id": f"{u['id']}@{seen[u['id']]}"} if seen[u["id"]] > 1 else u
        out.append({**u, "text": improve_text(u)})
    return out


def row(chunk_id, main, ids, text):
    return {"chunk_id": chunk_id, "article_id": main["id"], "article_ids": ids, "org": main["org"],
            "revision_date": main["revision_date"], "is_latest": main["is_latest"],
            "article_no": main["article_no"], "article_title": main.get("article_title", ""), "text": text}


class OrgDoc:
    """기관 하나의 조문을 이어 붙인 텍스트와 조문별 글자 구간."""
    def __init__(self, arts):
        self.arts, self.starts, parts, pos = arts, [], [], 0
        for a in arts:
            self.starts.append(pos); parts.append(a["text"]); pos += len(a["text"]) + len(SEP)
        self.ends = [s + len(a["text"]) for s, a in zip(self.starts, arts)]
        self.text = SEP.join(parts)

    def attribute(self, s, e):
        """[s, e) 구간에 걸친 조문들. (주 조문, 등장 순서 ID 목록)"""
        i = max(0, bisect.bisect_right(self.starts, s) - 1)
        hits = []
        while i < len(self.arts) and self.starts[i] < e:
            ov = min(e, self.ends[i]) - max(s, self.starts[i])
            if ov > 0: hits.append((ov, i))
            i += 1
        if not hits:  # 구분자만 걸친 경우: 가장 가까운 앞 조문
            i = max(0, bisect.bisect_right(self.starts, s) - 1); hits = [(1, i)]
        main = self.arts[max(hits, key=lambda x: (x[0], -x[1]))[1]]
        return main, [self.arts[i]["id"] for _, i in sorted(hits, key=lambda x: x[1])]


def build_fix(docs):
    rows = []
    for org, d in docs.items():
        step, k, s = FIX_SIZE - FIX_OVERLAP, 0, 0
        while s < len(d.text):
            e = min(len(d.text), s + FIX_SIZE)
            t = d.text[s:e].strip()
            if t:
                main, ids = d.attribute(s, e); rows.append(row(f"{org}|fix|{k}", main, ids, t)); k += 1
            if e == len(d.text): break
            s += step
    return rows


def sentences(text):
    """chat_rag 와 같은 분리 규칙((?<=[.!?。])\\s+ 또는 빈 줄)으로 나누되, 원문 위치를 함께 돌려준다."""
    out, pos = [], 0
    for m in re.finditer(r"(?<=[.!?。])\s+|\n{2,}", text):
        seg = text[pos:m.start()]
        if seg.strip():
            lead = len(seg) - len(seg.lstrip()); out.append((pos + lead, pos + len(seg.rstrip())))
        pos = m.end()
    seg = text[pos:]
    if seg.strip():
        lead = len(seg) - len(seg.lstrip()); out.append((pos + lead, pos + len(seg.rstrip())))
    return out


def build_sem(docs, device):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("jhgan/ko-sbert-nli", device=device)
    rows = []
    for n, (org, d) in enumerate(docs.items(), 1):
        spans = sentences(d.text)
        if len(spans) <= 1:
            groups = [spans]
        else:
            emb = model.encode([d.text[s:e] for s, e in spans], batch_size=64, show_progress_bar=False, convert_to_numpy=True)
            def gvec(a, b):
                v = emb[max(0, a):min(len(emb), b)].mean(axis=0); return v / (np.linalg.norm(v) + 1e-12)
            dist = [1.0 - float(gvec(i - SEM_BUF, i + SEM_BUF + 1) @ gvec(i + 1 - SEM_BUF, i + 2 + SEM_BUF)) for i in range(len(spans) - 1)]
            thr = float(np.percentile(dist, SEM_PCT)); groups, cur = [], [spans[0]]
            for i, x in enumerate(dist):
                if x > thr: groups.append(cur); cur = [spans[i + 1]]
                else: cur.append(spans[i + 1])
            groups.append(cur)
        k = 0
        for g in groups:
            if not g: continue
            s, e = g[0][0], g[-1][1]
            # chat_rag 는 문장을 공백으로 이어 붙인다. 위치 추적을 위해 원문 구간을 쓰되 줄바꿈 묶음만 공백으로 바꾼다.
            pieces = [(s, e)] if e - s <= SEM_MAX else [(p, min(e, p + SEM_MAX)) for p in range(s, e, SEM_MAX - SEM_OVERLAP)]
            for ps, pe in pieces:
                t = re.sub(r"\s*\n\s*", " ", d.text[ps:pe]).strip()
                if t:
                    main, ids = d.attribute(ps, pe); rows.append(row(f"{org}|sem|{k}", main, ids, t)); k += 1
        if n % 20 == 0: print(f"  sem {n}/{len(docs)} 기관", flush=True)
    return rows


def build_para(units):
    rows = []
    for u in units:
        text = u["text"]
        m = REV_RE.match(text); rev = m.group(0) if m else ""; body = text[len(rev):]
        h = TITLE_RE.match(body); title = h.group(0) if h else u["article_no"]
        rest = body[len(h.group(0)):] if h else body
        parts = [p.strip() for p in re.split(r"(?=[①-⑮])", rest) if p.strip()]
        if len(parts) <= 1:
            rows.append(row(u["id"], u, [u["id"]], text)); continue
        for k, p in enumerate(parts):
            rows.append(row(f"{u['id']}#{k}", u, [u["id"]], f"{rev}{title} {p}".strip()))
    return rows


def stats(name, rows):
    L = [len(r["text"]) for r in rows]
    return {"corpus": name, "chunks": len(rows), "articles_covered": len({r["article_id"] for r in rows}),
            "cross_article_chunks": sum(len(r["article_ids"]) > 1 for r in rows),
            "mean_len": round(sum(L) / len(L), 1) if L else 0, "max_len": max(L) if L else 0,
            "orgs": len({r["org"] for r in rows})}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="fix,sem,para")
    ap.add_argument("--orgs", type=int, default=0, help="동작 확인용: 앞에서 N개 기관만")
    ap.add_argument("--device", default=None, help="의미 단위 분할의 임베딩 장치 (기본: cuda 있으면 cuda)")
    ap.add_argument("--out", default=str(HERE / "corpora"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    units = load_units()
    orgs = list(dict.fromkeys(u["org"] for u in units))
    if a.orgs: keep = set(orgs[:a.orgs]); units = [u for u in units if u["org"] in keep]
    by_org = defaultdict(list)
    for u in units: by_org[u["org"]].append(u)
    docs = {o: OrgDoc(arts) for o, arts in by_org.items()}
    print(f"조문 {len(units)}개, 기관 {len(docs)}곳")
    device = a.device
    if device is None:
        import torch; device = "cuda" if torch.cuda.is_available() else "cpu"
    builders = {"fix": ("CH_FIX_A", lambda: build_fix(docs)), "sem": ("CH_SEM_A", lambda: build_sem(docs, device)), "para": ("CH_PARA_A", lambda: build_para(units))}
    sp = out / "chunk_variants_stats.json"
    all_stats = {s["corpus"]: s for s in json.loads(sp.read_text(encoding="utf-8"))} if sp.exists() else {}
    for key in a.only.split(","):
        name, fn = builders[key]; t0 = time.time(); rows = fn()
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        st = {**stats(name, rows), "build_sec": round(time.time() - t0, 1)}; all_stats[name] = st; print(st, flush=True)
    sp.write_text(json.dumps(list(all_stats.values()), ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
