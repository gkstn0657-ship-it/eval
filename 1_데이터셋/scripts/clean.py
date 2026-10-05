# -*- coding: utf-8 -*-
"""04_정제규칙/정제규칙.md 의 R-01~R-14 를 01_raw 에 적용해 05_clean/L0, L1, L2 를 산출한다.

L0 파싱       : R-01 표 셀 보존, R-11 메타데이터 승계, R-12 성명 마스킹
L1 노이즈 제거 : L0 + R-02 그림 자리표시, R-05 삭제조항 표기, R-06 PDF 머리글·쪽번호, R-07 목차 분리,
                R-08 번호 정규화, R-09 문자 정규화, R-14 조 제목·section_path 부착
L2 구조 분리   : L1 + R-03 부칙 분리, R-04 개정 표기 분리, R-10 개정판 정책 A/B, R-13 위임 규정, R-14 용어 사전
"""
import json, re, zipfile, subprocess, unicodedata, sys, io
from pathlib import Path
from collections import Counter, defaultdict
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "01_raw", ROOT / "05_clean"
sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------- 추출 (R-01)
def serialize_table(rows):
    """rows: list[list[str]] -> 행 단위 직렬화. 첫 행을 헤더로 보고 각 행에 '헤더: 값' 으로 붙인다."""
    rows = [[c.strip() for c in r] for r in rows if any(c.strip() for c in r)]
    if not rows: return ""
    header = rows[0]
    out = ["[표] " + " | ".join(header)]
    for r in rows[1:]:
        cells = []
        for i, v in enumerate(r):
            h = header[i] if i < len(header) and header[i] else f"열{i+1}"
            if v: cells.append(f"{h}: {v}" if h != v else v)
        if cells: out.append(" | ".join(cells))
    return "\n".join(out)

def hwp_tables(p):
    x = subprocess.run(["hwp5proc", "xml", str(p)], capture_output=True).stdout.decode("utf-8", "ignore")
    tables = []
    for t in re.findall(r"<TableControl.*?</TableControl>", x, re.S):
        grid = defaultdict(dict)
        for c in re.findall(r"<TableCell\b([^>]*)>(.*?)</TableCell>", t, re.S):
            attrs, body = c
            r = int(re.search(r'\brow="(\d+)"', attrs).group(1)); col = int(re.search(r'\bcol="(\d+)"', attrs).group(1))
            txt = " ".join(s.strip() for s in re.findall(r"<Text[^>]*>(.*?)</Text>", body, re.S) if s.strip())
            grid[r][col] = re.sub(r"\s+", " ", txt)
        rows = [[grid[r].get(c, "") for c in range(max(grid[r]) + 1)] for r in sorted(grid)] if grid else []
        tables.append(serialize_table(rows))
    return tables

def hwp_image_count(p):
    """BinData 에 실제 이미지가 있는지. 선 도형만 있는 <그림> 자리표시자(주택도시보증공사)와 구분한다 (R-02)."""
    try:
        import olefile
        return sum(1 for e in olefile.OleFileIO(str(p)).listdir() if e[0] == "BinData")
    except Exception:
        return -1

def extract_hwp(p):
    text = subprocess.run(["hwp5txt", str(p)], capture_output=True).stdout.decode("utf-8", "ignore")
    tables = hwp_tables(p)
    parts = text.split("<표>")
    if len(parts) - 1 != len(tables):
        # 개수가 다르면 안전하게 뒤에 덧붙인다
        text = text + "\n" + "\n\n".join(tables)
    else:
        text = parts[0] + "".join(("\n" + tables[i] + "\n") + parts[i + 1] for i in range(len(tables)))
    return [text], {"tables": len(tables), "bindata_images": hwp_image_count(p), "picture_placeholders": text.count("<그림>")}

def extract_hwpx(p):
    z = zipfile.ZipFile(p)
    secs = sorted(n for n in z.namelist() if re.match(r"Contents/section\d+\.xml", n))
    out, ntab = [], 0
    def walk(el):
        nonlocal ntab
        tag = el.tag.split("}")[-1]
        if tag == "tbl":
            ntab += 1
            rows = []
            for tr in el.iter():
                if tr.tag.split("}")[-1] == "tr":
                    rows.append(["".join(t.text or "" for t in tc.iter() if t.tag.split("}")[-1] == "t") for tc in tr if tc.tag.split("}")[-1] == "tc"])
            out.append("\n" + serialize_table(rows) + "\n"); return
        if tag == "t" and el.text: out.append(el.text)
        for ch in el: walk(ch)
        if tag == "p": out.append("\n")
    for n in secs: walk(ET.fromstring(z.read(n)))
    return ["".join(out)], {"tables": ntab}

def extract_pdf(p):
    import pdfplumber
    pages = []
    with pdfplumber.open(p) as pdf:
        for pg in pdf.pages:
            t = pg.extract_text() or ""
            try: tabs = [serialize_table(tb) for tb in pg.extract_tables() if tb]
            except Exception: tabs = []
            pages.append(t + ("\n" + "\n".join(tabs) if tabs else ""))
    return pages, {"pages": len(pages)}

# ---------------------------------------------------------------- R-12 성명 마스킹
NAME3 = re.compile(r"(?<![가-힣])([가-힣] [가-힣] [가-힣])(?: ([가-힣] [가-힣] [가-힣]))?(?: ([가-힣] [가-힣] [가-힣]))?(?![가-힣])")
def mask_names(text, org):
    if "KDN" in org:  # 제·개정 이력 표의 작성·검토·승인자. 다른 기관은 사람 확인 전이라 적용하지 않음 (R-12)
        n = len(NAME3.findall(text)); return NAME3.sub("[성명]", text), n
    return text, 0

# ---------------------------------------------------------------- L1 규칙
PAGENO = re.compile(r"^\s*(?:인사규정\s*)?[-–]?\s*\d{1,4}(?:\s*-\s*\d{1,3})?\s*[-–]?\s*(?:인사규정)?\s*$")
def clean_pdf_pages(pages):
    """R-06: 반복 머리글·바닥글, 쪽번호 줄 제거. page_breaks 기록."""
    first, last = Counter(), Counter()
    split = [[l for l in pg.splitlines()] for pg in pages]
    for ls in split:
        s = [l.strip() for l in ls if l.strip()]
        if s: first[s[0]] += 1; last[s[-1]] += 1
    rep = {k for k, v in list(first.items()) + list(last.items()) if v >= max(2, len(pages) * 0.5)}
    out, removed, breaks = [], 0, []
    for ls in split:
        keep = []
        for i, l in enumerate(ls):
            s = l.strip()
            edge = i < 2 or i >= len(ls) - 2
            if edge and (s in rep or PAGENO.fullmatch(s)): removed += 1; continue
            keep.append(l)
        breaks.append(sum(len(x) for x in out))
        out.append("\n".join(keep))
    return "\n".join(out), {"removed_header_footer_lines": removed, "page_breaks": breaks}

TOC_LINE = re.compile(r"^\s*제\s*\d+\s*조(?:의\s*\d+)?\s*[^\n]{0,40}?(?:\d{1,4}\s*-\s*\d{1,3}|\d{1,3})?\s*$")
def split_toc(text):
    """R-07: 조 제목만 나열된 줄이 5개 이상 연속이면 목차."""
    lines = text.splitlines(); toc, keep, i = [], [], 0
    while i < len(lines):
        j = i
        while j < len(lines) and (TOC_LINE.match(lines[j]) and len(lines[j]) < 60 or not lines[j].strip() or re.match(r"^\s*제\s*\d+\s*장", lines[j])) and j - i < 400:
            j += 1
        block = [l for l in lines[i:j] if TOC_LINE.match(l)]
        if len(block) >= 5 and not any(re.search(r"[.。]\s*$|한다|말한다|있다", l) for l in block):
            toc += lines[i:j]; i = j; continue
        keep.append(lines[i]); i += 1
    return "\n".join(keep), "\n".join(toc)

CIRC = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮"
def normalize_numbering(text):
    """R-08"""
    text = re.sub(r"제\s*(\d+)\s*조\s*의\s*(\d+)", r"제\1조의\2", text)
    text = re.sub(r"제\s*(\d+)\s*(조|장|절|항|호)(?![가-힣])", r"제\1\2", text)
    text = re.sub(r"(?m)^\s*\((\d{1,2})\)\s", lambda m: CIRC[int(m.group(1)) - 1] + " " if 1 <= int(m.group(1)) <= 15 else m.group(0), text)
    text = re.sub(r"(?m)^(\s*)(\d{1,2})\)\s", r"\1\2. ", text)
    text = re.sub(r"(?m)^\s*부\s*칙", "부칙", text)
    return text

def normalize_chars(text):
    """R-09: NFKC (원문자 보호), PUA·장식 문자 제거, 공백 정리."""
    prot = {c: f"{i}" for i, c in enumerate(CIRC)}
    for c, k in prot.items(): text = text.replace(c, k)
    text = unicodedata.normalize("NFKC", text)
    for c, k in prot.items(): text = text.replace(k, c)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Co" and ch not in "​﻿")
    text = re.sub(r"(?m)^\s*[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+\s*-\s*\d*\s*-?\s*$", "", text)
    text = re.sub(r"(?m)^[\s\-–—=_·]{3,}$", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"

def mark_deleted(text):
    """R-05: 제N조 삭제 <2011.11.29> -> 제N조 (삭제, 2011.11.29)"""
    return re.sub(r"(제\d+조(?:의\d+)?)\s*[<〈\[]?\s*삭\s*제\s*(?:[<〈\[]\s*)?([\d.\s]*\d)?\s*[>〉\]]?", lambda m: f"{m.group(1)} (삭제{', ' + m.group(2).strip() if m.group(2) else ''})", text)

# ---------------------------------------------------------------- L2 규칙
MARK = re.compile(r"[<〈\[]\s*((?:본조|단서|제목|전문|일부)?\s*(?:개정|신설|삭제|개폐정리))\s*([^>〉\]]*)[>〉\]]")
def strip_marks(text):
    """R-04"""
    marks = []
    def rep(m):
        dates = re.findall(r"\d{4}\s*[.년]\s*\d{1,2}\s*[.월]?\s*\d{0,2}", m.group(2))
        marks.append({"type": re.sub(r"\s", "", m.group(1)), "dates": [re.sub(r"\s", "", d).rstrip(".") for d in dates], "raw": m.group(0)})
        return ""
    return MARK.sub(rep, text), marks

DATE = re.compile(r"(\d{4})\s*[.년]\s*(\d{1,2})\s*[.월]\s*(\d{1,2})?")
def split_appendix(text):
    """R-03: 첫 '부칙' 줄부터 끝까지 분리. 블록별 시행일·타규정 개정 태그."""
    lines = text.splitlines()
    cands = [i for i, l in enumerate(lines) if re.match(r"^\s*부칙", l)]
    if not cands: return text, [], ""
    # 본문 시작: 첫 '제1장' 또는 '제1조(' 줄. 그 앞에 개정 공고문·목차·대비표가 있으면 preamble 로 분리
    starts = [i for i, l in enumerate(lines) if re.match(r"^\s*제1장|^\s*제1조\s*[\(（]", l)]
    pre_end = starts[0] if starts and starts[0] < len(lines) * 0.3 else 0
    preamble = "\n".join(lines[:pre_end]) if pre_end and re.search(r"부칙|대비표|목\s*차", "\n".join(lines[:pre_end])) else ""
    if not preamble: pre_end = 0
    # 부칙 시작: 마지막 장 제목 이후의 첫 부칙. 장 제목이 없으면 문서 40% 이후, 3줄 내 '시행' 이 있는 첫 부칙
    chaps = [i for i, l in enumerate(lines) if re.match(r"^\s*제\d+장", l) and i >= pre_end]
    after = [c for c in cands if c > (chaps[-1] if chaps else len(lines) * 0.4) and c >= pre_end]
    if not chaps: after = [c for c in after if re.search(r"시행", "\n".join(lines[c:c + 4]))] or after
    if not after: return "\n".join(lines[pre_end:]), [], preamble
    body, app = "\n".join(lines[pre_end:after[0]]), lines[after[0]:]
    blocks, cur = [], []
    for l in app:
        if re.match(r"^\s*부칙", l) and cur: blocks.append(cur); cur = []
        cur.append(l)
    if cur: blocks.append(cur)
    out = []
    for b in blocks:
        t = "\n".join(b)
        m = re.search(r"(\d{4}\s*[.년]\s*\d{1,2}\s*[.월]\s*\d{0,2})\s*일?\s*부터\s*시행", t)
        eff = re.sub(r"\s", "", m.group(1)).rstrip(".") if m else None
        other = bool(re.search(r"(?<!이 )(?:보수|복무|직제|취업|여비|연봉|임금)[가-힣 ]*(?:규정|규칙)[^\n]{0,30}(?:개정|신설)한다", t))
        out.append({"effective_date": eff, "other_rule_amendment": other, "text": t})
    return body, out, preamble

DELEG = re.compile(r"「?([가-힣A-Za-z0-9 ·]{2,25}?(?:규정|규칙|세칙|지침))」?\s*(?:으로|에서|이|에)\s*(?:따로\s*)?정(?:하는 바에 의|한다|하는)")
def delegated(text):
    """R-13"""
    return sorted({re.sub(r"\s+", " ", m).strip() for m in DELEG.findall(text) if not m.strip().startswith("이 ")})

DEF = re.compile(r"[“\"']([가-힣A-Za-z· ]{1,20})[”\"']\s*(?:이|라)?\s*(?:란|함은|라 함은|이란)\s*(.+?말한다\.?)")
def glossary(text):
    """R-14: 정의 조항에서 용어 추출"""
    return [{"term": t.strip(), "definition": d.strip()} for t, d in DEF.findall(text)]

# ---------------------------------------------------------------- 조 단위 분리 (R-14 제목·section_path)
ART = re.compile(r"(?m)^(?=제\s*\d+\s*조(?:\s*의\s*\d+)?\s*(?:[\(（]|\(삭제|삭제|<|〈))")
def articles(text, strict):
    units, section = [], []
    chunks = ART.split(text)
    pre = chunks[0]
    for ch in chunks[1:]:
        head = re.match(r"제\s*(\d+)\s*조(?:\s*의\s*(\d+))?\s*(?:[\(（]([^)）]*)[\)）])?", ch)
        no = f"제{head.group(1)}조" + (f"의{head.group(2)}" if head.group(2) else "")
        title = (head.group(3) or "").strip() if head else ""
        units.append({"article_no": no, "article_title": title, "text": ch.strip()})
    # section_path: 본문 내 장·절 제목을 순서대로 부여
    path, out = [], []
    pos_sections = [(m.start(), m.group(0).strip()) for m in re.finditer(r"(?m)^제\s*\d+\s*(장|절)[^\n]*", text)]
    cursor = 0
    for u in units:
        i = text.find(u["text"][:40], cursor); cursor = max(cursor, i)
        cur = [s for p, s in pos_sections if p <= i]
        chap = [s for s in cur if "장" in s[:6]]; sec = [s for s in cur if "절" in s[:6] and (not chap or cur.index(s) > cur.index(chap[-1]))]
        u["section_path"] = " > ".join(x for x in ([chap[-1]] if chap else []) + ([sec[-1]] if sec else []))
        out.append(u)
    return out, pre

# ---------------------------------------------------------------- 메인
def date_key(fn):
    m = re.search(r"(\d{2,4})\s*년\s*도?\s*(\d{1,2})\s*월\s*(\d{0,2})", fn)
    if not m: return (0, 0, 0)
    y = int(m.group(1)); y = y + 2000 if y < 100 else y
    return (y, int(m.group(2)), int(m.group(3) or 0))

TRANSCRIPTIONS = [json.loads(l) for l in (ROOT / "04_정제규칙" / "manual_transcriptions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

manifests = []
for mf in sorted(RAW.glob("*/manifest.jsonl")):
    for line in mf.read_text(encoding="utf-8").splitlines():
        m = json.loads(line); m["set"] = mf.parent.name; m["path"] = mf.parent / m["file"]; manifests.append(m)
latest = {}
for m in manifests:
    k = m["org"]; d = date_key(m["file"])
    if k not in latest or d > latest[k][0]: latest[k] = (d, m["file"])

report, gloss_all, synonyms_seed = [], [], [
    {"everyday": ["아기 낳으면", "출산", "애 낳고"], "terms": ["출산전후휴가", "육아휴직", "출산휴가"]},
    {"everyday": ["잠깐 쉬고 싶다", "쉬다", "휴가 길게"], "terms": ["휴직", "청원휴직", "직권휴직"]},
    {"everyday": ["부서 옮기다", "다른 팀으로", "자리 이동"], "terms": ["전보", "전직", "보직변경"]},
    {"everyday": ["윗 직급", "올라가다", "진급"], "terms": ["승진", "승격", "승진소요 최저연수"]},
    {"everyday": ["벌 받다", "잘못하면", "처벌"], "terms": ["징계", "파면", "해임", "정직", "감봉", "견책"]},
    {"everyday": ["잘리다", "해고", "나가야 하는 경우"], "terms": ["면직", "직권면직", "당연퇴직", "결격사유"]},
    {"everyday": ["처음 몇 달", "정식 직원 아님", "인턴 기간"], "terms": ["수습", "시용", "수습기간"]},
    {"everyday": ["병 걸려서", "아파서 오래", "병가"], "terms": ["질병휴직", "직권휴직", "휴직기간"]},
    {"everyday": ["자리를 뺏기다", "일을 못 하게 하다", "대기 발령"], "terms": ["직위해제", "대기"]},
    {"everyday": ["대학원", "공부하면서", "유학"], "terms": ["휴직", "연수휴직", "겸직"]},
    {"everyday": ["친척", "가족이 같은 회사", "형이 상사"], "terms": ["친족", "보직 제한", "결격사유"]},
    {"everyday": ["다른 회사 일", "투잡", "부업"], "terms": ["겸직", "영리업무", "겸직 금지"]},
]

import os
_SHARD = os.environ.get("CLEAN_SHARD")  # "i/n": 파일을 n개로 나눠 i번째만 처리 (병렬 실행용, 2026-10-04)
_INCR = os.environ.get("CLEAN_INCR")    # "1": L2 산출물이 이미 있으면 건너뜀 (증분 정제)
for _idx, m in enumerate(manifests):
    p = m["path"]; ext = p.suffix.lower(); org = m["org"]
    rel = Path(m["set"]) / m["file"]
    stem = rel.with_suffix("")
    if _SHARD and _idx % int(_SHARD.split("/")[1]) != int(_SHARD.split("/")[0]): continue
    if _INCR and (OUT / "L2" / stem.parent / (stem.name + ".articles.jsonl")).exists(): continue
    base_meta = {k: m[k] for k in ("org", "org_type", "rule_title", "rule_latest_date", "format", "sha256", "download_url", "source_page", "collected")}
    base_meta["source_file"] = str(rel).replace("\\", "/")
    dk = date_key(m["file"]); base_meta["revision_date"] = f"{dk[0]:04d}-{dk[1]:02d}" + (f"-{dk[2]:02d}" if dk[2] else "") if dk[0] else None
    base_meta["is_latest"] = latest[org][1] == m["file"]  # R-10 메타데이터
    stats = {"file": str(rel)}

    # ---- L0
    try:
        if ext == ".hwp": pages, x = extract_hwp(p)
        elif ext == ".hwpx": pages, x = extract_hwpx(p)
        else: pages, x = extract_pdf(p)
    except Exception as e:  # 손상 파일 등은 건너뛰고 계속 (2026-10-04 대량 수집 대응)
        print(f"SKIP {rel}: {type(e).__name__} {e}", flush=True); continue
    if not pages or sum(len(pg) for pg in pages) < 500:
        print(f"SKIP {rel}: 추출 텍스트 부족", flush=True); continue
    stats.update(x)
    raw_text = "\n".join(pages)
    l0, nmask = mask_names(raw_text, org); stats["masked_names"] = nmask
    l0_units, _ = articles(l0, strict=False)
    def dump(level, text, units, meta, extra=None):
        d = OUT / level / stem.parent; d.mkdir(parents=True, exist_ok=True)
        (d / (stem.name + ".txt")).write_text(text, encoding="utf-8")
        with open(d / (stem.name + ".articles.jsonl"), "w", encoding="utf-8") as f:
            for i, u in enumerate(units):
                f.write(json.dumps({"id": f"{org}|{meta['revision_date']}|{u['article_no']}", **u, **{k: meta[k] for k in ("org", "org_type", "revision_date", "is_latest", "source_file")}}, ensure_ascii=False) + "\n")
        (d / (stem.name + ".meta.json")).write_text(json.dumps({**meta, **(extra or {})}, ensure_ascii=False, indent=2), encoding="utf-8")
    dump("L0", l0, l0_units, {**base_meta, "level": "L0"})
    stats["L0_chars"] = len(l0); stats["L0_articles"] = len(l0_units)

    # ---- L1
    if ext == ".pdf":
        masked_pages = [mask_names(pg, org)[0] for pg in pages]
        t, x = clean_pdf_pages(masked_pages); stats.update(x); pb = x["page_breaks"]
    else:
        t, pb = l0, None
    t, toc = split_toc(t); stats["toc_lines"] = len(toc.splitlines()) if toc else 0
    t = mark_deleted(t)
    # R-02: 실제 이미지가 있고 사람 전사본이 있으면 치환, 이미지가 없는 도형 자리표시자는 제거, 그 외는 자리표시자 유지
    trans = [tr for tr in TRANSCRIPTIONS if tr["org"] == org]
    used = []
    if trans and stats.get("bindata_images", 0) > 0:
        for tr in sorted(trans, key=lambda x: x["placeholder_index"]):
            if "<그림>" in t:
                t = t.replace("<그림>", f"{tr['text']}\n(출처: 이미지 전사, {tr['source_image'].split('/')[-1]})", 1); used.append(tr)
    elif stats.get("bindata_images", 0) == 0:
        t = t.replace("<그림>", "")
    t = t.replace("<그림>", "[그림: 이미지, 내용 미추출]")
    stats["image_transcriptions"] = len(used)
    t = normalize_numbering(t)
    t = normalize_chars(t)
    l1_units, _ = articles(t, strict=True)
    dump("L1", t, l1_units, {**base_meta, "level": "L1"}, {"toc": toc, "page_breaks": pb, "image_transcriptions": used})
    stats["L1_chars"] = len(t); stats["L1_articles"] = len(l1_units)

    # ---- L2
    body, appendix, preamble = split_appendix(t)
    # 조 단위로 먼저 나눈 뒤 조별로 개정 표기를 분리해야 revision_marks 가 조문에 붙는다 (R-04)
    l2_units, _ = articles(body, strict=True)
    for u in l2_units:
        u["text"], um = strip_marks(u["text"]); u["revision_marks"] = um
    body, marks = strip_marks(body)
    body = normalize_chars(body)
    gl = glossary(body); gloss_all += [{"org": org, "revision_date": base_meta["revision_date"], **g} for g in gl]
    extra = {"preamble": preamble, "appendix_blocks": appendix, "effective_dates": [a["effective_date"] for a in appendix if a["effective_date"]],
             "other_rule_amendment_blocks": sum(a["other_rule_amendment"] for a in appendix),
             "revision_marks_total": len(marks) + sum(len(u["revision_marks"]) for u in l2_units),
             "delegated_to": delegated(t), "glossary": gl}
    dump("L2", body, l2_units, {**base_meta, "level": "L2"}, extra)
    stats.update({"L2_chars": len(body), "L2_articles": len(l2_units), "appendix_blocks": len(appendix),
                  "effective_dates": len(extra["effective_dates"]), "other_rule_blocks": extra["other_rule_amendment_blocks"],
                  "revision_marks": extra["revision_marks_total"], "delegated_to": len(extra["delegated_to"]), "glossary_terms": len(gl)})
    report.append(stats)
    print(f"{ext[1:]:4} {org[:10]:10} L0 {stats['L0_chars']:>6}/{stats['L0_articles']:>3}조  L1 {stats['L1_chars']:>6}/{stats['L1_articles']:>3}조  L2 {stats['L2_chars']:>6}/{stats['L2_articles']:>3}조  부칙{stats['appendix_blocks']:>3} 시행일{stats['effective_dates']:>3} 개정표기{stats['revision_marks']:>4} 위임{stats['delegated_to']:>2} 용어{stats['glossary_terms']:>3} 마스킹{nmask}")

# ---- R-10 정책 A/B 집계 인덱스, R-14 사전
L2 = OUT / "L2"
with open(L2 / "index_A_latest_only.jsonl", "w", encoding="utf-8") as fa, open(L2 / "index_B_all_revisions.jsonl", "w", encoding="utf-8") as fb:
    for f in sorted(L2.rglob("*.articles.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            fb.write(line + "\n")
            if json.loads(line)["is_latest"]: fa.write(line + "\n")
with open(L2 / "glossary.jsonl", "w", encoding="utf-8") as f:
    for g in gloss_all: f.write(json.dumps(g, ensure_ascii=False) + "\n")
with open(L2 / "synonyms_seed.jsonl", "w", encoding="utf-8") as f:
    for s in synonyms_seed: f.write(json.dumps({**s, "source": "03_질의정의 Q11 예시, 사람 작성"}, ensure_ascii=False) + "\n")
json.dump(report, open(OUT / "clean_report.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

tot = lambda k: sum(r.get(k, 0) or 0 for r in report)
nA = sum(1 for l in open(L2 / "index_A_latest_only.jsonl", encoding="utf-8")); nB = sum(1 for l in open(L2 / "index_B_all_revisions.jsonl", encoding="utf-8"))
print(f"\n합계: L0 {tot('L0_chars')}자/{tot('L0_articles')}조, L1 {tot('L1_chars')}자/{tot('L1_articles')}조, L2 본문 {tot('L2_chars')}자/{tot('L2_articles')}조")
print(f"부칙 블록 {tot('appendix_blocks')}, 시행일 추출 {tot('effective_dates')}, 타규정 개정 블록 {tot('other_rule_blocks')}, 개정표기 분리 {tot('revision_marks')}, 용어 {len(gloss_all)}, 머리글·쪽번호 제거 {tot('removed_header_footer_lines')}줄, 목차 {tot('toc_lines')}줄, 성명 마스킹 {tot('masked_names')}")
print(f"R-10 인덱스: A 최신판만 {nA}조, B 전체 {nB}조")
