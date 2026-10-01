# -*- coding: utf-8 -*-
"""01_raw 문서 자체 분석. 형식별 텍스트 추출 → 구조/노이즈/중복 지표 → 02_분석/ 산출."""
import json, re, csv, zipfile, subprocess, difflib, sys, io, html
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "01_raw"
OUT = ROOT / "02_분석"
TXT = OUT / "text"
TXT.mkdir(parents=True, exist_ok=True)

def extract_hwp(p: Path):
    r = subprocess.run(["hwp5txt", str(p)], capture_output=True)
    text = r.stdout.decode("utf-8", "ignore")
    meta = {}
    try:
        from hwp5.xmlmodel import Hwp5File
        h = Hwp5File(str(p)); c = Counter()
        n = len(list(h.bodytext.sections))
        for i in range(n):
            for rec in h.bodytext.section(i).records():
                c[rec["tagname"]] += 1
        meta = {"sections": n, "tables": c["HWPTAG_TABLE"], "paragraphs": c["HWPTAG_PARA_HEADER"],
                "txt_table_placeholders": text.count("<표>"), "txt_picture_placeholders": text.count("<그림>"),
                "footnote_shapes": c["HWPTAG_FOOTNOTE_SHAPE"], "hwp_version": ".".join(map(str, h.header.version))}
    except Exception as e:
        meta = {"meta_error": str(e)[:80]}
    try:
        r2 = subprocess.run(["hwp5proc", "xml", str(p)], capture_output=True)
        x = r2.stdout.decode("utf-8", "ignore")
        cells = re.findall(r"<TableCell.*?</TableCell>", x, re.S)
        meta["table_cells"] = len(cells)
        meta["table_cells_with_text"] = sum(1 for c in cells if "".join(re.findall(r"<Text[^>]*>(.*?)</Text>", c, re.S)).strip())
        meta["pictures"] = x.count("<ShapePicture") + x.count("<Picture")
    except Exception as e:
        meta["xml_error"] = str(e)[:80]
    return text, meta

def extract_hwpx(p: Path):
    z = zipfile.ZipFile(p)
    secs = sorted(n for n in z.namelist() if re.match(r"Contents/section\d+\.xml", n))
    text, tables, headers = [], 0, 0
    for n in secs:
        x = z.read(n).decode("utf-8", "ignore")
        tables += x.count("<hp:tbl")
        headers += x.count("<hp:header") + x.count("<hp:footer")
        paras = re.findall(r"<hp:p\b.*?</hp:p>", x, re.S)
        for para in paras:
            t = "".join(re.findall(r"<hp:t[^>]*>(.*?)</hp:t>", para, re.S))
            text.append(html.unescape(re.sub(r"<[^>]+>", "", t)))
    return "\n".join(text), {"sections": len(secs), "tables": tables, "header_footer_elems": headers}

def extract_pdf(p: Path):
    import pdfplumber
    text, tables, empty = [], 0, 0
    first, last, pageno = Counter(), Counter(), 0
    with pdfplumber.open(p) as pdf:
        pages = len(pdf.pages)
        for pg in pdf.pages:
            t = pg.extract_text() or ""
            if len(t.strip()) < 20: empty += 1
            ls = [l.strip() for l in t.splitlines() if l.strip()]
            if ls:
                first[ls[0]] += 1; last[ls[-1]] += 1
                pn = r"[-–\s]*\d{1,4}(\s*-\s*\d{1,3})?[-–\s]*"
                if re.fullmatch(pn + r"(인사규정)?|인사규정\s*" + pn, ls[-1]) or re.fullmatch(pn, ls[0]): pageno += 1
            text.append(t)
            try: tables += len(pg.find_tables())
            except Exception: pass
    rep_head = first.most_common(1)[0] if first else ("", 0)
    rep_foot = last.most_common(1)[0] if last else ("", 0)
    return "\n".join(text), {"pages": pages, "tables": tables, "text_poor_pages": empty,
                             "repeated_header": f"{rep_head[0][:20]} x{rep_head[1]}" if rep_head[1] > 1 else "",
                             "repeated_footer": f"{rep_foot[0][:20]} x{rep_foot[1]}" if rep_foot[1] > 1 else "",
                             "pageno_lines": pageno}

PAT = {
    "장": r"^\s*제\s*\d+\s*장",
    "절": r"^\s*제\s*\d+\s*절",
    "조": r"제\s*\d+\s*조(?:의\s*\d+)?\s*(?:\(|（)",
    "항_원문자": r"[①-⑮]",
    "호_숫자점": r"^\s*\d{1,2}\.\s",
    "목_가나다": r"^\s*[가-힣]\.\s",
    "별표": r"\[?별표\s*\d*\]?|별지\s*제?\s*\d*\s*호?\s*서식|서식\s*\d+",
    "부칙": r"^\s*부\s*칙",
    "개정표기": r"[<〈\(（]\s*(?:개정|신설|삭제)\s*[\d.\s~년월일,]+[>〉\)）]",
    "삭제조항": r"제\s*\d+\s*조\s*(?:\([^)]*\))?\s*삭제",
    "페이지번호": r"^\s*-?\s*\d{1,3}\s*-?\s*$",
    "날짜표기": r"\d{4}\s*[.년]\s*\d{1,2}\s*[.월]\s*\d{1,2}",
}

def structure(text: str):
    lines = text.splitlines()
    res = {}
    for k, pat in PAT.items():
        if pat.startswith("^"):
            res[k] = sum(1 for l in lines if re.search(pat, l))
        else:
            res[k] = len(re.findall(pat, text, re.M))
    res["성명패턴_후보"] = len(re.findall(r"(?<![가-힣])[가-힣]\s[가-힣]\s[가-힣](?![가-힣])", text))
    res["목차_후보줄"] = sum(1 for l in lines if re.search(r"제\s*\d+\s*조[^\n]{0,40}\d{1,4}\s*-\s*\d{1,3}\s*$", l))
    res["chars"] = len(text); res["lines"] = len(lines)
    res["blank_line_ratio"] = round(sum(1 for l in lines if not l.strip()) / max(1, len(lines)), 3)
    heads = re.findall(r"제\s*(\d+)\s*조", text)
    res["조_max_no"] = max(map(int, heads)) if heads else 0
    res["조_unique"] = len(set(heads))
    return res

def article_units(text: str):
    parts = re.split(r"(?=제\s*\d+\s*조(?:의\s*\d+)?\s*[\(（])", text)
    return [p.strip() for p in parts if re.match(r"제\s*\d+\s*조", p.strip())]

rows, texts = [], {}
for mf in sorted(RAW.glob("*/manifest.jsonl")):
    setname = mf.parent.name
    for line in mf.read_text(encoding="utf-8").splitlines():
        m = json.loads(line); p = mf.parent / m["file"]
        ext = p.suffix.lower()
        try:
            if ext == ".hwp": text, meta = extract_hwp(p)
            elif ext == ".hwpx": text, meta = extract_hwpx(p)
            else: text, meta = extract_pdf(p)
            err = ""
        except Exception as e:
            text, meta, err = "", {}, str(e)[:120]
        key = f"{setname}/{m['file']}"
        texts[key] = text
        (TXT / (key.replace("/", "__") + ".txt")).write_text(text, encoding="utf-8")
        row = {"set": setname, "org": m["org"], "org_type": m["org_type"], "file": m["file"], "format": ext.lstrip("."),
               "bytes": m["bytes"], "rule_latest_date": m["rule_latest_date"], "extract_error": err}
        row.update({f"meta_{k}": v for k, v in meta.items()})
        row.update(structure(text))
        rows.append(row)

# 개정판 간 중복/변경률 (기관별, 파일명 내 날짜순)
def date_key(fn):
    m = re.search(r"(\d{2,4})\s*년\s*도?\s*(\d{1,2})\s*월\s*(\d{0,2})", fn)
    if not m: return (0, 0, 0)
    y = int(m.group(1)); y = y + 2000 if y < 100 else y
    return (y, int(m.group(2)), int(m.group(3) or 0))

dup = []
byorg = defaultdict(list)
for k in texts: byorg[k.split("/")[1]].append(k)
for org, ks in byorg.items():
    ks = sorted(ks, key=lambda k: date_key(k))
    for a, b in zip(ks, ks[1:]):
        ua, ub = article_units(texts[a]), article_units(texts[b])
        sa, sb = set(ua), set(ub)
        sm = difflib.SequenceMatcher(None, texts[a], texts[b], autojunk=False)
        dup.append({"org": org, "older": a.split("/")[-1], "newer": b.split("/")[-1], "articles_older": len(ua),
                    "articles_newer": len(ub), "identical_articles": len(sa & sb),
                    "identical_article_ratio": round(len(sa & sb) / max(1, len(sb)), 3),
                    "char_similarity": round(sm.quick_ratio(), 3)})

# 조 길이 분포 (청킹은 후순위지만 정제 기준 참고용)
lens = [len(u) for t in texts.values() for u in article_units(t)]
lens.sort()
def pct(q): return lens[int(q * (len(lens) - 1))] if lens else 0
article_len = {"count": len(lens), "p10": pct(.1), "p50": pct(.5), "p90": pct(.9), "p99": pct(.99), "max": lens[-1] if lens else 0,
               "over_2000_chars": sum(1 for l in lens if l > 2000)}

# 번호 체계 변형 (기관별 양식 편차)
numbering = {}
for k, t in texts.items():
    org = k.split("/")[1]
    v = set()
    if re.search(r"[①-⑮]", t): v.add("항:원문자①")
    if re.search(r"^\s*\(\d{1,2}\)\s", t, re.M): v.add("항:(1)")
    if re.search(r"^\s*\d{1,2}\.\s", t, re.M): v.add("호:1.")
    if re.search(r"^\s*[가-힣]\.\s", t, re.M): v.add("목:가.")
    if re.search(r"^\s*\d{1,2}\)\s", t, re.M): v.add("호:1)")
    if re.search(r"제\s*\d+\s*조의\s*\d+", t): v.add("조의N(삽입조)")
    numbering.setdefault(org, set()).update(v)
numbering = {k: sorted(v) for k, v in numbering.items()}

with open(OUT / "inventory.csv", "w", newline="", encoding="utf-8-sig") as f:
    cols = sorted({c for r in rows for c in r}, key=lambda c: (c not in ("set", "org", "org_type", "file", "format", "bytes"), c))
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
with open(OUT / "revision_diff.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(dup[0].keys())); w.writeheader(); w.writerows(dup)
json.dump({"article_length_chars": article_len, "numbering_variants_by_org": numbering},
          open(OUT / "structure_summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

# 콘솔 요약
sys.stdout.reconfigure(encoding="utf-8")
fmt = Counter(r["format"] for r in rows)
print("files", len(rows), dict(fmt), "errors", sum(1 for r in rows if r["extract_error"]))
for r in rows:
    print(f"{r['format']:4} {r['org'][:10]:10} {r['chars']:>7} chars 조{r['조_unique']:>4} 항{r['항_원문자']:>4} 별표{r['별표']:>3} 부칙{r['부칙']:>2} 개정표기{r['개정표기']:>4} 삭제{r['삭제조항']:>2} 표{r.get('meta_tables','?')!s:>3} {r['file'][-28:]}")
print("article_len", article_len)
for d in dup: print(f"{d['org'][:10]:10} 동일조문비율 {d['identical_article_ratio']}  문자유사도 {d['char_similarity']}  {d['older'][-22:]} -> {d['newer'][-22:]}")
