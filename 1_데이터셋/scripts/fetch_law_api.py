# -*- coding: utf-8 -*-
"""국가법령정보센터(law.go.kr) Open API 로 교원·교직원 규정을 받아 조 단위 코퍼스로 만든다 (2026-10-13).

목록: scripts/corpus_teachers.py (법령 law / 행정규칙 admrul / 자치법규 ordin). manual 항목은 건너뛰고 수동 수집 목록에 적는다.
입력: 포폴_정리/.env 의 LAW_API_OC (law.go.kr Open API 아이디). 커밋하지 않는다.
산출:
  01_raw/교원_법령/<target>/<문서명>.json      API 원문(JSON) 그대로
  01_raw/교원_법령/manifest.jsonl              문서별 수집 기록
  01_raw/교원_법령/수동수집_목록.md            API 에 없거나 실패한 문서 (누리집에서 HWP 로 받을 것)
  05_clean/L2_교원/corpus.jsonl                조 단위 청크. 기존 L2P 행 형식(chunk_id, org, revision_date, article_no, article_title, text)에
                                               layer / issuer / audience / applies_to / doc_name / source_url 을 더한다.
                                               org 에는 문서명을 넣어 기존 기관 사전필터가 그대로 '문서 필터'로 동작하게 한다.
사용: python fetch_law_api.py              전체
      python fetch_law_api.py 교육공무원    문서명 부분 일치만 (해당 문서 행만 교체)
"""
import json, re, sys, time, datetime, pathlib
import xml.etree.ElementTree as ET
import requests
from corpus_teachers import as_dicts

sys.stdout.reconfigure(encoding="utf-8")
ROOT = pathlib.Path(__file__).resolve().parents[1]          # 1_데이터셋
REPO = ROOT.parent                                           # 포폴_정리
RAW = ROOT / "01_raw" / "교원_법령"
OUT = ROOT / "05_clean" / "L2_교원"; OUT.mkdir(parents=True, exist_ok=True)
TODAY = str(datetime.date.today())
API = "https://www.law.go.kr/DRF"
ORDIN_OWNER = re.compile("전남광주통합특별시교육청|전라남도교육청|전라남도교육감")   # 자치법규는 이 기관 소관만 (2026-07 통합 전·후)

def load_oc():
    p = REPO / ".env"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("LAW_API_OC="): return line.split("=", 1)[1].strip().strip('"\'')
    sys.exit("LAW_API_OC 가 없습니다. 포폴_정리/.env 에 LAW_API_OC=<law.go.kr OC> 를 적어 주세요.")
OC = load_oc()

S = requests.Session(); S.headers["User-Agent"] = "Mozilla/5.0"

def xml_to_dict(el):
    """XML 을 JSON 응답과 같은 모양의 dict 로. 같은 태그가 여러 개면 list, 자식이 없으면 문자열. 속성은 '@이름'."""
    kids = list(el)
    if not kids:
        return (el.text or "").strip()
    out = {}
    for a, v in el.attrib.items(): out["@" + a] = v
    for k in kids:
        v = xml_to_dict(k)
        if k.tag in out:
            if not isinstance(out[k.tag], list): out[k.tag] = [out[k.tag]]
            out[k.tag].append(v)
        else: out[k.tag] = v
    return out

def get_json(path, params):
    """이 OC 는 XML 만 열려 있어(JSON 은 HTML 안내 페이지) XML 로 받아 dict 로 바꾼다. 응답 모양은 JSON 과 같다."""
    params = {**params, "type": "XML"}
    err = None
    for i in range(3):
        try:
            r = S.get(f"{API}/{path}", params=params, timeout=60); r.raise_for_status()
            if not r.content.lstrip().startswith(b"<?xml") and b"<" not in r.content[:50]:
                raise ValueError("XML 이 아닌 응답")
            root = ET.fromstring(re.sub(rb"&(?!amp;|lt;|gt;|quot;|apos;|#)\w+;", b" ", r.content))
            return {root.tag: xml_to_dict(root)}
        except Exception as e:
            err = e; time.sleep(2 * (i + 1))
    raise err

def norm(s): return re.sub(r"[\s·‧ㆍ]", "", str(s or ""))
def as_list(v): return [] if v is None else (v if isinstance(v, list) else [v])
def text_of(v):
    if v is None: return ""
    if isinstance(v, list): return "\n".join(text_of(x) for x in v)
    if isinstance(v, dict): return "\n".join(text_of(x) for x in v.values())
    return str(v).strip()
def to_date(s):
    s = str(s or "").strip()
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}" if len(s) == 8 and s.isdigit() else (s or None)

# ---------------------------------------------------------------- 검색 (target 별로 응답 키가 다르다)
SEARCH_KEYS = {"law": ("law", "법령명한글", "법령일련번호", "법령ID"),
               "admrul": ("admrul", "행정규칙명", "행정규칙일련번호", "행정규칙일련번호"),
               "ordin": ("law", "자치법규명", "자치법규일련번호", "자치법규ID")}

def search(name, target):
    """이름으로 1건 찾기. 정확히 같은 이름 우선, 자치법규는 전남교육청 소관만, 없으면 이름 포함 첫 결과."""
    _, name_key, mst_key, id_key = SEARCH_KEYS[target]
    d = get_json("lawSearch.do", {"OC": OC, "target": target, "query": name, "display": 100})
    root = next(iter(d.values())) if d else {}
    items = [x for v in (root.values() if isinstance(root, dict) else []) for x in as_list(v) if isinstance(x, dict) and name_key in x]
    if target == "ordin":
        items = [x for x in items if ORDIN_OWNER.search(norm(x.get("지자체기관명", "")) + norm(x.get("자치법규명", "")))]
    if target == "law":
        items = [x for x in items if "현행" in str(x.get("현행연혁코드", "현행"))]
    exact = [x for x in items if norm(x.get(name_key)) == norm(name)]
    contains = [x for x in items if norm(name) in norm(x.get(name_key))]
    hit = (exact or contains or [None])[0]
    if not hit: return None
    return {"name": text_of(hit.get(name_key)), "mst": hit.get(mst_key), "id": hit.get(id_key),
            "effective": to_date(hit.get("시행일자")), "promulgated": to_date(hit.get("공포일자") or hit.get("발령일자")),
            "owner": text_of(hit.get("소관부처명") or hit.get("지자체기관명") or hit.get("소관부처") or "")}

# ---------------------------------------------------------------- 본문 → 조문
def find_articles(node):
    """응답 어디에 있든 조문 목록을 찾는다. 법령·행정규칙은 '조문단위', 자치법규는 '조문' 아래 '조'."""
    if isinstance(node, dict):
        if "조문단위" in node: return as_list(node["조문단위"])
        if "조" in node and isinstance(node.get("조"), (list, dict)): return as_list(node["조"])
        for v in node.values():
            r = find_articles(v)
            if r: return r
    elif isinstance(node, list):
        for v in node:
            r = find_articles(v)
            if r: return r
    return []

def flatten_article(a):
    """조문단위 1건 -> 본문. 조문내용 + 항(항내용 + 호 + 목). 원문 글자는 바꾸지 않는다."""
    parts = [text_of(a.get("조문내용") or a.get("조내용"))]
    for h in as_list(a.get("항")):
        parts.append(text_of(h.get("항내용")))
        for ho in as_list(h.get("호")):
            parts.append(text_of(ho.get("호내용")))
            for mok in as_list(ho.get("목")): parts.append(text_of(mok.get("목내용")))
    return "\n".join(p for p in parts if p)

def article_no(a):
    n = str(a.get("조문번호", "")).strip(); g = str(a.get("조문가지번호", "") or "").strip()
    if n.isdigit() and len(n) == 6 and not g:   # 자치법규: 000100 = 제1조, 000102 = 제1조의2
        n, g = str(int(n[:4])), str(int(n[4:]))
    return f"제{n}조" + (f"의{g}" if g and g != "0" else "")

ART_RE = re.compile(r"(?m)^(제\d+조(?:의\d+)?)\s*(?:\(([^)]*)\))?")
def split_plain(text):
    """조문단위가 없는 행정규칙: 본문 문자열을 '제N조' 머리로 나눈다."""
    out, ms = [], list(ART_RE.finditer(text))
    for i, m in enumerate(ms):
        body = text[m.start(): ms[i + 1].start() if i + 1 < len(ms) else len(text)].strip()
        out.append((m.group(1), (m.group(2) or "").strip(), body))
    return out

def fetch_one(doc):
    name, target = doc["name"], doc["target"]
    hit = search(name, target)
    if not hit: return None, "검색 결과 없음"
    params = {"OC": OC, "target": target}
    if target == "admrul": params["ID"] = hit["id"]
    else: params["MST"] = hit["mst"]
    full = get_json("lawService.do", params)
    root = next(iter(full.values())) if isinstance(full, dict) and full else {}
    info = root.get("기본정보") or root.get("행정규칙기본정보") or {}
    real = text_of(info.get("법령명_한글") or info.get("법령명한글") or info.get("행정규칙명") or info.get("자치법규명") or hit["name"])
    eff = to_date(info.get("시행일자")) or hit["effective"]
    url = {"law": f"https://www.law.go.kr/법령/{real}", "admrul": f"https://www.law.go.kr/행정규칙/{real}", "ordin": f"https://www.law.go.kr/자치법규/{real}"}[target]

    d = RAW / target; d.mkdir(parents=True, exist_ok=True)
    (d / f"{real.replace('/', '_')}.json").write_text(json.dumps(full, ensure_ascii=False, indent=1), encoding="utf-8")

    arts = find_articles(root)
    rows = []
    if arts:
        for a in arts:
            if str(a.get("조문여부", "조문")) not in ("조문", "Y"): continue   # 편·장·절 제목 행 (법령 '전문', 자치법규 'N')
            body = flatten_article(a); no = article_no(a); title = text_of(a.get("조문제목") or a.get("조제목"))
            if not body or re.fullmatch(r"\s*" + re.escape(no) + r"\s*(\([^)]*\))?\s*삭제.*", body, re.S): continue
            rows.append((no, title, body))
    else:
        plain = text_of(root.get("조문내용") or root.get("본문") or "")
        rows = split_plain(plain) or ([("전문", "", plain)] if plain else [])
    chunks = [{
        "chunk_id": f"{real}|{eff}|{no}", "article_id": f"{real}|{eff}|{no}",
        "org": real, "revision_date": eff, "is_latest": True,
        "article_no": no, "article_title": title, "text": body,
        "layer": doc["layer"], "issuer": doc["issuer"], "audience": doc["audience"], "applies_to": doc["applies_to"],
        "doc_name": real, "doc_type": target, "source_url": url,
    } for no, title, body in rows]
    with open(RAW / "manifest.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"doc_name": real, "requested": name, "target": target, "id": hit["id"], "mst": hit["mst"], "effective_date": eff,
                            "promulgated": hit["promulgated"], "owner": hit["owner"], "articles": len(chunks), "collected": TODAY, "source_url": url,
                            "license_note": "국가법령정보센터 Open API. 법령·행정규칙·자치법규는 저작권법 제7조에 따라 보호받지 못하는 저작물"}, ensure_ascii=False) + "\n")
    return chunks, f"{real}: 조문 {len(chunks)}개 (시행 {eff})"

if __name__ == "__main__":
    pick = sys.argv[1] if len(sys.argv) > 1 else ""
    docs = [d for d in as_dicts() if pick in d["name"]]
    RAW.mkdir(parents=True, exist_ok=True)
    all_rows, manual, failed = [], [], []
    print(f"{len(docs)}건 시작", flush=True)
    manual = [d for d in as_dicts() if d["target"] == "manual"]   # 수동 목록은 일부 실행 때도 전체를 적는다
    for doc in docs:
        if doc["target"] == "manual": continue
        try:
            chunks, msg = fetch_one(doc)
            if chunks is None: failed.append((doc, msg)); print(f"  [없음] {doc['name']}", flush=True)
            else: all_rows += chunks; print("  " + msg, flush=True)
        except Exception as e:
            failed.append((doc, f"{type(e).__name__}: {e}")); print(f"  [실패] {doc['name']}: {type(e).__name__} {e}", flush=True)
        time.sleep(0.4)

    out = OUT / "corpus.jsonl"
    if pick and out.exists():   # 일부만 다시 받을 때는 그 문서 행만 교체
        new = {r["org"] for r in all_rows}
        all_rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l and json.loads(l)["org"] not in new] + all_rows
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in all_rows), encoding="utf-8")

    lines = ["# 수동 수집 목록 (API 에 없거나 실패)", "", f"생성 {TODAY}. 받은 HWP/PDF 는 `01_raw/교원_교육청_HWP/<문서명>/` 에 둔다.", "",
             "| 문서명 | 층 | 대상 | 사유 |", "|---|---|---|---|"]
    lines += [f"| {d['name']} | {d['layer']} | {d['audience']} | 누리집 게시 자료 (API 없음) |" for d in manual]
    lines += [f"| {d['name']} | {d['layer']} | {d['audience']} | {why} |" for d, why in failed]
    (RAW / "수동수집_목록.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"완료: 청크 {len(all_rows)}개 -> {out} | 수동 {len(manual)}건, 실패 {len(failed)}건 -> 수동수집_목록.md", flush=True)
