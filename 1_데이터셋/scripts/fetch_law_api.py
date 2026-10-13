# -*- coding: utf-8 -*-
"""국가법령정보센터(law.go.kr) Open API 로 교원 관련 법령을 받아 조 단위 코퍼스로 만든다 (2026-10-13, 교원 규정 코퍼스 1층).

입력: 포폴_정리/.env 의 LAW_API_OC (law.go.kr Open API 아이디). 커밋하지 않는다.
산출:
  01_raw/교원_법령/<법령명>.json              API 원문(JSON) 그대로
  01_raw/교원_법령/manifest.jsonl             법령별 수집 기록 (법령ID, 시행일, 공포일, 수집일, 출처 URL)
  05_clean/L2_교원/법령.jsonl                 조 단위 청크. 기존 L2P 행 형식(chunk_id, org, revision_date, article_no, article_title, text)에
                                              layer / issuer / applies_to / law_name / source_url 을 더한다.
                                              org 에는 법령명을 넣어 기존 기관 사전필터가 그대로 '법령 필터'로 동작하게 한다.
사용: python fetch_law_api.py            (전체)
      python fetch_law_api.py 교육공무원법 (일부만, 법령명 부분 일치)
"""
import json, re, sys, time, datetime, pathlib
import requests

sys.stdout.reconfigure(encoding="utf-8")
ROOT = pathlib.Path(__file__).resolve().parents[1]          # 1_데이터셋
REPO = ROOT.parent                                           # 포폴_정리
RAW = ROOT / "01_raw" / "교원_법령"; RAW.mkdir(parents=True, exist_ok=True)
OUT = ROOT / "05_clean" / "L2_교원"; OUT.mkdir(parents=True, exist_ok=True)
TODAY = str(datetime.date.today())

def load_oc():
    p = REPO / ".env"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("LAW_API_OC="): return line.split("=", 1)[1].strip().strip('"\'')
    sys.exit("LAW_API_OC 가 없습니다. 포폴_정리/.env 에 LAW_API_OC=<law.go.kr OC> 를 적어 주세요.")
OC = load_oc()

# 1층 법령. applies_to: 공립 | 사립 | 전체. 공립 MVP 지만 사립학교법도 받아 두고 메타데이터로 거른다.
LAWS = [
    ("교육기본법", "전체"),
    ("초·중등교육법", "전체"), ("초·중등교육법 시행령", "전체"), ("초·중등교육법 시행규칙", "전체"),
    ("교육공무원법", "공립"), ("교육공무원임용령", "공립"), ("교육공무원 승진규정", "공립"), ("교육공무원 징계령", "공립"),
    ("국가공무원법", "공립"), ("국가공무원 복무규정", "공립"), ("공무원 보수규정", "공립"), ("공무원수당 등에 관한 규정", "공립"), ("공무원연금법", "공립"),
    ("교원의 지위 향상 및 교육활동 보호를 위한 특별법", "전체"), ("교원의 지위 향상 및 교육활동 보호를 위한 특별법 시행령", "전체"),
    ("학교폭력예방 및 대책에 관한 법률", "전체"), ("학교폭력예방 및 대책에 관한 법률 시행령", "전체"),
    ("학교안전사고 예방 및 보상에 관한 법률", "전체"),
    ("아동학대범죄의 처벌 등에 관한 특례법", "전체"),
    ("사립학교법", "사립"), ("사립학교법 시행령", "사립"),
]

S = requests.Session(); S.headers["User-Agent"] = "Mozilla/5.0"
def get_json(url, params):
    for i in range(3):
        try:
            r = S.get(url, params=params, timeout=60); r.raise_for_status()
            return r.json()
        except Exception as e:
            err = e; time.sleep(2 * (i + 1))
    raise err

def norm(s): return re.sub(r"\s+", "", s or "")

def search_law(name):
    """법령명으로 현행 법령 1건을 찾는다. 정확히 같은 이름을 우선, 없으면 첫 결과."""
    d = get_json("https://www.law.go.kr/DRF/lawSearch.do", {"OC": OC, "target": "law", "type": "JSON", "query": name, "display": 50})
    items = (d.get("LawSearch") or {}).get("law") or []
    if isinstance(items, dict): items = [items]
    items = [x for x in items if "현행" in str(x.get("현행연혁코드", "현행"))]
    exact = [x for x in items if norm(x.get("법령명한글")) == norm(name)]
    return (exact or items or [None])[0]

def as_list(v):
    if v is None: return []
    return v if isinstance(v, list) else [v]

def text_of(v):
    """API 는 같은 필드를 문자열 또는 문자열 목록으로 준다."""
    if v is None: return ""
    if isinstance(v, list): return "\n".join(text_of(x) for x in v)
    return str(v).strip()

def flatten_article(a):
    """조문단위 1건 -> 본문. 조문내용 + 항(항내용 + 호 + 목). 원문 글자는 바꾸지 않는다."""
    parts = [text_of(a.get("조문내용"))]
    for h in as_list(a.get("항")):
        parts.append(text_of(h.get("항내용")))
        for ho in as_list(h.get("호")):
            parts.append(text_of(ho.get("호내용")))
            for mok in as_list(ho.get("목")):
                parts.append(text_of(mok.get("목내용")))
    return "\n".join(p for p in parts if p)

def article_no(a):
    n = str(a.get("조문번호", "")).strip(); g = str(a.get("조문가지번호", "") or "").strip()
    return f"제{n}조" + (f"의{g}" if g and g != "0" else "")

def to_date(s):
    s = str(s or "").strip()
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}" if len(s) == 8 and s.isdigit() else (s or None)

def fetch_one(name, applies_to):
    hit = search_law(name)
    if not hit: print(f"  [없음] {name}", flush=True); return []
    mst = hit.get("법령일련번호"); law_id = hit.get("법령ID")
    full = get_json("https://www.law.go.kr/DRF/lawService.do", {"OC": OC, "target": "law", "type": "JSON", "MST": mst})
    law = full.get("법령") or {}
    info = law.get("기본정보") or {}
    real_name = text_of(info.get("법령명_한글") or info.get("법령명한글") or hit.get("법령명한글") or name)
    eff = to_date(info.get("시행일자") or hit.get("시행일자")); prom = to_date(info.get("공포일자") or hit.get("공포일자"))
    source_url = f"https://www.law.go.kr/법령/{real_name}"
    (RAW / f"{real_name}.json").write_text(json.dumps(full, ensure_ascii=False, indent=1), encoding="utf-8")
    with open(RAW / "manifest.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"law_name": real_name, "law_id": law_id, "mst": mst, "effective_date": eff, "promulgated": prom,
                            "applies_to": applies_to, "collected": TODAY, "source_url": source_url,
                            "license_note": "국가법령정보센터 Open API. 법령은 저작권법 제7조에 따라 보호받지 못하는 저작물"}, ensure_ascii=False) + "\n")
    rows = []
    for a in as_list((law.get("조문") or {}).get("조문단위")):
        if str(a.get("조문여부", "조문")) != "조문": continue      # 편·장·절 제목 행은 뺀다
        body = flatten_article(a)
        if not body or "삭제" == body.replace(article_no(a), "").strip(" <>()").strip(): continue
        no = article_no(a); title = text_of(a.get("조문제목"))
        rows.append({
            "chunk_id": f"{real_name}|{eff}|{no}", "article_id": f"{real_name}|{eff}|{no}",
            "org": real_name, "revision_date": eff, "is_latest": True,
            "article_no": no, "article_title": title, "text": body,
            "layer": "법령", "issuer": "국가", "applies_to": applies_to, "law_name": real_name, "source_url": source_url,
        })
    print(f"  {real_name}: 조문 {len(rows)}개 (시행 {eff})", flush=True)
    return rows

if __name__ == "__main__":
    pick = sys.argv[1] if len(sys.argv) > 1 else ""
    targets = [(n, a) for n, a in LAWS if pick in n]
    print(f"법령 {len(targets)}건 수집 시작", flush=True)
    all_rows = []
    for name, applies_to in targets:
        try: all_rows += fetch_one(name, applies_to)
        except Exception as e: print(f"  [실패] {name}: {type(e).__name__} {e}", flush=True)
        time.sleep(0.5)
    out = OUT / "법령.jsonl"
    if pick and out.exists():   # 일부만 다시 받을 때는 그 법령 행만 교체
        keep = [l for l in out.read_text(encoding="utf-8").splitlines() if l and json.loads(l)["org"] not in {r["org"] for r in all_rows}]
        all_rows = [json.loads(l) for l in keep] + all_rows
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in all_rows), encoding="utf-8")
    print(f"완료: {len(all_rows)} 청크 -> {out}", flush=True)
