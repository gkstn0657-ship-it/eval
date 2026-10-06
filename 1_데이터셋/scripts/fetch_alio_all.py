# -*- coding: utf-8 -*-
"""ALIO 인사규정 목록(302기관) 중 미수집 기관의 인사규정을 수집한다 (2026-10-04, 코퍼스 확장).
기관당 최신 3개 개정판. HWP/HWPX 우선, 없으면 PDF, zip 만 있으면 풀어서 HWP/HWPX/PDF 최신 3개.
4개 스레드 병렬, 이어받기(manifest 에 있는 기관은 건너뜀).
산출: 01_raw/공공기관_인사규정_전체/<기관>/..., manifest.jsonl (기존 manifest 와 같은 필드)
"""
import requests, json, re, html, hashlib, time, pathlib, zipfile, io, sys, datetime, threading
from concurrent.futures import ThreadPoolExecutor
sys.stdout.reconfigure(encoding="utf-8")
ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "01_raw" / "공공기관_인사규정_전체"; RAW.mkdir(parents=True, exist_ok=True)
H = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.alio.go.kr/occasional/ruleList.do"}
TYPE = {"A2001": "시장형 공기업", "A2002": "준시장형 공기업", "A2003": "기금관리형 준정부기관", "A2004": "위탁집행형 준정부기관", "A2005": "기타공공기관"}
MAX_VER = 3; TODAY = str(datetime.date.today()); WORKERS = 4
L = json.load(open(ROOT / "scripts" / "alio_인사규정_목록_20261001.json", encoding="utf-8"))
have = {json.loads(l)["org"] for mf in (ROOT / "01_raw").glob("*/manifest.jsonl") for l in open(mf, encoding="utf-8")}
rules = {}
for r in L:
    if re.sub(r"^\d+\.\s*", "", r["title"]).strip() != "인사규정" or r["pname"] in have or "해산" in r["pname"]: continue
    if r["pname"] not in rules or r.get("idate", "") > rules[r["pname"]].get("idate", ""): rules[r["pname"]] = r
print("대상 기관", len(rules), flush=True)
def safe(s): return re.sub(r'[\\/:*?"<>|]', "_", s).replace("(주)", "").replace("주식회사", "").strip(" _")
def get(url):
    err = None
    for i in range(3):
        try: return requests.get(url, headers=H, timeout=60)
        except Exception as e: err = e; time.sleep(2 * (i + 1))
    raise err
def date_key(name):
    m = re.search(r"(\d{2,4})\s*년도?\s*(\d{1,2})\s*월(?:\s*(\d{1,2})\s*일)?", name)
    if not m: return (0, 0, 0)
    y = int(m.group(1)); y = y + 2000 if y < 100 else y
    return (y, int(m.group(2)), int(m.group(3) or 0))
mf = RAW / "manifest.jsonl"; lock = threading.Lock()
def work(item):
    n, (pname, x) = item
    url = f"https://www.alio.go.kr/item/itemBoard21110.do?apbald={x['apbaId']}&nowcode=21110&reportFormNo=21110&table_name=COMM_RULE&idx_name=RULENO&idx={x['seq']}&reportGbn=N&bid_type=K1100"
    try:
        s = get(url).text
        files = [(fn, html.unescape(f)) for fn, f in re.findall(r"rulefiledown\.json\?fileNo=(\d+)\"[\s\S]{0,400}?previewAjax\('\d+',\s*'([^']+)'", s)]
        org = safe(pname); d = RAW / org; d.mkdir(exist_ok=True); rows = []
        def add(name, data, fileno, note=""):
            name = safe(name); p = d / name
            if not p.exists(): p.write_bytes(data)
            rows.append({"org": pname, "org_type": TYPE.get(x["apbaType"], x["apbaType"]), "rule_title": "인사규정", "rule_latest_date": x.get("ruleStDa"),
                "file": f"{org}/{name}", "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "format": p.suffix.lower().lstrip("."),
                "source_page": url, "download_url": f"https://www.alio.go.kr/download/rulefiledown.json?fileNo={fileno}", "alio_seq": x["seq"],
                "collected": TODAY, "license_note": "ALIO 공개 내부규정" + note + ". 공공기관 경영정보 공개 자료, 재배포 전 공공누리 유형 확인 필요"})
        pick = lambda exts: sorted([f for f in files if f[1].lower().endswith(exts)], key=lambda f: date_key(f[1]))[-MAX_VER:]
        sel = pick((".hwp", ".hwpx")) or pick((".pdf",)); kind = "hwp" if sel and sel[0][1].lower().endswith((".hwp", ".hwpx")) else ("pdf" if sel else "")
        if sel:
            for fileno, fname in sel: add(fname, get(f"https://www.alio.go.kr/download/rulefiledown.json?fileNo={fileno}").content, fileno)
        else:
            zips = [f for f in files if f[1].lower().endswith(".zip")]
            if zips:
                fileno, zname = zips[-1]; z = zipfile.ZipFile(io.BytesIO(get(f"https://www.alio.go.kr/download/rulefiledown.json?fileNo={fileno}").content))
                inner = []
                for i in z.infolist():
                    if i.file_size == 0: continue
                    nm = i.filename
                    if not (i.flag_bits & 0x800):
                        try: nm = nm.encode("cp437").decode("cp949")
                        except Exception: pass
                    inner.append((nm.split("/")[-1], i))
                cand = []
                for exts in ((".hwp", ".hwpx"), (".pdf",)):
                    cand = sorted([t for t in inner if t[0].lower().endswith(exts)], key=lambda t: date_key(t[0]))[-MAX_VER:]
                    if cand: break
                for nm, i in cand: add(nm, z.read(i), fileno, "(zip 제공, 압축 해제)")
                kind = "zip"
        with lock:
            with open(mf, "a", encoding="utf-8") as f:
                for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        msg = f"{n:3} {pname} | 파일 {len(files)} | {kind} {len(rows)}개"
    except Exception as e:
        msg = f"{n:3} {pname} | 실패 {e}"
    print(msg, flush=True); return msg
done = {json.loads(l)["org"] for l in open(mf, encoding="utf-8")} if mf.exists() else set()
items = [(n, kv) for n, kv in enumerate(sorted(rules.items()), 1) if kv[0] not in done]
print("이어받기: 완료", len(done), "남음", len(items), flush=True)
with ThreadPoolExecutor(WORKERS) as ex: log = list(ex.map(work, items))
(RAW / "fetch_log.txt").write_text("\n".join(log), encoding="utf-8")
print("완료. manifest 행:", sum(1 for _ in open(mf, encoding="utf-8")), flush=True)
