import requests, json, re, html, hashlib, os, time, pathlib
H={"User-Agent":"Mozilla/5.0","Referer":"https://www.alio.go.kr/occasional/ruleList.do"}
RAW=pathlib.Path(r"c:\Users\SSAFY\Desktop\RAG 포폴 소스\포폴_정리\1_데이터셋\01_raw\공기업_인사규정_HWP")
RAW.mkdir(parents=True, exist_ok=True)
allr=json.load(open('rules_all.json',encoding='utf-8'))
TYPE={'A2001':'시장형 공기업','A2002':'준시장형 공기업'}
targets=[x for x in allr if x['apbaType'] in TYPE and re.sub(r'^\d+\.\s*','',x['title']).strip()=='인사규정' and '해산' not in x['pname']]
print('targets', len(targets))
MAX_VER=3
manifest=[]; log=[]
for x in targets:
    url=f"https://www.alio.go.kr/item/itemBoard21110.do?apbald={x['apbaId']}&nowcode=21110&reportFormNo=21110&table_name=COMM_RULE&idx_name=RULENO&idx={x['seq']}&reportGbn=N&bid_type=K1100"
    s=requests.get(url,headers=H,timeout=30).text
    files=re.findall(r'rulefiledown\.json\?fileNo=(\d+)"[\s\S]{0,400}?previewAjax\(\'\d+\',\s*\'([^\']+)\'',s)
    files=[(n,html.unescape(f)) for n,f in files]
    hw=[f for f in files if f[1].lower().endswith(('.hwp','.hwpx'))]
    sel=hw[-MAX_VER:]  # 페이지 순서상 뒤쪽이 최신
    org=re.sub(r'[\/:*?"<>|]','_',x['pname']).replace('(주)','').replace('주식회사','').strip(' _')
    log.append(f"{x['pname']} | 파일 {len(files)}개(HWP {len(hw)}개) | 선택 {len(sel)}개")
    d=RAW/org; d.mkdir(exist_ok=True)
    for fileNo,fname in sel:
        fname=re.sub(r'[\/:*?"<>|]','_',fname)
        p=d/fname
        if not p.exists():
            r=requests.get(f"https://www.alio.go.kr/download/rulefiledown.json?fileNo={fileNo}",headers=H,timeout=60)
            p.write_bytes(r.content); time.sleep(0.5)
        b=p.read_bytes()
        manifest.append({"org":x['pname'],"org_type":TYPE[x['apbaType']],"rule_title":x['title'],"rule_latest_date":x['ruleStDa'],
            "file":f"{org}/{fname}","bytes":len(b),"sha256":hashlib.sha256(b).hexdigest(),
            "format":p.suffix.lower().lstrip('.'),"source_page":url,"download_url":f"https://www.alio.go.kr/download/rulefiledown.json?fileNo={fileNo}",
            "alio_seq":x['seq'],"collected":"2026-10-01","license_note":"ALIO 공개 내부규정. 공공기관 경영정보 공개 자료, 재배포 전 공공누리 유형 확인 필요"})
with open(RAW/"manifest.jsonl","w",encoding="utf-8") as f:
    for m in manifest: f.write(json.dumps(m,ensure_ascii=False)+"\n")
print("\n".join(log)); print("files", len(manifest), "total MB", round(sum(m['bytes'] for m in manifest)/1e6,1))
