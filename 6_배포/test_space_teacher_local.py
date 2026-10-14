# -*- coding: utf-8 -*-
"""조립된 space_teacher/ 를 업로드 전에 로컬에서 점검한다 (Gradio 없이 FastAPI 만, LLM 키 없으면 근거만).
사용: PYTHONIOENCODING=utf-8 python 6_배포/test_space_teacher_local.py
"""
import os, sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parent / ("space_merged" if "--merged" in sys.argv else "space_teacher")
os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface")); os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["RAG_KEY"] = "test-key"
for p in ("3_성능테스트", "4_결과분석", "5_가드레일/v2", "."): sys.path.insert(0, str(ROOT / p))
os.chdir(ROOT)
import hitl_serve; hitl_serve.CORPUS = "TEACH_A"
import teacher_route
from fastapi.testclient import TestClient
cl = TestClient(teacher_route.api)
print("health:", cl.get("/health").json())
print("orgs:", len(cl.get("/orgs", headers={"X-RAG-KEY": "test-key"}).json()), "/ 키 없음:", cl.get("/orgs").status_code)
for q, aud, doc in [("육아휴직은 자녀가 몇 살까지 가능해?", "교원", ""), ("병가는 1년에 며칠까지 쓸 수 있어?", "행정직", ""),
                    ("외부 강의 하려면 신고해야 해?", "교원", ""), ("기간제교원 임용 기간은 최대 몇 년이야?", "교원", "교육공무원법")]:
    r = cl.get("/ask", params={"q": q, "audience": aud, "doc": doc, "applies_to": "공립", "k": 3}, headers={"X-RAG-KEY": "test-key"}).json()
    print(f"\n[{aud}/{doc or '-'}] {q} ({r['sec']}s, gate={r['gate']['action']}, note={r['gate'].get('note')})")
    for h in r["hits"]: print("  -", h["org"], h["article_no"], f"({h['title']})", f"[{h['audience']}/{h['applies_to']}]")
    if r.get("answer"): print("  답:", r["answer"][:200])
