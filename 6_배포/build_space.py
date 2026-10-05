# -*- coding: utf-8 -*-
"""HF Space 업로드용 묶음을 6_배포/space/ 에 조립한다. 코드는 복사본이므로 원본을 고친 뒤 다시 실행한다.
사용: python 6_배포/build_space.py
"""
import shutil
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; OUT = HERE / "space"
FILES = [
    "3_성능테스트/run_eval.py", "3_성능테스트/corpora/L2P_A.jsonl", "3_성능테스트/cache/bge-m3.npz",
    "5_가드레일/guardrails.py", "4_결과분석/hitl_serve.py", "4_결과분석/hitl.html",
]
for name in ("app.py", "demo.html", "requirements.txt", "README.md", ".gitattributes"):
    (OUT).mkdir(exist_ok=True); shutil.copy2(HERE / name, OUT / name)
for rel in FILES:
    dst = OUT / rel; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT / rel, dst)
(OUT / ".gitignore").write_text("__pycache__/\n*.log\n3_성능테스트/results*/\n3_성능테스트/hitl_*.jsonl\n", encoding="utf-8")
print("조립 완료:", OUT)
