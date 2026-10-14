# -*- coding: utf-8 -*-
"""기존 Space(runningturtle123/public-agency-ragchat)에 공기업 데모(/demo)와 교원 API(/teacher)를 함께 올리는 묶음을
6_배포/space_merged/ 에 조립하고(옵션) 업로드한다. HF 가 새 Gradio Space 를 PRO 전용으로 바꿔 기존 Space 를 같이 쓴다.

조립: python 6_배포/build_space_merged.py
업로드: python 6_배포/build_space_merged.py --upload
  RAG_KEY 는 6_배포/.rag_key(없으면 새로 생성, 커밋 금지)를 Space 시크릿에 넣는다. HF_TOKEN 은 저장된 로그인 토큰.
"""
import sys, shutil, hashlib, secrets, os, json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; OUT = HERE / "space_merged"
REPO = "runningturtle123/public-agency-ragchat"
FILES = ["3_성능테스트/run_eval.py", "3_성능테스트/corpora/L2P_A.jsonl",
         "5_가드레일/guardrails.py", "5_가드레일/v2/guard.py", "4_결과분석/hitl_serve.py", "4_결과분석/hitl.html"]

if OUT.exists(): shutil.rmtree(OUT)
OUT.mkdir()
for src, dst in (("app_merged.py", "app.py"), ("teacher/teacher_route.py", "teacher_route.py"), ("guard_route.py", "guard_route.py"),
                 ("guard_data.json", "guard_data.json"), ("demo.html", "demo.html"), ("requirements.txt", "requirements.txt"),
                 ("SPACE_README.md", "README.md"), (".gitattributes", ".gitattributes")):
    shutil.copy2(HERE / src, OUT / dst)
for rel in FILES:
    d = OUT / rel; d.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT / rel, d)
# 교원 코퍼스 + 두 코퍼스 청크를 모두 담은 임베딩 캐시
teach = ROOT / "1_데이터셋" / "05_clean" / "L2_교원" / "corpus.jsonl"
shutil.copy2(teach, OUT / "3_성능테스트" / "corpora" / "TEACH_A.jsonl")
big = np.load(ROOT / "3_성능테스트" / "cache" / "bge-m3.npz", allow_pickle=True)["d"].item()
need = {}
for cf in (OUT / "3_성능테스트" / "corpora" / "L2P_A.jsonl", OUT / "3_성능테스트" / "corpora" / "TEACH_A.jsonl"):
    for l in cf.read_text(encoding="utf-8").splitlines():
        if l.strip():
            k = hashlib.sha1(json.loads(l)["text"].encode()).hexdigest()
            if k not in big: sys.exit(f"임베딩 캐시에 없는 청크: {cf.name}. smoke_teacher_corpus.py 를 먼저 돌려 주세요.")
            need[k] = big[k]
(OUT / "3_성능테스트" / "cache").mkdir(parents=True)
np.savez_compressed(OUT / "3_성능테스트" / "cache" / "bge-m3.npz", d=np.array(need, dtype=object))
(OUT / ".gitignore").write_text("__pycache__/\n*.log\nguard_log.jsonl\n3_성능테스트/results*/\n3_성능테스트/hitl_*.jsonl\n", encoding="utf-8")
print(f"조립 완료: {OUT} (캐시 {len(need)}개)")

if "--upload" in sys.argv:
    from huggingface_hub import HfApi, get_token
    api = HfApi()
    key_file = HERE / ".rag_key"
    rag_key = key_file.read_text(encoding="utf-8").strip() if key_file.exists() else secrets.token_urlsafe(32)
    key_file.write_text(rag_key, encoding="utf-8")
    api.add_space_secret(REPO, "RAG_KEY", rag_key)
    for name, val in (("ANTHROPIC_API_KEY", os.environ.get("ANTHROPIC_API_KEY")), ("HF_TOKEN", os.environ.get("HF_TOKEN") or get_token())):
        if val: api.add_space_secret(REPO, name, val); print(f"시크릿 {name} 설정")
    api.upload_folder(repo_id=REPO, repo_type="space", folder_path=str(OUT), commit_message="교원 규정 API(/teacher) 추가, 데모는 /demo 로 유지 (가드레일 v3)",
                      delete_patterns=["*.py", "*.html", "*.json", "*.md", "*.txt"])
    print(f"업로드 완료: https://huggingface.co/spaces/{REPO}  (RAG_KEY 는 6_배포/.rag_key)")
