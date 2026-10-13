# -*- coding: utf-8 -*-
"""교원 규정 Space 업로드용 묶음을 6_배포/space_teacher/ 에 조립하고(옵션) HF Space 에 올린다.

조립: python 6_배포/build_space_teacher.py
업로드: python 6_배포/build_space_teacher.py --upload [--repo runningturtle123/teacher-rules-rag]
  업로드는 `hf auth login` 된 계정으로 한다. RAG_KEY 를 새로 만들어 Space 시크릿에 넣고 6_배포/.rag_key 에 저장한다(커밋 금지).
  ANTHROPIC_API_KEY / HF_TOKEN 시크릿은 환경변수에 있으면 같이 올린다.
"""
import sys, shutil, hashlib, secrets, os
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; OUT = HERE / "space_teacher"; SRC = HERE / "teacher"
REPO = sys.argv[sys.argv.index("--repo") + 1] if "--repo" in sys.argv else "runningturtle123/teacher-rules-rag"

FILES = ["3_성능테스트/run_eval.py", "5_가드레일/guardrails.py", "5_가드레일/v2/guard.py", "4_결과분석/hitl_serve.py", "4_결과분석/hitl.html"]
if OUT.exists(): shutil.rmtree(OUT)
OUT.mkdir()
for name in ("app.py", "teacher_route.py", "README.md"): shutil.copy2(SRC / name, OUT / name)
shutil.copy2(HERE / "requirements.txt", OUT / "requirements.txt")
for rel in FILES:
    dst = OUT / rel; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT / rel, dst)

# 코퍼스 + 그 청크만 담은 임베딩 캐시 (전체 캐시 202MB 중 교원 코퍼스 4,299개만 → 약 17MB)
corpus_src = ROOT / "1_데이터셋" / "05_clean" / "L2_교원" / "corpus.jsonl"
(OUT / "3_성능테스트" / "corpora").mkdir(parents=True); shutil.copy2(corpus_src, OUT / "3_성능테스트" / "corpora" / "TEACH_A.jsonl")
import json
texts = [json.loads(l)["text"] for l in corpus_src.read_text(encoding="utf-8").splitlines() if l.strip()]
big = np.load(ROOT / "3_성능테스트" / "cache" / "bge-m3.npz", allow_pickle=True)["d"].item()
keys = [hashlib.sha1(t.encode()).hexdigest() for t in texts]
missing = [k for k in keys if k not in big]
if missing: sys.exit(f"임베딩 캐시에 없는 청크 {len(missing)}개. 먼저 3_성능테스트/smoke_teacher_corpus.py 를 돌려 주세요.")
small = {k: big[k] for k in keys}
(OUT / "3_성능테스트" / "cache").mkdir(); np.savez_compressed(OUT / "3_성능테스트" / "cache" / "bge-m3.npz", d=np.array(small, dtype=object))
(OUT / ".gitattributes").write_text("*.npz filter=lfs diff=lfs merge=lfs -text\n", encoding="utf-8")
(OUT / ".gitignore").write_text("__pycache__/\n*.log\n*.jsonl.log\nguard_log.jsonl\n", encoding="utf-8")
print(f"조립 완료: {OUT} (청크 {len(texts)}, 캐시 {len(small)})")

if "--upload" in sys.argv:
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(REPO, repo_type="space", space_sdk="gradio", exist_ok=True, private=False)
    key_file = HERE / ".rag_key"
    rag_key = key_file.read_text(encoding="utf-8").strip() if key_file.exists() else secrets.token_urlsafe(32)
    key_file.write_text(rag_key, encoding="utf-8")
    api.add_space_secret(REPO, "RAG_KEY", rag_key)
    from huggingface_hub import get_token
    secrets_env = {"ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY"), "HF_TOKEN": os.environ.get("HF_TOKEN") or get_token()}
    for name, val in secrets_env.items():
        if val: api.add_space_secret(REPO, name, val); print(f"시크릿 {name} 설정")
    api.upload_folder(repo_id=REPO, repo_type="space", folder_path=str(OUT), commit_message="교원 규정 RAG Space 업로드")
    print(f"업로드 완료: https://huggingface.co/spaces/{REPO}  (RAG_KEY 는 6_배포/.rag_key)")
