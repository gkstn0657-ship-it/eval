# -*- coding: utf-8 -*-
"""코퍼스 텍스트의 bge-m3 임베딩 캐시(cache/bge-m3.npz)를 GPU fp16 으로 미리 채운다.
run_eval.py 는 임베딩 모델을 CPU 에 올리므로(VRAM 3GB, CE 와 동시 적재 불가) 대규모 코퍼스는 이 스크립트로 먼저 캐시한다.
사용: python precache_embeddings.py L2P_A [L2P_B ...]
"""
import sys, json, hashlib, numpy as np, torch
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent; CACHE = HERE / "cache"; CACHE.mkdir(exist_ok=True)
f = CACHE / "bge-m3.npz"
cache = dict(np.load(f, allow_pickle=True)["d"].item()) if f.exists() else {}
texts = []
for name in sys.argv[1:] or ["L2P_A"]:
    texts += [json.loads(l)["text"] for l in open(HERE / "corpora" / f"{name}.jsonl", encoding="utf-8")]
todo = sorted({t for t in texts if hashlib.sha1(t.encode()).hexdigest() not in cache}, key=len)  # 길이순: 배치 패딩 최소화
print(f"텍스트 {len(texts)}, 캐시 {len(cache)}, 신규 {len(todo)}")
if todo:
    from sentence_transformers import SentenceTransformer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = SentenceTransformer("BAAI/bge-m3", device=dev)
    if dev == "cuda": m.half()
    B = 16 if dev == "cuda" else 32  # VRAM 3GB: 긴 조문(최대 1,200자)이라 16 이 안전
    for i in range(0, len(todo), B):
        vecs = m.encode(todo[i:i + B], batch_size=B, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        for t, v in zip(todo[i:i + B], vecs): cache[hashlib.sha1(t.encode()).hexdigest()] = v.astype(np.float32)
        if (i // B) % 25 == 0: print(f"  {i + len(vecs)}/{len(todo)}", flush=True)
        if (i // B) % 125 == 0 and i: np.savez_compressed(f, d=np.array(cache, dtype=object))  # 2,000개마다 중간 저장
    np.savez_compressed(f, d=np.array(cache, dtype=object))
print("캐시", len(cache), "저장")
