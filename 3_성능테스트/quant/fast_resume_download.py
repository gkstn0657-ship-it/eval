# -*- coding: utf-8 -*-
"""HF 다운로드 이어받기(병렬 구간). 연결 하나당 속도가 제한된 네트워크용.
huggingface_hub 가 남긴 .incomplete 파일(앞부분)을 살리고, 남은 구간을 64MB 조각으로 나눠 여러 연결로 동시에 받는다.
끝나면 sha256 을 원본 저장소 값과 대조하고 정식 파일명으로 옮긴다.
사용: python fast_resume_download.py <repo_id> <local_dir> [연결수=32]
"""
import sys, os, glob, hashlib, time, threading, urllib.request
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import HfApi, hf_hub_url
repo, local = sys.argv[1], sys.argv[2]; W = int(sys.argv[3]) if len(sys.argv) > 3 else 32
CH = 64 * 1024 * 1024
info = HfApi().model_info(repo, files_metadata=True)
files = [(s.rfilename, s.size, s.lfs.sha256) for s in info.siblings if s.rfilename.endswith(".safetensors")]
dl = os.path.join(local, ".cache", "huggingface", "download")
done_bytes = [0]; lock = threading.Lock(); t0 = time.time()

def fetch(url, start, end, path):
    for attempt in range(8):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=120) as r, open(path, "wb") as f:
                while True:
                    b = r.read(1 << 20)
                    if not b: break
                    f.write(b)
                    with lock: done_bytes[0] += len(b)
            if os.path.getsize(path) == end - start + 1: return
        except Exception as e:
            time.sleep(2 + attempt * 2)
    raise RuntimeError(f"구간 실패 {start}-{end}")

jobs, plan = [], []
for name, size, sha in files:
    final = os.path.join(local, name)
    if os.path.exists(final) and os.path.getsize(final) == size: continue
    inc = [p for p in glob.glob(os.path.join(dl, "*.incomplete")) if sha in os.path.basename(p)]
    have = os.path.getsize(inc[0]) if inc else 0
    url = hf_hub_url(repo, name)
    parts = []
    for s in range(have, size, CH):
        e = min(s + CH, size) - 1; p = f"{final}.part{s}"
        parts.append(p); jobs.append((url, s, e, p))
    plan.append((name, size, sha, final, inc[0] if inc else None, parts))
remain = sum(e - s + 1 for _, s, e, _ in jobs)
print(f"남은 {remain/1e9:.2f}GB, 조각 {len(jobs)}개, 연결 {W}", flush=True)

def mon():
    while True:
        time.sleep(30)
        el = time.time() - t0; d = done_bytes[0]
        print(f"  {d/1e9:.2f}/{remain/1e9:.2f}GB  {d/el/1e6:.1f}MB/s", flush=True)
        if d >= remain: break
threading.Thread(target=mon, daemon=True).start()
with ThreadPoolExecutor(W) as ex: list(ex.map(lambda j: fetch(*j), jobs))

for name, size, sha, final, inc, parts in plan:
    h = hashlib.sha256(); tmp = final + ".assembling"
    with open(tmp, "wb") as out:
        for src in ([inc] if inc else []) + parts:
            with open(src, "rb") as f:
                while True:
                    b = f.read(1 << 24)
                    if not b: break
                    h.update(b); out.write(b)
    if os.path.getsize(tmp) != size or h.hexdigest() != sha:
        raise RuntimeError(f"검증 실패 {name}: size {os.path.getsize(tmp)} vs {size}")
    os.replace(tmp, final)
    for p in parts + ([inc] if inc else []): os.remove(p)
    print(f"검증 완료 {name}", flush=True)
print(f"qwen14b done {time.time()-t0:.0f}s", flush=True)
