# -*- coding: utf-8 -*-
"""Hugging Face Space(Gradio SDK, ZeroGPU) 진입점 — 한 Space 에 두 API.
  /demo/…     공공기관 인사규정 데모 (포트폴리오, 화면 demo.html + 가드레일 v3 /demo/ask)
  /teacher/…  교원·교직원 규정 API (티타임용, teacher_route.py, 호출 키 필요)
HF 가 새 Gradio Space 를 PRO 전용으로 바꿔(2026-10) 기존 Space 하나에 같이 올린다. 모델(bge-m3·재랭커)은 공유한다.
"""
import os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["GUARDRAILS"] = "off"   # 데모(hitl_serve.ask)는 쓰지 않고 guard_route.ask_v3 로 대체한다
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # ZeroGPU: @spaces.GPU 밖에서는 CPU
sys.path.insert(0, str(ROOT / "3_성능테스트")); sys.path.insert(0, str(ROOT / "4_결과분석"))
sys.path.insert(0, str(ROOT / "5_가드레일" / "v2")); sys.path.insert(0, str(ROOT))
try:  # ZeroGPU 는 기동 시 @spaces.GPU 함수가 하나라도 있어야 한다
    import spaces
    @spaces.GPU
    def _zerogpu_marker(): return "unused"
except ImportError:
    pass

import hitl_serve  # noqa: E402
import guard_route  # noqa: E402
import teacher_route  # noqa: E402
import gradio as gr  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.responses import HTMLResponse, RedirectResponse  # noqa: E402

E = hitl_serve.E; E._device = lambda: "cpu"
# 기동 시 미리 로드. 임베딩은 cache/bge-m3.npz 에 두 코퍼스 것이 다 있어 모델 다운로드만 기다린다.
c1 = hitl_serve.corpus(); c1.bge; c1.bm25
c2 = teacher_route.corpus(); c2.bge; c2.bm25
E.st_model("BAAI/bge-m3"); E.ce_model()

# D: 재랭킹만 ZeroGPU 로 (실측 2026-10-14: 이 Space 는 requirements 의 torch 가 CPU 빌드이고 CUDA_VISIBLE_DEVICES="" 라
# "No CUDA GPUs are available" 로 매번 실패해 CPU 로 떨어졌다. 시도 자체가 0.5초를 먹으므로 USE_ZEROGPU=1 일 때만 켠다.)
try:
    if os.environ.get("USE_ZEROGPU") != "1": raise ImportError("ZeroGPU 재랭킹 비활성")
    import spaces, torch
    _ce_gpu = {}
    @spaces.GPU(duration=20)
    def _ce_predict_gpu(pairs):
        if "m" not in _ce_gpu:
            from sentence_transformers import CrossEncoder
            _ce_gpu["m"] = CrossEncoder("BAAI/bge-reranker-v2-m3", device="cuda", max_length=E.CE_MAX_LEN)
        return _ce_gpu["m"].predict(pairs, batch_size=32, show_progress_bar=False)
    def ce_predict_auto(pairs):
        try: return _ce_predict_gpu(pairs)
        except Exception as e:
            print(f"ZeroGPU 재랭킹 실패, CPU 로: {type(e).__name__}: {str(e)[:120]}", flush=True)
            return teacher_route._ce_cpu(pairs)
    teacher_route.CE_PREDICT = ce_predict_auto
    print("재랭커: ZeroGPU 우선, CPU 대체", flush=True)
except ImportError:
    print("재랭커: CPU", flush=True)
print(f"준비 완료: 데모 {len(c1.rows)}청크 / 교원 {len(c2.rows)}청크 / LLM {teacher_route.pick_llm()[1]}", flush=True)

api = FastAPI()
# ----- 공기업 데모 (/demo) -----
@api.get("/demo/", response_class=HTMLResponse)
def demo_index():
    return HTMLResponse((ROOT / "demo.html").read_text(encoding="utf-8"), headers={"Cache-Control": "no-store"})
@api.get("/demo")  # 슬래시 없이 열면 상대 경로 orgs/ask 가 /orgs 로 풀려 404 → 항상 /demo/ 로 보낸다
def demo_slash(): return RedirectResponse("/demo/", status_code=301)
api.get("/demo/orgs")(hitl_serve.orgs)
api.get("/demo/ask")(guard_route.ask_v3)
@api.get("/")
def root(): return RedirectResponse("/demo/", status_code=302)
# ----- 교원 API (/teacher) -----
api.get("/teacher/ask")(teacher_route.ask)
api.get("/teacher/orgs")(teacher_route.orgs)
api.get("/teacher/health")(teacher_route.health)

with gr.Blocks() as demo:
    gr.Markdown("# 공공기관 인사규정 RAG · 교원 규정 RAG")
    gr.HTML('<p>데모: <a href="/demo/" target="_blank" rel="noopener"><b>/demo/</b></a> · 교원 API: <code>/teacher/ask</code> (티타임 전용, 호출 키 필요)</p>')

app, _, _ = demo.launch(server_name="0.0.0.0", server_port=7860, prevent_thread_lock=True, ssr_mode=False)
for r in reversed(list(api.router.routes)):
    if getattr(r, "path", "").startswith(("/demo", "/teacher")) or getattr(r, "path", "") == "/":
        app.router.routes.insert(0, r)
demo.block_thread()
