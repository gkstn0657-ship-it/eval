# -*- coding: utf-8 -*-
"""Hugging Face Space(Gradio SDK) 진입점 — 교원·교직원 규정 챗봇 API (티타임용).
Gradio 는 Space 요건용 껍데기이고 API 는 teacher_route 의 FastAPI 가 담당한다. 화면은 티타임(teachers-time.vercel.app)에 있다.
"""
import os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["GUARDRAILS"] = "on"
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # CPU Space
sys.path.insert(0, str(ROOT / "3_성능테스트")); sys.path.insert(0, str(ROOT / "4_결과분석"))
sys.path.insert(0, str(ROOT / "5_가드레일" / "v2")); sys.path.insert(0, str(ROOT))

import hitl_serve  # noqa: E402
hitl_serve.CORPUS = "TEACH_A"   # corpus() 가 처음 불릴 때 이 이름의 corpora/*.jsonl 을 읽는다
import teacher_route  # noqa: E402
import gradio as gr  # noqa: E402

# 기동 시 미리 로드 (bge-m3, 재랭커, BM25). 임베딩은 cache/bge-m3.npz 에 있어 모델 다운로드만 기다린다.
E = hitl_serve.E; E._device = lambda: "cpu"
c = hitl_serve.corpus(); c.bge; c.bm25; E.st_model("BAAI/bge-m3"); E.ce_model()
print(f"준비 완료: 청크 {len(c.rows)} / 문서 {len(c.org_names)} / LLM {teacher_route.pick_llm()[1]}", flush=True)

with gr.Blocks() as demo:
    gr.Markdown("# 교원·교직원 규정 RAG (티타임용 API)")
    gr.Markdown("화면은 티타임 앱에 있습니다. 이 Space 는 `/ask`, `/orgs`, `/health` API 만 제공합니다.")

app, _, _ = demo.launch(server_name="0.0.0.0", server_port=7860, prevent_thread_lock=True, ssr_mode=False)
for r in reversed(list(teacher_route.api.router.routes)):
    if getattr(r, "path", None) in ("/ask", "/orgs", "/health"): app.router.routes.insert(0, r)
demo.block_thread()
