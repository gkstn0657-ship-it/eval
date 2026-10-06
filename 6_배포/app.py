# -*- coding: utf-8 -*-
"""Hugging Face Space(Gradio SDK) 진입점. hybrid_prefilter 데모(4_결과분석/hitl_serve.py)를 그대로 띄운다.
Gradio는 Space 요건용 껍데기(/gradio)이고, 화면과 API는 hitl_serve 의 FastAPI 가 담당한다.
"""
import os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# run_eval.py 가 setdefault 로 로컬 경로(E:\hf_cache)·오프라인을 켜므로 import 전에 Space 환경으로 고정
os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["GUARDRAILS"] = "off"  # 공개 데모는 가드레일 없이 검색 결과로 바로 답변 (2026-10-06 결정)
# ZeroGPU Space: GPU 는 @spaces.GPU 함수 안에서만 쓸 수 있어, 로컬 검증과 같은 CPU 경로로 고정한다
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(ROOT / "4_결과분석"))
try:  # ZeroGPU 는 기동 시 @spaces.GPU 함수가 하나라도 있어야 한다 (없으면 RUNTIME_ERROR)
    import spaces
    @spaces.GPU
    def _zerogpu_marker(): return "unused"
except ImportError:
    pass

import hitl_serve  # noqa: E402
import gradio as gr  # noqa: E402
import uvicorn  # noqa: E402

# 첫 방문자가 모델 다운로드를 기다리지 않도록 기동 시 미리 로드 (bge-m3, bge-reranker-v2-m3, BM25)
E = hitl_serve.E
E._device = lambda: "cpu"  # spaces 패키지가 torch.cuda 를 패치해 is_available() 이 True 로 보일 수 있어 CPU 로 강제
c = hitl_serve.corpus(); c.bge; c.bm25
E.st_model("BAAI/bge-m3"); ce = E.ce_model()
import torch
_probe = ce.predict([("정년이 몇 살이야?", "제31조(정년) 직원의 정년은 60세로 한다."), ("정년이 몇 살이야?", "제5조(목적) 이 규정은 직원의 임용을 정한다.")], show_progress_bar=False)
print(f"진단: cuda.is_available={torch.cuda.is_available()} ce.device={ce.model.device} ce.dtype={next(ce.model.parameters()).dtype} probe={[round(float(x), 4) for x in _probe]}", flush=True)
print("모델·코퍼스 준비 완료", flush=True)

# 공개용 화면(demo.html)으로 교체. 로컬 HITL 비교 화면(hitl.html)의 "/" 라우트를 뺀다.
from fastapi.responses import HTMLResponse  # noqa: E402
hitl_serve.app.router.routes = [r for r in hitl_serve.app.router.routes if getattr(r, "path", None) != "/"]
@hitl_serve.app.get("/", response_class=HTMLResponse)
def public_index():
    return HTMLResponse((ROOT / "demo.html").read_text(encoding="utf-8"), headers={"Cache-Control": "no-store"})

with gr.Blocks() as demo:
    gr.Markdown("# 공공기관 인사규정 RAG (hybrid_prefilter)")
    gr.HTML('<p style="font-size:1.1em">데모 화면: <a href="/" target="_blank" rel="noopener"><b>새 창에서 열기</b></a> &nbsp;·&nbsp; 직접 주소 <code>https://runningturtle123-public-agency-ragchat.hf.space</code></p>')
    # Space 페이지가 여는 루트(/)에서 데모 화면으로 바로 이동

# ZeroGPU 검사는 Gradio 의 launch() 를 거쳐야 통과한다 (uvicorn 직접 실행 시 RUNTIME_ERROR).
# launch() 가 만든 FastAPI 앱에 데모 서버(hitl_serve)의 화면·API 를 기본 주소에 붙인다.
from starlette.routing import Mount  # noqa: E402
# ssr_mode=False: Spaces 는 기본으로 Node SSR 서버를 앞에 두는데, 그러면 데모 요청이 Python 까지 오지 않고 Gradio 페이지로 렌더링된다
app, _, _ = demo.launch(server_name="0.0.0.0", server_port=7860, prevent_thread_lock=True, ssr_mode=False)
# 데모 화면과 API 를 기본 주소(/)에 바로 붙인다. Gradio 의 "/" 화면은 가려지고 나머지 Gradio 경로(/gradio_api 등)는 그대로 둔다.
from starlette.routing import Route
from starlette.responses import RedirectResponse
for r in reversed([r for r in hitl_serve.app.router.routes if getattr(r, "path", None) in ("/", "/orgs", "/ask")]):
    app.router.routes.insert(0, r)
# 예전 주소(/demo/...)로 들어오면 새 주소로 보낸다
app.router.routes.insert(0, Route("/demo{rest:path}", lambda req: RedirectResponse((req.path_params["rest"] or "/") + (("?" + req.url.query) if req.url.query else ""), status_code=301)))
demo.block_thread()
