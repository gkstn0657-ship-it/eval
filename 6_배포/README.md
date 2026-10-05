---
title: 공공기관 인사규정 RAG (hybrid_prefilter)
emoji: 🏛️
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 5.49.1
python_version: "3.11"
app_file: app.py
pinned: false
---

# 공공기관 인사규정 RAG — hybrid_prefilter 데모

공공기관 292곳의 인사규정(최신판 조문 22,336청크)에 대해 질문하면,
`hybrid_prefilter` 파이프라인이 근거 조문을 찾아 답변을 만든다. 공개 데모는 가드레일을 끈 상태다(평가 코드의 가드레일은 `5_가드레일/`에 그대로 있다).

- 검색: 기관 사전 필터 → BAAI/bge-m3 dense + BM25 → RRF → bge-reranker-v2-m3 재랭킹
- 답변 생성: Qwen2.5-72B-Instruct, HF Inference API (`HF_TOKEN` 시크릿 필요. 없으면 검색 근거만 표시). 답변 평가에 쓴 로컬 qwen2.5:7b-instruct 는 라우터에 없어 같은 계열 72B 를 쓴다. `HF_MODEL` 변수로 교체 가능

평가 코드와 결과: https://github.com/gkstn0657-ship-it (포폴정리 저장소 `3_성능테스트/`, `5_가드레일/`)
