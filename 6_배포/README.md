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
`hybrid_prefilter` 파이프라인이 근거 조문을 찾고 가드레일을 거쳐 답변을 만든다.

- 검색: 기관 사전 필터 → BAAI/bge-m3 dense + BM25 → RRF → bge-reranker-v2-m3 재랭킹
- 가드레일: 범위 밖·복수 기관 질의 거절, 생성 답변의 조문 인용 검증
- 답변 생성: Claude Haiku 4.5 (`ANTHROPIC_API_KEY` 시크릿 필요. 없으면 검색 근거만 표시)

평가 코드와 결과: https://github.com/gkstn0657-ship-it (포폴정리 저장소 `3_성능테스트/`, `5_가드레일/`)
