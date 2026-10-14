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

## 교원·교직원 규정 API (`/teacher/…`, 티타임용, 2026-10-14 추가)

고등학교 교원·행정직 질문용. 코퍼스는 법령 59 · 교육부 행정규칙 6 · 전남광주통합특별시교육청/(구)전라남도교육청 자치법규 30 (조문 4,299개, 국가법령정보센터 Open API). 검색은 문서·직종·설립 마스크 → 같은 hybrid 경로, 답변은 가드레일 v3. 호출 키(`RAG_KEY` 시크릿, 헤더 `X-RAG-KEY`)가 있어야 한다.

```
GET /teacher/ask?q=&doc=&audience=교원|행정직|공무직&applies_to=공립|사립&k=5
GET /teacher/orgs   GET /teacher/health
```
