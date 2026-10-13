---
title: 교원·교직원 규정 RAG (티타임)
emoji: 🏫
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.49.1
python_version: "3.11"
app_file: app.py
pinned: false
---

# 교원·교직원 규정 RAG — 티타임용 API

고등학교 교원·행정직이 묻는 규정 질문에 근거 조문과 함께 답한다. 화면은 티타임(teachers-time.vercel.app)에 있고 이 Space 는 API 만 제공한다.

- 코퍼스: 법령 59 · 교육부 행정규칙 6 · 전남광주통합특별시교육청/(구)전라남도교육청 자치법규 30 (조문 4,299개, 국가법령정보센터 Open API)
- 검색: 문서·직종(교원/행정직/공무직)·설립(공립/사립) 마스크 → bge-m3 dense + BM25 → RRF → bge-reranker-v2-m3
- 답변: 가드레일 v3. LLM 은 조문에서 답이 되는 문장만 고르고, 서버가 그 문장이 원문에 있는지 검증한 뒤 원문만으로 답을 조립한다.
- 환경변수(Secrets): `RAG_KEY`(티타임과 공유하는 호출 키), `ANTHROPIC_API_KEY`(claude-haiku-4-5, 권장) 또는 `HF_TOKEN`(Qwen2.5-72B)

API
```
GET /ask?q=질문&doc=문서명&audience=교원|행정직|공무직&applies_to=공립|사립&k=5   (헤더 X-RAG-KEY)
GET /orgs      문서명 목록
GET /health
```
