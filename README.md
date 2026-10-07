이전 프로젝트 (Dense 단독 검색 · Vanilla RAG) https://github.com/gkstn0657-ship-it/chat_rag
<br>
<br>
<br>
현 프로젝트 배포 링크: https://runningturtle123-public-agency-ragchat.hf.space/demo/

---

# 공공기관 인사규정 RAG 평가 저장소

공공기관 인사규정(ALIO 공개 원문)으로 검색 파이프라인을 같은 기준에서 측정하고, 결과를 근거로 설계를 결정한 기록이다. 진행 과정은 `worklog/` 와 커밋 메시지에, 결론은 `4_결과분석/` 에 있다.

## 저장소 구성

| 폴더 | 내용 |
|---|---|
| `1_데이터셋/` | 원문 수집(`01_raw`), 데이터 분석(`02_분석`), 페르소나·질의 유형(`03_질의정의`), 정제 규칙(`04_정제규칙`), 정제 결과(`05_clean`), 평가 세트(`06_eval`), 외부 질의(`07_외부질의`) |
| `2_파이프라인설계/` | (비어 있음) 파이프라인은 `pipelines/` 의 기존 레포를 그대로 쓴다 |
| `3_성능테스트/` | 코퍼스 생성, 측정 스크립트, 사전등록, 결과(`results*/`), 양자화(`quant/`) |
| `4_결과분석/` | 결과 분석 문서, 대시보드(`dashboard.html`), 선행 연구 대조, 청킹·기관 필터 근거 |
| `5_가드레일/` | 가드레일 정책 v2 와 하네스(`v2/`). 이전 가드레일은 폐기 |
| `5_회고/` | 회고 |
| `6_배포/` | HF Space 배포 묶음 (`app.py`, `demo.html`, `SPACE_README.md`, `build_space.py`) |
| `pipelines/` | 서브모듈: chat_rag(Vanilla), hybrid-search-eval(하이브리드+재랭커), multimodal-rag |
| `worklog/` | 날짜별 진행 기록, 인계 문서 |

## 1_데이터셋

기준 원문은 공공기관 인사규정 HWP·PDF 원본이다. 현업 문서 형태(표·부칙·개정판 혼재)에서 측정하기 위해 골랐다.

| 폴더 | 내용 | 원칙 |
|---|---|---|
| `01_raw/` | ALIO 원본. 처음 20개 기관 41건(`공기업_인사규정_HWP·PDF`), 확장 292개 기관 741건(`공공기관_인사규정_전체`) | 수정 금지. `manifest.jsonl` 에 출처·해시·수집일 |
| `02_분석/` | 형식·구조·노이즈·개정판 중복 분석 | 정제 규칙의 근거 |
| `03_질의정의/` | 페르소나 5종, 질의 유형 11종 | 정제·평가의 입력 |
| `04_정제규칙/` | 규칙 R-01~R-14, 정제 수준 L0/L1/L2, 이미지 표 전사본 | 규칙마다 근거와 실패 질의 명시 |
| `05_clean/` | 정제 결과. `L0/L1/L2/<세트>/<기관>/<파일>.{txt,articles.jsonl,meta.json}`, `L2/index_A_latest_only.jsonl`(최신판), `L2/index_B_all_revisions.jsonl`(전체 개정판), `L2/glossary.jsonl`, `L2/synonyms_seed.jsonl`, `_images/`, `clean_report.json` | 정제 수준은 실험 변수 |
| `06_eval/` | 골든셋 dev/holdout(봉인)·employee·kgs_faq·matrix 등, 구성 근거 `평가데이터셋_구성근거.md` | 측정 전 확정 |
| `07_외부질의/` | 인사혁신처 FAQ 579건과 선별본, 외부 AI 작성 질문 200건 | |

조문 레코드 필드: `id`(기관|개정일|조번호), `article_no`, `article_title`, `section_path`, `text`, `org`, `revision_date`, `is_latest`, `source_file`, L2 는 `revision_marks`.

재실행: `cd 1_데이터셋 && python scripts/clean.py` (hwp5txt, hwp5proc, pdfplumber 필요). 수집: `scripts/fetch_alio_hwp.py`.

## 3_성능테스트

- 코퍼스: `python build_corpora.py` → `corpora/{L0,L1,L2,L2P}_{A,B}.jsonl`. 청킹 비교용은 `build_chunk_variants.py`.
- 임베딩 캐시: `python precache_embeddings.py L2P_A` (GPU).
- 검색 측정: `python run_eval.py --corpora L2P_A --pipelines hybrid_prefilter [--gold <세트>] [--tag <이름>]` → `results_<이름>/summary.json, per_query.json`.
- 답변 생성 측정: `run_gen_eval.py`(기준), `run_gen_quant.py`(양자화 모델 비교). 양자화 절차는 `quant/`.
- 사전등록: `사전등록.md`, 각 실험 계획서(`*_실험계획.md`). 측정 전에 쓰고 결과를 본 뒤 바꾸지 않는다.

## 4_결과분석

`python build_dashboard.py [--local]` 로 `dashboard.html` 생성. 문장은 `narrative.json`. 검색 데모 서버는 `python serve.py`.

## 5_가드레일 (v2)

| 파일 | 역할 |
|---|---|
| `가드레일_정책_v2.md` | 정책과 검토. 이전 `정책.md` 는 폐기 |
| `v2/guard.py` | 하네스. 입력 규칙 → LLM 근거 추출(JSON) → 원문 대조 검증 → 처리 결정 → 템플릿 조립 → 최종 검사. 답변 본문은 조문 원문과 고정 문구로만 만들어진다 |
| `v2/test_guard.py` | 단위 테스트 (`cd 5_가드레일/v2 && python -m unittest test_guard -v`) |
| `v2/eval_guard.py` | 외부 질문 200건(사람 라벨)으로 측정. 설계용 104 / 봉인용 96(`split.json`). 봉인용은 `--sealed --confirm-open` 으로만 열리고 `sealed_open.log` 에 기록 |

## 6_배포

HF Space(`runningturtle123/public-agency-ragchat`). 검색은 기관 사전 필터 → bge-m3 + BM25 → RRF → bge-reranker-v2-m3, 답변 생성은 HF Inference API(`HF_TOKEN` 필요, 없으면 검색 근거만 표시). 업로드 묶음은 `python 6_배포/build_space.py` 로 `6_배포/space/` 에 조립한다. Space 카드(`SPACE_README.md`)는 조립 시 `README.md` 로 복사된다.

## 라이선스

원문은 ALIO 공개 내부규정이다. 공개 저장소에 포함하기 전 각 기관의 공공누리 유형 확인이 필요하며, 확인 전까지 원본 재배포 여부는 보류한다. 인사혁신처 FAQ 는 공공 FAQ 를 재구성한 것이다.
