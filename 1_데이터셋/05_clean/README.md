# 05_clean: 정제 결과

`scripts/clean.py`가 `04_정제규칙/정제규칙.md`의 R-01~R-14를 01_raw 41파일에 적용한 결과. 세 수준을 각각 산출한다.

## 산출 구조

```
05_clean/
  L0/<세트>/<기관>/<파일>.txt            파싱만. 표 셀 보존, 성명 마스킹
                                .articles.jsonl   조 단위 (느슨한 분리기, 부칙 혼입 상태 그대로)
                                .meta.json        manifest 승계 메타데이터 + is_latest
  L1/...                                 노이즈 제거. meta에 toc, page_breaks
  L2/...                                 본문만. meta에 preamble, appendix_blocks, effective_dates,
                                         delegated_to, glossary. articles에 revision_marks
  L2/index_A_latest_only.jsonl           R-10 정책 A: 기관별 최신판 조문만
  L2/index_B_all_revisions.jsonl         R-10 정책 B: 전체 개정판 조문 (is_latest 로 구분)
  L2/glossary.jsonl                      정의 조항에서 추출한 용어 (R-14)
  L2/synonyms_seed.jsonl                 일상어-규정어 대응 시드 12건, 사람 작성 (R-14)
  _images/                               HWP BinData 에서 추출한 이미지 표 원본(도로공사 4장)
  clean_report.json                      파일별 수치
```

조문 레코드 필드: `id`(기관|개정일|조번호), `article_no`, `article_title`, `section_path`, `text`, `org`, `org_type`, `revision_date`, `is_latest`, `source_file`, L2는 `revision_marks` 추가.

## 수준별 결과

| 수준 | 글자 수 | 조문 수 | 적용 규칙 | 비고 |
|---|---|---|---|---|
| L0 | 969,294 | 3,083 | R-01, R-11, R-12 | raw 추출(898,338자)보다 늘어남. 표 셀 텍스트 복원분 |
| L1 | 949,079 | 3,077 | + R-02, R-05~R-09, R-14 제목 | 머리글·쪽번호 140줄, 목차 107줄 제거, 번호·문자 정규화, 이미지 표 4건 전사본 삽입 |
| L2 | 661,496 | 2,600 | + R-03, R-04, R-10, R-13, R-14 사전 | 부칙 1,253블록(시행일 1,105건) 분리, 개정 표기 6,633개 분리(조문별 revision_marks 포함) |

L0→L2에서 본문이 32% 줄었다. 줄어든 분량은 삭제가 아니라 meta.json의 부칙·개정 표기·preamble로 이동했다.

- R-10 인덱스: A(최신판만) 1,214조, B(전체 개정판) 2,600조. B는 A의 2.1배이며 그 차이가 near-duplicate 조문이다.
- R-13 위임 규정: 파일당 2~31건 추출(시행세칙, 보수규정, 복무규정, 직제규정 등).
- R-14 용어: 324건. 정의 조항이 없는 기관(석유공사, 가스기술공사, 전력기술, 마사회, 가스공사)은 0.
- R-12 성명 마스킹: 한전KDN 102건. 다른 파일은 적용하지 않음(사람 확인 전).

## 확인된 한계

- **PDF 2단 레이아웃**: 한국마사회 PDF는 두 단의 줄이 섞여 추출된다(`제6장 신분보장 할 경우에는 「근로기준법」에…`). 개정 표기가 줄을 넘어 걸치고 조문 분리가 흔들린다. L1 123조 → L2 65조로 줄어든 이유의 일부. 레이아웃 인식 파서(열 분리)가 필요하다.
- **개정 공고문 선행 문서**: 토지주택공사 2024년판은 규정 본문 앞에 개정 공고·신구 대비표가 붙어 있다. `preamble`로 분리했으나 대비표 내용 자체는 검색 대상에서 빠진다.
- **개정 표기 없는 PDF**: 전력기술·에스알은 본문에 개정 표기가 없다(개정 이력을 부칙 또는 이력표로만 관리). Q7 개정 이력 질의는 부칙의 시행일로만 답할 수 있다.
- **이미지 표**: 실제 이미지는 도로공사 4장만(두 개정판 동일). `04_정제규칙/manual_transcriptions.jsonl`의 사람 전사본으로 치환하고 출처를 meta.json `image_transcriptions`에 남겼다. 주택도시보증공사의 `<그림>` 3개는 선 도형이라 제거. 전사본은 부칙 안에 있어 L2에서는 appendix_blocks로 이동한다.
- **표 직렬화**: `헤더: 값 | 헤더: 값` 한 가지 형식만 산출. 임베딩 모델별 비교는 성능 테스트에서 한다.
- **느슨한 조 분리기(L0)**: 부칙 안의 `제1조(시행일)`도 조문으로 센다. L0 3,083조 중 상당수가 부칙 조문이며, 이것이 L0와 L2를 비교할 때 보여야 하는 차이다.

## 재실행

```bash
cd 1_데이터셋
python scripts/clean.py      # hwp5txt, hwp5proc(pyhwp), pdfplumber 필요
```
