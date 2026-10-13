# -*- coding: utf-8 -*-
"""교원·교직원 규정 코퍼스(1_데이터셋/05_clean/L2_교원/corpus.jsonl) 검색 점검 (2026-10-13).

run_eval.py 는 손대지 않고, 코퍼스를 corpora/TEACH_A.jsonl 로 복사해 같은 경로(bge-m3 + BM25 → RRF → bge-reranker)로
질문 12개의 상위 5개 조문을 본다. 골든셋 전 '검색이 말이 되는가' 확인용이며 지표는 내지 않는다.
산출: results_teacher_smoke/smoke.json, smoke.md
실행: PYTHONIOENCODING=utf-8 python smoke_teacher_corpus.py
"""
import os, sys, json, shutil, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
# run_eval 은 E:\hf_cache 를 기본으로 잡는다. 이 PC 는 ~/.cache/huggingface 에 모델이 있다.
os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface")); os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(HERE))
import run_eval as E  # noqa: E402

# run_eval.st_model 은 VRAM 3GB 기준으로 임베딩을 CPU 에 둔다. 이 PC(8GB)는 GPU 로 임베딩해 색인을 빠르게 한다. run_eval 은 손대지 않는다.
def _st_model_gpu(name):
    if name not in E._models:
        from sentence_transformers import SentenceTransformer
        E._models[name] = SentenceTransformer(name, device=E._device())
    return E._models[name]
E.st_model = _st_model_gpu

SRC = ROOT / "1_데이터셋" / "05_clean" / "L2_교원" / "corpus.jsonl"
DST = HERE / "corpora" / "TEACH_A.jsonl"
OUT = HERE / "results_teacher_smoke"; OUT.mkdir(exist_ok=True)

# (질문, 기관 필터(문서명) 또는 None, 기대 조문 메모)
QUERIES = [
    ("교사 육아휴직은 자녀가 몇 살까지 가능해?", None, "교육공무원법 제44조"),
    ("육아휴직은 자녀가 몇 살까지 가능해?", "교육공무원법", "교육공무원법 제44조 (필터)"),
    ("교원 연가 일수는 재직기간별로 며칠이야?", None, "국가공무원 복무규정 제15조"),
    ("교사가 외부 강의를 하려면 미리 신고해야 해?", None, "청탁금지법 제10조 / 공무원 행동강령"),
    ("기간제교원 임용 기간은 최대 몇 년이야?", None, "교육공무원법 제32조"),
    ("학교폭력 신고를 받으면 학교장은 언제까지 교육청에 보고해야 해?", None, "학교폭력예방법 시행령"),
    ("교육활동 침해 학생에게 할 수 있는 조치는 뭐가 있어?", None, "교원지위법 제25조"),
    ("행정실 직원 병가는 1년에 며칠까지 쓸 수 있어?", None, "지방공무원 복무규정 / 전남 복무 조례"),
    ("병가는 1년에 며칠까지 쓸 수 있어?", "전라남도교육감 소속 지방공무원 복무 조례", "전남 복무 조례 (필터)"),
    ("교사 정년은 몇 살이야?", None, "교육공무원법 제47조"),
    ("교원 특별휴가는 어떤 경우에 며칠 받아?", None, "교원휴가에 관한 예규"),
    ("학교운영위원회 위원 임기는 몇 년이야?", None, "전라남도립학교 운영위원회 조례"),
    ("공무원이 받아도 되는 선물 금액 한도는 얼마야?", None, "청탁금지법 제8조"),
    ("우리 반 학생 생활기록부 출결은 어떻게 적어?", None, "학교생활기록 작성 및 관리지침"),
]

def main():
    shutil.copyfile(SRC, DST)
    t0 = time.time(); c = E.Corpus("TEACH_A"); c.bge; c.bm25; E.ce_model()
    print(f"코퍼스 {len(c.rows)}청크 / 문서 {len(c.org_names)}개, 색인 {time.time()-t0:.0f}초", flush=True)
    results = []
    for q, org, note in QUERIES:
        E._CTX["org"] = org
        t = time.time(); idx, info = E.run_hybrid_prefilter(c, q); sec = round(time.time() - t, 2)
        E._CTX["org"] = None
        seen, hits = set(), []
        for i in idx:
            r = c.rows[i]
            if r["article_id"] in seen: continue
            seen.add(r["article_id"])
            hits.append({"doc": r["org"], "article": r["article_no"], "title": r.get("article_title", ""), "audience": r.get("audience"),
                         "text": r["text"][:160].replace("\n", " ")})
            if len(hits) >= 5: break
        results.append({"q": q, "org": org, "expect": note, "sec": sec, "top1_ce": info.get("top1_ce"), "hits": hits})
        print(f"[{sec}s ce={info.get('top1_ce')}] {q} -> {hits[0]['doc']} {hits[0]['article']}({hits[0]['title']})", flush=True)
    (OUT / "smoke.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    md = ["# 교원 코퍼스 검색 점검 (hybrid_prefilter, 상위 5 조문)", "", f"코퍼스 {len(c.rows)}청크 / 문서 {len(c.org_names)}개", ""]
    for r in results:
        md += [f"## {r['q']}" + (f"  (필터: {r['org']})" if r["org"] else ""), f"기대: {r['expect']} · {r['sec']}s · top1_ce {r['top1_ce']}", ""]
        md += [f"{i+1}. **{h['doc']} {h['article']}({h['title']})** [{h['audience']}] {h['text']}" for i, h in enumerate(r["hits"])] + [""]
    (OUT / "smoke.md").write_text("\n".join(md), encoding="utf-8")
    print("저장:", OUT / "smoke.md", flush=True)

if __name__ == "__main__":
    main()
