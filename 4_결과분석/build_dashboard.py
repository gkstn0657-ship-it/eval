# -*- coding: utf-8 -*-
"""3_성능테스트/results 를 읽어 4_결과분석/dashboard.html 을 만든다. 문장(narrative.json)은 결과를 본 뒤 사람이 쓴다."""
import json, sys, html
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
R = ROOT / "3_성능테스트" / "results"
summary = json.load(open(R / "summary.json", encoding="utf-8"))
per_query = json.load(open(R / "per_query.json", encoding="utf-8"))
corpora = json.load(open(ROOT / "3_성능테스트" / "corpora" / "corpora_stats.json", encoding="utf-8"))
narr = json.load(open(HERE / "narrative.json", encoding="utf-8"))
LOCAL = "--local" in sys.argv
PIPE_NAME = {"vanilla": "Vanilla RAG", "vanilla_bge": "Vanilla (임베딩 bge-m3)", "hybrid": "하이브리드 + CE", "baseline_dense": "dense 단독", "baseline_bm25": "BM25 단독", "baseline_random": "random"}
LEVEL_NAME = {"L0": "L0 파싱만", "L1": "L1 노이즈 제거", "L2": "L2 구조 분리"}
TYPE_NAME = {"Q1": "조항 조회", "Q2": "요건 판단", "Q3": "수치 확인", "Q4": "절차·기한", "Q5": "용어 정의", "Q6": "기관 비교", "Q7": "개정 이력", "Q8": "별표", "Q10": "삭제 조항", "Q11": "일상어"}
cond = {(c["corpus"], c["pipeline"]): c for c in summary["conditions"]}
bytype = {(c["corpus"], c["pipeline"], c["type"]): c for c in summary["by_type"]}
def pct(x): return "–" if x is None else f"{x*100:.0f}"
def f3(x): return "–" if x is None else f"{x:.3f}"

# ---- 차트 1: 정제 강도별 nDCG@5 (정책 A), 파이프라인 2개 + dense 단독
def bar_group_svg(groups, series, getv, w=640, h=260):
    """groups: [(key,label)], series: [(key,label,cssvar)]. 가로 축 그룹, 막대 묶음. 값 0~1."""
    pad_l, pad_b, pad_t = 36, 44, 12
    gw = (w - pad_l - 12) / len(groups); bw = min(28, (gw - 16) / len(series))
    out = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="막대 차트" style="width:100%;height:auto;max-width:100%">']
    for y in (0, .25, .5, .75, 1):
        yy = pad_t + (1 - y) * (h - pad_t - pad_b)
        out.append(f'<line x1="{pad_l}" x2="{w-8}" y1="{yy:.1f}" y2="{yy:.1f}" stroke="var(--grid)" stroke-width="1"/>')
        out.append(f'<text x="{pad_l-6}" y="{yy+4:.1f}" text-anchor="end" font-size="11" fill="var(--muted)">{y:.2f}</text>')
    for gi, (gk, gl) in enumerate(groups):
        x0 = pad_l + gi * gw + (gw - bw * len(series) - 4 * (len(series) - 1)) / 2
        for si, (sk, sl, var) in enumerate(series):
            v = getv(gk, sk) or 0
            bh = v * (h - pad_t - pad_b); x = x0 + si * (bw + 4); y = pad_t + (h - pad_t - pad_b) - bh
            out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{bh:.1f}" rx="3" fill="var({var})"><title>{html.escape(gl)} · {html.escape(sl)}: {v:.3f}</title></rect>')
            out.append(f'<text x="{x+bw/2:.1f}" y="{y-4:.1f}" text-anchor="middle" font-size="11" fill="var(--fg2)">{v:.2f}</text>')
        out.append(f'<text x="{pad_l + gi*gw + gw/2:.1f}" y="{h-pad_b+18}" text-anchor="middle" font-size="12" fill="var(--fg2)">{html.escape(gl)}</text>')
    out.append(f'<line x1="{pad_l}" x2="{w-8}" y1="{h-pad_b}" y2="{h-pad_b}" stroke="var(--axis)"/>')
    out.append("</svg>")
    return "".join(out)

SERIES = [("hybrid", "하이브리드 + CE", "--s1"), ("vanilla", "Vanilla RAG", "--s2")]
chart_levels = bar_group_svg([(f"{l}_A", LEVEL_NAME[l]) for l in ("L0", "L1", "L2")], SERIES, lambda g, s: cond[(g, s)]["ndcg5"])
chart_policy = bar_group_svg([("L2_A", "A 최신판만"), ("L2_B", "B 전체 개정판")], SERIES, lambda g, s: cond[(g, s)]["ndcg5"])

# ---- 유형별 히트맵 표 (L2_A, Recall@5)
types = [t for t in TYPE_NAME if any(k[2] == t for k in bytype)]
def heat(v):
    if v is None: return ""
    step = min(6, int(v * 6.99)); return f"background:var(--seq{step});color:{'var(--fg-on-dark)' if step>=4 else 'var(--fg)'}"
rows_type = []
for t in types:
    cells = "".join(f'<td style="{heat(bytype.get(("L2_A",p,t),{}).get("recall5"))}">{pct(bytype.get(("L2_A",p,t),{}).get("recall5"))}</td>' for p in ("hybrid", "vanilla", "baseline_bm25"))
    n = bytype.get(("L2_A", "hybrid", t), {}).get("n", "")
    rows_type.append(f"<tr><th scope='row'>{t} {TYPE_NAME[t]}<span class='n'>n={n}</span></th>{cells}</tr>")

# ---- 조건표
rows_cond = []
for c in summary["conditions"]:
    if c["pipeline"] == "baseline_random": continue
    lv, pol = c["corpus"].split("_")
    rows_cond.append(f"<tr><td>{LEVEL_NAME[lv]}</td><td>{'A 최신판만' if pol=='A' else 'B 전체'}</td><td>{PIPE_NAME[c['pipeline']]}</td><td class='num'>{pct(c['recall5'])}%</td><td class='num'>{f3(c['mrr'])}</td><td class='num'><b>{f3(c['ndcg5'])}</b></td></tr>")
rand = [c for c in summary["conditions"] if c["pipeline"] == "baseline_random"]
rand_max = max(c["ndcg5"] for c in rand)

# ---- 비교표
rows_cmp = []
for c in summary["comparisons"]:
    sig = '<span class="pill ok">유의</span>' if c["significant"] else '<span class="pill">비유의</span>'
    rows_cmp.append(f"<tr><td>{html.escape(c['label'])}</td><td class='num'>{c['delta_ndcg5']:+.3f}</td><td class='num'>[{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]</td><td>{sig}</td></tr>")

# ---- Q9 거절
rows_q9 = "".join(f"<tr><td>{c['corpus']}</td><td>{PIPE_NAME[c['pipeline']]}</td><td class='num'>{pct(c['abstain_rate'])}%</td></tr>" for c in summary["q9"] if c["corpus"] in ("L0_A", "L2_A") and c["pipeline"] != "baseline_dense")

# ---- 실패 사례 (L2_A hybrid, 정답 못 찾은 질의)
gold = {q["qid"]: q for q in (json.loads(l) for l in (ROOT / "1_데이터셋/06_eval/goldenset_dev.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}
fails = [r for r in per_query if r["corpus"] == "L2_A" and r["pipeline"] == "hybrid" and r.get("recall5") == 0.0]
rows_fail = "".join(f"<tr><td>{r['qid']}</td><td>{html.escape(gold[r['qid']]['query'])}</td><td>{html.escape(r['gold'][0].split('|')[0])} {html.escape(r['gold'][0].split('|')[2])}</td><td>{html.escape(r['top5'][0].split('|')[0] + ' ' + r['top5'][0].split('|')[2]) if r['top5'] else ''}</td><td>{r.get('first_rank') or '>10'}</td></tr>" for r in fails)


SEARCH_PANEL = """
<section id="search"><h2>직접 검색해 보기</h2><p class="lead">파이프라인을 고르고 질의를 던져 상위 5개 조문을 본다. 측정에 쓴 코드와 같은 구현이다.</p>
<div class="card">
<form id="sf" class="sform"><input id="sq" type="text" placeholder="예: 가스공사는 몇 살까지 다닐 수 있어?" autocomplete="off" required>
<select id="sp" aria-label="파이프라인 선택"><option value="hybrid">하이브리드 + CE (hybrid-search-eval)</option><option value="vanilla">Vanilla RAG (chat_rag)</option><option value="vanilla_bge">Vanilla RAG, 임베딩만 bge-m3 (참고)</option></select>
<button type="submit" id="sb">검색</button></form>
<p class="note" id="sinfo">질의를 입력하고 검색을 누르면 결과가 아래에 나타납니다.</p>
<div id="sans" class="ans" hidden><div class="alabel">답변 <span id="amodel"></span></div><div id="atext"></div></div>
<div id="sres" class="cols"></div>
</div></section>
<style>
.sform{display:flex;gap:8px;flex-wrap:wrap} .sform input{flex:1 1 320px;min-width:0;padding:8px 10px;border:1px solid var(--axis);border-radius:6px;background:var(--bg);color:var(--fg);font:inherit}
.sform select,.sform button{padding:8px 10px;border:1px solid var(--axis);border-radius:6px;background:var(--bg);color:var(--fg);font:inherit} .sform button{background:var(--s1);color:#fff;border-color:var(--s1);cursor:pointer}
.pipes{display:flex;gap:14px;flex-wrap:wrap;align-items:center;margin-top:10px;font-size:.9rem} .pipes .plabel{color:var(--muted);font-size:.8rem;letter-spacing:.04em} .pipes label{display:flex;gap:6px;align-items:center;cursor:pointer}
.cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px;margin-top:12px} .col h3{font-size:.95rem;margin:0 0 6px} .col .meta{font-size:.78rem;color:var(--muted);margin-bottom:6px}
.hit{border-top:1px solid var(--grid);padding:8px 0} .hit .h{font-size:.85rem;font-weight:600} .hit .h .org{color:var(--fg2);font-weight:400} .hit .t{font-size:.82rem;color:var(--fg2);margin-top:2px;max-height:5.2em;overflow:hidden}
.hit.abst{opacity:.6}
.ans{margin-top:12px;padding:12px 14px;border:1px solid var(--s1);border-radius:8px;background:var(--bg)} .ans .alabel{font-size:.8rem;letter-spacing:.04em;color:var(--s1);font-weight:600;margin-bottom:6px} .ans .alabel span{color:var(--muted);font-weight:400} #atext{white-space:pre-wrap;font-size:.92rem}
</style>
<script>
(function(){
 const f=document.getElementById('sf'),q=document.getElementById('sq'),sp=document.getElementById('sp'),ans=document.getElementById('sans'),atext=document.getElementById('atext'),amodel=document.getElementById('amodel'),info=document.getElementById('sinfo'),res=document.getElementById('sres'),b=document.getElementById('sb');
 const esc=s=>String(s).replace(/[&<>"]/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[m]));
 f.addEventListener('submit',async e=>{e.preventDefault();b.disabled=true;info.textContent='검색 중…';res.innerHTML='';ans.hidden=true;atext.textContent='';
  try{const r=await fetch('/search?q='+encodeURIComponent(q.value)+'&corpus=L2_A&k=5&pipelines='+sp.value);if(!r.ok)throw new Error(r.status);const d=await r.json();
   const total=Object.values(d.pipelines).reduce((a,p)=>a+p.hits.length,0);info.textContent='기관 필터: '+(d.org_detected||'없음 (질의에 기관명 없음)')+' · 문서: 부칙·개정표기 분리본, 최신판만 · 결과 '+total+'건';
   res.innerHTML=Object.values(d.pipelines).map(p=>{const i=p.info;const meta=('abstain' in i)?('1위 점수 '+(i.top1_score??i.top1_ce)+(i.abstain?' · 임계값 미달, 거절':'')):('1위 CE 점수 '+i.top1_ce+(i.abstain?' · 거절':''));
    return '<div class="col"><h3>'+esc(p.label)+'</h3><div class="meta">'+esc(p.desc||'')+'</div><div class="meta">'+esc(meta)+'</div>'+(p.hits.length?'':'<div class="hit">검색 결과 없음</div>')+p.hits.map(h=>'<div class="hit'+(i.abstain?' abst':'')+'"><div class="h">'+h.rank+'. '+esc(h.article_no)+(h.title?'('+esc(h.title)+')':'')+' <span class="org">'+esc(h.org)+' · '+esc(h.revision_date)+'</span></div><div class="t">'+esc(h.text)+'</div></div>').join('')+'</div>';}).join('');
   ans.hidden=false;amodel.textContent='생성 중… (로컬 qwen2.5:7b-instruct, 15~30초)';
   try{const a=await fetch('/answer?q='+encodeURIComponent(q.value)+'&pipeline='+sp.value+'&corpus=L2_A&k=5');const ad=await a.json();if(ad.error){atext.textContent=ad.error;}else{amodel.textContent=ad.model+(ad.abstain?' · 검색 임계값 미달이라 참고 문서 없이 답함':'');atext.textContent=ad.answer;}}catch(e2){atext.textContent='답변 생성 실패: '+e2;}
  }catch(err){info.textContent='검색 서버에 연결할 수 없습니다. 로컬에서 python 4_결과분석/serve.py 를 실행한 뒤 http://127.0.0.1:8765 로 여세요.';}
  finally{b.disabled=false;}});
})();
</script>
"""

kpi = narr["kpi"]
def section(title, lead, body): return f"<section><h2>{html.escape(title)}</h2><p class='lead'>{html.escape(lead)}</p>{body}</section>"

page = f"""<title>{html.escape(narr['title'])}</title>
<style>
/* 레이아웃: 상단 결론 4타일 → 정제 강도 차트 → 개정판 정책 차트 → 유형별 히트맵 → 조건 전체표 → 통계 비교 → 실패 사례. 단일 컬럼, 최대 960px. */
:root{{--bg:#f7f7f4;--card:#fcfcfb;--fg:#0b0b0b;--fg2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--line:rgba(11,11,11,.10);
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--ok:#0ca30c;--fg-on-dark:#fff;
--seq0:#f0efec;--seq1:#cde2fb;--seq2:#9ec5f4;--seq3:#6da7ec;--seq4:#3987e5;--seq5:#256abf;--seq6:#184f95}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#0d0d0d;--card:#1a1a19;--fg:#fff;--fg2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--line:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--s3:#199e70;--seq0:#383835;--seq1:#184f95;--seq2:#1c5cab;--seq3:#256abf;--seq4:#2a78d6;--seq5:#3987e5;--seq6:#6da7ec;color-scheme:dark}}}}
:root[data-theme="dark"]{{--bg:#0d0d0d;--card:#1a1a19;--fg:#fff;--fg2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--line:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--s3:#199e70;--seq0:#383835;--seq1:#184f95;--seq2:#1c5cab;--seq3:#256abf;--seq4:#2a78d6;--seq5:#3987e5;--seq6:#6da7ec;color-scheme:dark}}
body{{background:var(--bg);color:var(--fg);font-family:system-ui,-apple-system,"Segoe UI","Malgun Gothic",sans-serif;line-height:1.55;padding-block:24px;padding-inline:16px}}
main{{max-width:960px;margin:0 auto;display:grid;gap:28px}}
h1{{font-size:1.6rem;margin:0 0 4px;text-wrap:balance}} h2{{font-size:1.15rem;margin:0 0 6px}} .sub{{color:var(--fg2);margin:0}}
.lead{{margin:0 0 12px;color:var(--fg)}} .note{{color:var(--fg2);font-size:.9rem;margin:8px 0 0}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}}
.tile{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:14px 16px}} .tile .v{{font-size:1.5rem;font-weight:600;line-height:1.1}} .tile .k{{font-size:.8rem;letter-spacing:.04em;text-transform:uppercase;color:var(--muted)}} .tile .d{{font-size:.9rem;color:var(--fg2);margin-top:4px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:14px 16px}}
.legend{{display:flex;gap:16px;flex-wrap:wrap;font-size:.85rem;color:var(--fg2);margin:0 0 6px}} .legend i{{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:middle}}
.tw{{overflow-x:auto}} table{{border-collapse:collapse;width:100%;font-size:.9rem}} th,td{{padding:6px 8px;border-bottom:1px solid var(--grid);text-align:left;vertical-align:top}} th{{color:var(--fg2);font-weight:600}} td.num,th.num{{text-align:right;font-variant-numeric:tabular-nums}} .n{{color:var(--muted);font-size:.78rem;margin-left:6px}}
.pill{{display:inline-block;padding:1px 8px;border-radius:999px;font-size:.78rem;border:1px solid var(--line);color:var(--fg2)}} .pill.ok{{border-color:var(--ok);color:var(--ok)}}
ul{{margin:8px 0 0;padding-left:18px}} li{{margin:2px 0}}
footer{{color:var(--muted);font-size:.8rem}}
</style>
<main>
<header><h1>{html.escape(narr['title'])}</h1><p class="sub">{html.escape(narr['subtitle'])}</p></header>
{SEARCH_PANEL if LOCAL else ""}
<div class="tiles">
{''.join(f'<div class="tile"><div class="k">{html.escape(k["k"])}</div><div class="v">{html.escape(k["v"])}</div><div class="d">{html.escape(k["d"])}</div></div>' for k in kpi)}
</div>
{section("정제 강도별 검색 성능 (최신판만, nDCG@5)", narr["levels"], '<div class="card"><div class="legend"><span><i style="background:var(--s1)"></i>하이브리드 + CE</span><span><i style="background:var(--s2)"></i>Vanilla RAG</span></div>' + chart_levels + f'<p class="note">{html.escape(narr["levels_note"])}</p></div>')}
{section("개정판 정책 비교 (L2, nDCG@5)", narr["policy"], '<div class="card"><div class="legend"><span><i style="background:var(--s1)"></i>하이브리드 + CE</span><span><i style="background:var(--s2)"></i>Vanilla RAG</span></div>' + chart_policy + f'<p class="note">{html.escape(narr["policy_note"])}</p></div>')}
{section("질의 유형별 Recall@5 (L2 최신판)", narr["types"], '<div class="card tw"><table><thead><tr><th>유형</th><th>하이브리드 + CE</th><th>Vanilla RAG</th><th>BM25 단독</th></tr></thead><tbody>' + ''.join(rows_type) + '</tbody></table>' + f'<p class="note">{html.escape(narr["types_note"])}</p></div>')}
{section("통계 비교 (쌍대 부트스트랩 95% CI, nDCG@5)", narr["stats"], '<div class="card tw"><table><thead><tr><th>비교</th><th class="num">Δ</th><th class="num">95% CI</th><th>판정</th></tr></thead><tbody>' + ''.join(rows_cmp) + '</tbody></table></div>')}
{section("범위 외 질의 거절률 (Q9, n=4)", narr["q9"], '<div class="card tw"><table><thead><tr><th>코퍼스</th><th>파이프라인</th><th class="num">거절률</th></tr></thead><tbody>' + rows_q9 + '</tbody></table></div>')}
{section("하이브리드 + CE 가 상위 5개 안에 정답을 못 넣은 질의 (L2 최신판)", narr["fails"], '<div class="card tw"><table><thead><tr><th>qid</th><th>질의</th><th>정답</th><th>1위로 찾은 것</th><th>정답 순위</th></tr></thead><tbody>' + (rows_fail or '<tr><td colspan="5">없음</td></tr>') + '</tbody></table></div>')}
{section("전체 조건 (36개 중 random 제외)", narr["all"], '<div class="card tw"><table><thead><tr><th>정제</th><th>개정판</th><th>파이프라인</th><th class="num">Recall@5</th><th class="num">MRR</th><th class="num">nDCG@5</th></tr></thead><tbody>' + ''.join(rows_cond) + '</tbody></table>' + f'<p class="note">random 베이스라인 nDCG@5 최대 {rand_max:.3f}. 측정 질의 {summary["n_scored"]}개(범위 외 4개 제외), dev 세트. holdout 44개는 봉인 상태.</p></div>')}
{section("부가: Vanilla 의 임베딩만 하이브리드와 같은 bge-m3 로 바꾸면", narr["embed"], '<div class="card tw"><table><thead><tr><th>Vanilla 임베딩 (L2 최신판)</th><th class="num">Recall@5</th><th class="num">MRR</th><th class="num">nDCG@5</th></tr></thead><tbody>' + ''.join(f"<tr><td>{PIPE_NAME[p]}</td><td class='num'>{pct(cond[('L2_A',p)]['recall5'])}%</td><td class='num'>{f3(cond[('L2_A',p)]['mrr'])}</td><td class='num'><b>{f3(cond[('L2_A',p)]['ndcg5'])}</b></td></tr>" for p in ("vanilla","vanilla_bge")) + f"<tr><td>하이브리드 + CE (참고)</td><td class='num'>{pct(cond[('L2_A','hybrid')]['recall5'])}%</td><td class='num'>{f3(cond[('L2_A','hybrid')]['mrr'])}</td><td class='num'><b>{f3(cond[('L2_A','hybrid')]['ndcg5'])}</b></td></tr>" + '</tbody></table>' + f'<p class="note">{html.escape(narr["embed_note"])}</p></div>')}
<section><h2>한계</h2><ul>{''.join(f'<li>{html.escape(x)}</li>' for x in narr['limits'])}</ul></section>
<footer>데이터: 공기업 인사규정 41건(ALIO). 코퍼스 {', '.join(c['corpus']+' '+str(c['chunks'])+'청크' for c in corpora)}. 사전등록 2026-10-01.</footer>
</main>"""
out = HERE / ("dashboard_local.html" if LOCAL else "dashboard.html"); out.write_text(page, encoding="utf-8"); print(out.name, len(page), "chars")
