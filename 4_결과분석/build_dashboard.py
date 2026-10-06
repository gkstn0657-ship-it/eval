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
def section(title, lead, body, data=""): return f"<section><h2>{html.escape(title)}</h2>" + (f"<p class='data'>{html.escape(data)}</p>" if data else "") + f"<p class='lead'>{html.escape(lead)}</p>{body}</section>"
D20 = "20개 기관"; DEV = "평가 dev 44문항(채점 40)"
PIPE_NAME = {"vanilla": "Vanilla RAG", "vanilla_bge": "Vanilla (임베딩 bge-m3)", "hybrid_nobm25": "하이브리드+CE − BM25", "hybrid_nofilter": "하이브리드+CE − 기관 필터", "vanilla_filter": "Vanilla + 기관 필터", "vanilla_bge_filter": "Vanilla (bge-m3) + 기관 필터", "hybrid": "하이브리드 + CE", "baseline_dense": "dense 단독", "baseline_bm25": "BM25 단독", "baseline_random": "random"}
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

# ---- 구성요소 분해 비교 (추가 조건이 측정돼 있을 때만)
ABL_HTML = ""
if ("L2_A", "hybrid_nobm25") in cond:
    ABL = [("hybrid", "하이브리드 + CE", "--s1"), ("hybrid_nobm25", "하이브리드+CE − BM25", "--s4")]
    if ("L2_A", "hybrid_nofilter") in cond: ABL += [("hybrid_nofilter", "하이브리드+CE − 기관 필터", "--s7")]
    _mf = ROOT / "3_성능테스트" / "results_meta_bm25" / "org20_dev_L2_A" / "summary.json"
    if _mf.exists():
        _m = json.load(open(_mf, encoding="utf-8"))
        cond[("L2_A", "meta_bm25")] = next(x for x in _m["conditions"] if x["cond"] == "M1_meta_concat")
        for x in _m["by_type"]:
            if x["cond"] == "M1_meta_concat": bytype[("L2_A", "meta_bm25", x["type"])] = x
        ABL += [("meta_bm25", "하이브리드+CE − 기관 필터, BM25 에 기관명", "--s8")]
    if ("L2_A", "vanilla_bge_filter") in cond: ABL += [("vanilla_bge_filter", "Vanilla(bge-m3) + 기관 필터", "--s5")]
    ABL += [("vanilla_bge", "Vanilla (bge-m3)", "--s3")]
    if ("L2_A", "vanilla_filter") in cond: ABL += [("vanilla_filter", "Vanilla + 기관 필터", "--s6")]
    ABL += [("vanilla", "Vanilla RAG", "--s2")]
    chart_abl = bar_group_svg([("L2_A", "L2 최신판")], ABL, lambda g, s: cond[(g, s)]["ndcg5"], w=640, h=280)
    abl_rows = "".join(f"<tr><th scope='row'>{t} {TYPE_NAME[t]}</th>" + "".join(f"<td class='num'>{pct(bytype.get(('L2_A',p,t),{}).get('recall5'))}</td>" for p, _, _ in ABL) + "</tr>" for t in types)
    ABL_HTML = section("구성요소 분해 비교 (nDCG@5, L2 최신판)", narr.get("ablation", ""), data=f"{D20} · 코퍼스 L2 최신판 1,233청크 · {DEV}", body=
        '<div class="card"><div class="legend">' + "".join(f'<span><i style="background:var({v})"></i>{html.escape(l)}</span>' for _, l, v in ABL) + '</div>' + chart_abl + f'<p class="note">{html.escape(narr.get("ablation_note", ""))}</p></div>'
        + '<div class="card tw" style="margin-top:12px"><table><thead><tr><th>유형 (Recall@5)</th>' + "".join(f"<th>{html.escape(l)}</th>" for _, l, _ in ABL) + '</tr></thead><tbody>' + abl_rows + '</tbody></table></div>')

# ---- 개선안 적용 결과 (results_improved 가 있을 때만)
gpath = R / "gen_summary.json"
IMP_HTML = ""
RI = ROOT / "3_성능테스트" / "results_improved"
if (RI / "summary.json").exists():
    S2 = json.load(open(RI / "summary.json", encoding="utf-8")); c2 = {(c["corpus"], c["pipeline"]): c for c in S2["conditions"]}; b2 = {(c["corpus"], c["pipeline"], c["type"]): c for c in S2["by_type"]}
    pr = [("hybrid", "하이브리드 + CE"), ("vanilla_bge_filter", "Vanilla(bge-m3) + 기관 필터"), ("vanilla", "Vanilla RAG")]
    trs = "".join(f"<tr><td>{l}</td><td class='num'>{f3(c2[('L2_A',k)]['ndcg5'])}</td><td class='num'><b>{f3(c2[('L2P_A',k)]['ndcg5'])}</b></td><td class='num'>{c2[('L2P_A',k)]['ndcg5']-c2[('L2_A',k)]['ndcg5']:+.3f}</td></tr>" for k, l in pr)
    cmps = {c["label"]: c for c in S2["comparisons"]}
    tt = "".join(f"<tr><th scope='row'>{t} {TYPE_NAME[t]}</th>" + "".join(f"<td class='num'>{pct(b2.get(('L2_A',k,t),{}).get('recall5'))} → <b>{pct(b2.get(('L2P_A',k,t),{}).get('recall5'))}</b></td>" for k, _ in pr) + "</tr>" for t in types)
    gen_imp = ""
    if (RI / "gen_summary.json").exists() and gpath.exists():
        G1 = json.load(open(RI / "gen_summary.json", encoding="utf-8"))["pipelines"]; G0 = json.load(open(gpath, encoding="utf-8"))["pipelines"]
        rows_g = [("총점 (100점)", "total_mean", "{:.1f}"), ("출처 조문 일치율", "citation_match_rate", None), ("범위 외 질의 거절률 (n=4)", "q9_refusal_rate", None), ("범위 외 질의에 소관 규정 안내 (n=4)", "q9_guides_rule_rate", None), ("범위 외 질의 환각률 (n=4)", "q9_hallucination_rate", None)]
        def fm(v, f): return "–" if v is None else (f.format(v) if f else pct(v) + "%")
        gen_imp = '<div class="card tw" style="margin-top:12px"><table><thead><tr><th>답변 생성 (하이브리드)</th><th class="num">개선 전</th><th class="num">개선 후</th><th>답변 생성 (Vanilla)</th><th class="num">개선 전</th><th class="num">개선 후</th></tr></thead><tbody>' + "".join(f"<tr><td>{lab}</td><td class='num'>{fm(G0['hybrid'].get(k),f)}</td><td class='num'><b>{fm(G1['hybrid'].get(k),f)}</b></td><td>{lab}</td><td class='num'>{fm(G0['vanilla'].get(k),f)}</td><td class='num'><b>{fm(G1['vanilla'].get(k),f)}</b></td></tr>" for lab, k, f in rows_g) + '</tbody></table>' + f'<p class="note">{html.escape(narr.get("improve_gen_note",""))}</p></div>'
    IMP_HTML = section("개선안 3건 적용 결과 (L2 → L2P, 최신판만)", narr.get("improve", ""), data=f"{D20} · 코퍼스 L2 1,233 → L2P 1,416청크 · {DEV}", body=
        '<div class="card tw"><table><thead><tr><th>파이프라인</th><th class="num">개선 전 nDCG@5</th><th class="num">개선 후</th><th class="num">Δ</th></tr></thead><tbody>' + trs + '</tbody></table>'
        + f'<p class="note">{html.escape(narr.get("improve_note",""))}</p></div>'
        + '<div class="card tw" style="margin-top:12px"><table><thead><tr><th>유형 (Recall@5, 전 → 후)</th>' + "".join(f"<th>{l}</th>" for _, l in pr) + '</tr></thead><tbody>' + tt + '</tbody></table></div>' + gen_imp)

# ---- 기관 필터 대체 실험: 메타데이터 BM25 (results_meta_bm25 가 있을 때만)
META_HTML = ""
RM = ROOT / "3_성능테스트" / "results_meta_bm25"
META_GROUPS = [("org292_dev", "292기관 · dev"), ("org292_employee", "292기관 · 직원 질문"), ("org20_dev", "20기관 · dev"), ("org20_employee", "20기관 · 직원 질문")]
META_SERIES = [("B1_hybrid_prefilter", "기관 필터(사전) + 하이브리드+CE", "--s1"), ("M1_meta_concat", "하이브리드+CE − 기관 필터, BM25 에 기관명", "--s8"), ("B0_hybrid_nofilter", "하이브리드+CE − 기관 필터", "--s7")]
msum = {g: json.load(open(RM / g / "summary.json", encoding="utf-8")) for g, _ in META_GROUPS if (RM / g / "summary.json").exists()}
if msum:
    mc = {(g, x["cond"]): x for g, sm in msum.items() for x in sm["conditions"]}
    groups = [(g, l) for g, l in META_GROUPS if g in msum]
    chart_meta = bar_group_svg(groups, META_SERIES, lambda g, s: (mc.get((g, s)) or {}).get("ndcg5"), w=760, h=300)
    head = "<tr><th>파이프라인</th>" + "".join(f"<th class='num'>{html.escape(l)}<br><span class='n'>nDCG@5 · 기관적중</span></th>" for _, l in groups) + "</tr>"
    body = "".join("<tr><td>" + html.escape(sl) + "</td>" + "".join(
        f"<td class='num'><b>{f3((mc.get((g, sk)) or {}).get('ndcg5'))}</b> · {pct((mc.get((g, sk)) or {}).get('org_hit5'))}%</td>" for g, _ in groups) + "</tr>" for sk, sl, _ in META_SERIES)
    def mcell(c): return "–" if not c else f"{c['delta']:+.3f} [{c['ci95'][0]:+.2f}, {c['ci95'][1]:+.2f}] " + ('<span class="pill ok">유의</span>' if c["significant"] else '<span class="pill">비유의</span>')
    cmp_rows = ""
    for lab_m in ("M1_meta_concat",):
        for lab_b, bname in (("B0_hybrid_nofilter", "필터 없음 대비"), ("B1_hybrid_prefilter", "기관 필터 대비")):
            cs = {g: next((c for c in msum[g]["comparisons"] if c["label"] == f"{lab_m} vs {lab_b}"), None) for g, _ in groups}
            cmp_rows += f"<tr><td>{html.escape(dict((k, l) for k, l, _ in META_SERIES)[lab_m])} · {bname}</td>" + "".join(f"<td>{mcell(cs[g])}</td>" for g, _ in groups) + "</tr>"
    META_HTML = section("기관 필터 대체 실험: 메타데이터를 BM25 에 넣으면 (nDCG@5, L2 + 개선안 최신판)", narr.get("meta", ""), data=f"292개 기관 · 코퍼스 L2P 최신판 22,336청크 · {DEV}", body=
        '<div class="card"><div class="legend">' + "".join(f'<span><i style="background:var({v})"></i>{html.escape(l)}</span>' for _, l, v in META_SERIES) + '</div>' + chart_meta + f'<p class="note">{html.escape(narr.get("meta_note", ""))}</p></div>'
        + '<div class="card tw" style="margin-top:12px"><table><thead>' + head + '</thead><tbody>' + body + '</tbody></table><p class="note">기관적중: 상위 5개 중 정답 기관 조문 비율.</p></div>'
        + '<div class="card tw" style="margin-top:12px"><table><thead><tr><th>비교 (Δ nDCG@5, 95% CI)</th>' + "".join(f"<th>{html.escape(l)}</th>" for _, l in groups) + '</tr></thead><tbody>' + cmp_rows + '</tbody></table></div>')

# ---- holdout 개봉 결과 (results_holdout 가 있을 때만)
HO_HTML = ""
RH = ROOT / "3_성능테스트" / "results_holdout"
if (RH / "summary.json").exists():
    SH = json.load(open(RH / "summary.json", encoding="utf-8")); ch = {(c["corpus"], c["pipeline"]): c for c in SH["conditions"]}
    # dev 쪽: L2P_A 는 results_improved 에, 나머지는 results 에
    cdev = dict(cond)
    if (RI / "summary.json").exists():
        for c in json.load(open(RI / "summary.json", encoding="utf-8"))["conditions"]: cdev[(c["corpus"], c["pipeline"])] = c
    rows_ho = [("L2_A", "hybrid"), ("L2_A", "hybrid_nofilter"), ("L2_A", "vanilla_bge_filter"), ("L2_A", "vanilla_bge"), ("L2_A", "vanilla"), ("L2P_A", "hybrid"), ("L0_A", "hybrid"), ("L2_B", "hybrid")]
    trs = "".join(f"<tr><td>{LEVEL_NAME.get(c.split('_')[0], 'L2 + 개선안')} · {'최신판만' if c.endswith('A') else '전체 개정판'}</td><td>{PIPE_NAME[p]}</td><td class='num'>{f3(cdev.get((c,p),{}).get('ndcg5'))}</td><td class='num'><b>{f3(ch.get((c,p),{}).get('ndcg5'))}</b></td><td class='num'>{(ch[(c,p)]['ndcg5']-cdev[(c,p)]['ndcg5']):+.3f}</td><td class='num'>{pct(ch.get((c,p),{}).get('recall5'))}%</td></tr>" for c, p in rows_ho if (c, p) in ch and (c, p) in cdev)
    cmph = {c["label"]: c for c in SH["comparisons"]}; cmpd = {c["label"]: c for c in summary["comparisons"]}
    keys = ["hybrid vs vanilla (L2_A)", "hybrid vs hybrid_nofilter (L2_A)", "vanilla_bge_filter vs vanilla_bge (L2_A)", "hybrid vs hybrid_nobm25 (L2_A)", "L2 vs L0, hybrid (A)", "B vs A, hybrid (L2)", "vanilla_bge vs vanilla (L2_A)"]
    def cell(c): return "–" if not c else f"{c['delta_ndcg5']:+.3f} [{c['ci95'][0]:+.2f}, {c['ci95'][1]:+.2f}] " + ('<span class="pill ok">유의</span>' if c["significant"] else '<span class="pill">비유의</span>')
    trc = "".join(f"<tr><td>{html.escape(k)}</td><td>{cell(cmpd.get(k))}</td><td>{cell(cmph.get(k))}</td></tr>" for k in keys if k in cmph or k in cmpd)
    HO_HTML = section("홀드아웃 개봉 (봉인했던 44문항, 1회 측정)", narr.get("holdout", ""), data=f"{D20} · 코퍼스 5종 1,233~2,624청크 · 평가 holdout 44문항(채점 40)", body=
        '<div class="card tw"><table><thead><tr><th>문서</th><th>파이프라인</th><th class="num">dev nDCG@5</th><th class="num">holdout</th><th class="num">Δ</th><th class="num">holdout Recall@5</th></tr></thead><tbody>' + trs + '</tbody></table>' + f'<p class="note">{html.escape(narr.get("holdout_note",""))}</p></div>'
        + '<div class="card tw" style="margin-top:12px"><table><thead><tr><th>비교 (Δ nDCG@5, 95% CI)</th><th>dev</th><th>holdout</th></tr></thead><tbody>' + trc + '</tbody></table></div>')

# ---- 답변 생성 품질 (results/gen_summary.json 이 있을 때만)
GEN_HTML = ""
gpath = R / "gen_summary.json"
if gpath.exists():
    G = json.load(open(gpath, encoding="utf-8")); gp = G["pipelines"]
    def gm(p, k, f="{:.0f}"): v = gp[p].get(k); return "–" if v is None else f.format(v)
    rows = [("총점 (100점, 범위 외 제외)", "total_mean", "{:.1f}"), ("정확성 /40", "accuracy", "{:.1f}"), ("완전성 /25", "completeness", "{:.1f}"), ("출처정확도 /20", "citation", "{:.1f}"), ("간결성 /15", "concise", "{:.1f}"),
            ("80점 이상 비율", "pass80_rate", None), ("환각 비율 (판정)", "hallucination_rate", None), ("출처 조문 일치율 (자동)", "citation_match_rate", None), ("기대 답 숫자 포함률 (자동)", "hint_numbers_covered", None),
            ("범위 외 질의 거절률 (n=4)", "q9_refusal_rate", None), ("범위 외 질의 환각률 (n=4)", "q9_hallucination_rate", None)]
    trs = "".join(f"<tr><td>{lab}</td><td class='num'>{gm('hybrid',k,f) if f else pct(gp['hybrid'].get(k))+'%'}</td><td class='num'>{gm('vanilla',k,f) if f else pct(gp['vanilla'].get(k))+'%'}</td></tr>" for lab, k, f in rows)
    miss = "".join(f"<tr><td>{PIPE_NAME[p]}</td><td class='num'>{gp[p]['answer_when_retrieval_missed']['n']}</td><td class='num'>{pct(gp[p]['answer_when_retrieval_missed']['hallucination_rate'])}%</td></tr>" for p in ("hybrid", "vanilla"))
    bt = {(x["pipeline"], x["type"]): x for x in G["by_type"]}
    tt = "".join(f"<tr><th scope='row'>{t} {TYPE_NAME.get(t, t)}</th>" + "".join(f"<td class='num'>{'–' if bt.get((p,t),{}).get('total_mean') is None else f'{bt[(p,t)][chr(116)+chr(111)+chr(116)+chr(97)+chr(108)+chr(95)+chr(109)+chr(101)+chr(97)+chr(110)]:.0f}'}</td><td class='num'>{pct(bt.get((p,t),{}).get('hallucination_rate'))}%</td>" for p in ("hybrid", "vanilla")) + "</tr>" for t in list(TYPE_NAME) + ["Q9"] if any(k[1] == t for k in bt))
    cmp_ = G["comparison"]
    GEN_HTML = section("답변 생성 품질 (로컬 qwen2.5:7b-instruct, dev 44문항)", narr.get("gen", ""), data=f"{D20} · 코퍼스 L2 최신판 1,233청크 · 평가 dev 44문항", body=
        '<div class="card tw"><table><thead><tr><th>지표</th><th class="num">하이브리드 + CE</th><th class="num">Vanilla RAG</th></tr></thead><tbody>' + trs + '</tbody></table>'
        + f"<p class='note'>총점 차이 {cmp_['delta']:+.1f}점, 95% CI [{cmp_['ci95'][0]:+.1f}, {cmp_['ci95'][1]:+.1f}] (n={cmp_['n']}). {html.escape(narr.get('gen_note',''))}</p></div>"
        + '<div class="card tw" style="margin-top:12px"><table><thead><tr><th>검색이 정답을 못 찾은 질의에서</th><th class="num">건수</th><th class="num">환각률</th></tr></thead><tbody>' + miss + '</tbody></table></div>'
        + '<div class="card tw" style="margin-top:12px"><table><thead><tr><th>유형</th><th class="num">하이브리드 총점</th><th class="num">환각</th><th class="num">Vanilla 총점</th><th class="num">환각</th></tr></thead><tbody>' + tt + '</tbody></table></div>')

kpi = narr["kpi"]

page = f"""<title>{html.escape(narr['title'])}</title>
<style>
/* 레이아웃: 상단 결론 4타일 → 정제 강도 차트 → 개정판 정책 차트 → 유형별 히트맵 → 조건 전체표 → 통계 비교 → 실패 사례. 단일 컬럼, 최대 960px. */
:root{{--bg:#f7f7f4;--card:#fcfcfb;--fg:#0b0b0b;--fg2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--line:rgba(11,11,11,.10);
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#4a3aa7;--s5:#e87ba4;--s6:#eda100;--s7:#e34948;--s8:#0f9bb3;--ok:#0ca30c;--fg-on-dark:#fff;
--seq0:#f0efec;--seq1:#cde2fb;--seq2:#9ec5f4;--seq3:#6da7ec;--seq4:#3987e5;--seq5:#256abf;--seq6:#184f95}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#0d0d0d;--card:#1a1a19;--fg:#fff;--fg2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--line:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#9085e9;--s5:#d55181;--s6:#c98500;--s7:#e66767;--s8:#3bb8cc;--seq0:#383835;--seq1:#184f95;--seq2:#1c5cab;--seq3:#256abf;--seq4:#2a78d6;--seq5:#3987e5;--seq6:#6da7ec;color-scheme:dark}}}}
:root[data-theme="dark"]{{--bg:#0d0d0d;--card:#1a1a19;--fg:#fff;--fg2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--line:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#9085e9;--s5:#d55181;--s6:#c98500;--s7:#e66767;--s8:#3bb8cc;--seq0:#383835;--seq1:#184f95;--seq2:#1c5cab;--seq3:#256abf;--seq4:#2a78d6;--seq5:#3987e5;--seq6:#6da7ec;color-scheme:dark}}
body{{background:var(--bg);color:var(--fg);font-family:system-ui,-apple-system,"Segoe UI","Malgun Gothic",sans-serif;line-height:1.55;padding-block:24px;padding-inline:16px}}
main{{max-width:960px;margin:0 auto;display:grid;gap:28px}}
h1{{font-size:1.6rem;margin:0 0 4px;text-wrap:balance}} h2{{font-size:1.15rem;margin:0 0 6px}} .sub{{color:var(--fg2);margin:0}}
.lead{{margin:0 0 12px;color:var(--fg)}} .data{{margin:0 0 6px;color:var(--muted);font-size:.82rem}} .note{{color:var(--fg2);font-size:.9rem;margin:8px 0 0}}
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
{section("정제 강도별 검색 성능 (최신판만, nDCG@5)", narr["levels"], data=f"{D20} · 코퍼스 L0 1,584 / L1 1,578 / L2 1,233청크 · {DEV}", body= '<div class="card"><div class="legend"><span><i style="background:var(--s1)"></i>하이브리드 + CE</span><span><i style="background:var(--s2)"></i>Vanilla RAG</span></div>' + chart_levels + f'<p class="note">{html.escape(narr["levels_note"])}</p></div>')}
{section("개정판 정책 비교 (L2, nDCG@5)", narr["policy"], data=f"{D20} · 코퍼스 L2 최신판 1,233 / 전체 개정판 2,624청크 · {DEV}", body= '<div class="card"><div class="legend"><span><i style="background:var(--s1)"></i>하이브리드 + CE</span><span><i style="background:var(--s2)"></i>Vanilla RAG</span></div>' + chart_policy + f'<p class="note">{html.escape(narr["policy_note"])}</p></div>')}
{section("질의 유형별 Recall@5 (L2 최신판)", narr["types"], data=f"{D20} · 코퍼스 L2 최신판 1,233청크 · {DEV}, 유형당 4문항", body= '<div class="card tw"><table><thead><tr><th>유형</th><th>하이브리드 + CE</th><th>Vanilla RAG</th><th>BM25 단독</th></tr></thead><tbody>' + ''.join(rows_type) + '</tbody></table>' + f'<p class="note">{html.escape(narr["types_note"])}</p></div>')}
{section("통계 비교 (쌍대 부트스트랩 95% CI, nDCG@5)", narr["stats"], data=f"{D20} · 코퍼스 L0~L2 · {DEV}", body= '<div class="card tw"><table><thead><tr><th>비교</th><th class="num">Δ</th><th class="num">95% CI</th><th>판정</th></tr></thead><tbody>' + ''.join(rows_cmp) + '</tbody></table></div>')}
{ABL_HTML}
{META_HTML}
{IMP_HTML}
{HO_HTML}
{GEN_HTML}
{section("범위 외 질의 거절률 (Q9, n=4)", narr["q9"], data=f"{D20} · 평가 dev 범위 외 4문항", body= '<div class="card tw"><table><thead><tr><th>코퍼스</th><th>파이프라인</th><th class="num">거절률</th></tr></thead><tbody>' + rows_q9 + '</tbody></table></div>')}
{section("하이브리드 + CE 가 상위 5개 안에 정답을 못 넣은 질의 (L2 최신판)", narr["fails"], data=f"{D20} · 코퍼스 L2 최신판 1,233청크 · {DEV}", body= '<div class="card tw"><table><thead><tr><th>qid</th><th>질의</th><th>정답</th><th>1위로 찾은 것</th><th>정답 순위</th></tr></thead><tbody>' + (rows_fail or '<tr><td colspan="5">없음</td></tr>') + '</tbody></table></div>')}
{section("전체 조건 (36개 중 random 제외)", narr["all"], data=f"{D20} · 코퍼스 6종 1,233~3,228청크 · {DEV}", body= '<div class="card tw"><table><thead><tr><th>정제</th><th>개정판</th><th>파이프라인</th><th class="num">Recall@5</th><th class="num">MRR</th><th class="num">nDCG@5</th></tr></thead><tbody>' + ''.join(rows_cond) + '</tbody></table>' + f'<p class="note">random 베이스라인 nDCG@5 최대 {rand_max:.3f}. 측정 질의 {summary["n_scored"]}개(범위 외 4개 제외), dev 세트. holdout 44개 결과는 위 홀드아웃 절.</p></div>')}
{section("부가: Vanilla 의 임베딩만 하이브리드와 같은 bge-m3 로 바꾸면", narr["embed"], data=f"{D20} · 코퍼스 L2 최신판 1,233청크 · {DEV}", body= '<div class="card tw"><table><thead><tr><th>Vanilla 임베딩 (L2 최신판)</th><th class="num">Recall@5</th><th class="num">MRR</th><th class="num">nDCG@5</th></tr></thead><tbody>' + ''.join(f"<tr><td>{PIPE_NAME[p]}</td><td class='num'>{pct(cond[('L2_A',p)]['recall5'])}%</td><td class='num'>{f3(cond[('L2_A',p)]['mrr'])}</td><td class='num'><b>{f3(cond[('L2_A',p)]['ndcg5'])}</b></td></tr>" for p in ("vanilla","vanilla_bge")) + f"<tr><td>하이브리드 + CE (참고)</td><td class='num'>{pct(cond[('L2_A','hybrid')]['recall5'])}%</td><td class='num'>{f3(cond[('L2_A','hybrid')]['mrr'])}</td><td class='num'><b>{f3(cond[('L2_A','hybrid')]['ndcg5'])}</b></td></tr>" + '</tbody></table>' + f'<p class="note">{html.escape(narr["embed_note"])}</p></div>')}
<section><h2>한계</h2><ul>{''.join(f'<li>{html.escape(x)}</li>' for x in narr['limits'])}</ul></section>
<footer>데이터: 20개 기관 실험은 공기업 인사규정 41건(ALIO), 292개 기관 실험은 공공기관 인사규정 전체 수집본. 사전등록 2026-10-01.</footer>
</main>"""
out = HERE / ("dashboard_local.html" if LOCAL else "dashboard.html"); out.write_text(page, encoding="utf-8"); print(out.name, len(page), "chars")
