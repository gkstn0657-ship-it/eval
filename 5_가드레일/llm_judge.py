# -*- coding: utf-8 -*-
"""G2 2단계: 로컬 LLM(Ollama)로 '검색된 조문이 질문에 답하는가' 판정. GPU 미사용(num_gpu=0)."""
import json, urllib.request
OLLAMA = "http://127.0.0.1:11434"
MODEL = "hf.co/mykor/A.X-4.0-Light-gguf:Q4_K_M"
SYSTEM = "너는 규정 검색 결과를 검수하는 심사자다. 질문에 대한 답이 아래 조문 안에 실제로 적혀 있는지만 판단한다. 조문에 없는 내용을 추측하지 않는다."
def _chat(user, num_predict=4, timeout=600):
    body = {"model": MODEL, "stream": False, "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            "options": {"num_gpu": 0, "num_thread": 12, "temperature": 0, "num_predict": num_predict, "num_ctx": 4096}}
    req = urllib.request.Request(OLLAMA + "/api/chat", data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read().decode("utf-8"))["message"]["content"].strip()
def judge_relevance(question, articles, max_chars=1500):
    """articles: [(라벨, 본문)]. 반환 (True/False, 원문응답). 하나라도 질문에 답하는 내용이 있으면 True."""
    ctx = "\n\n".join(f"[조문 {i+1}] {lab}\n{txt[:max_chars]}" for i, (lab, txt) in enumerate(articles))
    ans = _chat(f"질문: {question}\n\n{ctx}\n\n위 조문 중 질문에 대한 답이 실제로 적혀 있는 조문이 있으면 '예', 없으면 '아니오'라고만 답하라.")
    return ans.startswith("예"), ans
