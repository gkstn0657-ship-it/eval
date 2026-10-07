#!/usr/bin/env bash
# 속도 단축판: 결과물은 quantize_qwen14b.sh 와 같고 순서만 겹친다.
#  - 일반 4비트를 먼저 만들어 GPU 생성을 시작하고, 그동안 CPU 에서 보정 행렬 → 보정판 4비트를 만든다.
#  - 7B 채점(J7)은 GPU 가 비는 틈(14B 생성 사이)에 끼워 넣는다.
set -uo pipefail
HF="C:/Users/SSAFY/hf_models/Qwen2.5-14B-Instruct"; OUT="C:/Users/SSAFY/hf_models/gguf"; mkdir -p "$OUT"
LT="C:/Users/SSAFY/llama_tools"; Q="$(cd "$(dirname "$0")" && pwd)"; T="$(dirname "$Q")"
F16="$OUT/qwen2.5-14b-instruct-f16.gguf"; R="$T/results_gen_quant"
t(){ echo "[$(date +%T)] $*"; }
reg(){ sed "s#^FROM .*#FROM $OUT/$1.gguf#" "$Q/base_qwen25.Modelfile" > "$OUT/$2.Modelfile"; ollama create "$2" -f "$OUT/$2.Modelfile"; }
cd "$T"
t "F16 변환";    [ -f "$F16" ] || python "$LT/src/convert_hf_to_gguf.py" "$HF" --outtype f16 --outfile "$F16" || exit 1
t "4비트 양자화"; [ -f "$OUT/q14-q4_k_m.gguf" ] || "$LT/bin/llama-quantize.exe" "$F16" "$OUT/q14-q4_k_m.gguf" Q4_K_M 20 || exit 1
reg q14-q4_k_m q14-q4 || exit 1; t "q14-q4 등록"
( cp "$Q/calib_hr.txt" "$OUT/calib_hr.txt"   # llama-imatrix 는 한글 경로 파일을 열지 못해 영문 경로로 복사
  t "보정 행렬(CPU)"; { [ -f "$OUT/imatrix_hr.dat" ] || "$LT/bin/llama-imatrix.exe" -m "$F16" -f "$OUT/calib_hr.txt" -o "$OUT/imatrix_hr.dat" -c 512 -t 10; }   && t "보정판 4비트" && { [ -f "$OUT/q14-q4_k_m-imat.gguf" ] || "$LT/bin/llama-quantize.exe" --imatrix "$OUT/imatrix_hr.dat" "$F16" "$OUT/q14-q4_k_m-imat.gguf" Q4_K_M 10; }   && reg q14-q4_k_m-imat q14-q4i && t "q14-q4i 등록" ) > "$R/imatrix.log" 2>&1 &
t "생성 Q14-4";  python -u run_gen_quant.py --gen q14-q4 Q14-4 > "$R/gen_Q14-4.log" 2>&1
t "채점 J7 (Q14-4)"; python -u run_gen_quant.py --judge qwen2.5:7b-instruct J7 >> "$R/judge_J7.log" 2>&1
wait; grep -q "q14-q4i 등록" "$R/imatrix.log" || { t "보정판 실패"; tail -5 "$R/imatrix.log"; exit 1; }
t "생성 Q14-4i"; python -u run_gen_quant.py --gen q14-q4i Q14-4i > "$R/gen_Q14-4i.log" 2>&1
t "채점 J7 (Q14-4i)"; python -u run_gen_quant.py --judge qwen2.5:7b-instruct J7 >> "$R/judge_J7.log" 2>&1
t "채점 J14";    python -u run_gen_quant.py --judge q14-q4 J14 > "$R/judge_J14.log" 2>&1
python run_gen_quant.py --summary > "$R/summary.log" 2>&1; t "집계 끝"
