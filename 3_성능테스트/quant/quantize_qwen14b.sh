#!/usr/bin/env bash
# 생성모델_양자화_실험계획.md 2단계: Qwen2.5-14B-Instruct 를 GGUF 로 변환하고 3종으로 양자화해 Ollama 에 등록한다.
# 도구: llama.cpp b11433 (convert_hf_to_gguf.py, llama-imatrix, llama-quantize), Ollama 0.35
set -euo pipefail
HF=C:/Users/SSAFY/hf_models/Qwen2.5-14B-Instruct      # HF 원본(bf16 safetensors)
OUT=C:/Users/SSAFY/hf_models/gguf; mkdir -p $OUT
LT=C:/Users/SSAFY/llama_tools                         # src/(변환기), bin/(실행 파일)
HERE=$(cd "$(dirname "$0")" && pwd)
F16=$OUT/qwen2.5-14b-instruct-f16.gguf
t(){ echo "[$(date +%T)] $*"; }

t "1) F16 GGUF 변환"
[ -f $F16 ] || python $LT/src/convert_hf_to_gguf.py $HF --outtype f16 --outfile $F16
t "2) 도메인 보정 행렬(imatrix): 인사규정 조문 80개, 평가 정답 조문 제외"
[ -f $OUT/imatrix_hr.dat ] || $LT/bin/llama-imatrix.exe -m $F16 -f $HERE/calib_hr.txt -o $OUT/imatrix_hr.dat -c 512 -t 20
t "3) 양자화 2종 (4비트, 4비트+보정)"
[ -f $OUT/q14-q4_k_m.gguf ]  || $LT/bin/llama-quantize.exe $F16 $OUT/q14-q4_k_m.gguf Q4_K_M 20
[ -f $OUT/q14-q4_k_m-imat.gguf ] || $LT/bin/llama-quantize.exe --imatrix $OUT/imatrix_hr.dat $F16 $OUT/q14-q4_k_m-imat.gguf Q4_K_M 20
ls -la $OUT
t "4) Ollama 등록 (기준 7B 와 같은 Qwen 대화 형식)"
for pair in "q14-q4_k_m:q14-q4" "q14-q4_k_m-imat:q14-q4i"; do
  f=${pair%%:*}; name=${pair##*:}
  sed "s#^FROM .*#FROM $OUT/$f.gguf#" $HERE/base_qwen25.Modelfile > $OUT/$name.Modelfile
  ollama create $name -f $OUT/$name.Modelfile
done
ollama list
t "끝"
