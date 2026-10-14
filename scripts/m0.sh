#!/usr/bin/env bash
# M0 全流程（在 Apple Silicon Mac 上執行）。每一步都可以單獨重跑：
#   bash scripts/m0.sh questions | answers | dataset | baseline | train | fuse | eval | serve
set -euo pipefail
cd "$(dirname "$0")/.."

GEN_MODEL=mlx-community/Qwen3.5-9B-4bit   # 出題、寫答案的「老師」
BASE_MODEL=mlx-community/Qwen3.5-4B-bf16  # 要微調的學生（需與 configs/lora_m0.yaml 一致）
ADAPTER=out/adapters-m0
FUSED=out/coach-m0-bf16
FINAL=out/coach-m0-q4

start_gen_server() {
  uv run mlx_lm.server --model "$GEN_MODEL" --port 8080 \
    --chat-template-args '{"enable_thinking":false}' >out/gen_server.log 2>&1 &
  SERVER_PID=$!
  trap 'kill $SERVER_PID 2>/dev/null' EXIT
  until curl -s localhost:8080/v1/models >/dev/null; do sleep 2; done
}

mkdir -p out
case "${1:-}" in
  questions) start_gen_server; uv run python -m coach.generate questions --n 3 ;;
  answers)   start_gen_server; uv run python -m coach.generate answers ;;
  dataset)   uv run python -m coach.build_dataset ;;
  baseline)  uv run python -m coach.eval_mcq --model "$BASE_MODEL" ;;
  train)     uv run mlx_lm.lora -c configs/lora_m0.yaml ;;
  fuse)
    uv run mlx_lm.fuse --model "$BASE_MODEL" --adapter-path "$ADAPTER" --save-path "$FUSED"
    uv run mlx_lm.convert --hf-path "$FUSED" --mlx-path "$FINAL" -q --q-bits 4 --q-group-size 64
    du -sh "$FINAL" ;;
  eval)
    uv run python -m coach.eval_mcq --model "$BASE_MODEL" --adapter "$ADAPTER"
    uv run python -m coach.eval_mcq --model "$FINAL" ;;
  serve)
    uv run mlx_lm.server --model "$FINAL" --port 8080 --chat-template-args '{"enable_thinking":false}' &
    trap 'kill $! 2>/dev/null' EXIT
    until curl -s localhost:8080/v1/models >/dev/null; do sleep 2; done
    uv run python -m coach.app ;;
  *) echo "usage: $0 {questions|answers|dataset|baseline|train|fuse|eval|serve}"; exit 1 ;;
esac
