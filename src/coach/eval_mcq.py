"""羽球規則選擇題評測（data/eval/mcq_rules.jsonl），用來比較微調前後與量化前後的模型。

    python -m coach.eval_mcq --model mlx-community/Qwen3.5-4B-bf16
    python -m coach.eval_mcq --model mlx-community/Qwen3.5-4B-bf16 --adapter out/adapters-m0
    python -m coach.eval_mcq --model out/coach-m0-q4
"""

import argparse
import json
import re

from mlx_lm import generate, load

from coach.generate import COACH_SYSTEM

PROMPT = "{question}\n\n{options}\n\n請只回答一個選項字母（A、B、C 或 D）。"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--data", default="data/eval/mcq_rules.jsonl")
    args = ap.parse_args()

    model, tokenizer = load(args.model, adapter_path=args.adapter)
    with open(args.data, encoding="utf-8") as f:
        items = [json.loads(l) for l in f if l.strip()]

    correct = 0
    for it in items:
        options = "\n".join(f"{k}. {v}" for k, v in it["options"].items())
        messages = [
            {"role": "system", "content": COACH_SYSTEM},
            {"role": "user", "content": PROMPT.format(question=it["question"], options=options)},
        ]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)
        out = generate(model, tokenizer, prompt=prompt, max_tokens=16)
        m = re.search(r"[ABCD]", re.sub(r"<think>.*?</think>", "", out, flags=re.S))
        pred = m.group(0) if m else "?"
        ok = pred == it["answer"]
        correct += ok
        print(f"{'✓' if ok else '✗'} {pred}/{it['answer']} {it['question']}")

    print(f"\naccuracy: {correct}/{len(items)} = {correct / len(items):.1%}")


if __name__ == "__main__":
    main()
