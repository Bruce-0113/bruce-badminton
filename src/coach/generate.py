"""用本機的大模型（OpenAI 相容 API，預設 mlx_lm.server）合成羽球問答。

兩步驟：每個種子先出 N 個問題，再逐題寫教練回答。輸出為 jsonl，中斷後重跑會從斷點續跑。

    python -m coach.generate questions --n 3
    python -m coach.generate answers
"""

import argparse
import json
import re
from pathlib import Path

from openai import OpenAI

from coach.seeds import iter_seeds

RAW = Path("data/raw")
QUESTIONS = RAW / "questions.jsonl"
QA = RAW / "qa.jsonl"

COACH_SYSTEM = (
    "你是一位資深的專業羽球教練，熟悉 BWF 規則、技術、步法、單雙打戰術、體能訓練與運動傷害預防。"
    "一律使用台灣繁體中文與台灣慣用的羽球術語回答。"
    "回答要具體、可執行：先給結論，再說明步驟或要點，接著列出常見錯誤；"
    "涉及疼痛或受傷時，提醒停止訓練並就醫。不確定的事要直說，不要編造。"
)

QUESTION_PROMPT = """請以「{persona}」的身分，針對羽球主題「{category} — {subtopic}」，寫出 {n} 個真實、具體、彼此不同的問題，像是會在球場上問教練的那種。
要求：
- 使用台灣繁體中文
- 每題一行，不要編號、不要其他說明
- 至少一題要描述自己的具體情況（例如程度、遇到的問題、身體狀況）"""

THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def chat(client, messages, max_tokens):
    resp = client.chat.completions.create(
        model="default_model",
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.8,
        top_p=0.95,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    choice = resp.choices[0]
    return THINK_RE.sub("", choice.message.content or "").strip(), choice.finish_reason


def done_ids(path, key):
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as f:
        return {json.loads(line)[key] for line in f if line.strip()}


def gen_questions(client, n):
    done = done_ids(QUESTIONS, "seed_id")
    seeds = [s for s in iter_seeds() if s["id"] not in done]
    print(f"{len(done)} seeds done, {len(seeds)} to go")
    with QUESTIONS.open("a", encoding="utf-8") as f:
        for i, s in enumerate(seeds, 1):
            text, _ = chat(client, [{"role": "user", "content": QUESTION_PROMPT.format(n=n, **s)}], 400)
            lines = [re.sub(r"^\s*(\d+[.、)]|[-*•])\s*", "", l).strip() for l in text.splitlines()]
            for j, q in enumerate([l for l in lines if len(l) >= 6][:n]):
                f.write(json.dumps({"seed_id": s["id"], "qid": f"{s['id']}#{j}", "category": s["category"], "question": q}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i}/{len(seeds)}] {s['id']}")


def gen_answers(client, max_tokens):
    done = done_ids(QA, "qid")
    with QUESTIONS.open(encoding="utf-8") as f:
        todo = [q for q in map(json.loads, f) if q["qid"] not in done]
    print(f"{len(done)} answers done, {len(todo)} to go")
    with QA.open("a", encoding="utf-8") as f:
        for i, q in enumerate(todo, 1):
            answer, finish = chat(client, [{"role": "system", "content": COACH_SYSTEM}, {"role": "user", "content": q["question"]}], max_tokens)
            f.write(json.dumps({**q, "answer": answer, "finish_reason": finish}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i}/{len(todo)}] {q['question'][:40]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["questions", "answers"])
    ap.add_argument("--base-url", default="http://localhost:8080/v1")
    ap.add_argument("--n", type=int, default=3, help="每個種子出幾題")
    ap.add_argument("--max-tokens", type=int, default=900)
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    client = OpenAI(base_url=args.base_url, api_key="local")
    if args.step == "questions":
        gen_questions(client, args.n)
    else:
        gen_answers(client, args.max_tokens)


if __name__ == "__main__":
    main()
