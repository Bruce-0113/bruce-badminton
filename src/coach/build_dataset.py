"""data/raw/qa.jsonl → data/m0/{train,valid}.jsonl（mlx_lm.lora 的 chat 格式）。

清理：轉台灣繁體（OpenCC s2twp）、丟掉被截斷或太短的回答、問題去重。
"""

import json
import random
import re
from pathlib import Path

from opencc import OpenCC

from coach.generate import COACH_SYSTEM, QA

OUT = Path("data/m0")
S2TW = OpenCC("s2twp")


def norm(text):
    return re.sub(r"[\s\W_]+", "", text).lower()


def main(valid_ratio=0.05, seed=0):
    rows, seen = [], set()
    stats = {"total": 0, "truncated": 0, "short": 0, "dup": 0, "simplified_fixed": 0}
    with QA.open(encoding="utf-8") as f:
        for r in map(json.loads, f):
            stats["total"] += 1
            if r.get("finish_reason") == "length":
                stats["truncated"] += 1
                continue
            q, a = S2TW.convert(r["question"]), S2TW.convert(r["answer"])
            if len(a) < 80:
                stats["short"] += 1
                continue
            if norm(q) in seen:
                stats["dup"] += 1
                continue
            seen.add(norm(q))
            stats["simplified_fixed"] += (q, a) != (r["question"], r["answer"])
            rows.append({"messages": [
                {"role": "system", "content": COACH_SYSTEM},
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ]})

    random.Random(seed).shuffle(rows)
    n_valid = max(1, int(len(rows) * valid_ratio))
    OUT.mkdir(parents=True, exist_ok=True)
    for name, part in [("valid", rows[:n_valid]), ("train", rows[n_valid:])]:
        with (OUT / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in part)
    print(stats, f"-> train {len(rows) - n_valid}, valid {n_valid}")


if __name__ == "__main__":
    main()
