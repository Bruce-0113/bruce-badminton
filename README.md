# 🏸 羽球教練 LLM：M0（全程在 Mac 上，免雲端費用）

用本機的 Qwen3.5-9B 合成羽球問答，再用 LoRA 微調 Qwen3.5-4B，最後量化成 MLX 4-bit。
9B 寫答案、4B 學它，本質上就是**序列級知識蒸餾**；最後一步是**量化**。剪枝留到之後（見 [PLAN.md](PLAN.md)）。

需求：Apple Silicon Mac（24GB）、[uv](https://docs.astral.sh/uv/)。

```bash
uv sync
bash scripts/m0.sh questions   # 360 個種子 × 3 題 ≈ 1,080 題       （約 0.5–1 小時）
bash scripts/m0.sh answers     # 9B 逐題寫教練回答，可中斷續跑      （約 4–8 小時，建議睡前跑）
bash scripts/m0.sh dataset     # 轉繁體、去重、過濾 → data/m0/
bash scripts/m0.sh baseline    # 微調前的 4B 規則題分數（對照組）
bash scripts/m0.sh train       # LoRA 微調                          （約 2–4 小時）
bash scripts/m0.sh fuse        # 合併 LoRA → 量化成 4-bit（約 2.5GB）
bash scripts/m0.sh eval        # 微調後、量化後的規則題分數
bash scripts/m0.sh serve       # 開 Gradio 聊天介面
```

時間都是估計值，實際依晶片而定。每一步在做什麼、為什麼這樣做，詳見 [docs/m0-steps.md](docs/m0-steps.md)。

## 檔案

| 路徑 | 用途 |
|---|---|
| `src/coach/seeds.py` | 13 大類羽球主題 × 6 種提問者角色 |
| `src/coach/generate.py` | 透過 OpenAI 相容 API 合成問答（預設連 `mlx_lm.server`，也可換成 LM Studio / Ollama） |
| `src/coach/build_dataset.py` | OpenCC 轉台灣繁體、丟掉被截斷的回答、問題去重、切 train/valid |
| `configs/lora_m0.yaml` | `mlx_lm.lora` 設定 |
| `src/coach/eval_mcq.py` | 規則選擇題評測 |
| `data/eval/mcq_rules.jsonl` | 20 題手寫 BWF 規則選擇題（**建議找教練或裁判複核**） |
| `src/coach/app.py` | Gradio 聊天介面 |

## 調整

- **資料量**：`generate.py questions --n 5` 可以多出題；資料量變了，記得調 `lora_m0.yaml` 的 `iters`（約 筆數 × 3 ÷ 8）。
- **記憶體不足**：把 `batch_size` 改成 1、`max_seq_length` 改成 1536。
- **答案品質**：生成模型換成更大的（例如透過 LM Studio 跑別的模型），改 `scripts/m0.sh` 的 `GEN_MODEL` 或 `--base-url`。

## 授權

程式碼以 [MIT License](LICENSE) 釋出。使用的模型（Qwen3.5）有各自的授權，以 Hugging Face 上的模型頁面為準；本 repo 不包含任何模型權重或合成資料。
