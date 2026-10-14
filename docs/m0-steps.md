# M0 實作步驟

這份文件記錄 M0 的做法：每一步做什麼、為什麼這樣做、輸入與輸出，以及可能遇到的問題。指令速查請看 [README](../README.md)，長期計畫請看 [PLAN.md](../PLAN.md)。

## 整體流程

```
seeds.py（13 類主題 × 6 種角色 = 360 個種子）
   │
   ▼  ① 出題：Qwen3.5-9B-4bit（mlx_lm.server）
data/raw/questions.jsonl（約 1,080 題）
   │
   ▼  ② 寫答案：同一個 9B 模型，帶教練 system prompt
data/raw/qa.jsonl
   │
   ▼  ③ 清理：OpenCC 轉台灣繁體、丟掉被截斷或太短的回答、問題去重
data/m0/train.jsonl, valid.jsonl
   │
   ▼  ④ LoRA 微調 Qwen3.5-4B（mlx_lm.lora）
out/adapters-m0/
   │
   ▼  ⑤ 合併 LoRA（mlx_lm.fuse）→ 量化成 4-bit（mlx_lm.convert -q）
out/coach-m0-q4/（約 2.5GB）
   │
   ▼  ⑥ 評測（規則選擇題，比較微調前、微調後、量化後）
   ▼  ⑦ 部署（mlx_lm.server + Gradio）
```

## 和壓縮技術的對應

| 技術 | M0 有沒有做 | 做法 |
|---|---|---|
| 知識蒸餾 | ✅ 序列級 | 9B（老師）寫答案，4B（學生）用 LoRA 學它的回答。不需要 logits，所以老師和學生不必共用 tokenizer |
| 量化 | ✅ | MLX affine 4-bit，group size 64 |
| 剪枝 | ❌ | 需要雲端 GPU 做剪枝後的恢復訓練，留待之後（PLAN.md 的 M2） |

## 步驟說明

### 0. 環境準備

```bash
brew install uv          # 或參考 https://docs.astral.sh/uv/
uv sync                  # 安裝 mlx-lm 0.32.0、openai、opencc、gradio
```

第一次執行時會從 Hugging Face 下載模型：9B-4bit 約 5GB，4B-bf16 約 8GB。

### 1. 出題：`bash scripts/m0.sh questions`

- **做什麼**：在背景啟動 `mlx_lm.server` 載入 `mlx-community/Qwen3.5-9B-4bit`。每個種子（主題 × 角色）請模型以該角色身分提出 3 個問題。
- **為什麼用「主題 × 角色」**：只給主題的話，問題會很相似。換成初學者、家長、校隊選手、銀髮族等不同角色，問題的口吻和程度會更多元。
- **輸出**：`data/raw/questions.jsonl`，每行一題，帶 `seed_id` 和 `qid`。
- **細節**：
  - 關閉 Qwen 的 thinking 模式（`enable_thinking: false`），若輸出中仍有 `<think>` 區塊會再清除一次。
  - 會自動去掉題號和項目符號。
- **續跑**：已完成的種子會被略過，中斷後重跑同一個指令即可。

### 2. 寫答案：`bash scripts/m0.sh answers`

- **做什麼**：用同一個 9B 模型，加上「資深羽球教練」的 system prompt（`generate.py` 的 `COACH_SYSTEM`），逐題作答。
- **回答格式**：先講結論，再列步驟或要點、常見錯誤；涉及受傷時提醒就醫。
- **輸出**：`data/raw/qa.jsonl`，每筆會記錄 `finish_reason`。值為 `length` 表示回答被截斷，步驟 3 會丟掉。
- **時間**：最久的一步，約 4–8 小時，建議睡前跑。可以隨時中斷，續跑會從斷點開始。
- **換模型**：任何 OpenAI 相容的 API 都能用（LM Studio、Ollama 等），改 `--base-url` 即可。

### 3. 清理資料：`bash scripts/m0.sh dataset`

- **轉繁體**：OpenCC `s2twp`，簡體轉台灣繁體並換成台灣用語（例如「杀球」→「殺球」）。
- **過濾**：去掉被截斷的回答和太短的回答（少於 80 字）。
- **去重**：問題去掉空白和標點後完全相同就視為重複。
- **切分**：打亂後 95% 當訓練集、5% 當驗證集。
- **輸出格式**：`mlx_lm.lora` 吃的 chat 格式 `{"messages": [system, user, assistant]}`。
- 執行結束會印出統計數字（總數、截斷、太短、重複），可以用來檢查資料品質。

### 4. 對照組：`bash scripts/m0.sh baseline`

先記錄**微調前**的 Qwen3.5-4B 在 20 題規則選擇題上的分數，之後才能判斷微調有沒有幫助。

### 5. LoRA 微調：`bash scripts/m0.sh train`

設定檔是 `configs/lora_m0.yaml`：

| 參數 | 值 | 說明 |
|---|---|---|
| `model` | `mlx-community/Qwen3.5-4B-bf16` | 在 bf16 上訓練，量化留到最後一步 |
| `mask_prompt` | `true` | 只對教練的回答計算 loss，不學 system prompt 和問題 |
| `num_layers` | 16 | 只在最後 16 層加 LoRA |
| `batch_size` × `grad_accumulation_steps` | 2 × 4 | 等效 batch 8 |
| `iters` | 600 | 約 1,500 筆 × 3 epoch ÷ 8；資料量改變時要跟著調 |
| `learning_rate` | 1e-5 | |
| `lora_parameters` | rank 16、scale 20、dropout 0.05 | |
| `grad_checkpoint` | `true` | 用時間換記憶體，24GB 夠用 |

- **觀察**：每 100 步會印 validation loss。持續下降代表正常；train loss 降、val loss 升代表過擬合，要減少 `iters`。
- **輸出**：`out/adapters-m0/`。

### 6. 合併與量化：`bash scripts/m0.sh fuse`

1. `mlx_lm.fuse`：把 LoRA 權重合併回 bf16 模型，輸出到 `out/coach-m0-bf16/`。
2. `mlx_lm.convert -q --q-bits 4 --q-group-size 64`：量化成 4-bit，輸出到 `out/coach-m0-q4/`。

**為什麼先合併再量化**：直接在 4-bit 模型上訓練（QLoRA）比較省記憶體，但合併時會有額外誤差。24GB 跑得動 bf16 訓練，所以選精度較好的做法。

### 7. 評測：`bash scripts/m0.sh eval`

評測集是 `data/eval/mcq_rules.jsonl`，20 題手寫的 BWF 規則選擇題，內容包含計分、發球高度、場地尺寸、球的規格、休息時間和違例判定。用貪婪解碼（不抽樣），只取回答中的第一個 A–D 字母。

會跑兩組，再和步驟 4 的對照組比較：

| 模型 | 預期 |
|---|---|
| 微調前（baseline） | 對照組 |
| 微調後（bf16 + adapter） | 應該 ≥ baseline |
| 量化後（4-bit） | 應該只比微調後低一點點 |

**限制**：

- 這 20 題只測規則記憶。技術和戰術的回答品質，要在聊天介面上實際問問看。
- 合成資料本身可能有規則錯誤，因為 9B 也會記錯，微調可能把錯誤學進去。所以規則題分數下降時，要回頭檢查 `qa.jsonl` 裡的規則類回答。
- 評測題是手寫的，建議找教練或裁判複核。

### 8. 使用：`bash scripts/m0.sh serve`

用 `mlx_lm.server` 載入量化後的模型，提供 OpenAI 相容 API；再開 Gradio 聊天介面（`src/coach/app.py`），預設網址是 http://127.0.0.1:7860。

## 常見問題

| 問題 | 解法 |
|---|---|
| 訓練時記憶體不足 | `batch_size: 1`、`max_seq_length: 1536`，或把 `num_layers` 降到 8 |
| 回答出現簡體字 | 步驟 3 已經轉過繁體；如果是模型自己生成簡體，就多跑一些資料，或在 system prompt 加強要求 |
| 回答出現 `<think>` | 確認 server 有帶 `--chat-template-args '{"enable_thinking":false}'` |
| 生成太慢 | 減少題數（`--n 2`），或降低 `--max-tokens` |
| 想重新生成 | 刪掉 `data/raw/*.jsonl` 再跑 |

## 下一步（有預算時）

請看 [PLAN.md](../PLAN.md)：

- M2：結構化剪枝（Qwen3.5-9B → 約 5B）加上 logit 蒸餾
- M4：用 RAG 讓規則題附上 BWF 條文

M0 會當作它們的對照組。
