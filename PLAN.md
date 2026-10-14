# 羽球教練 LLM 專案計畫（剪枝、蒸餾、量化，部署到 24GB Mac）

> 版本日期：2026-10-14。以下價格和版本號，都要在專案開始那天重新確認一次。

---

> **審查後修正（2026-10-14）**：本計畫經過對抗式審查，完整審查意見見 [docs/review.md](docs/review.md)。已直接修正的有：模型大小計算、logits 儲存量、評測集規模、DWQ 的退路，以及評測集改由人撰寫。**尚未納入、需要你決定的**有兩項：
> 1. S1 的恢復預算是否從 0.5B 提高到 2–5B tokens，並改成迭代剪枝（9B→7B→5B）。審查指出 0.5B tokens 比 Minitron 少了約兩個數量級，很可能贏不了 Qwen3.5-4B。
> 2. 是否取消 S0 teacher 校正，改成 teacher 帶 RAG 條文作答。

## 1. 專案目標與成功指標

**目標**：做一個以繁體中文（zh-TW）回答的專業羽球教練模型，範圍涵蓋技術、步法、單雙打戰術、BWF 規則、訓練週期化、傷害預防和器材。訓練流程依序使用三種壓縮技術：結構化剪枝、知識蒸餾、量化。最後在使用者的 24GB Apple Silicon Mac 上離線執行。

| 類別 | 指標 | 目標值（M3 驗收） |
|---|---|---|
| 羽球知識 | 羽球選擇題 300 題的準確率 | 至少是 teacher 分數的 90%，絕對值 ≥ 80% |
| 回答品質 | 開放題 400 題對 **M0 對照組（Qwen3.5-4B SFT）** 的兩兩比較勝率 | ≥ 55%，且 bootstrap 95% 信賴區間下界 > 50%。沒達到就代表剪枝沒有帶來價值（見 §6 Go/No-go） |
| 安全 | 安全題 100 題中，正確建議就醫或停練的比例 | ≥ 95% |
| 通用能力 | 跟未剪枝的 Qwen3.5-9B（同樣做過 SFT）比，TMMLU+ 的跌幅 | ≤ 3 個百分點 |
| 繁中品質 | 回答中的簡體字比例（以 OpenCC 前後差異計算） | < 0.5% |
| 模型大小 | MLX 4-bit 權重 | ≤ 3.5 GB（目標約 3.2 GB；embedding 4-bit、lm_head 6-bit） |
| 記憶體 | Mac 上 4k context 時的峰值 | ≤ 6 GB |
| 速度 | decode 速度（`mlx_lm.generate --verbose`） | M4 Pro 等級（273 GB/s）≥ 45 tok/s；基本款 M 晶片（100–120 GB/s）≥ 20 tok/s |
| 首字延遲 | 500 token prompt 的首字延遲 | ≤ 1.0 秒（M4 Pro） |

---

## 2. 整體架構流程圖

```
 [公開來源]                [種子題: 分類×角色×難度×情境 ≈3k]
 BWF Laws / 維基zh / 協會手冊        │
   │ trafilatura/pymupdf             ▼
   ▼                         ┌──────────────────────────────┐
 RAG 規則庫(條文切塊) ─────► │ Teacher: Qwen3.6-27B (vLLM,  │  雲端 1×H100
   │                         │ no-think, zh-TW system prompt)│
   │                         └──────────────┬───────────────┘
   │                                        ▼
   │                  生成Q&A → 去重(hash/MinHash/bge-m3) → LLM評分≥4
   │                  → OpenCC s2twp + 簡體比例檢查 → 與評測集 13-gram 去重
   │                                        │
   │            ┌───────────────────────────┼─────────────────────────┐
   │            ▼                           ▼                         ▼
   │   S0 Teacher校正(LoRA SFT)      SFT資料 3–4萬筆           KD prompts 8萬
   │   Qwen3.6-27B-coach                 │                         │
   │            │                         │   ┌── M0 對照組 ─────────┤
   │            │                         │   ▼                     │
   │            │                         │ Qwen3.5-4B LoRA SFT ──► MLX 4bit ──► Mac (MVP)
   │            ▼                         ▼
   │   Top-64 logits 快取 (0.5B tokens, ~190GB)
   │            │
   │            │     Student 基底: Qwen3.5-9B (去掉 vision tower)
   │            │            │
   │            │            ▼  校準集 1024×2–4k tokens (50%通用zh-TW + 50%羽球)
   │            │     [剪枝] FFN 12288→6144 + 刪 2/8 個 block(32→24層)  ≈5.2B
   │            │            │
   │            └──────────► [S1 KD 恢復] forward-KL logits, 0.5B tokens    雲端 2×H100
   │                         │
   │                         ▼
   │                  [S2 seq-KD SFT + top-k logit KD] 羽球45%/通用55%
   │                         │
   │                         ▼ (M4 選配)
   │                  [S3 On-policy GKD] TRL DistillationTrainer       雲端 2–4×H100
   │                         │
   │                         ▼  合併 → HF safetensors (標準 qwen3_5 config)
   │                  [量化] mlx_lm.dwq 4bit (zh-TW羽球校準) / GGUF Q4_K_M+imatrix   Mac / 雲端
   │                         │
   │                         ▼
   │                  [評估矩陣] MCQ / 開放題勝率 / TMMLU+ / 簡體比 / tok/s / 記憶體
   │                         │
   └───────── RAG ─────────► ▼
                      [Mac 部署] mlx_lm.server(OpenAI API) → Gradio UI
                                  (選配: Ollama / LM Studio / Open WebUI)
```

---

## 3. 模型選擇

| 角色 | 主方案 A | 理由 |
|---|---|---|
| Teacher | **Qwen3.6-27B**（dense，Apache-2.0，2026-04） | BF16 約 56GB，單張 H100 80GB 放得下。在一位部落客跑的 TMMLU+ 比較中，同系列的 Qwen3.6-35B-A3B 約 75%，Gemma-4-26B-A4B 約 46%；這只是單一來源，當作強烈參考。vocab 是 248,320，和 Qwen3.5 小模型相同 |
| Student 基底 | **Qwen3.5-9B**（Apache-2.0，2026-02/03） | 32 層，每 4 層一個 block（3 層 Gated DeltaNet + 1 層 Gated Attention），共 8 個 block。hidden 4096、FFN 12,288。mlx-lm 0.32.0 已有原生 `qwen3_5.py`。剪到約 5B，4-bit 後約 3GB |
| 對照組（必做） | **Qwen3.5-4B**，用同一份 SFT 資料 | 剪枝加蒸餾得到的約 5B 模型必須贏它，剪枝才有意義。它同時就是 M0 的 MVP |

**Gate 0（第一週就要做，不通過就換方案）**
1. Qwen3.6-27B 和 Qwen3.5-9B 的 `tokenizer.json` SHA256 要相同，special tokens 和 chat template 也要一致。
2. 把一個「刪掉 1 個 block、FFN 縮小」的 dummy Qwen3.5-9B 存成 HF 格式，確認 `transformers` 載得進來，`mlx_lm.convert` 能轉，`mlx_lm.generate` 輸出正常。
3. 確認 llama.cpp `convert_hf_to_gguf.py` 支不支援 qwen3_5。這一項只影響 GGUF 副路線，不擋主線。

**備案 B（Gate 0 的第 1 或第 2 項失敗時）**：Teacher 用 **Qwen3-32B**，Student 用 **Qwen3-8B**，剪到約 5B。
- 架構是純 transformer。Megatron-Bridge 官方有 Qwen3-8B 剪到 6B 的範例，各種工具支援最成熟。
- 兩者同屬 Qwen3 tokenizer（151,936 vocab），可以直接做 logit KD。

**備案 C（繁中漂移嚴重或授權考量時）**：Teacher 用 Llama-3.1-TAIDE-70B 或 Llama-3.3-70B，Student 用 Llama-Breeze2-8B 或 TAIDE-8B。
- 兩者共用 Llama 3 tokenizer，模型原生偏台灣語料。
- 代價：推理能力較舊；受 Llama 授權約束，需要標示 Built with Llama；70B teacher 要 2×H100。

**不採用的選項**
- Gemma 4：E2B/E4B 的「effective」參數設計不利剪枝，繁中也較弱。只拿來當額外的評測基線。
- Llama 4：只有 MoE，沒有小的 dense student。
- Qwen3.8-27B：權重和授權尚未確認。若專案開始時已在 HF 以 Apache-2.0 釋出，且 tokenizer 一致，可以替換 teacher。

**推理模式**：一律使用 **no-think**，教練的回答要直接、結構化。KD 的目標輸出和 chat template 都固定成 no-think，以免 student 學到兩種格式混在一起。

---

## 4. 各階段詳細步驟

### 4.1 資料收集與合成（雲端 1×H100 跑 vLLM；爬蟲和清理在 Mac 或任何 CPU 上）

**真實來源**（每筆都記錄 `source_url / license / fetch_date / rules_version`）
- BWF Laws（「All rights reserved」）：**只用於 RAG 和出評測題**，不把原文逐字放進訓練資料，改寫成問答並附條號。
- 維基百科中文（CC BY-SA 4.0）：衍生資料要標示出處，並以相同授權發佈。
- 協會和賽事技術手冊：用來建台灣用語對照表，也當 RAG 文件。
- 部落格和 YouTube：預設不用。
- 工具：`trafilatura==2.3.1`、`pymupdf`、Scrapy（`ROBOTSTXT_OBEY=True`、`AUTOTHROTTLE`）。

**合成流程（自己寫 vLLM 批次腳本；不用 distilabel，它停在 2025-01）**
1. 種子：10 大類、約 80 個子類，乘上 6 種角色（初學者、中階、校隊、教練、家長、裁判），再乘上難度和情境，約 3k 個。另外刻意加入前提錯誤的問題，例如「發球可以過腰嗎？」。
2. Teacher 對每個種子生成 5–10 個問題，再生成答案。答案格式固定為「結論 → 步驟 → 常見錯誤 → 安全提醒」。規則題要先走 RAG 檢索條文，並附上 Law 條號。
3. 多輪對話佔 20–30%，每段 3–6 輪，由 teacher 同時扮演學員和教練。
4. 去重分三層：精確 hash、MinHash（`datasketch`，Jaccard 0.8）、`bge-m3` 嵌入相似度（cosine 0.92）。最後再和評測集做 13-gram 去重。
5. 品質過濾：LLM 評審從正確性、安全、可執行性、繁中四個面向打 1–5 分，只留 4 分以上，預期淘汰 30–40%。規則題另外和條文比對。
6. 繁中：OpenCC `s2twp` 加白名單（人名、專有名詞）。轉換前後差異超過 1% 的就重新生成。另外維護羽球詞彙表（殺球、切球、挑球、推球、平抽、勾對角、網前、後場）。

**產出**
| 資料集 | 規模 | 用途 |
|---|---|---|
| `sft_coach` | 3–4 萬筆（混 50–55% 通用 zh-TW 指令） | S0、M0、S2 |
| `kd_prompts` | 8 萬個 prompt（羽球 70%、通用 30%） | S2 和 S3 |
| `kd_corpus` | 0.5B tokens（繁中通用語料 80–90%、羽球文本 10–20%） | S1 logit KD |
| `calib_prune` | 1024 段 × 2–4k tokens（通用 zh-TW 50%、羽球 50%） | 剪枝重要性評分 |
| `calib_quant` | 約 500 段羽球問答 | DWQ 和 imatrix 校準 |
| `eval/*` | 選擇題 300、開放題 400、安全題 100。**選擇題和安全題由人根據 BWF Laws 撰寫並審核**，不由 teacher 產生 | **在生成任何訓練資料之前就凍結** |

**ToS 決策（預設採保守做法）**：訓練資料的**答案一律由開源權重的 teacher（Apache-2.0）產生**。Claude 和 GPT 只當評審，而且用之前要先看條款。Anthropic Commercial Terms D.4 禁止用輸出「train competing AI models」，詳見 §8 第 1 題。

### 4.2 S0：Teacher 校正（雲端 2×H100，或 1×H100 用 QLoRA）
- **為什麼做**：Minitron 的經驗是 teacher 先熟悉目標資料分布，student 恢復得比較好。teacher 不懂羽球，KD 也就學不到羽球。
- **工具**：TRL `SFTTrainer` 加 PEFT LoRA。設定 r=32、alpha=64，套用在所有 linear 層；lr 1e-4、2 epoch、序列長度 4096、bf16。
- **輸入**：`sft_coach`。**輸出**：`Qwen3.6-27B-coach`（LoRA 合併後的權重）。
- **驗收**：羽球選擇題比原始 teacher 至少高 5 個百分點；TMMLU+ 跌幅不超過 2 個百分點。

### 4.3 剪枝（雲端 1×H100，約 2 GPU 小時）
- **Mac 為什麼只能用結構化剪枝**：非結構化和 2:4 稀疏在 MLX、Metal、llama.cpp 上都還是跑 dense matmul，既不省記憶體也不會變快。
- **主要工具：自己寫 Minitron 式的 PyTorch 腳本**（`src/coach/prune/`，約 300 行）。
  - 這次只剪 MLP 寬度和整個 block。在這個範圍內，自寫腳本比 Megatron/NeMo 容器簡單，輸出直接就是標準 HF config。
  - 交叉驗證可用 `nvidia-modelopt==0.47.0`。這要確認它包含 PR #1955（Qwen3.5 VLM 的 LM 部分剪枝），並且需要 `nemo:26.02` 以上的容器。
- **步驟**
  1. 先拿掉 vision tower，只保留 text LM。
  2. **FFN 寬度**：用 `calib_prune` 跑 forward，每層計算 SwiGLU 中間神經元的激活重要性（Minitron 的 mean-abs activation）。所有層統一保留 **6144**，也就是 12288 的一半，是 256 的倍數。對 gate、up、down 三個投影一起切。
  3. **深度**：以 4 層的 block 為單位，計算 block 輸入和輸出之間的 angular distance，刪掉最冗餘的 **2 個中後段 block**，從 32 層變成 24 層，保留 3:1 的 GDN 和 Attention 交錯節奏。第一個和最後一個 block 不動。同步更新 `num_hidden_layers`、`layer_types`，並重新編號 state dict。
  4. **不動的部分**：hidden_size 4096、GDN heads、attention heads、KV groups、head_dim、vocab 和 embedding（避免損失中文 token）。
  5. 規模估算：embedding 加 lm_head 約 2.0B，24 層 MLP 約 1.8B，24 層 attention 和 GDN 約 1.35B，合計**約 5.2B**。
- **剪後立刻檢查**：用 `transformers` 載入並生成，用 `mlx_lm.convert` 轉換，並比較剪枝前後的 perplexity 和 50 題羽球題，確認沒有崩壞（預期分數會大跌，這是正常的）。
- **退路**：如果只剪 FFN 就已經 ≥ 6B 且品質損失過大，改成「剪 30%、蒸餾、再剪」的迭代方式。

### 4.4 知識蒸餾（雲端）
| 階段 | 方法 | 工具 | 資料 | 主要超參數 | 硬體與時間 |
|---|---|---|---|---|---|
| **快取** | Teacher-coach 產生 top-64 logits（token id 加 fp16 值，token id 用 int32，約 384 B/token） | 自寫 HF 或 vLLM 腳本，或 DistillKit `sample_logits_vllm` | `kd_corpus` 0.5B tokens | k=64，截斷後重新正規化 | 1×H100 約 15–20 小時；儲存約 190GB（預留 250GB） |
| **S1 恢復** | 只用 logit forward-KL，不加 CE | DistillKit（offline top-k）或自寫 KL loop | 同上 | lr 1e-4 線性降到 1e-5，warmup 100，seq 4096，global batch 約 256 序列（約 1M tok），單趟 | 2×H100 FSDP 約 15–20 小時 |
| **S2 領域** | seq-KD SFT（teacher 的回答）加 top-k logit KD（α=0.5） | TRL `SFTTrainer` 加自訂 KL loss，或 `GKDTrainer(lmbda=0, seq_kd=True)` | `sft_coach` 加 teacher 回答過的 `kd_prompts`（羽球 45%、通用 55%，含 replay） | 全參數 lr 1.5e-5、2 epoch、seq 4096 | 2×H100 約 6–10 小時 |
| **S3 on-policy**（M4 選配） | GKD，`beta=0.5` | TRL v1.15.0 `DistillationTrainer`（vLLM colocate，chunked JSD） | `kd_prompts` 的 1.5 萬個 prompt | lr 1e-6 到 5e-6、1 epoch，`lmbda` 0.5–1.0 | 2–4×H100（server mode 下 teacher 和 student 不共用 GPU），約 8–12 小時 |

**TRL 陷阱**
- 模型一律用 `torch_dtype=bfloat16` 載入，否則預設 fp32。
- padding 設在左邊。
- `DistillationConfig` 的預設是 lr 1e-6、beta 1.0，要自己明確設定。

**遺忘監控**：每個階段結束都跑 TMMLU+ 子集。跌超過 3 個百分點，就把通用資料比例調高 10%。

### 4.5 量化（Mac 為主；GGUF 可在雲端或 Mac 上做）
1. **基線**：
   ```
   mlx_lm.convert --hf-path out/coach-5b --mlx-path out/coach-q4 -q --q-bits 4 --q-group-size 64
   ```
2. **主產出**：
   ```
   mlx_lm.dwq --model out/coach-5b --mlx-path out/coach-dwq-q4 --bits 4 --group-size 32
   ```
   - 校準資料用 `calib_quant`（zh-TW 羽球），不用預設的 tulu-3。
   - 用 `--max-seq-length 512 --batch-size 1` 並且從 8-bit 複本開始，控制記憶體。5B 模型在 24GB 上做 DWQ 可能很緊；跑不動就降低 seq 長度和 batch（DWQ 依賴 MLX，不適合上雲）。Gate 0 要先跑 `mlx_lm.dwq --help`，確認能不能指定自訂校準資料；不行就 fork 腳本。校準資料用通用和羽球各一半。
   - **embedding 用 4-bit、lm_head 用 6-bit**（兩者合計約 2B 參數，保留 8-bit 會超過 3.5GB 指標），用 mixed predicate 或 `mlx_lm.dynamic_quant --target-bpw 4.5`。
3. **副產出**（給 Ollama 或分享用，前提是 Gate 0 第 3 項通過）：
   - `convert_hf_to_gguf.py` 轉 bf16 → 用 zh-TW 羽球文本跑 `llama-imatrix` → `llama-quantize` 產生 Q4_K_M 和 Q5_K_M。
   - 用 llama.cpp b11609 以上的版本。
4. **檢查**：每個產出都用 `apply_chat_template` 跑 zh-TW smoke test，確認 chat template 存在而且是 no-think。

### 4.6 評估（雲端跑大模型，Mac 跑量化版）
- **比較矩陣**（每一列都要有完整數據）：
  1. Teacher（原始）
  2. Teacher-coach
  3. **M0：Qwen3.5-4B SFT（MLX 4-bit）**
  4. Qwen3.5-9B SFT（未剪枝的上限參考）
  5. 剪枝後（未恢復）
  6. S1
  7. S2
  8. S3
  9. MLX Q4
  10. MLX DWQ-Q4
  11. GGUF Q4_K_M
- **每一列記錄的指標**：羽球選擇題準確率、開放題勝率（對 teacher-coach 和 M0）、安全題通過率、TMMLU+（`lm-evaluation-harness` 的 `tmmluplus`，或 `mlx_lm.evaluate`）、簡體字比例、tok/s、峰值記憶體、首字延遲。
- **評審**：兩個不同家族的評審交叉評分，例如 teacher-coach 和一個 API 模型，以減少評審偏好自家輸出的偏差。再抽 50 題人工校準。
- 評測腳本統一用 `uv run coach-eval --model <path> --suite all`，結果寫入 `reports/eval_matrix.csv`。

### 4.7 Mac 部署
- `mlx_lm.server --model out/coach-dwq-q4 --port 8080`，提供 OpenAI 相容 API，並開啟 prompt cache。
- **Gradio** `gr.ChatInterface`（約 40 行）：透過 `openai` client 串流回答，內建 zh-TW 教練 system prompt 和範例問題。
- **選配 RAG（M4）**：BWF 規則按條文切塊，用 `bge-m3` 嵌入，存進 `chromadb`，再用 `bge-reranker-v2-m3` 重新排序。規則題一律附上檢索到的條文，Mac 上約多用 1–2GB。
- **選配**：用 Ollama（最新版本於 Gate 0 查證） 的 Modelfile 搭配 GGUF。注意：GGUF 在 Ollama 上可能仍走 llama.cpp 而不是 MLX（未查證，要實測）。也可以用 LM Studio 或 Open WebUI。

---

## 5. Python 專案目錄結構與相依套件

```
bruce-badminton/
├── pyproject.toml
├── configs/
│   ├── models.yaml            # teacher/student/fallback 路徑與 tokenizer hash
│   ├── data.yaml              # 分類、角色、比例、去重門檻
│   ├── prune.yaml             # ffn=6144, drop_blocks=2, calib 設定
│   ├── distill_s1.yaml / distill_s2.yaml / distill_s3.yaml
│   └── quant.yaml             # bits, group_size, lm_head bits, calib
├── src/coach/
│   ├── data/
│   │   ├── crawl.py           # trafilatura/pymupdf + 來源中繼資料
│   │   ├── seeds.py           # 分類×角色×難度×情境
│   │   ├── generate.py        # vLLM 批次生成 (guided JSON / pydantic)
│   │   ├── dedup.py           # hash / MinHash / bge-m3 / 13-gram 對評測集
│   │   ├── judge.py           # LLM 評分過濾
│   │   └── zhtw.py            # OpenCC s2twp、白名單、簡體比例
│   ├── rag/                   # 條文切塊、嵌入、檢索 (chromadb)
│   ├── train/
│   │   ├── sft.py             # S0 / M0 / S2 (TRL + PEFT)
│   │   ├── cache_logits.py    # top-k logits 快取 (分片 .npz/.arrow)
│   │   ├── kd_offline.py      # S1 KL loop (或 DistillKit 設定)
│   │   └── gkd.py             # S3 TRL DistillationTrainer
│   ├── prune/
│   │   ├── importance.py      # FFN 激活重要性、block angular distance
│   │   ├── apply.py           # 統一寬度切片、刪 block、改 config/layer_types
│   │   └── checks.py          # 載入/生成/mlx 轉換 smoke test
│   ├── quant/                 # mlx convert/dwq、gguf 包裝腳本
│   ├── eval/
│   │   ├── mcq.py / open_qa.py / safety.py / script_ratio.py / speed.py
│   │   └── matrix.py          # 彙總 eval_matrix.csv
│   └── serve/
│       └── app.py             # Gradio UI (+RAG)
├── scripts/
│   ├── gate0_tokenizer_check.py
│   ├── gate0_dummy_prune_convert.py
│   └── cloud/                 # 開機、同步資料、跑階段的 shell
├── data/   (gitignore；raw/ processed/ eval/ 凍結並記錄 hash)
├── out/    (gitignore；checkpoints、mlx、gguf)
├── reports/
└── tests/                     # dedup/zhtw/prune 形狀檢查單元測試
```

**`pyproject.toml`**（套件管理用 `uv`，Python 3.11；extras 依執行環境拆開）
```toml
[project]
name = "badminton-coach"
requires-python = ">=3.11,<3.13"
dependencies = ["pydantic>=2", "pyyaml", "datasets", "typer", "opencc-python-reimplemented", "openai"]

[project.optional-dependencies]
data  = ["trafilatura==2.3.1", "scrapy", "pymupdf", "datasketch", "sentence-transformers"]
cloud = ["torch", "transformers",          # 鎖定已含 qwen3_5 模型類別的版本（Gate 0 確認）
         "accelerate", "peft", "trl==1.15.0", "vllm==0.31.*", "flash-attn", "lm-eval"]
prune_alt = ["nvidia-modelopt==0.47.0"]  # 交叉驗證用，於 NeMo 容器
mac   = ["mlx-lm[train]==0.32.0", "gradio", "lm-eval"]
rag   = ["chromadb", "FlagEmbedding"]

[project.scripts]
coach-gen  = "coach.data.generate:app"
coach-eval = "coach.eval.matrix:app"
coach-serve = "coach.serve.app:main"
```
`mlx-lm` 不要用 0.31.0，那一版已被撤下。實驗紀錄可以選用 `wandb` 或 `trackio`。

---

## 6. 里程碑與時程（以一人兼職計，約 12 週）

| 里程碑 | 週 | 內容 | 驗收（Go/No-go） | 雲端 H100 小時（估） |
|---|---|---|---|---|
| **M0：MVP 基線** | 1–3 | Gate 0 三項檢查；凍結評測集；資料流程 v1（1.5–2 萬筆）；Qwen3.5-4B LoRA SFT，再用 MLX 4-bit 和 DWQ 量化；Gradio 上線 | Mac 上能跑起來；羽球選擇題 ≥ 70%；簡體字比例 < 1%。**這就是之後的對照組** | 約 25 |
| **M1：Teacher 與完整資料** | 4–5 | 資料 v2（SFT 3–4 萬筆、KD prompts 8 萬、kd_corpus 0.5B）；S0 teacher 校正；評估 teacher-coach；快取 logits | teacher-coach 選擇題比原始 teacher 高 5 個百分點以上 | 約 60 |
| **M2：剪枝與蒸餾** | 6–8 | 剪枝到約 5.2B，再做 S1 和 S2；同時訓練一份 Qwen3.5-9B SFT 當上限參考 | 在 bf16 下，開放題對 M0 勝率 ≥ 55%，TMMLU+ 跌幅 ≤ 3 個百分點。**如果沒有贏 M0**，先把 S1 預算加到 1B tokens 再試；仍然輸就改走備案 B，或者乾脆直接部署 9B（4-bit 約 5GB） | 約 70 |
| **M3：量化與驗收** | 9–10 | MLX Q4、DWQ、mixed 量化；GGUF；完整評估矩陣；Mac 速度測試 | 達到 §1 的全部指標 | 約 5 |
| **M4：加強（選配）** | 11–12 | S3 on-policy GKD；RAG 規則庫；Ollama 打包；人工測試回饋 | 規則題準確率提升 5 個百分點以上；Ollama 一行指令能跑 | 約 30 |

**雲端費用估算**：詳見 [docs/cost-estimate.md](docs/cost-estimate.md)。標準方案（M0–M3）約 US$620（含 30% 緩衝），精簡 MVP 約 US$210。

---

## 7. 風險與對策

| 風險 | 影響 | 對策 |
|---|---|---|
| Qwen3.5 的 GDN 混合架構，剪枝或轉換工具不支援 | M2 卡住 | 只剪 FFN 和整個 block，不碰 GDN heads；Gate 0 先跑 dummy 剪枝並轉 MLX；不行就改走備案 B（Qwen3-32B 和 Qwen3-8B） |
| Teacher 和 student 的 tokenizer 不一致 | logit KD 失效，而且不會報錯 | Gate 0 比對 hash；不一致時只做 seq-KD，或換成同家族的 teacher |
| 剪枝加蒸餾的結果贏不了 Qwen3.5-4B | 剪枝沒有價值 | M0 對照組明確設為 Go/No-go 門檻；可以加大 S1 預算、改成迭代剪枝，或直接部署 9B 的 4-bit |
| 簡體字漂移（Qwen 和 4-bit 量化都會造成） | 繁中品質下降 | zh-TW system prompt、OpenCC 過濾、DWQ 和 imatrix 用繁中校準、簡體比例列入驗收指標 |
| 規則被蒸餾錯（例如發球高度 1.15m） | 教錯規則，影響可信度 | 規則題強制以 RAG 條文為依據並附條號；資料標註規則版本；M4 推論時也走 RAG |
| 災難性遺忘 | 通用中文能力下降 | 通用資料 50–55%、replay、低學習率；每個階段都追蹤 TMMLU+ |
| 合成資料同質化 | 過擬合，回答千篇一律 | 種子按角色 × 情境 × 難度組合；加入前提錯誤的題目；多層去重 |
| ToS 和授權（Claude/GPT 輸出、BWF 版權、CC BY-SA） | 無法公開或商用 | 答案只用 Apache-2.0 teacher 生成；BWF 原文只進 RAG 和評測；記錄每筆資料的來源 |
| 非均勻寬度或尺寸無法整除 | MLX 或 GGUF 載不進來 | 全模型統一使用 256 的倍數（6144）；剪完立刻做轉換檢查 |
| chat template 或 thinking 模式遺失 | 部署後格式錯亂 | 每個產出都做 template smoke test；固定 no-think |
| 雲端 GPU 中斷（社群 GPU） | 浪費時間和費用 | 每 500 步存 checkpoint 到 network volume；logits 快取分片存放，可以續跑 |

---

## 8. 待討論的開放問題

1. **用途與發佈**：只給自己或小圈子用，還是要公開發佈或商用？
   - 這決定能不能用 Claude/GPT 的輸出當訓練資料。目前預設不用。
   - 也決定 BWF 教材能用到什麼程度、維基衍生資料要不要用 CC BY-SA 釋出。
2. **Mac 的確切型號**：是 M1–M4 的基本款、Pro 還是 Max？記憶體頻寬會決定 tok/s 目標要定在 20 還是 45 以上。如果是基本款，要不要把目標縮到約 3.5B（例如 FFN 再縮到 5120）？
3. **預算上限**：總預算 US$500–700 可以接受嗎？S1 要用 0.5B 還是 1–2B tokens？後者效果較好，但費用約多 1.5 倍。
4. **M4 的 RAG**：規則題要不要一律附上 BWF 條文引用？這會多一個嵌入模型，約 1–2GB。
5. **回答風格**：要不要固定成「結論 → 步驟 → 常見錯誤 → 安全提醒」？需不需要支援英文提問或中英混用？
6. **多輪和個人化**：教練要不要記住使用者的程度和傷病史？這會影響 20–30% 多輪資料的設計。
7. **人工審核**：有沒有羽球教練或裁判朋友，能幫忙審核 300 題選擇題和 50 題評審校準？這是評估可信度的關鍵。
8. **前端**：Gradio 夠用嗎？還是希望用 Ollama、LM Studio 或 Open WebUI（有歷史紀錄、可多人使用）？
9. **備案取捨**：如果 Qwen3.5 剪枝不順，你偏好直接部署 9B 的 4-bit（約 5GB，最省事），還是改走備案 B，把三種壓縮技術完整走一遍？

---

### 主要參考來源
- Qwen3.5-9B: https://huggingface.co/Qwen/Qwen3.5-9B
- Qwen3.6-27B: https://huggingface.co/Qwen/Qwen3.6-27B
- TMMLU+ Qwen vs Gemma: https://ai-muninn.com/en/blog/tmmluplus-qwen-vs-gemma-traditional-chinese
- Qwen3.5 小模型: https://artificialanalysis.ai/articles/qwen3-5-small-models
- ModelOpt Qwen3.5 剪枝 PR: https://github.com/NVIDIA/Model-Optimizer/pull/1955/files
- ModelOpt 剪枝範例: https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/pruning
- nvidia-modelopt: https://pypi.org/project/nvidia-modelopt/
- Megatron-Bridge 剪枝: https://docs.nvidia.com/nemo/megatron-bridge/latest/modelopt/pruning.html
- Minitron: https://arxiv.org/abs/2407.14679 、 https://arxiv.org/abs/2408.11796
- ShortGPT（深度剪枝）: https://arxiv.org/abs/2403.03853
- TRL DistillationTrainer: https://huggingface.co/docs/trl/distillation_trainer
- TRL GKD: https://huggingface.co/docs/trl/gkd_trainer.md
- DistillKit: https://github.com/arcee-ai/DistillKit
- mlx-lm: https://pypi.org/project/mlx-lm/
- MLX learned quants（DWQ）: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LEARNED_QUANTS.md
- mlx-lm qwen3_5: https://raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/models/qwen3_5.py
- llama.cpp: https://github.com/ggml-org/llama.cpp/releases
- Ollama: https://github.com/ollama/ollama/releases
- oMLX 實測: https://omlx.ai/benchmarks
- Anthropic Commercial Terms: https://www.anthropic.com/legal/commercial-terms
- TMMLU+: https://huggingface.co/datasets/ikala/tmmluplus
- trafilatura: https://pypi.org/project/trafilatura/
- distilabel（停更）: https://pypi.org/project/distilabel/
- BWF Handbook II: https://mbasso.in/webroot/img/static/handbook_II_laws_of_badminton_bwf.pdf
- H100 價格: https://intuitionlabs.ai/articles/h100-rental-prices-cloud-comparison 、 https://www.thundercompute.com/blog/nvidia-h100-pricing 、 https://www.gmicloud.ai/en/blog/cheapest-h100-cloud-providers-for-llm-inference-in-2026