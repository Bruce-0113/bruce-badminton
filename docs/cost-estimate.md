# 雲端 GPU 費用估算（2026-10-14 查價，匯率以 1 USD ≈ 31.8 TWD 計）

> 先講結論：三種情境含 30% 緩衝後，**精簡 MVP 約 US$210（約 NT$6,700）**、**標準約 US$620（約 NT$19,600）**、**完整版約 US$1,410（約 NT$44,900）**。原計畫寫的「H100 每小時 $1.5–2.5」現在偏樂觀。RunPod H100 SXM 社群價是 $2.69，Lambda 已漲到 $3.99。本估算以 **$2.69/H100 小時** 為基準價。
>
> 吞吐量和 MFU（實際算力佔理論峰值的比例）都是推估，網路上查不到 Qwen3.6-27B 在 H100 上的公開實測數字。**第一天先花約 $3 跑 1 小時 benchmark**，再用實測值修正下面所有時數。

## 1. 2026 年 10 月各平台 GPU 時價（USD/小時，每張 GPU）

| GPU | RunPod 社群雲 | RunPod 安全雲 | Lambda 隨需 | Vast.ai（市集） | Modal（按秒計費） | Colab Pro |
|---|---|---|---|---|---|---|
| H100 SXM 80GB | **2.69** | 3.49 | 3.99（SXM 只賣 8 卡節點） | 起價約 2.27；未驗證主機約 0.9–1.9 | 約 3.95 | 常常租不到 |
| H100 PCIe 80GB | 1.99 | 2.89 | 3.29 | 起價約 2.80 | — | — |
| A100 80GB | 1.19（PCIe）/ 1.39（SXM） | 1.59 | 2.79（SXM） | 約 1.0–1.5 | 約 2.50 | 約 7.5–15 CU/時 ≈ $0.75–1.5 |
| L40S 48GB | 0.79 | 1.09 | — | — | 約 1.95 | — |
| RTX 4090 24GB | 0.34 | 0.74 | — | 約 0.3 | — | — |
| RTX 5090 32GB | 0.69 | 0.99 | — | 約 0.34–0.63（spot 低到 0.10） | — | — |
| 儲存 | Network volume $0.07/GB/月（1TB 以下） | | | | | 100 CU 售價 $9.99，90 天後過期 |

**Spot 和 On-demand 怎麼選**
- **RunPod**：2026-10-08 的官網頁面沒有列 spot 價，以 console 為準。第三方資料說隨需價比 spot 貴 40–75%。
- **Vast.ai**：interruptible 實例比 on-demand 便宜 30–50%，但主機可能幾分鐘前通知就收回 GPU。
- **Lambda**：沒有 spot。
- **Modal**：按秒計費、閒置時自動縮到零，適合零碎的評估工作，但單價大約是 RunPod 的 1.5 倍。
- **Colab Pro**：有 session 時數上限，H100 經常租不到，也沒有 200GB 以上的持久磁碟。只適合 4B LoRA 或小模型評估。

**哪些工作適合 spot**：vLLM 生成和 logits 快取都已經分片、可以續跑，適合用 spot 或 interruptible。S1 和 S2 是 2 卡 FSDP 訓練，被中斷的代價高，建議用社群雲或安全雲的 on-demand SXM（有 NVLink）。

## 2. 各階段的計算假設

| 階段 | 基準硬體 | 吞吐量假設與計算 |
|---|---|---|
| ① 合成資料（vLLM，Teacher Qwen3.6-27B） | 1×H100，用 **FP8** | 官方 vLLM recipe 跑 **BF16 版要 TP=2（2 張卡）**。為了省錢，生成階段用 FP8，單卡就放得下。離線大批次時，輸出約 **1,200 tok/s**、prefill 約 15k tok/s。目前唯一的參考數字是 Qwen3.5-27B FP8 在單張 H100 上 312 tok/s，但那是有延遲要求的線上服務情境，離線大批次會高很多，要實測。 |
| ② Teacher logit 快取 | 1×H100，用 BF16（只放 text 權重，約 54GB，只做 forward） | 每 token 計算量 2×27e9 = 5.4e10 FLOPs。以 989 TFLOPs × 40% MFU 計約 7,400 tok/s，加上 top-64 和寫檔的額外開銷約 15%。**0.5B tokens ≈ 20 小時**。儲存：64×(id 4B + fp16 2B) = 384 B/token，0.5B ≈ 190GB；id 打包成 3 bytes 可降到約 160GB。 |
| ③ 剪枝重要性計算 | 1×A100 80GB 就夠 | 9B 模型 forward 跑 1024×4k ≈ 4M tokens，計算量約 7e16 FLOPs，幾分鐘就算完。時間主要花在下載、改 config 和 smoke test 上。 |
| ④ S0 Teacher LoRA | 2×H100 | 35k 筆 × 約 1k tokens × 2 epoch = 70M tokens。LoRA 每 token 約 4×27e9 FLOPs，合計 7.6e18。以 35% MFU 計約 6 GPU 小時，加上評估和開銷 → **約 10 GPU 小時**。 |
| ④ S1 KD 恢復 | 2×H100 SXM FSDP | 每 token 6×5.2e9 = 3.1e10 FLOPs。GDN kernel 和讀 logits 會拖慢速度，以 20% MFU 計約 6.3k tok/s/GPU。**0.5B ≈ 22 GPU 小時**（兩張卡平行，實際約 11 小時）。全參數 Adam 約 83GB，所以一定要 2 卡。 |
| ④ S2 seq-KD + logit KD | 2×H100 | (3.5 萬 + 8 萬) × 約 900 tokens ≈ 100M，跑 2 epoch = 200M tokens → 約 9 GPU 小時。再加上這 100M tokens 的 teacher logits 快取約 4 小時。 |
| ④ S3 GKD | 3×H100 | TRL 的 on-policy 流程效率低：15k prompts × 約 600 tokens 的 rollout，還要 teacher 打分 → 3 卡 × 10 小時 ≈ **30 GPU 小時**。 |
| ④ M0（4B LoRA）和 9B SFT 參考線 | L40S，或 Mac | 4B：40–70M tokens 約 3–4 小時（L40S）。9B LoRA：70M tokens 約 3 H100 小時。 |
| ⑤ 量化 | **Mac，免費** | 5B 做 DWQ：8-bit 教師約 5.2GB + 4-bit 學生約 3GB + seq 512 的 activation，24GB 放得下，M4 Pro 約 1–3 小時。GGUF imatrix 也可以在 Mac 上用 Metal 跑。 |
| ⑥ 評估 | H100（27B）或 L40S（≤9B） | TMMLU+ 約 22.7k 題，每題約 300 tokens，合計約 7M tokens prefill，加上 530 題生成和 pairwise 評審。27B 每列約 1 小時，5–9B 每列約 0.3 小時。 |

## 3. 三種情境的範圍

| 項目 | 精簡 MVP | 標準（原計畫 M0–M3） | 完整版（加 M4） |
|---|---|---|---|
| SFT 資料量 | 2 萬筆，KD prompts 3 萬 | 3–4 萬筆，KD prompts 8 萬 | 同標準，再做 best-of-2 重生成 |
| S0 teacher 校正 | 不做，改用 RAG 和 system prompt | 做 | 做，3 epoch |
| logits 快取和 S1 的 token 數 | 0.2B | 0.5B | 1.5B |
| S2 | 只做 seq-KD | seq-KD + logit KD | 同標準，資料更多 |
| S3 GKD | 不做 | 不做 | 做 |
| 剪枝 | 單一設定 | 單一設定 | 3 種設定，各跑 50M tokens 短 KD 比較 |
| 9B SFT 上限參考 | 不做 | 做 | 做 |

## 4. 各階段 GPU 小時與費用（H100 以 $2.69、A100 以 $1.39、L40S 以 $0.79 計）

| 階段 | 精簡 MVP | 標準 | 完整版 |
|---|---|---|---|
| ① 合成資料：輸出 tokens ÷ 1,200 tok/s，加 prefill | 50M 輸出、120M prefill → **14 H100 小時，$38** | 約 122M 輸出（SFT 候選 48M + KD 回答 48M + 出題 5M + 評審 21M）、300M prefill → **34 小時，$91** | 約 220M 輸出 → **60 小時，$161** |
| ② logit 快取 | 0.2B → **9 小時，$24** | 0.5B 約 20 小時 + S2 用的 4 小時 → **24 小時，$65** | 1.5B 約 60 小時 + 6 小時 → **66 小時，$178** |
| ③ 剪枝 | 1 A100 小時，$1.4 | 2 A100 小時，$2.8 | 2 A100 小時 + 7 H100 小時的設定比較 → **$22** |
| ④ S0 Teacher LoRA | 0 | **10 小時，$27** | **15 小時，$40** |
| ④ M0 4B LoRA | 3 L40S 小時，$2.4（Mac 上免費） | 4 L40S 小時，$3.2 | $3.2 |
| ④ 9B SFT 參考線 | 0 | **3 小時，$8** | **4 小時，$11** |
| ④ S1 KD 恢復 | **9 小時，$24** | **22 小時，$59** | **66 小時，$178** |
| ④ S2 | 90M tokens → **5 小時，$13** | 200M → **10 小時，$27** | **15 小時，$40** |
| ④ S3 GKD | 0 | 0 | **30 小時，$81** |
| ⑤ 量化 | Mac，$0 | Mac，$0 | Mac，$0（GGUF 可選，1 L40S 小時約 $1） |
| ⑥ 評估 | 3 小時，$8 | 8 小時，$22 | 15 小時，$40 |
| **H100 小時合計** | **約 40** | **約 111** | **約 287** |
| **GPU 小計** | **$111** | **$305** | **$755** |
| 儲存（network volume） | 約 200GB × 1.5 個月 ≈ $21 | 約 400GB × 2.5 個月 ≈ $70 | 約 850GB × 3 個月 ≈ $180 |
| API 評審（可選） | $30 | $100 | $150 |
| **小計** | **$162（約 NT$5,150）** | **$475（約 NT$15,100）** | **$1,085（約 NT$34,500）** |
| **加 30% 緩衝後總計** | **約 $210（約 NT$6,700）** | **約 $620（約 NT$19,600）** | **約 $1,410（約 NT$44,900）** |

**換平台後的 H100 費用**（只算 GPU，不含儲存、API 和緩衝）

| H100 單價 | MVP（40 小時） | 標準（111 小時） | 完整（287 小時） |
|---|---|---|---|
| $1.5（Vast interruptible 或未驗證主機） | $60 | $167 | $431 |
| $1.99（RunPod H100 PCIe 社群雲，只適合單卡工作，速度慢約 20–30%） | $80 | $221 | $571 |
| **$2.69（RunPod SXM 社群雲，基準價）** | **$108** | **$299** | **$772** |
| $3.49（RunPod 安全雲） | $140 | $387 | $1,002 |
| $3.95–3.99（Modal 或 Lambda） | $159 | $443 | $1,145 |

**省錢建議**
1. ① 和 ② 用 Vast.ai 或 RunPod 的 interruptible 單卡。這兩階段約佔總時數 40–45%，可以省 30–50%。
2. 出題和評審改用 Qwen3.6-35B-A3B（MoE）。參考數字顯示它的吞吐量約是 dense 27B 的 3 倍。訓練用的答案仍然由 27B 產生，所以 seq-KD 的一致性不受影響。
3. logits 改存 top-32，儲存量減半。
4. 9B SFT 參考線和評估改用 A100 或 L40S。

## 5. Mac（24GB）上可以免費做的工作

| 工作 | 可行性 |
|---|---|
| 爬蟲、trafilatura/pymupdf 清理、OpenCC、MinHash 去重、bge-m3 嵌入（MPS） | 完全可以 |
| Gate 0 tokenizer SHA256 比對 | 完全可以 |
| Gate 0 dummy 剪枝和 MLX 轉換 | 用隨機初始化的小型同架構 config 做，或在 CPU 上用 low-mem 模式。bf16 的 9B 約 18GB，在 24GB 上很緊 |
| M0 的 Qwen3.5-4B LoRA（`mlx_lm.lora`） | 可以，40M tokens 大約要半天到一天。不想等就花 $3 用 L40S |
| 量化：`mlx_lm.convert`、DWQ、`dynamic_quant`、GGUF `llama-imatrix` 和 `llama-quantize` | 可以，**原計畫說「DWQ 跑不動就到雲端」要修正**。DWQ 依賴 MLX，MLX 的 CUDA backend 能不能跑 DWQ 還沒查證，所以退路應該是降低 seq 長度和 batch，不是上雲 |
| 量化版評估：選擇題、tok/s、記憶體、首字延遲、Gradio、RAG | 完全可以，而且速度和記憶體本來就只能在 Mac 上量 |
| 9B 剪枝重要性計算 | 理論上可以，但記憶體太緊，建議花約 $1.4 租 1 小時 A100 |

## 6. 要回頭修正的計畫內容

1. **Teacher 不是 BF16 單卡就能跑**：官方 vLLM recipe 的 BF16 版本含 vision 權重，要 2 張 H100。所以生成用 FP8 單卡；logits 快取用 BF16，只載入 text 權重的單卡，或 TP=2。整個 KD 都用同一種精度，前後才一致。
2. **原計畫「約 190 H100 小時、$400–700」**：「標準」的 GPU 費用約 $305，但加上儲存、API 和 30% 緩衝後約 $620，落在原本的區間內。「完整版」會超過 $1,000，因為 1.5B tokens 讓快取和 S1 的時數各增加 3 倍。
3. **S1 預算加倍的影響**：S1 從 0.5B 加到 1B tokens，會多出約 20 小時快取和 22 小時訓練，等於多 42 H100 小時，約 +$113，儲存也會多出約 190GB。
4. **最好有預算警示**：RunPod 和 Vast 都要預先儲值。建議每個階段都設停機腳本，跑完自動關機，避免閒置計費；閒置往往是實際帳單超支的主因。

**來源**
- [RunPod Pricing（2026-10-08 更新）](https://www.runpod.io/gpu-cloud/pricing)
- [deploybase RunPod spot/on-demand](https://deploybase.ai/articles/runpod-gpu-pricing)
- [diyai RunPod 2026](https://diyai.io/ai-tools/hosting/runpod-pricing/)
- [Spheron Lambda H100 2026](https://www.spheron.network/blog/lambda-cloud-h100-pricing-2026/)
- [deploybase Lambda](https://deploybase.ai/articles/lambda-cloud-gpu-pricing)
- [computeprices Vast H100](https://computeprices.com/providers/vast/gpus/h100)
- [computeprices Vast 5090](https://computeprices.com/providers/vast/gpus/rtx5090)
- [Beam RTX 5090 pricing](https://www.beam.cloud/blog/nvidia-rtx-5090-pricing)
- [vantaige Vast.ai](https://vantaige.io/ai-tool/vast-ai)
- [budgetforge Modal 2026](https://www.budgetforge.dev/tools/modal-pricing-2026)
- [computeprices Modal L40S](https://computeprices.com/providers/modal/gpus/l40s)
- [explainx Colab CU 2026](https://explainx.ai/blog/google-ai-pro-colab-a100-h100-200-compute-units-claimed-vs-verified-2026)
- [deploybase Colab](https://deploybase.ai/articles/google-colab-fine-tune-llm-free-vs-pro-gpu-comparison)
- [markaicode Qwen3.6-27B H100](https://markaicode.com/benchmarks/hugging-face-qwen-3-h100-latency-benchmark/)
- [markaicode Qwen3.5 27B vs 35B-A3B](https://markaicode.com/benchmarks/hugging-face-qwen-25-h100-throughput-benchmark/)
- [Vultr Qwen3.6-27B](https://docs.vultr.com/models/alibaba-qwen-36/qwen3.6-27b/nvidia)
- [longforecast USD/TWD 2026-09-15](https://longforecast.com/usd-to-twd)