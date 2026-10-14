**對抗式審查：羽球教練 LLM 壓縮計畫**

**高｜剪枝後的恢復預算差了兩個數量級，結果很可能贏不了 Qwen3.5-4B**
- 問題：Minitron 把 8B 剪到 4B，用了約 94B tokens 來恢復。本計畫一次把 FFN 砍掉 50%、深度砍掉 25%，卻只給 0.5B tokens。TMMLU+ 跌幅 ≤3 個百分點這個目標不切實際。另外，Qwen3.5-4B 是官方用兆級 tokens 訓練出來的，5.2B 只比它大 30%，尺寸上的好處很小。
- 修正：改成迭代剪枝，先 9B→約 7B 做 KD，再降到 5B。S1 預算拉到 2–5B tokens，雲端費用約多 US$300–800。或者改用 online KD，不建快取。事先寫明「輸給 M0 也算一個有效結論」。M2 的 Go/No-go 也要再算一次預算。

**高｜模型大小的計算互相矛盾**
- 問題：已在 config 確認 `vocab_size=248320`、`tie_word_embeddings=false`，所以 embedding 加 lm_head 約 2.03B。這部分若保留 8-bit 就約 2.1GB，其他 3.2B 參數以 4.5 bpw 計約 1.8GB，合計約 3.9GB，超過 ≤3.5GB 的指標。
- 修正：embedding 只做查表，用 4-bit 就夠；lm_head 用 6-bit。這樣總計約 3.2GB。不要做 vocab 裁剪，它會破壞 KD 需要的 tokenizer 一致性；真要做，只能放在所有 KD 結束之後。

**高｜評測有循環**
- 問題：選擇題和開放題都由 teacher 的流程產生。評審又是 teacher-coach，而 student 正是從它蒸餾出來的，等於自己評自己，有自我偏好偏差。S0 的「+5 個百分點」也是 teacher 在自己出的題上自我驗證。
- 修正：選擇題和安全題由人根據 BWF Laws 和教練教材撰寫，並由人審核。主評審改用另一家族的模型。資料生成要嚴格排除評測題的原始文件。

**中｜S0 用 teacher 自己的輸出做 SFT，幾乎學不到新知識**
- 問題：S0 的資料是 teacher 自己產生的，主要只會改變回答格式，對羽球知識幫助有限。
- 修正：S0 只用有真實來源依據的資料。另一個做法是取消 S0，改在產生 logits 和答案時讓 teacher 帶著 RAG 檢索到的條文作答，大約能省 30 H100 小時。

**中｜logits 快取大小算少了**
- 問題：vocab 有 248k，超過 uint16 的上限 65,535，token id 要用 int32。每個 token 是 64×(4+2)=384 B，總量約 192GB，不是 150GB。
- 修正：儲存預算調到 250GB 以上，或改成 k=32，或改用 online KD。

**中｜統計檢定力不足**
- 問題：安全題 30 題、門檻 95%，等於最多只能錯 1 題。開放題 200 題時，55% 勝率的信賴區間約 ±7 個百分點，達到 55% 也不代表顯著。
- 修正：安全題增加到 100 題以上。開放題增加到 400 題，勝率要附 bootstrap 信賴區間，並說明平手怎麼處理。

**中｜DWQ 不一定能吃自訂校準資料**
- 問題：官方文件列出的 `mlx_lm.dwq` 參數只有 `--model --mlx-path --bits --group-size --num-samples --batch-size --max-seq-length`，沒有資料路徑參數。計畫假設可以直接換成 zh-TW 羽球校準資料，這一點沒有根據。
- 修正：在 Gate 0 跑 `--help` 確認，必要時 fork 腳本。校準資料改成通用和羽球各一半，避免偏科。另外要確認 GDN 的小參數（`A_log`、`dt_bias`、`conv1d`、norm）在 MLX 和 GGUF 都沒有被量化。

**中｜兩個模型的 architectures 都是 `Qwen3_5ForConditionalGeneration`**
- 問題：拿掉 vision tower 不只是刪權重，還要改 architectures 類別和權重 key 的前綴，例如 `model.language_model.*`。
- 修正：Gate 0 的 dummy 測試必須涵蓋「純文字類別」從存檔到 `transformers` 載入，再到 `mlx_lm.convert` 的整條路徑。

**中｜套件和工具版本未經查證**
- Ollama：查到的最新版是 v0.30.8（2026-06），計畫寫的 v0.40.4 查不到。
- `vllm 0.31`、`trl 1.15.0`、`mlx-lm 0.32.0`、llama.cpp b11609：都沒有查證。
- TRL `DistillationTrainer` 的匯入路徑正從 `trl.experimental.distillation` 搬到頂層 `trl`，而 TRL v1.12.0 是誤發的版本，不要用。
- Async 版的蒸餾 trainer 需要 `vllm>=0.22` 和 `transformers>=5.2`。
- 修正：在 Gate 0 用 `pip index versions` 實際鎖定版本，`transformers` 要求 ≥5.x。

**中｜Qwen3.6-27B 的授權沒有確認**
- 問題：搜尋結果沒有找到官方的授權欄位。
- 修正：直接到 HF 讀 LICENSE 並把結論記錄下來。Gate 0 第 1 項除了比對 `tokenizer.json` 的 hash，也要比對 chat template。這次已在 config 確認兩個模型的 vocab 都是 248,320。

**低｜llama.cpp 對 Qwen3.5 的轉換支援**
- 問題：社群有人需要自己打補丁才能轉某些 Qwen3.5 checkpoint，而且剪過層數的模型沒人驗證過。
- 修正：維持「GGUF 不擋主線」的安排。

**低｜ToS 疑慮**
- 問題：用 Claude 或 GPT 的分數決定哪些資料留下來訓練，可能被認定為「用輸出訓練模型」。
- 修正：API 模型只用於評測，資料過濾改用開源評審。

**低｜過度工程**
- 問題：9B 的 4-bit 只要約 5GB，在 24GB Mac 上直接就能跑，壓縮的主要價值是學習。
- 修正：評估矩陣從 11 列砍到 6 列：Teacher、M0、9B-SFT、S2、DWQ、GGUF。S3 只在 S2 通過後才做。

**遺漏的步驟**
1. 規則要標註 BWF Laws 的版本日期，並測試「不知道就拒答」的能力。
2. 多輪對話要有專門的評測。
3. 剪枝前先做一個 baseline：未剪枝的 9B 只做 SFT 再 4-bit。這是使用者最可能實際採用的方案。
4. 每次刪 block 後，要確認 `layer_types` 仍維持「3 層 GDN + 1 層 Attention」的交錯，並且最後一層仍是 full attention。

**參考來源**
- Qwen3.6-27B config：https://huggingface.co/Qwen/Qwen3.6-27B/raw/main/config.json
- Qwen3.5-9B config：https://huggingface.co/Qwen/Qwen3.5-9B/raw/main/config.json
- MLX DWQ 說明：https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LEARNED_QUANTS.md
- Ollama 版本：https://www.promptquorum.com/prompt-bites/ollama-latest-version
- TRL v1.11.0：https://pypi.org/project/trl/1.11.0/
- TRL 版本紀錄：https://releases.sh/release/rel_-_Lo5QLOolB09xyuoL6kA
- TRL Async 蒸餾：https://huggingface.co/docs/trl/async_distillation_trainer
- TRL 大型 teacher 蒸餾：https://huggingfacetb-trl-distillation-trainer.hf.space/
- llama.cpp 社群補丁討論：https://github.com/ggml-org/llama.cpp/discussions/20711
- Minitron：https://arxiv.org/abs/2408.11796