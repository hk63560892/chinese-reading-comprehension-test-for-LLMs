## 專案概覽
code：（https://drive.google.com/drive/folders/1Tu9DLGUefoeaiS8vrMcnarfHIR8dj4-5?usp=drive_link）
本專案基於Alpaca-2系列的開源模型進行訓練。與[先前的版本](https://github.com/ymcui/Chinese-LLaMA-Alpaca)相比，本專案具有以下特點：

#### 📖 優化的中文詞表

- 此模型在[先前的項目](https://github.com/ymcui/Chinese-LLaMA-Alpaca)基礎上擴展了LLaMA和Alpaca模型的詞表（大小為55296），以增強對中文字詞的覆蓋。
- 新設計的詞表提高了模型對中文文本的編解碼效率。

#### ⚡ 採用FlashAttention-2的高效注意力機制

- 此模型採用了[FlashAttention-2](https://github.com/Dao-AILab/flash-attention)來實現高效的注意力機制，尤其適用於長文本處理。
- 此技術不僅提供更快的速度，還優化了記憶體使用。

#### 🚄 PI和YaRN技術的超長上下文擴展

- 此模型基於[位置插值PI](https://arxiv.org/abs/2306.15595)和[YaRN](https://arxiv.org/abs/2309.00071)技術，進一步推出了支持長達64K上下文的模型。
- 這些技術使模型能夠在不增加複雜度的情況下處理更長的文本。

#### 🤖 簡化的中英雙語系統提示語

- 相比[先前的模型](https://github.com/ymcui/Chinese-LLaMA-Alpaca)，此模型簡化了系統提示語，以便更好地適應中英雙語環境。

#### 👮 人類偏好對齊

- 透過人類反饋的強化學習（RLHF），此模型改進了模型傳遞正確價值觀的能力。
- 推出了Alpaca-2-RLHF系列模型，使用方式與SFT模型一致。

## 文件結構

- `inference.py`: 模型推理腳本，用於生成答案。
- `preprocessing.py`: 數據預處理腳本，將資料轉換為適合模型的格式。
- `datasets/`: 存放原始和處理後的數據。
- `README.md`: 本文件，提供專案概述和使用說明。

## 使用方法

### 推理

執行`inference.py`以生成答案：
python inference.py --model_dir <模型路徑> [--cuda_ids <CUDA設備ID>] [--enable_8bit_quantization] [--enable_4bit_quantization]

## `preprocessing.py` 腳本

`preprocessing.py` 負責從 Excel 文件中讀取數據，進行處理，並轉換成 JSON 格式以供模型使用。

### 主要功能

1. **讀取 Excel 文件**：使用 pandas 從指定路徑讀取 Excel 文件。
2. **數據處理**：轉換答案格式，刪除不必要的列。
3. **生成 JSON 格式**：將處理後的數據轉換為 JSON 格式。
4. **保存 JSON 文件**：將轉換後的 JSON 數據保存至文件。
## `model_config.json` 配置文件

此配置文件定義了用於 LLaMA 模型的參數和設定。

### 主要參數

- **模型架構**：使用 `LlamaForCausalLM` 作為模型架構。
- **隱藏層活化函數**：活化函數設定為 `silu`。
- **隱藏層大小**：隱藏層的維度設定為 4096。
- **中間層大小**：中間層的維度設定為 11008。
- **最大位置嵌入**：設定為 4096。
- **注意力頭數量**：設定為 64。
- **隱藏層數量**：設定為 64。
- **詞彙大小**：設定為 55296。

### 完整配置


{
  "architectures": ["LlamaForCausalLM"],
  "bos_token_id": 1,
  "eos_token_id": 2,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 11008,
  "max_position_embeddings": 4096,
  "model_type": "llama",
  "num_attention_heads": 64,
  "num_hidden_layers": 64,
  "num_key_value_heads": 64,
  "pad_token_id": 0,
  "pretraining_tp": 1,
  "rms_norm_eps": 3e-05,
  "rope_scaling": null,
  "tie_word_embeddings": false,
  "torch_dtype": "float16",
  "transformers_version": "4.31.0",
  "use_cache": true,
  "vocab_size": 55296
}
### 預訓練

- 此模型在原版Llama-2的基礎上，利用大規模無標註數據進行增量訓練。
- 訓練代碼參考了🤗transformers的[run_clm.py](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py)。

### 指令精調

- 在Chinese-LLaMA-2的基礎上，利用有標註指令數據進行進一步精調，得到Chinese-Alpaca-2系列模型。
- 訓練代碼參考了[Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca)。

## 引用

[paper](https://arxiv.org/abs/2304.08177)
