# 🚀 Hybrid Trading System for Taiwan Stock Index (^TWII)

這是一個先進的演算法交易系統，結合了用於價格預測的 **LSTM-SSAM** (Long Short-Term Memory with Sequential Self-Attention) 以及用於交易決策的 **Pro Trader RL** (Reinforcement Learning)。

## ✨ 核心特色 (Key Features)

| 特色 | 說明 |
|---------|-------------|
| **本地資料整合** | TWII 歷史資料採本地 CSV 管理 (`twii_data_from_2000_01_01.csv`)，確保成交量單位 (億元) 正確，並具備自動更新機制 |
| **嚴謹訓練流程** | **Data Leakage Prevention**: LSTM 模型訓練時的資料縮放 (Scaling) 嚴格限制在訓練集內，防止 Look-ahead Bias |
| **LSTM-SSAM 預測** | T+1 與 T+5 價格預測，並使用 MC Dropout 進行不確定性估計 |
| **遷移學習 (Transfer Learning)** | 使用全球指數進行預訓練 (Pre-train) → 針對 ^TWII 進行微調 (Fine-tune) |
| **特徵融合 (Feature Fusion)** | 整合 23 種特徵，包含 LSTM 預測值與信心分數 |
| **PPO Agent** | 分離的買入 (Buy) 與賣出 (Sell) 代理人，並具備類別平衡機制 |
| **回測 (Backtesting)** | 完整的模擬回測，包含停損機制與績效指標計算 |

## 📊 績效結果 (2023-Present)

| 指標 (Metric) | 數值 (Value) |
|--------|-------|
| **總報酬率 (ROI)** | 85.49% |
| **年化報酬率 (Annualized Return)** | 23.53% |
| **夏普值 (Sharpe Ratio)** | 1.47 |
| **最大回撤 (Max Drawdown)** | -17.23% |
| **勝率 (Win Rate)** | 100% (5 次交易) |

## 🏗️ 系統架構 (Architecture)

```
┌─────────────────────────────────────────────────────────────────┐
│                     HYBRID TRADING SYSTEM                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │  LSTM T+1    │    │  LSTM T+5    │    │    技術指標       │  │
│  │   預測模型    │    │  + MC Dropout│    │  (Indicators)    │  │
│  └──────┬───────┘    └──────┬───────┘    └────────┬─────────┘  │
│         │                   │                      │            │
│         └───────────────────┼──────────────────────┘            │
│                             │                                    │
│                    ┌────────▼────────┐                          │
│                    │    23 特徵融合   │                          │
│                    │  (Feature Fusion)│                         │
│                    └────────┬────────┘                          │
│                             │                                    │
│         ┌───────────────────┴───────────────────┐               │
│         │                                       │               │
│  ┌──────▼──────┐                        ┌──────▼──────┐        │
│  │  Buy Agent  │                        │  Sell Agent │        │
│  │    (PPO)    │                        │    (PPO)    │        │
│  └──────┬──────┘                        └──────┬──────┘        │
│         │                                      │                │
│         └──────────────────┬───────────────────┘                │
│                            │                                     │
│                   ┌────────▼────────┐                           │
│                   │    交易訊號      │                           │
│                   └─────────────────┘                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 專案結構 (Project Structure)

```
hybrid-trader-v02/
├── ptrl_hybrid_system.py        # 核心系統 (資料載入/特徵計算/訓練邏輯)
├── update_twii_data.py          # 資料更新腳本 (自動抓取最新 TWII 數據)
├── twii_data_from_2000_01_01.csv # 本地 TWII 歷史資料庫 (Volume: 億元)
├── train_v3_models.py           # V3 訓練腳本 (使用本地資料)
├── train_v4_models.py           # V4 訓練腳本 (使用本地資料)
├── daily_ops_dual.py            # 每日維運
├── backtest_v3_no_filter.py     # V3 回測 (使用本地資料)
├── backtest_v4_no_filter.py     # V4 回測 (使用本地資料)
└── backtest_v4_dca_hybrid_no_filter.py # DCA 混合回測 (使用本地資料)
```

## 🛠️ 安裝說明 (Installation)

### 建議使用虛擬環境 (Virtual Environment)
在 Windows 上使用虛擬環境可以避免套件版本衝突，強烈建議使用。

**方法一：使用自動腳本 (推薦)**
```powershell
.\setup_env.ps1
```

**方法二：手動設定**
```powershell
# 1. 建立虛擬環境
python -m venv venv

# 2. 啟動虛擬環境
.\venv\Scripts\Activate.ps1

# 3. 安裝套件
pip install -r requirements.txt
```

### ⚡ GPU 加速設定 (重要)
本專案建議使用 NVIDIA 顯卡進行訓練加速。

**方法一：使用 setup_env.ps1 (自動)**
腳本會自動安裝支援 CUDA 11.8 的 PyTorch 版本。

**方法二：手動安裝**
若您手動執行 `pip install -r requirements.txt`，預設會安裝 CPU 版本。請執行以下指令將其替換為 GPU 版本：

```powershell
# 1. 移除 CPU 版本
pip uninstall torch torchvision torchaudio -y

# 2. 安裝 GPU 版本 (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 系統需求 (Dependencies)

```
tensorflow>=2.10
stable-baselines3>=2.0
gymnasium
yfinance
pandas
numpy
ta
torch
tqdm
matplotlib
psutil
```

## 🚀 快速開始 (Quick Start)

### 1. 訓練 LSTM 模型 (長週期)

```bash
python train_lstm_models.py
```

### 2. 訓練 RL 模型 (V3 vs V4)

本專案提供兩個版本的 RL 訓練腳本，請依需求選擇：

| 特性 | V3 (Lightweight) | V4 (Standard) |
|------|------------------|---------------|
| **用途** | 輕量版，適合快速實驗 | 標準版，適合完整訓練 |
| **Buy Fine-tune** | 200,000 步 | 1,000,000 步 |
| **Sell Fine-tune** | 100,000 步 | 300,000 步 |
| **指令** | `python train_v3_models.py` | `python train_v4_models.py` |
| **輸出目錄** | `models_hybrid_v3/` | `models_hybrid_v4/` |

### 3. 每日維運 (Daily Operations)

自動化腳本能完成「LSTM 重訓 → 特徵工程 → RL 推論 → 報告生成」全流程。

- **雙策略比較 (推薦)**: 同時執行 V3 與 V4 模型並產生綜合建議。
  ```bash
  python daily_ops_dual.py
  ```

- **單策略運行**:
  ```bash
  python daily_ops_v3.py  # 僅 V3
  python daily_ops_v4.py  # 僅 V4
  ```

**功能特點 (v2.7)：**
- **全時推論模式**: 無論 Donchian 濾網狀態，AI 都會執行預測並顯示意圖
- **濾網狀態標記**: `BUY`, `WAIT`, `FILTERED (AI: BUY)`, `FILTERED (AI: WAIT)`
- **情境分析**: Sell Agent 針對三種持倉情境 (成本區/獲利+10%/虧損-5%) 提供建議
- **數據匯出**: 自動匯出 `raw_data.csv` 和 `processed_features.csv` 供除錯檢查
- **成交量修補**: 自動偵測 yfinance 的 Volume=0 異常並用昨日數據填補
- **統一訓練天數**: 三個 daily_ops 腳本均使用 T+5/2200天, T+1/2000天
- 輸出 JSON 與 TXT 戰情報告 (`daily_runs/YYYY-MM-DD/reports/`)

### 4. 無濾網回測 (No-Filter Backtest)

測試 AI 在每天都可進場 (無 Donchian 濾網限制) 的情況下的績效：

```bash
python backtest_v3_no_filter.py  # V3 無濾網回測
python backtest_v4_no_filter.py  # V4 無濾網回測
```

**功能特點 (v2.9)：**

| 功能 | 說明 |
|------|------|
| **自訂日期範圍** | 透過 `--start` 和 `--end` 參數指定回測期間 |
| **動態檔名** | 輸出檔案自動包含日期範圍，避免覆蓋 |
| **Benchmark 比較** | 策略績效 vs Buy & Hold 並排顯示 |
| **績效摘要** | 控制台、圖表、CSV 三處同步顯示 |

**使用範例：**
```bash
# 預設日期 (2024-01-01 至今)
python backtest_v4_no_filter.py

# 自訂開始日期
python backtest_v4_no_filter.py --start 2020-01-01

# 自訂完整日期範圍
python backtest_v4_no_filter.py --start 2015-01-01 --end 2023-12-31
```

### 5. DCA + AI 混合策略回測

測試「定期定額 + AI 自由操作」混合策略的績效：

```bash
python backtest_v4_dca_hybrid_no_filter.py
```

**策略說明：**
- 每年年初獲得 60 萬新資金
- 50% 分 12 個月定期定額投入 (買入後不賣)
- 50% 由 AI 自由決定買賣時機

**比較基準：**
1. 純定期定額：每月 5 萬元
2. 年初一次投入：每年 60 萬 Buy & Hold

**輸出檔案：**
```
results_backtest_v4_dca_hybrid_no_filter/
├── backtest_v4_dca_hybrid_no_filter_20240102_20251205.png
├── metrics_v4_dca_hybrid_no_filter_20240102_20251205.csv
└── trades_v4_dca_hybrid_no_filter_20240102_20251205.csv
```

### 🔍 回測腳本功能比較

| 功能 | `backtest_v3_no_filter.py` | `backtest_v4_no_filter.py` | `backtest_v4_dca_hybrid_no_filter.py` |
|------|:---:|:---:|:---:|
| 自訂日期範圍 | ✅ | ✅ | ✅ |
| 動態檔名 | ✅ | ✅ | ✅ |
| Benchmark 比較 | ✅ | ✅ | ✅ |
| DCA + AI 混合策略 | ❌ | ❌ | ✅ |
| **LSTM 模型日期篩選** | ❌ | ❌ | ✅ |

> [!IMPORTANT]
> **LSTM 模型日期篩選**：只有 `backtest_v4_dca_hybrid_no_filter.py` 會根據回測 start_date 來選擇 LSTM 模型，確保只使用 `train_end < start_date` 的模型，避免資料洩漏 (look-ahead bias)。其他兩個腳本使用當天日期選擇模型。

## 📈 訓練流程 (Training Pipeline)

### Phase 1: 數據整合 (Unified Data Source)
- **本地數據**: ^TWII 使用本地 `twii_data_from_2000_01_01.csv`，確保成交量單位正確 (億元)。
- **自動更新**: 系統自動檢查並透過 `update_twii_data.py` 補齊最新交易日資料。
- **國際指數**: 下載 4 個全球指數：^GSPC, ^IXIC, ^SOX, ^DJI (from yfinance)
- **影響範圍**: 涵蓋 V3/V4 訓練、所有回測腳本以及每日維運腳本 (Daily Ops)。

### Phase 2: 特徵工程 (Feature Engineering)
- 包含 23 種特徵：
  - 標準化 OHLC 價格
  - 唐奇安通道 (Donchian Channel)、超級趨勢 (SuperTrend)
  - 平均K線 (Heikin-Ashi) 型態
  - RSI, MFI, ATR 指標
  - 相對強度 (Relative Strength) 指標
  - **LSTM_Pred_1d**: T+1 預測漲幅
  - **LSTM_Pred_5d**: T+5 預測漲幅
  - **LSTM_Conf_5d**: T+5 信心度 (MC Dropout)

### Phase 3: 預訓練 (Pre-training)
- Buy Agent: 1,000,000 步 (類別平衡採樣)
- Sell Agent: 500,000 步

### Phase 4: 微調與回測 (Fine-tuning & Backtesting)
- 微調：針對 ^TWII (2000-2022) 進行訓練，Learning Rate = 1e-5
- 回測：驗證數據集 (2023-Present)

### Phase 5: 訓練監控 (Training Monitoring)
本系統整合了 **TensorBoard** 進行訓練過程的即時監控。

**自動記錄的指標：**
- `rollout/ep_rew_mean`: 平均獎勵
- `train/loss`: 總損失
- `train/policy_gradient_loss`: 策略梯度損失
- `train/value_loss`: 價值函數損失
- `train/entropy_loss`: 熵損失
- `eval/mean_reward`: 驗證集平均獎勵 (EvalCallback)

**如何使用 TensorBoard：**
```powershell
# 在專案目錄下執行
tensorboard --logdir ./tensorboard_logs/

# 然後開啟瀏覽器前往
# http://localhost:6006
```

**日誌存放位置：**
- `./tensorboard_logs/`: TensorBoard 日誌
- `./logs/`: EvalCallback 評估結果
- `models_hybrid/best_tuned/`: 驗證集最佳模型

---

## 📊 輸出結果 (Output)

執行 `ptrl_hybrid_system.py` 後，您將獲得：

- `models_hybrid/ppo_buy_twii_final.zip`: 微調後的 Buy Model
- `models_hybrid/ppo_sell_twii_final.zip`: 微調後的 Sell Model
- `results_hybrid/final_performance.png`: 績效圖表
- `tensorboard_logs/`: 訓練過程日誌 (可用 TensorBoard 查看)

## 🔧 V3 vs V4 版本比較

| 項目 | V3 (Lightweight) | V4 (Standard) | 原始版 (ptrl_hybrid_system.py) |
|-----|------------------|-----------------|--------------------------------|
| **Pre-train Buy** | 1,000,000 | 1,000,000 | 1,000,000 |
| **Pre-train Sell** | 500,000 | 500,000 | 500,000 |
| **Fine-tune Buy** | **200,000** | **1,000,000** | 1,000,000 |
| **Fine-tune Sell** | **100,000** | **300,000** | 300,000 |
| **信心度門檻** | [0.001, 0.010] v2.5 | [0.001, 0.010] v2.5 | [0.005, 0.015] (舊版) |
| **特徵快取** | 強制清除 | 強制清除 | 使用快取 (需手動清除) |
| **模型路徑** | `models_hybrid_v3` | `models_hybrid_v4` | `models_hybrid` |

---

## 🔮 LSTM 信心度解讀指南 (Confidence Interpretation)

### 計算原理 (Methodology)
信心度 (`LSTM_Conf_5d`) 是基於 **蒙地卡羅 Dropout (MC Dropout)** 計算的：
1. 對同一筆資料進行 30 次預測（每次 Dropout 隨機遮蔽不同神經元）
2. 計算這 30 次預測的**變異係數 (CV)** = 標準差 ÷ 平均值
3. CV 越小 → 模型越穩定 → 信心度越高

### 門檻設定 (v2.5)
```python
# ptrl_hybrid_system.py (Line 336-340)
threshold_high = 0.001  # CV <= 0.1% → 信心度 = 1.0
threshold_low  = 0.010  # CV >= 1.0% → 信心度 = 0.0
score = 1.0 - (cv - 0.001) / (0.010 - 0.001)
conf_5d = np.clip(score, 0.0, 1.0)
```

### 分數對照表

| 信心度 | CV 範圍 | 解讀 | 建議 |
|--------|---------|------|------|
| **0.9+** | < 0.2% | 🟢 **極高信心** - 模型非常確定 | 預測可靠度高，可作為主要參考 |
| **0.8-0.9** | 0.2%-0.3% | 🟢 **高信心** - 模型相當穩定 | 預測值得信賴 |
| **0.7-0.8** | 0.3%-0.4% | 🟡 **中等偏高** - 正常水準 | 預測可參考，但需結合其他指標 |
| **0.6-0.7** | 0.4%-0.6% | 🟡 **中等** - 略有不確定性 | 預測僅供輔助參考 |
| **< 0.6** | > 0.6% | 🔴 **低信心** - 模型不確定 | 預測不穩定，謹慎採信 |

### 實際應用建議
1. **信心度 0.8+**：可以更積極地參考 LSTM 的漲跌預測
2. **信心度 0.7**：預測方向可參考，但點位預估需打折扣
3. **信心度 < 0.6**：模型對當天的判斷較不確定，可能是因為市場處於異常波動期

---

## 📚 參考文獻 (References)

- **Pro Trader RL**: [Paper Implementation](https://arxiv.org/abs/xxxx)
- **LSTM-SSAM**: Sequential Self-Attention for time series prediction
- **MC Dropout**: Uncertainty estimation via Monte Carlo Dropout

## 📄 授權 (License)

MIT License

## 👤 作者 (Author)

Phil Liang

---

*Built with Python, TensorFlow, Stable-Baselines3, and ❤️*
