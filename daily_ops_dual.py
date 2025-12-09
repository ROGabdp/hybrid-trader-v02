# -*- coding: utf-8 -*-
"""
================================================================================
Daily Operations with Dual Strategy & Versioning (v2.1 - Patched)
================================================================================
每日維運腳本 - 雙策略推論與版本控管

修正紀錄 (v2.2):
1. [Fix] Step 1 改為直接呼叫 model registry 腳本，並傳入動態日期 (確保模型更新至今日)
2. [Fix] Step 2 補上 target_scaler 的載入與注入 (防止 inverse_transform 失敗)
3. [Safety] 增加 import 檢查與錯誤處理
4. [Fix] yfinance end_date 加一天 (因為 yf.download 的 end 是 exclusive)
5. [Fix] 使用實際下載資料的最後日期作為工作區日期 (避免週末/盤中執行時日期不符)
6. [Safety] meta.json 載入加上 try-except 防護

作者：Phil Liang (Fixed by Gemini)
日期：2025-12-07 (v2.2 Updated)
================================================================================
"""

import os
import sys
import shutil
import pickle
import subprocess
import json
import glob
from datetime import datetime, timedelta

# 設定 UTF-8 輸出
sys.stdout.reconfigure(encoding='utf-8')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow import keras
from keras import layers

# =============================================================================
# 引用主系統
# =============================================================================
import ptrl_hybrid_system as core_system

# =============================================================================
# 設定路徑
# =============================================================================
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
DAILY_RUNS_PATH = os.path.join(PROJECT_PATH, 'daily_runs')

# RL 模型路徑 (V3 vs V4)
STRATEGY_A_PATH = os.path.join(PROJECT_PATH, 'models_hybrid_v3')  # V3 (高步數微調)
STRATEGY_B_PATH = os.path.join(PROJECT_PATH, 'models_hybrid_v4')  # V4 (輕量化微調)

# LSTM 訓練腳本名稱 (必須存在於同一目錄下)
SCRIPT_5D = "twii_model_registry_5d.py"
SCRIPT_1D = "twii_model_registry_multivariate.py"

# LSTM 模型預設輸出路徑 (訓練腳本預設會存到這裡)
DEFAULT_LSTM_5D_DIR = os.path.join(PROJECT_PATH, 'saved_models_5d')
DEFAULT_LSTM_1D_DIR = os.path.join(PROJECT_PATH, 'saved_models_multivariate')


# =============================================================================
# Step 0: 建立當日專屬工作區
# =============================================================================
def create_daily_workspace(date_str: str) -> dict:
    daily_path = os.path.join(DAILY_RUNS_PATH, date_str)
    paths = {
        'root': daily_path,
        'lstm_models': os.path.join(daily_path, 'lstm_models'),
        'lstm_5d': os.path.join(daily_path, 'lstm_models', 'saved_models_5d'),
        'lstm_1d': os.path.join(daily_path, 'lstm_models', 'saved_models_multivariate'),
        'cache': os.path.join(daily_path, 'cache'),
        'reports': os.path.join(daily_path, 'reports'),
    }
    for key, path in paths.items():
        os.makedirs(path, exist_ok=True)
    print(f"[Workspace] 建立當日工作區: {daily_path}")
    return paths


# =============================================================================
# Step 1: LSTM 全量重訓與封存 (v2.3 - 動態天數 + 全量學習)
# =============================================================================
def train_and_archive_lstm(workspace: dict, end_date: str):
    print("\n" + "=" * 60)
    print("📚 Step 1: LSTM 全量重訓與封存")
    print("=" * 60)
    
    # [v2.3] 動態計算起始日期
    # T+5 模型：使用過去 2200 天（約 2020-01 起）- 捕捉更長趨勢
    # T+1 模型：使用過去 2000 天（約 2020-07 起）- 專注近期市場
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    start_5d = (end_dt - timedelta(days=2200)).strftime('%Y-%m-%d')
    start_1d = (end_dt - timedelta(days=2000)).strftime('%Y-%m-%d')
    
    # [v2.3] 啟用全量學習：split_ratio = 0.99
    # 只保留 1% 作為 Early Stopping 監控，確保模型學習到最新資料
    split_ratio = "0.99"
    
    # 1. 執行 T+5 訓練 (傳入動態日期 + split_ratio)
    print(f"\n[Training] T+5 Model ({start_5d} ~ {end_date}, split={split_ratio})...")
    script_5d_path = os.path.join(PROJECT_PATH, SCRIPT_5D)
    cmd_5d = [sys.executable, script_5d_path, "train", "--start", start_5d, "--end", end_date, "--split_ratio", split_ratio]
    try:
        subprocess.run(cmd_5d, check=True, timeout=1200, cwd=PROJECT_PATH)  # 確保工作目錄正確
        print("[Training] ✅ T+5 訓練完成")
    except subprocess.CalledProcessError as e:
        print(f"[Error] T+5 訓練失敗: {e}")
        return False
    except FileNotFoundError:
        print(f"[Error] 找不到訓練腳本: {script_5d_path}")
        return False
    except Exception as e:
        print(f"[Error] 執行錯誤: {e}")
        return False

    # 2. 執行 T+1 訓練 (傳入動態日期 + split_ratio)
    print(f"\n[Training] T+1 Model ({start_1d} ~ {end_date}, split={split_ratio})...")
    script_1d_path = os.path.join(PROJECT_PATH, SCRIPT_1D)
    cmd_1d = [sys.executable, script_1d_path, "train", "--start", start_1d, "--end", end_date, "--split_ratio", split_ratio]
    try:
        subprocess.run(cmd_1d, check=True, timeout=1200, cwd=PROJECT_PATH)
        print("[Training] ✅ T+1 訓練完成")
    except subprocess.CalledProcessError as e:
        print(f"[Error] T+1 訓練失敗: {e}")
        return False
    except FileNotFoundError:
        print(f"[Error] 找不到訓練腳本: {script_1d_path}")
        return False

    # 3. 封存模型 (Copy from default dir to daily dir)
    print("\n[Archive] 封存模型到當日工作區...")
    
    def archive_dir(src_dir, dest_dir):
        if os.path.exists(src_dir):
            if os.path.exists(dest_dir):
                shutil.rmtree(dest_dir) # 清空舊的
            shutil.copytree(src_dir, dest_dir)
            print(f"  ✅ 已封存: {os.path.basename(src_dir)} -> {dest_dir}")
        else:
            print(f"  ⚠️ 來源目錄不存在: {src_dir}")

    archive_dir(DEFAULT_LSTM_5D_DIR, workspace['lstm_5d'])
    archive_dir(DEFAULT_LSTM_1D_DIR, workspace['lstm_1d'])
    
    return True


# =============================================================================
# Step 2: 隔離式特徵工程 (修正版)
# =============================================================================
def isolated_feature_engineering(workspace: dict, end_date: str) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("🔧 Step 2: 隔離式特徵工程 (模型注入)")
    print("=" * 60)
    
    # [修正] 直接從原始訓練腳本引用正確的 Layer 定義
    # 這樣確保數學運算邏輯 (Attention Score 計算) 與訓練時完全一致
    try:
        from twii_model_registry_5d import SelfAttention
        print("[System] 成功引用原始 SelfAttention 類別")
    except ImportError:
        print("[Error] 無法引用 twii_model_registry_5d，請確認檔案是否存在")
        sys.exit(1)

    # 輔助函式：載入整組模型元件
    def load_model_components(model_dir):
        keras_files = glob.glob(os.path.join(model_dir, "*.keras"))
        if not keras_files: return None, None, None, None
        
        # 找最新的模型檔
        latest_keras = sorted(keras_files)[-1]
        print(f"  ...Loading {os.path.basename(latest_keras)}")
        
        # [修正] 載入模型時使用正確的 Custom Object
        model = keras.models.load_model(latest_keras, custom_objects={'SelfAttention': SelfAttention})

        # 載入 Meta (加上錯誤防護)
        meta_file = latest_keras.replace('model_', 'meta_').replace('.keras', '.json')
        meta = {}
        if os.path.exists(meta_file):
            try:
                with open(meta_file, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
            except Exception as e:
                print(f"  ⚠️ 載入 meta 失敗: {e}")

        # 載入 Feature Scaler
        scaler_feat_file = latest_keras.replace('model_', 'feature_scaler_').replace('.keras', '.pkl')
        if not os.path.exists(scaler_feat_file):
             scaler_feat_file = latest_keras.replace('model_', 'scaler_').replace('.keras', '.pkl')
        
        scaler_feat = None
        if os.path.exists(scaler_feat_file):
            with open(scaler_feat_file, 'rb') as f:
                scaler_feat = pickle.load(f)

        # 載入 Target Scaler
        scaler_tgt_file = latest_keras.replace('model_', 'target_scaler_').replace('.keras', '.pkl')
        if not os.path.exists(scaler_tgt_file):
             scaler_tgt = scaler_feat
        else:
             with open(scaler_tgt_file, 'rb') as f:
                 scaler_tgt = pickle.load(f)

        return model, scaler_feat, scaler_tgt, meta

    # 1. 載入模型
    print("\n[Model Injection] 載入當日封存的 LSTM 模型...")
    m5d, sf5d, st5d, meta5d = load_model_components(workspace['lstm_5d'])
    m1d, sf1d, st1d, meta1d = load_model_components(workspace['lstm_1d'])
    
    if m5d is None or m1d is None:
        print("[Error] 模型載入失敗，無法進行特徵工程")
        sys.exit(1)

    # 2. 注入主系統
    print("\n[Model Injection] 注入 core_system._LSTM_MODELS...")
    if not hasattr(core_system, '_LSTM_MODELS'):
        core_system._LSTM_MODELS = {}
    
    core_system._LSTM_MODELS.update({
        'model_5d': m5d, 'scaler_feat_5d': sf5d, 'scaler_tgt_5d': st5d, 'meta_5d': meta5d,
        'model_1d': m1d, 'scaler_feat_1d': sf1d, 'scaler_tgt_1d': st1d, 'meta_1d': meta1d,
        'loaded': True
    })
    print("  ✅ 注入完成 (含 Target Scalers)")

    # 3. 下載數據 & 計算特徵
    # [修正] yfinance 的 end 參數是 exclusive，需要加一天才能包含當日
    end_dt = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)
    download_end = end_dt.strftime('%Y-%m-%d')
    print(f"\n[Compute] Loading Data (2020-01-01 ~ {end_date})...")
    # [Modify] 使用本地資料載入函數
    raw_df = core_system._load_local_twii_data(start_date="2020-01-01")
    
    # 篩選日期範圍 (確保不超過 end_date)
    end_dt_ts = pd.Timestamp(end_date)
    raw_df = raw_df[raw_df.index <= end_dt_ts]
    
    # [修正] 取得實際下載資料的最後日期 (避免週末/盤中執行時日期不符)
    actual_last_date = raw_df.index[-1].strftime('%Y-%m-%d')
    print(f"[Data] 實際資料最後日期: {actual_last_date}")

    # [Note] Local CSV 成交量已是正確單位 (億元) 且已補值，無需額外處理
    
    # [v2.6] 匯出原始數據 CSV (修補後)
    raw_csv_path = os.path.join(workspace['cache'], 'raw_data.csv')
    raw_df.to_csv(raw_csv_path)
    print(f"[Export] 原始數據已存檔: {raw_csv_path}")
    
    print(f"[Compute] 計算特徵中 (使用當日模型)...")
    # 強制不使用快取，確保重新計算
    df = core_system.calculate_features(raw_df, raw_df, ticker="^TWII", use_cache=False)
    
    # [v2.6] 匯出處理後特徵數據 CSV
    features_csv_path = os.path.join(workspace['cache'], 'processed_features.csv')
    df.to_csv(features_csv_path)
    print(f"[Export] 特徵數據已存檔: {features_csv_path}")
    
    # 存入當日快取 (pkl 格式，供後續載入使用)
    cache_file = os.path.join(workspace['cache'], 'twii_features.pkl')
    with open(cache_file, 'wb') as f:
        pickle.dump(df, f)
    print(f"[Cache] 特徵已存檔: {cache_file}")
    
    return df, actual_last_date  # [修正] 回傳實際日期供報告使用


# =============================================================================
# Step 3: 雙模型推論 (v2.4 - 濾網 + 情境分析)
# =============================================================================
def dual_inference(workspace: dict, df: pd.DataFrame) -> dict:
    print("\n" + "=" * 60)
    print("🎯 Step 3: 雙模型推論 (含濾網與情境分析)")
    print("=" * 60)
    
    from stable_baselines3 import PPO
    
    # 準備特徵
    FEATURE_COLS = core_system.FEATURE_COLS
    latest = df.iloc[-1]
    
    # [v2.4] 獲取濾網狀態
    signal_buy_filter = bool(latest.get('Signal_Buy_Filter', False))
    print(f"  [濾網] Signal_Buy_Filter = {signal_buy_filter}")
    
    # 確保特徵欄位對齊
    features = []
    for col in FEATURE_COLS:
        val = latest.get(col, 0.0)
        features.append(val)
    features = np.array(features, dtype=np.float32).reshape(1, -1)
    
    # 處理 NaN/Inf
    features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
    
    results = {'filter_status': signal_buy_filter}
    
    # [v2.4] 三種持倉情境
    SELL_SCENARIOS = {
        'cost': 1.00,    # 成本區 (剛進場)
        'profit': 1.10,  # 獲利中 (+10%)
        'loss': 0.95,    # 虧損中 (-5%)
    }
    
    def run_strategy(name, path, key):
        buy_path = os.path.join(path, 'ppo_buy_twii_final.zip')
        sell_path = os.path.join(path, 'ppo_sell_twii_final.zip')
        
        if not os.path.exists(buy_path):
            results[key] = {'error': 'Model not found'}
            print(f"  [Warning] {name}: 模型不存在")
            return

        try:
            buy_agent = PPO.load(buy_path)
            sell_agent = PPO.load(sell_path)
            
            # =====================================================================
            # Buy Logic (v2.5 - 全時推論 + 狀態標記)
            # =====================================================================
            # 步驟 A: 無論濾網狀態，一律執行 AI 預測
            b_act, _ = buy_agent.predict(features, deterministic=True)
            b_obs = buy_agent.policy.obs_to_tensor(features)[0]
            b_prob = buy_agent.policy.get_distribution(b_obs).distribution.probs.detach().cpu().numpy()[0]
            
            ai_action = 'BUY' if b_act[0] == 1 else 'WAIT'
            buy_prob = float(b_prob[1]) if b_act[0] == 1 else float(b_prob[0])
            
            # 步驟 B: 根據濾網狀態決定最終顯示字串
            if signal_buy_filter:
                # 濾網通過
                buy_signal = ai_action  # "BUY" 或 "WAIT"
            else:
                # 濾網未過：標記為 FILTERED 但顯示 AI 原始判斷
                buy_signal = f"FILTERED (AI: {ai_action})"
            
            print(f"  [{name}] Buy: {buy_signal} ({buy_prob:.1%})")
            
            # =====================================================================
            # Sell Logic (情境分析) - 保持不變
            # =====================================================================
            sell_scenarios = {}
            for scenario_name, return_value in SELL_SCENARIOS.items():
                s_feat = np.concatenate([features[0], [return_value]]).reshape(1, -1)
                s_act, _ = sell_agent.predict(s_feat, deterministic=True)
                sell_scenarios[scenario_name] = 'SELL' if s_act[0] == 1 else 'HOLD'
            
            print(f"  [{name}] Sell: 成本={sell_scenarios['cost']} | 獲利={sell_scenarios['profit']} | 虧損={sell_scenarios['loss']}")
            
            results[key] = {
                'name': name,
                'buy_signal': buy_signal,
                'buy_prob': buy_prob,
                'ai_action': ai_action,  # 新增：AI 原始判斷
                'sell_scenarios': sell_scenarios,
            }
            
        except Exception as e:
            results[key] = {'error': str(e)}
            print(f"  [Error] {name}: {e}")
            import traceback
            traceback.print_exc()

    # 執行 A (V3)
    run_strategy("V3 (Fine-tune 1M)", STRATEGY_A_PATH, 'A')
    
    # 執行 B (V4)
    run_strategy("V4 (Fine-tune 200K)", STRATEGY_B_PATH, 'B')
    
    return results


# =============================================================================
# Step 4: 輸出報告 (v2.4 - 濾網 + 情境分析)
# =============================================================================
def generate_report(workspace: dict, df: pd.DataFrame, res: dict, date_str: str):
    print("\n" + "=" * 60)
    print("📊 Step 4: 戰情儀表板 (v2.4)")
    print("=" * 60)
    
    last = df.iloc[-1]
    filter_status = res.get('filter_status', False)
    
    lines = []
    lines.append("=" * 50)
    lines.append(f"📅 日期: {date_str}")
    lines.append("=" * 50)
    lines.append(f"📊 收盤: {last['Close']:.2f} | 量: {last['Volume']:.2f} 億元")
    lines.append("-" * 50)
    
    # 濾網狀態
    filter_icon = "✅" if filter_status else "🚫"
    filter_text = "通過 (Donchian 突破)" if filter_status else "未通過 (非突破日)"
    lines.append(f"� [濾網狀態] {filter_icon} {filter_text}")
    lines.append("-" * 50)
    
    # LSTM
    lines.append("�🔮 [分析師 LSTM]")
    lines.append(f"   T+1 漲跌: {last.get('LSTM_Pred_1d', 0)*100:+.2f}%")
    lines.append(f"   T+5 漲跌: {last.get('LSTM_Pred_5d', 0)*100:+.2f}%")
    lines.append(f"   信心度:   {last.get('LSTM_Conf_5d', 0)*100:.1f}%")
    lines.append("-" * 50)
    
    # RL 策略 (含情境分析)
    lines.append("🤖 [操盤手 RL]")
    
    def format_strategy(key, label):
        if key not in res or 'error' in res[key]:
            return [f"   {label}: ❌ 模型載入失敗"]
        
        r = res[key]
        result_lines = []
        
        # Buy (v2.5 全時推論格式)
        buy_signal = r['buy_signal']
        buy_prob = r['buy_prob']
        
        if buy_signal == 'BUY':
            buy_icon = "🚀"
        elif buy_signal == 'WAIT':
            buy_icon = "💤"
        elif 'FILTERED' in buy_signal:
            buy_icon = "🚫"
        else:
            buy_icon = "❓"
        
        result_lines.append(f"   🛒 {label} 買入: {buy_icon} {buy_signal} ({buy_prob:.1%})")
        
        # Sell (情境矩陣)
        ss = r.get('sell_scenarios', {})
        result_lines.append(f"   📦 {label} 賣出:")
        result_lines.append(f"      ├─ 成本區 (0%):  {ss.get('cost', 'N/A')}")
        result_lines.append(f"      ├─ 獲利中 (+10%): {ss.get('profit', 'N/A')}")
        result_lines.append(f"      └─ 虧損中 (-5%):  {ss.get('loss', 'N/A')}")
        
        return result_lines
    
    lines.extend(format_strategy('A', 'V3'))
    lines.append("")
    lines.extend(format_strategy('B', 'V4'))
    lines.append("-" * 50)
    
    # 綜合建議 (使用 ai_action 而非 buy_signal 判斷 AI 意圖)
    ai_a = res.get('A', {}).get('ai_action', 'N/A')
    ai_b = res.get('B', {}).get('ai_action', 'N/A')
    
    if not filter_status:
        # 濾網未過，但顯示 AI 想法
        if ai_a == 'BUY' and ai_b == 'BUY':
            advice = "🚫 濾網攔截 | AI 意圖: 雙買進 (被擋下)"
        elif ai_a == 'BUY' or ai_b == 'BUY':
            advice = "🚫 濾網攔截 | AI 意圖: 有意買進 (被擋下)"
        else:
            advice = "🚫 濾網攔截 | AI 意圖: 觀望"
    elif ai_a == 'BUY' and ai_b == 'BUY':
        advice = "⭐⭐ V3+V4 雙買進 (Strong Buy) ⭐⭐"
    elif ai_a == 'WAIT' and ai_b == 'WAIT':
        advice = "💤 空手觀望 (Wait)"
    elif ai_a == 'BUY':
        advice = "⚠️ 僅 V3 買進 (V3 Only)"
    elif ai_b == 'BUY':
        advice = "⚠️ 僅 V4 買進 (V4 Only)"
    else:
        advice = "❓ 訊號不明"
        
    lines.append(f"💡 綜合建議: {advice}")
    lines.append("=" * 50)
    
    report = "\n".join(lines)
    print(report)
    
    # 存檔 TXT
    txt_path = os.path.join(workspace['reports'], 'summary.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    # 存檔 JSON
    json_path = os.path.join(workspace['reports'], 'summary.json')
    json_data = {
        'date': date_str,
        'generated_at': datetime.now().isoformat(),
        'filter_status': filter_status,
        'market': {
            'close': float(last.get('Close', 0)),
            'volume': float(last.get('Volume', 0)),
        },
        'lstm': {
            'pred_1d': float(last.get('LSTM_Pred_1d', 0)),
            'pred_5d': float(last.get('LSTM_Pred_5d', 0)),
            'conf_5d': float(last.get('LSTM_Conf_5d', 0)),
        },
        'strategies': {
            'A': res.get('A', {}),
            'B': res.get('B', {}),
        },
        'advice': advice,
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n[Report] 已儲存: {txt_path}")
    print(f"[Report] 已儲存: {json_path}")


# =============================================================================
# Main
# =============================================================================
def main():
    today = datetime.now()
    # 處理週末 (往前推到週五) - 用於初步估計日期
    if today.weekday() == 5: today -= timedelta(days=1)
    elif today.weekday() == 6: today -= timedelta(days=2)
    
    date_str = today.strftime('%Y-%m-%d')
    print(f"🚀 啟動每日維運系統 - {date_str}")
    
    # Step 0
    ws = create_daily_workspace(date_str)
    
    # Step 1 (Train up to Today)
    train_and_archive_lstm(ws, date_str)
    
    # Step 2 - [修正] 接收實際資料日期
    df, actual_date = isolated_feature_engineering(ws, date_str)
    
    # [修正] 如果實際日期與預估日期不同，顯示警告
    if actual_date != date_str:
        print(f"[Warning] 預估日期 {date_str} 與實際資料日期 {actual_date} 不同")
        print(f"[Info] 報告將使用實際資料日期: {actual_date}")
    
    # Step 3
    res = dual_inference(ws, df)
    
    # Step 4 - [修正] 使用實際資料日期生成報告
    generate_report(ws, df, res, actual_date)

if __name__ == "__main__":
    main()