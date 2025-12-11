# -*- coding: utf-8 -*-
"""
================================================================================
Daily Operations - Intraday Version (盤中完整訓練+預測版)
================================================================================
盤中執行腳本 - 完整獨立運作

流程:
1. 從 yfinance 下載當日 OHLC 資料 (盤中即時)
2. 成交量使用 CSV 歷史資料的前 5 日平均 (因盤中無法取得準確成交量)
3. 使用上述資料訓練 LSTM 模型 (T+5 及 T+1)
4. 使用新訓練的 LSTM 進行特徵工程與預測
5. 輸出結果到 intraday_runs/{date}_{time}/ (獨立資料夾，不影響 daily_runs)
6. 不寫入 twii_data_from_2000_01_01.csv

作者：Phil Liang
日期：2025-12-11
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
INTRADAY_RUNS_PATH = os.path.join(PROJECT_PATH, 'intraday_runs')  # 獨立資料夾
CSV_FILE = os.path.join(PROJECT_PATH, 'twii_data_from_2000_01_01.csv')

# RL 模型路徑 (V3 vs V4)
STRATEGY_A_PATH = os.path.join(PROJECT_PATH, 'models_hybrid_v3')  # V3 (輕量化微調)
STRATEGY_B_PATH = os.path.join(PROJECT_PATH, 'models_hybrid_v4')  # V4 (標準完整微調)

# LSTM 訓練腳本名稱
SCRIPT_5D = "twii_model_registry_5d.py"
SCRIPT_1D = "twii_model_registry_multivariate.py"

# LSTM 模型預設輸出路徑
DEFAULT_LSTM_5D_DIR = os.path.join(PROJECT_PATH, 'saved_models_5d')
DEFAULT_LSTM_1D_DIR = os.path.join(PROJECT_PATH, 'saved_models_multivariate')


# =============================================================================
# Step 0: 建立盤中專屬工作區 (使用 intraday_runs)
# =============================================================================
def create_intraday_workspace(date_str: str, time_str: str) -> dict:
    folder_name = f"{date_str}_{time_str}"
    intraday_path = os.path.join(INTRADAY_RUNS_PATH, folder_name)
    paths = {
        'root': intraday_path,
        'lstm_models': os.path.join(intraday_path, 'lstm_models'),
        'lstm_5d': os.path.join(intraday_path, 'lstm_models', 'saved_models_5d'),
        'lstm_1d': os.path.join(intraday_path, 'lstm_models', 'saved_models_multivariate'),
        'cache': os.path.join(intraday_path, 'cache'),
        'reports': os.path.join(intraday_path, 'reports'),
    }
    for key, path in paths.items():
        os.makedirs(path, exist_ok=True)
    print(f"[Workspace] 建立盤中工作區: {intraday_path}")
    return paths


# =============================================================================
# 輔助函式: 取得盤中 OHLC (from yfinance)
# =============================================================================
def fetch_intraday_ohlc(ticker: str = "^TWII") -> tuple:
    """
    從 yfinance 下載當日 OHLC 資料
    
    Returns:
        tuple: (date_str, open, high, low, close) or None if failed
    """
    print(f"\n[Download] 正在從 yfinance 下載 {ticker} 當日資料...")
    
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="5d")
        
        if df.empty:
            print(f"[Error] yfinance 無法取得 {ticker} 資料")
            return None
        
        latest = df.iloc[-1]
        latest_date = df.index[-1].strftime('%Y-%m-%d')
        
        print(f"  📅 日期: {latest_date}")
        print(f"  📈 Open: {latest['Open']:.2f}")
        print(f"  📊 High: {latest['High']:.2f}")
        print(f"  📉 Low: {latest['Low']:.2f}")
        print(f"  💰 Close: {latest['Close']:.2f}")
        
        return (latest_date, latest['Open'], latest['High'], latest['Low'], latest['Close'])
        
    except Exception as e:
        print(f"[Error] yfinance 下載失敗: {e}")
        return None


# =============================================================================
# 輔助函式: 取得前 5 日成交量平均
# =============================================================================
def get_avg_volume_from_csv(n_days: int = 5) -> float:
    """
    從 CSV 檔案讀取最近 N 日的成交量平均
    """
    print(f"\n[Volume] 從 CSV 計算前 {n_days} 日成交量平均...")
    
    try:
        df = pd.read_csv(CSV_FILE)
        volumes = df['volume'].tail(n_days)
        avg_vol = volumes.mean()
        
        print(f"  📊 前 {n_days} 日成交量: {volumes.tolist()}")
        print(f"  📈 平均成交量: {avg_vol:.2f} 億元")
        
        return avg_vol
        
    except Exception as e:
        print(f"[Error] 讀取 CSV 失敗: {e}")
        return 3000.0


# =============================================================================
# 輔助函式: 建立暫存 CSV (用於 LSTM 訓練)
# =============================================================================
def create_temp_csv_with_intraday(intraday_data: tuple, avg_volume: float, workspace: dict) -> str:
    """
    建立暫存 CSV 檔案，包含歷史資料 + 盤中資料
    僅用於 LSTM 訓練，不影響原始 CSV
    
    Returns:
        str: 暫存 CSV 路徑
    """
    print("\n[TempCSV] 建立暫存訓練資料...")
    
    # 1. 讀取原始 CSV
    df = pd.read_csv(CSV_FILE)
    
    # 2. 加入當日資料
    date_str, o, h, l, c = intraday_data
    # 轉換日期格式為 CSV 格式 (YYYY/M/D)
    dt = datetime.strptime(date_str, '%Y-%m-%d')
    csv_date = f"{dt.year}/{dt.month}/{dt.day}"
    
    # 檢查是否已存在當日資料
    last_date = df['date'].iloc[-1]
    last_dt = datetime.strptime(last_date, '%Y/%m/%d')
    
    if last_dt.date() == dt.date():
        # 更新最後一筆
        print(f"  [Info] 更新 {csv_date} 的資料為盤中數據")
        df.iloc[-1] = [csv_date, o, h, l, c, avg_volume]
    else:
        # 新增一筆
        print(f"  [Info] 加入盤中資料: {csv_date}")
        new_row = pd.DataFrame({
            'date': [csv_date],
            'open': [o],
            'high': [h],
            'low': [l],
            'close': [c],
            'volume': [avg_volume]
        })
        df = pd.concat([df, new_row], ignore_index=True)
    
    # 3. 存檔到暫存位置
    temp_csv_path = os.path.join(workspace['cache'], 'temp_twii_data.csv')
    df.to_csv(temp_csv_path, index=False)
    print(f"  ✅ 暫存 CSV 已建立: {temp_csv_path}")
    print(f"  📊 資料範圍: {df['date'].iloc[0]} ~ {df['date'].iloc[-1]} ({len(df)} 筆)")
    
    return temp_csv_path


# =============================================================================
# Step 1: LSTM 訓練 (使用 CSV 交換策略)
# =============================================================================
def train_lstm_with_intraday(workspace: dict, temp_csv_path: str, end_date: str):
    """
    使用含盤中資料訓練 LSTM 模型
    
    策略：暫時交換 CSV 檔案
    1. 備份原始 CSV
    2. 用暫存 CSV 覆蓋原始 CSV
    3. 執行訓練
    4. 恢復原始 CSV (無論成功與否)
    """
    print("\n" + "=" * 60)
    print("📚 Step 1: LSTM 訓練 (含盤中資料)")
    print("=" * 60)
    
    # 計算訓練日期範圍
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    start_5d = (end_dt - timedelta(days=2200)).strftime('%Y-%m-%d')
    start_1d = (end_dt - timedelta(days=2000)).strftime('%Y-%m-%d')
    split_ratio = "0.99"
    
    # 備份原始 CSV
    backup_csv_path = CSV_FILE + ".bak"
    print(f"\n[Backup] 備份原始 CSV -> {backup_csv_path}")
    shutil.copy2(CSV_FILE, backup_csv_path)
    
    # 用暫存 CSV 覆蓋原始 CSV
    print(f"[Swap] 覆蓋原始 CSV 為盤中資料")
    shutil.copy2(temp_csv_path, CSV_FILE)
    
    training_success = True
    
    try:
        # 訓練 T+5 模型
        print(f"\n[Training] T+5 Model ({start_5d} ~ {end_date})...")
        script_5d_path = os.path.join(PROJECT_PATH, SCRIPT_5D)
        cmd_5d = [
            sys.executable, script_5d_path, "train",
            "--start", start_5d,
            "--end", end_date,
            "--split_ratio", split_ratio
        ]
        subprocess.run(cmd_5d, check=True, timeout=1200, cwd=PROJECT_PATH)
        print("[Training] ✅ T+5 訓練完成")

        # 訓練 T+1 模型
        print(f"\n[Training] T+1 Model ({start_1d} ~ {end_date})...")
        script_1d_path = os.path.join(PROJECT_PATH, SCRIPT_1D)
        cmd_1d = [
            sys.executable, script_1d_path, "train",
            "--start", start_1d,
            "--end", end_date,
            "--split_ratio", split_ratio
        ]
        subprocess.run(cmd_1d, check=True, timeout=1200, cwd=PROJECT_PATH)
        print("[Training] ✅ T+1 訓練完成")
        
    except subprocess.CalledProcessError as e:
        print(f"[Error] 訓練失敗: {e}")
        training_success = False
    except FileNotFoundError as e:
        print(f"[Error] 找不到訓練腳本: {e}")
        training_success = False
    except Exception as e:
        print(f"[Error] 執行錯誤: {e}")
        training_success = False
    finally:
        # 無論成功與否，都要恢復原始 CSV
        print(f"\n[Restore] 恢復原始 CSV")
        shutil.copy2(backup_csv_path, CSV_FILE)
        os.remove(backup_csv_path)
        print("[Restore] ✅ 原始 CSV 已恢復")
    
    if not training_success:
        return False

    # 封存模型到盤中工作區
    print("\n[Archive] 封存模型到盤中工作區...")
    
    def archive_dir(src_dir, dest_dir):
        if os.path.exists(src_dir):
            if os.path.exists(dest_dir):
                shutil.rmtree(dest_dir)
            shutil.copytree(src_dir, dest_dir)
            print(f"  ✅ 已封存: {os.path.basename(src_dir)} -> {dest_dir}")
        else:
            print(f"  ⚠️ 來源目錄不存在: {src_dir}")

    archive_dir(DEFAULT_LSTM_5D_DIR, workspace['lstm_5d'])
    archive_dir(DEFAULT_LSTM_1D_DIR, workspace['lstm_1d'])
    
    return True


# =============================================================================
# Step 2: 隔離式特徵工程 (使用盤中訓練的模型)
# =============================================================================
def isolated_feature_engineering_intraday(workspace: dict, intraday_data: tuple, avg_volume: float) -> pd.DataFrame:
    """
    盤中版特徵工程 - 使用剛訓練完的 LSTM 模型
    """
    print("\n" + "=" * 60)
    print("🔧 Step 2: 特徵工程 (使用盤中訓練模型)")
    print("=" * 60)
    
    # 引用 SelfAttention
    try:
        from twii_model_registry_5d import SelfAttention
        print("[System] 成功引用原始 SelfAttention 類別")
    except ImportError:
        print("[Error] 無法引用 twii_model_registry_5d")
        sys.exit(1)

    def load_model_components(model_dir):
        keras_files = glob.glob(os.path.join(model_dir, "*.keras"))
        if not keras_files: return None, None, None, None
        
        latest_keras = sorted(keras_files)[-1]
        print(f"  ...Loading {os.path.basename(latest_keras)}")
        
        model = keras.models.load_model(latest_keras, custom_objects={'SelfAttention': SelfAttention})

        meta_file = latest_keras.replace('model_', 'meta_').replace('.keras', '.json')
        meta = {}
        if os.path.exists(meta_file):
            try:
                with open(meta_file, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
            except Exception as e:
                print(f"  ⚠️ 載入 meta 失敗: {e}")

        scaler_feat_file = latest_keras.replace('model_', 'feature_scaler_').replace('.keras', '.pkl')
        if not os.path.exists(scaler_feat_file):
             scaler_feat_file = latest_keras.replace('model_', 'scaler_').replace('.keras', '.pkl')
        
        scaler_feat = None
        if os.path.exists(scaler_feat_file):
            with open(scaler_feat_file, 'rb') as f:
                scaler_feat = pickle.load(f)

        scaler_tgt_file = latest_keras.replace('model_', 'target_scaler_').replace('.keras', '.pkl')
        if not os.path.exists(scaler_tgt_file):
             scaler_tgt = scaler_feat
        else:
             with open(scaler_tgt_file, 'rb') as f:
                 scaler_tgt = pickle.load(f)

        return model, scaler_feat, scaler_tgt, meta

    # 1. 從盤中工作區載入模型 (剛訓練完的)
    print("\n[Model Injection] 載入盤中訓練的 LSTM 模型...")
    m5d, sf5d, st5d, meta5d = load_model_components(workspace['lstm_5d'])
    m1d, sf1d, st1d, meta1d = load_model_components(workspace['lstm_1d'])
    
    if m5d is None or m1d is None:
        print("[Error] 模型載入失敗")
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

    # 3. 合併歷史與盤中資料
    print("\n[Merge] 合併歷史資料與盤中資料...")
    df = pd.read_csv(CSV_FILE)
    df['date'] = pd.to_datetime(df['date'], format='%Y/%m/%d')
    df = df.set_index('date')
    df = df.rename(columns={
        'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'
    })
    
    date_str, o, h, l, c = intraday_data
    intraday_dt = pd.Timestamp(date_str)
    
    if intraday_dt in df.index:
        df.loc[intraday_dt] = [o, h, l, c, avg_volume]
    else:
        new_row = pd.DataFrame({
            'Open': [o], 'High': [h], 'Low': [l], 'Close': [c], 'Volume': [avg_volume]
        }, index=[intraday_dt])
        df = pd.concat([df, new_row])
    
    df = df.sort_index()
    raw_df = df
    
    # 4. 計算特徵
    print(f"\n[Compute] 計算特徵中 (使用盤中訓練模型)...")
    df = core_system.calculate_features(raw_df, raw_df, ticker="^TWII", use_cache=False)
    
    # 匯出
    features_csv_path = os.path.join(workspace['cache'], 'intraday_features.csv')
    df.to_csv(features_csv_path)
    print(f"[Export] 盤中特徵數據已存檔: {features_csv_path}")
    
    return df


# =============================================================================
# Step 3: 雙模型推論
# =============================================================================
def dual_inference(workspace: dict, df: pd.DataFrame) -> dict:
    print("\n" + "=" * 60)
    print("🎯 Step 3: 雙模型推論 (盤中預測)")
    print("=" * 60)
    
    from stable_baselines3 import PPO
    
    FEATURE_COLS = core_system.FEATURE_COLS
    latest = df.iloc[-1]
    
    signal_buy_filter = bool(latest.get('Signal_Buy_Filter', False))
    print(f"  [濾網] Signal_Buy_Filter = {signal_buy_filter}")
    
    features = []
    for col in FEATURE_COLS:
        val = latest.get(col, 0.0)
        features.append(val)
    features = np.array(features, dtype=np.float32).reshape(1, -1)
    features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
    
    results = {'filter_status': signal_buy_filter}
    
    SELL_SCENARIOS = {
        'cost': 1.00,
        'profit': 1.10,
        'loss': 0.95,
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
            
            b_act, _ = buy_agent.predict(features, deterministic=True)
            b_obs = buy_agent.policy.obs_to_tensor(features)[0]
            b_prob = buy_agent.policy.get_distribution(b_obs).distribution.probs.detach().cpu().numpy()[0]
            
            ai_action = 'BUY' if b_act[0] == 1 else 'WAIT'
            buy_prob = float(b_prob[1]) if b_act[0] == 1 else float(b_prob[0])
            
            if signal_buy_filter:
                buy_signal = ai_action
            else:
                buy_signal = f"FILTERED (AI: {ai_action})"
            
            print(f"  [{name}] Buy: {buy_signal} ({buy_prob:.1%})")
            
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
                'ai_action': ai_action,
                'sell_scenarios': sell_scenarios,
            }
            
        except Exception as e:
            results[key] = {'error': str(e)}
            print(f"  [Error] {name}: {e}")
            import traceback
            traceback.print_exc()

    run_strategy("V3 (Lightweight 200K)", STRATEGY_A_PATH, 'A')
    run_strategy("V4 (Standard 1M)", STRATEGY_B_PATH, 'B')
    
    return results


# =============================================================================
# Step 4: 輸出盤中報告
# =============================================================================
def generate_intraday_report(workspace: dict, df: pd.DataFrame, res: dict, date_str: str, intraday_data: tuple, avg_volume: float):
    print("\n" + "=" * 60)
    print("📊 Step 4: 盤中戰情儀表板")
    print("=" * 60)
    
    last = df.iloc[-1]
    filter_status = res.get('filter_status', False)
    
    _, o, h, l, c = intraday_data
    
    lines = []
    lines.append("=" * 50)
    lines.append(f"📅 盤中即時分析 - {date_str}")
    lines.append(f"⏰ 更新時間: {datetime.now().strftime('%H:%M:%S')}")
    lines.append("=" * 50)
    lines.append(f"📊 Open:  {o:.2f}")
    lines.append(f"📈 High:  {h:.2f}")
    lines.append(f"📉 Low:   {l:.2f}")
    lines.append(f"💰 Close: {c:.2f} (即時)")
    lines.append(f"📦 Volume: {avg_volume:.2f} 億元 (前5日平均估計)")
    lines.append("-" * 50)
    
    filter_icon = "✅" if filter_status else "🚫"
    filter_text = "通過 (Donchian 突破)" if filter_status else "未通過 (非突破日)"
    lines.append(f"🎯 [濾網狀態] {filter_icon} {filter_text}")
    lines.append("-" * 50)
    
    lines.append("🔮 [分析師 LSTM] (盤中訓練)")
    lines.append(f"   T+1 漲跌: {last.get('LSTM_Pred_1d', 0)*100:+.2f}%")
    lines.append(f"   T+5 漲跌: {last.get('LSTM_Pred_5d', 0)*100:+.2f}%")
    lines.append(f"   信心度:   {last.get('LSTM_Conf_5d', 0)*100:.1f}%")
    lines.append("-" * 50)
    
    lines.append("🤖 [操盤手 RL]")
    
    def format_strategy(key, label):
        if key not in res or 'error' in res[key]:
            return [f"   {label}: ❌ 模型載入失敗"]
        
        r = res[key]
        result_lines = []
        
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
    
    ai_a = res.get('A', {}).get('ai_action', 'N/A')
    ai_b = res.get('B', {}).get('ai_action', 'N/A')
    
    if not filter_status:
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
        
    lines.append(f"💡 盤中建議: {advice}")
    lines.append("=" * 50)
    lines.append("")
    lines.append("⚠️ 注意：此為盤中即時分析")
    lines.append("   • LSTM 使用盤中價格 + 5日均量訓練")
    lines.append("   • 成交量為預估值，實際結果可能有差異")
    lines.append("📌 收盤後執行 daily_ops_dual.py 取得正式分析")
    
    report = "\n".join(lines)
    print(report)
    
    # 存檔 TXT
    txt_path = os.path.join(workspace['reports'], 'intraday_summary.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    # 存檔 JSON
    json_path = os.path.join(workspace['reports'], 'intraday_summary.json')
    json_data = {
        'date': date_str,
        'generated_at': datetime.now().isoformat(),
        'type': 'intraday_with_training',
        'workspace': workspace['root'],
        'filter_status': filter_status,
        'market': {
            'open': float(o),
            'high': float(h),
            'low': float(l),
            'close': float(c),
            'volume_estimated': float(avg_volume),
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
    now = datetime.now()
    date_str = now.strftime('%Y-%m-%d')
    time_str = now.strftime('%H%M%S')
    
    print("=" * 60)
    print(f"🚀 盤中完整分析系統啟動 - {date_str}")
    print(f"⏰ 執行時間: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Step 0: 建立盤中專屬工作區
    ws = create_intraday_workspace(date_str, time_str)
    
    # Step 0.5: 取得盤中 OHLC
    intraday_data = fetch_intraday_ohlc("^TWII")
    if intraday_data is None:
        print("[Error] 無法取得盤中資料，結束執行")
        sys.exit(1)
    
    actual_date = intraday_data[0]
    
    # Step 0.6: 取得前 5 日成交量平均
    avg_volume = get_avg_volume_from_csv(n_days=5)
    
    # Step 0.7: 建立暫存 CSV (用於訓練)
    temp_csv_path = create_temp_csv_with_intraday(intraday_data, avg_volume, ws)
    
    # Step 1: LSTM 訓練 (使用暫存 CSV)
    success = train_lstm_with_intraday(ws, temp_csv_path, actual_date)
    if not success:
        print("[Error] LSTM 訓練失敗，無法進行預測")
        sys.exit(1)
    
    # Step 2: 特徵工程 (使用剛訓練完的模型)
    df = isolated_feature_engineering_intraday(ws, intraday_data, avg_volume)
    
    # Step 3: 雙模型推論
    res = dual_inference(ws, df)
    
    # Step 4: 輸出報告
    generate_intraday_report(ws, df, res, actual_date, intraday_data, avg_volume)
    
    print(f"\n🎉 盤中分析完成！結果存放於: {ws['root']}")


if __name__ == "__main__":
    main()
