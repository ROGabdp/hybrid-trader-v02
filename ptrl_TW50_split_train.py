# -*- coding: utf-8 -*-
"""
Pro Trader RL (Split Training Version)
Implements custom Train/Test split:
- Test Set: 2012-01-01 to 2017-12-31
- Training Set: 2000-01-01 to 2011-12-31 AND 2018-01-01 to Present
"""

import os
import sys
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import ta
from ta.volatility import AverageTrueRange
from ta.momentum import RSIIndicator
from ta.volume import MFIIndicator
import gymnasium as gym
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import glob
import multiprocessing

# --- 0. 環境與 GPU 設定 ---
def setup_environment():
    if torch.cuda.is_available():
        print(f"✅ CUDA is available! Device: {torch.cuda.get_device_name(0)}")
        device = "cuda"
    else:
        print("⚠️ CUDA not available. Using CPU.")
        device = "cpu"

    PROJECT_PATH = os.getcwd()
    MODELS_PATH = os.path.join(PROJECT_PATH, 'models_paper') 
    RESULTS_PATH = os.path.join(PROJECT_PATH, 'results_paper')
    DATA_PATH = os.path.join(PROJECT_PATH, 'data')

    for path in [MODELS_PATH, RESULTS_PATH, DATA_PATH]:
        if not os.path.exists(path):
            os.makedirs(path)
            
    return PROJECT_PATH, MODELS_PATH, RESULTS_PATH, DATA_PATH, device

# --- 1. 資料下載 (Start Date: 2000-01-01) ---
def fetch_tw50_data(data_path, start_date="2000-01-01"):
    tickers = [
        "0050.TW", "^TWII",
        "2330.TW", "2317.TW", "2454.TW", "2881.TW", "2382.TW", "2308.TW", "2882.TW", "2412.TW", "2891.TW", "3711.TW",
        "2886.TW", "2884.TW", "2303.TW", "2885.TW", "3231.TW", "3034.TW", "5880.TW", "2892.TW", "3008.TW", "2357.TW",
        "2002.TW", "2890.TW", "1101.TW", "2880.TW", "2883.TW", "2887.TW", "2345.TW", "3045.TW", "5871.TW", "3037.TW",
        "2912.TW", "1216.TW", "6505.TW", "4938.TW", "5876.TW", "1303.TW", "2395.TW", "2379.TW", "1301.TW", "3017.TW",
        "1326.TW", "2603.TW", "1590.TW", "3661.TW", "2327.TW", "4904.TW", "2801.TW", "1605.TW", "1504.TW", "2207.TW"
    ]
    print(f"開始下載 {len(tickers)} 檔標的資料 (Start: {start_date})...")
    data = yf.download(tickers, start=start_date, group_by='ticker', auto_adjust=True, threads=True, progress=False)
    clean_data = {}
    if "^TWII" in data.columns.levels[0]:
        market_index = data["^TWII"].copy()
        if isinstance(market_index.columns, pd.MultiIndex):
            market_index.columns = market_index.columns.get_level_values(0)
        clean_data["^TWII"] = market_index

    for t in tqdm(tickers):
        if t == "^TWII": continue
        try:
            df = data[t].copy()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.rename(columns={"Open": "Open", "High": "High", "Low": "Low", "Close": "Close", "Volume": "Volume"})
            df = df.dropna()
            if len(df) > 250:
                clean_data[t] = df
        except Exception as e:
            print(f"Skipping {t}: {e}")
    return clean_data, clean_data.get("^TWII")

# --- 2. 特徵工程 (Feature Engineering) ---
FEATURE_COLS = [
    'Norm_Close', 'Norm_Open', 'Norm_High', 'Norm_Low',
    'Norm_DC_Lower',
    'Norm_HA_Open', 'Norm_HA_High', 'Norm_HA_Low', 'Norm_HA_Close',
    'Norm_SuperTrend_1', 'Norm_SuperTrend_2',
    'Norm_RSI', 'Norm_MFI',
    'Norm_ATR_Change',
    'Norm_RS_Ratio',
    'RS_ROC_5', 'RS_ROC_10', 'RS_ROC_20', 'RS_ROC_60', 'RS_ROC_120'
]

# --- Helper Functions for Indicators ---
def calculate_heikin_ashi(df):
    ha_close = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    
    ha_open = [df['Open'].iloc[0]]
    for i in range(1, len(df)):
        ha_open.append((ha_open[-1] + ha_close.iloc[i-1]) / 2)
    ha_open = pd.Series(ha_open, index=df.index)
    
    ha_high = pd.concat([df['High'], ha_open, ha_close], axis=1).max(axis=1)
    ha_low = pd.concat([df['Low'], ha_open, ha_close], axis=1).min(axis=1)
    
    return pd.DataFrame({
        'HA_open': ha_open,
        'HA_high': ha_high,
        'HA_low': ha_low,
        'HA_close': ha_close
    })

def calculate_supertrend(df, length=14, multiplier=3.0):
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    atr = AverageTrueRange(high, low, close, window=length).average_true_range()
    atr = atr.fillna(method='bfill')
    
    hl2 = (high + low) / 2
    basic_upperband = hl2 + (multiplier * atr)
    basic_lowerband = hl2 - (multiplier * atr)
    
    final_upperband = basic_upperband.copy()
    final_lowerband = basic_lowerband.copy()
    
    trend = np.zeros(len(df))
    
    for i in range(1, len(df)):
        if basic_upperband.iloc[i] < final_upperband.iloc[i-1] or close.iloc[i-1] > final_upperband.iloc[i-1]:
            final_upperband.iloc[i] = basic_upperband.iloc[i]
        else:
            final_upperband.iloc[i] = final_upperband.iloc[i-1]
            
        if basic_lowerband.iloc[i] > final_lowerband.iloc[i-1] or close.iloc[i-1] < final_lowerband.iloc[i-1]:
            final_lowerband.iloc[i] = basic_lowerband.iloc[i]
        else:
            final_lowerband.iloc[i] = final_lowerband.iloc[i-1]
            
        if close.iloc[i] > final_upperband.iloc[i-1]:
            trend[i] = 1
        elif close.iloc[i] < final_lowerband.iloc[i-1]:
            trend[i] = -1
        else:
            trend[i] = trend[i-1]
            
    st = pd.Series(np.where(trend == 1, final_lowerband, final_upperband), index=df.index)
    return pd.DataFrame({'SUPERT_': st})

def calculate_features(df_in, benchmark_df):
    df = df_in.copy()
    # Donchian Channel
    df['DC_Upper'] = df['High'].rolling(20).max().shift(1)
    df['DC_Lower'] = df['Low'].rolling(20).min().shift(1)
    df['DC_Upper_10'] = df['High'].rolling(10).max().shift(1) # Filter Signal
    
    # Fill NA
    df['DC_Upper'] = df['DC_Upper'].fillna(method='bfill')
    df['DC_Lower'] = df['DC_Lower'].fillna(method='bfill')
    df['DC_Upper_10'] = df['DC_Upper_10'].fillna(method='bfill')

    # Indicators
    df['ATR'] = AverageTrueRange(df['High'], df['Low'], df['Close'], window=10).average_true_range()
    df['RSI'] = RSIIndicator(df['Close'], window=14).rsi()
    try:
        df['MFI'] = MFIIndicator(df['High'], df['Low'], df['Close'], df['Volume'], window=14).money_flow_index()
    except:
        df['MFI'] = 50.0
    
    ha = calculate_heikin_ashi(df)
    df['HA_Open'] = ha['HA_open']
    df['HA_High'] = ha['HA_high']
    df['HA_Low'] = ha['HA_low']
    df['HA_Close'] = ha['HA_close']

    st1 = calculate_supertrend(df, length=14, multiplier=2.0)
    st2 = calculate_supertrend(df, length=21, multiplier=1.0)
    df['SuperTrend_1'] = st1.iloc[:, 0]
    df['SuperTrend_2'] = st2.iloc[:, 0]

    # Normalization
    base_price = df['DC_Upper'].replace(0, np.nan).fillna(method='bfill')
    price_cols = ['Close', 'Open', 'High', 'Low', 'DC_Lower',
                  'HA_Open', 'HA_High', 'HA_Low', 'HA_Close',
                  'SuperTrend_1', 'SuperTrend_2']
    for col in price_cols:
        df[f'Norm_{col}'] = df[col] / base_price

    df['Norm_RSI'] = df['RSI'] / 100.0
    df['Norm_MFI'] = df['MFI'] / 100.0
    df['Norm_ATR_Change'] = (df['ATR'] / df['ATR'].shift(1)).fillna(1.0)

    if benchmark_df is not None:
        bench_close = benchmark_df['Close'].reindex(df.index).fillna(method='ffill')
        df['RS_Raw'] = df['Close'] / bench_close
        rs_min = df['RS_Raw'].rolling(250).min()
        rs_max = df['RS_Raw'].rolling(250).max()
        denominator = (rs_max - rs_min).replace(0, np.nan).fillna(1.0) + 1e-9
        df['Norm_RS_Ratio'] = (df['RS_Raw'] - rs_min) / denominator
        df['Norm_RS_Ratio'] = df['Norm_RS_Ratio'].fillna(0.5)
        for period in [5, 10, 20, 60, 120]:
            df[f'RS_ROC_{period}'] = df['RS_Raw'].pct_change(period).fillna(0)
    else:
        df['Norm_RS_Ratio'] = 0.0
        for period in [5, 10, 20, 60, 120]:
            df[f'RS_ROC_{period}'] = 0.0

    # Signals & Labels
    df['Signal_Buy_Filter'] = (df['High'] > df['DC_Upper_10'])
    df['Next_20d_Max'] = df['High'].shift(-20).rolling(20).max() / df['Close'] - 1
    
    feature_cols_to_check = [c for c in df.columns if c != 'Next_20d_Max']
    return df.dropna(subset=feature_cols_to_check)

# --- 3. RL 環境定義 ---
class BuyEnvPaper(gym.Env):
    def __init__(self, data_dict, is_training=True):
        super().__init__()
        self.samples = []
        self.pos_samples = []
        self.neg_samples = []
        
        # 檢查欄位
        sample_df = next(iter(data_dict.values()))
        missing = [c for c in FEATURE_COLS if c not in sample_df.columns]
        if missing: raise ValueError(f"Missing columns: {missing}")

        for t, df in data_dict.items():
            df = df.dropna(subset=['Next_20d_Max'])
            signals = df[df['Signal_Buy_Filter'] == True]
            if len(signals) > 0:
                states = signals[FEATURE_COLS].values.astype(np.float32)
                future_rets = signals['Next_20d_Max'].values.astype(np.float32)
                for i in range(len(signals)):
                    sample = (states[i], future_rets[i])
                    self.samples.append(sample)
                    if future_rets[i] >= 0.10:
                        self.pos_samples.append(sample)
                    else:
                        self.neg_samples.append(sample)

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(len(FEATURE_COLS),), dtype=np.float32)
        self.is_training = is_training
        self.idx = 0
        self.current_sample = None

    def reset(self, seed=None, options=None):
        if self.is_training:
            if np.random.rand() < 0.5 and len(self.pos_samples) > 0:
                idx = np.random.randint(0, len(self.pos_samples))
                self.current_sample = self.pos_samples[idx]
            elif len(self.neg_samples) > 0:
                idx = np.random.randint(0, len(self.neg_samples))
                self.current_sample = self.neg_samples[idx]
            else:
                idx = np.random.randint(0, len(self.samples))
                self.current_sample = self.samples[idx]
        else:
            self.idx = (self.idx + 1) % len(self.samples)
            self.current_sample = self.samples[self.idx]
            
        return self.current_sample[0], {}

    def step(self, action):
        _, max_ret = self.current_sample
        reward = 0.0
        is_success = (max_ret >= 0.10)

        if action == 1:
            if is_success: reward = 1.0
            else: reward = 0.0
        else:
            if not is_success: reward = 1.0
            else: reward = 0.0

        return self.samples[self.idx][0], reward, True, False, {}

class SellEnvPaper(gym.Env):
    def __init__(self, data_dict):
        super().__init__()
        self.episodes = []
        for t, df in data_dict.items():
            buy_indices = np.where(df['Signal_Buy_Filter'])[0]
            feature_data = df[FEATURE_COLS].values.astype(np.float32)
            close_prices = df['Close'].values.astype(np.float32)
            
            for idx in buy_indices:
                if idx + 120 < len(df):
                    episode_features = feature_data[idx : idx+120]
                    episode_prices = close_prices[idx : idx+120]
                    buy_price = episode_prices[0]
                    returns = episode_prices / buy_price
                    
                    self.episodes.append({
                        'features': episode_features,
                        'prices': episode_prices,
                        'returns': returns,
                        'buy_price': buy_price
                    })
                    
        self.action_space = spaces.Discrete(2)
        self.obs_dim = len(FEATURE_COLS) + 1 
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.obs_dim,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.current_episode = self.episodes[np.random.randint(0, len(self.episodes))]
        self.day = 0
        return self._get_observation(), {}

    def _get_observation(self):
        market_features = self.current_episode['features'][self.day]
        current_return = self.current_episode['returns'][self.day]
        obs = np.concatenate([market_features, [current_return]])
        return obs.astype(np.float32)

    def step(self, action):
        current_return = self.current_episode['returns'][self.day]
        reward = 0.0
        done = False
        is_profitable = (current_return >= 1.10)

        if action == 1: # Sell
            if is_profitable:
                max_ret_in_window = np.max(self.current_episode['returns'])
                if max_ret_in_window == 1.10:
                    reward = 1.0
                else:
                    ratio = (current_return - 1.10) / (max_ret_in_window - 1.10)
                    reward = 1.0 + ratio
            else:
                reward = -1.0
            done = True
            
        elif self.day >= 119: # Forced sell
            if is_profitable:
                max_ret_in_window = np.max(self.current_episode['returns'])
                if max_ret_in_window == 1.10:
                    reward = 1.0
                else:
                    ratio = (current_return - 1.10) / (max_ret_in_window - 1.10)
                    reward = 1.0 + ratio
            else:
                reward = -1.0
            done = True
            
        else: # Hold
            if not is_profitable:
                reward = 0.5
            else:
                reward = 0.0
            self.day += 1

        next_obs = self._get_observation() if not done else self._get_observation()
        return next_obs, reward, done, False, {}

# --- 4. 回測類別 ---
class DetailedBacktesterPaper:
    def __init__(self, data_dict, buy_model, sell_model, initial_capital=1000000):
        self.data_dict = data_dict
        self.buy_model = buy_model
        self.sell_model = sell_model
        self.cash = initial_capital
        self.inventory = {}
        self.max_positions = 10
        self.position_size_pct = 0.10
        sample_ticker = list(data_dict.keys())[0]
        self.dates = sorted(data_dict[sample_ticker].index)
        self.trade_logs = []

    def run(self):
        print(f"執行回測 (Split Logic)...")
        records = []
        for current_date in tqdm(self.dates):
            # 1. 賣出檢查
            for ticker in list(self.inventory.keys()):
                info = self.inventory[ticker]
                df = self.data_dict[ticker]
                if current_date not in df.index: continue
                row = df.loc[current_date]
                price = row['Close']
                info['days_held'] += 1
                current_return = price / info['cost_price']
                
                # Stop Loss
                if current_return < 0.90:
                    self._sell(ticker, price, "StopLoss_Dip", current_date)
                    continue
                if info['days_held'] >= 20 and current_return < 1.10:
                    self._sell(ticker, price, "StopLoss_Sideways", current_date)
                    continue

                # AI Signal
                market_features = row[FEATURE_COLS].values.astype(np.float32)
                sell_state = np.concatenate([market_features, [current_return]])
                obs_tensor = torch.as_tensor(sell_state).unsqueeze(0).to(self.sell_model.device)
                with torch.no_grad():
                    distribution = self.sell_model.policy.get_distribution(obs_tensor)
                    probs = distribution.distribution.probs.cpu().numpy()[0]
                
                if (probs[1] - probs[0]) > 0.85:
                    self._sell(ticker, price, "AI_Signal", current_date)

            # 2. 買入檢查
            current_equity = 0
            for t, info in self.inventory.items():
                if current_date in self.data_dict[t].index:
                    current_equity += info['shares'] * self.data_dict[t].loc[current_date]['Close']
                else:
                    current_equity += info['shares'] * info['cost_price']
            total_asset = self.cash + current_equity

            if len(self.inventory) < self.max_positions and self.cash > 10000:
                tickers = list(self.data_dict.keys())
                np.random.shuffle(tickers)
                for ticker in tickers:
                    if ticker in self.inventory: continue
                    df = self.data_dict[ticker]
                    if current_date not in df.index: continue
                    row = df.loc[current_date]
                    if row['Signal_Buy_Filter']:
                        state = row[FEATURE_COLS].values.astype(np.float32)
                        action, _ = self.buy_model.predict(state, deterministic=True)
                        if action == 1:
                            target_amt = total_asset * self.position_size_pct
                            invest_amt = min(target_amt, self.cash)
                            if invest_amt > row['Close'] * 1000:
                                shares = int(invest_amt / row['Close'])
                                cost = shares * row['Close'] * (1 + 0.001425)
                                self.cash -= cost
                                self.inventory[ticker] = {'shares': shares, 'cost_price': row['Close'], 'days_held': 0}
                                self.trade_logs.append({
                                    'Date': current_date, 'Ticker': ticker, 'Action': 'Buy',
                                    'Price': row['Close'], 'Reason': 'AI_Signal'
                                })
                                if len(self.inventory) >= self.max_positions: break
            
            # 紀錄
            daily_equity = 0
            held_stocks = []
            for t, info in self.inventory.items():
                if current_date in self.data_dict[t].index:
                    val = info['shares'] * self.data_dict[t].loc[current_date]['Close']
                else:
                    val = info['shares'] * info['cost_price']
                daily_equity += val
                held_stocks.append(t)
            
            records.append({
                'Date': current_date, 'Total_Value': self.cash + daily_equity,
                'Cash': self.cash, 'Invested_Amount': daily_equity, 'Holdings_Count': len(held_stocks)
            })
            
        if not records:
            return pd.DataFrame(columns=['Total_Value', 'Cash', 'Invested_Amount', 'Holdings_Count']), self.trade_logs
        return pd.DataFrame(records).set_index('Date'), self.trade_logs

    def _sell(self, ticker, price, reason, date):
        info = self.inventory[ticker]
        revenue = info['shares'] * price * (1 - 0.004)
        self.cash += revenue
        del self.inventory[ticker]
        self.trade_logs.append({
            'Date': date, 'Ticker': ticker, 'Action': 'Sell',
            'Price': price, 'Reason': reason
        })

# --- Main Execution ---
if __name__ == "__main__":
    # 1. 設定環境
    PROJECT_PATH, MODELS_PATH, RESULTS_PATH, DATA_PATH, device = setup_environment()
    
    # 2. 下載與處理資料
    raw_data_dict, market_index_df = fetch_tw50_data(DATA_PATH, start_date="2000-01-01")
    
    print("正在計算特徵...")
    processed_data = {}
    bench_df = raw_data_dict.get("^TWII")
    for ticker, df in raw_data_dict.items():
        if ticker != "^TWII":
            try:
                processed_data[ticker] = calculate_features(df, bench_df)
            except Exception as e:
                print(f"Error processing {ticker}: {e}")
    
    # --- Data Splitting Logic ---
    print("\n=== Data Splitting ===")
    print("Test Period: 2012-01-01 to 2017-12-31")
    print("Train Period: 2000-2011 AND 2018-Present")
    
    train_data = {}
    test_data = {}
    
    for ticker, df in processed_data.items():
        if ticker == "0050.TW": continue
        
        # Create masks
        test_mask = (df.index >= '2012-01-01') & (df.index <= '2017-12-31')
        train_mask = ~test_mask
        
        df_train = df[train_mask].copy()
        df_test = df[test_mask].copy()
        
        if not df_train.empty:
            train_data[ticker] = df_train
        if not df_test.empty:
            test_data[ticker] = df_test
            
    print(f"Training Tickers: {len(train_data)}")
    print(f"Testing Tickers: {len(test_data)}")

    # 3. 準備訓練環境
    import psutil
    n_cpu = multiprocessing.cpu_count()
    
    # Memory-based limit
    # Estimate: Each env needs ~3GB RAM (Data increased by 70% -> ~2.7GB/env) + 6GB System Buffer
    mem = psutil.virtual_memory()
    total_mem_gb = mem.total / (1024 ** 3)
    max_envs_mem = int((total_mem_gb - 6) / 3.0) 
    if max_envs_mem < 1: max_envs_mem = 1
    
    # Combine CPU and Memory limits
    n_envs = min(14, max(1, n_cpu - 1), max_envs_mem)
    
    print(f"System Memory: {total_mem_gb:.1f} GB")
    print(f"Detected {n_cpu} CPUs. Adjusted n_envs to {n_envs} (Mem limit: {max_envs_mem}) to avoid OOM.")
    
    # Train Env uses TRAIN DATA
    buy_env = make_vec_env(BuyEnvPaper, n_envs=n_envs, seed=0, vec_env_cls=SubprocVecEnv, env_kwargs={'data_dict': train_data, 'is_training': True})
    sell_env = make_vec_env(SellEnvPaper, n_envs=n_envs, seed=0, vec_env_cls=SubprocVecEnv, env_kwargs={'data_dict': train_data})
    
    # Eval Env uses TEST DATA (Strict Out-of-Sample Evaluation)
    eval_env = make_vec_env(BuyEnvPaper, n_envs=1, seed=0, vec_env_cls=SubprocVecEnv, env_kwargs={'data_dict': test_data, 'is_training': True})
    eval_sell_env = make_vec_env(SellEnvPaper, n_envs=1, seed=0, vec_env_cls=SubprocVecEnv, env_kwargs={'data_dict': test_data})
    
    # 4. 設定 PPO 參數
    ppo_params = {
        "learning_rate": 0.0001,
        "n_steps": 2048 // n_envs,
        "batch_size": 512,
        "ent_coef": 0.01,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "vf_coef": 0.5,
        "verbose": 1,
        "device": device,
        "policy_kwargs": dict(net_arch=[64, 64, 64])
    }
    if ppo_params["n_steps"] < 128: ppo_params["n_steps"] = 128
    
    # 5. 訓練 Buy Agent
    print("\n=== 檢查 Buy Agent 模型 (Split) ===")
    TOTAL_TIMESTEPS_BUY = 2_000_000
    TOTAL_TIMESTEPS_SELL = 1_000_000
    
    BUY_RESULTS_PATH = os.path.join(RESULTS_PATH, "buy_split")
    SELL_RESULTS_PATH = os.path.join(RESULTS_PATH, "sell_split")
    for p in [BUY_RESULTS_PATH, SELL_RESULTS_PATH]:
        if not os.path.exists(p): os.makedirs(p)

    checkpoint_callback = CheckpointCallback(save_freq=5000, save_path=MODELS_PATH, name_prefix="ppo_buy_split")
    eval_callback = EvalCallback(eval_env, best_model_save_path=os.path.join(MODELS_PATH, "best_buy_split"),
                                 log_path=BUY_RESULTS_PATH, eval_freq=2000, n_eval_episodes=100, deterministic=True, render=False)
    callbacks = CallbackList([checkpoint_callback, eval_callback])
    
    # Check for existing models
    final_buy_path = os.path.join(MODELS_PATH, "ppo_buy_split_final.zip")
    best_buy_path = os.path.join(MODELS_PATH, "best_buy_split", "best_model.zip")
    
    buy_model = None
    should_train_buy = True
    
    if os.path.exists(final_buy_path) or os.path.exists(best_buy_path):
        print(f"\n⚠️ 偵測到已存在的 Buy Agent 模型！")
        print(f"路徑: {final_buy_path if os.path.exists(final_buy_path) else best_buy_path}")
        print("請選擇操作:")
        print("1. 使用現有模型 (跳過訓練)")
        print("2. 接續先前中斷的訓練 (Resume)")
        print("3. 將模型刪除，重新開始訓練 (Restart)")
        
        choice = input("請輸入選項 (1/2/3): ").strip()
        
        if choice == '1':
            print("✅ 選擇使用現有模型，跳過訓練...")
            if os.path.exists(best_buy_path):
                buy_model = PPO.load(best_buy_path, device=device)
                print(f"已載入最佳模型: {best_buy_path}")
            else:
                buy_model = PPO.load(final_buy_path, device=device)
                print(f"已載入最終模型: {final_buy_path}")
            should_train_buy = False
            
        elif choice == '2':
            print("🔄 選擇接續訓練...")
            ckpt_files = glob.glob(os.path.join(MODELS_PATH, "ppo_buy_split_*_steps.zip"))
            latest_ckpt = None
            if ckpt_files:
                latest_ckpt = max(ckpt_files, key=lambda x: int(x.split('_')[-2]))
            
            if latest_ckpt:
                buy_model = PPO.load(latest_ckpt, env=buy_env, device=device)
                print(f"已載入最新存檔準備接續訓練: {latest_ckpt}")
            elif os.path.exists(best_buy_path):
                buy_model = PPO.load(best_buy_path, env=buy_env, device=device)
                print(f"⚠️ 找不到 checkpoint，已載入最佳模型: {best_buy_path}")
            else:
                buy_model = PPO.load(final_buy_path, env=buy_env, device=device)
                print(f"已載入最終模型準備接續訓練: {final_buy_path}")
            should_train_buy = True
            
        elif choice == '3':
            print("🗑️ 選擇刪除舊模型並重新訓練...")
            for p in [final_buy_path, best_buy_path]:
                if os.path.exists(p):
                    try:
                        os.remove(p)
                        print(f"已刪除: {p}")
                    except OSError as e:
                        print(f"無法刪除 {p}: {e}")
            
            ckpt_pattern = os.path.join(MODELS_PATH, "ppo_buy_split_*_steps.zip")
            for p in glob.glob(ckpt_pattern):
                try:
                    os.remove(p)
                    print(f"已刪除 Checkpoint: {p}")
                except OSError as e:
                    print(f"無法刪除 Checkpoint {p}: {e}")

            buy_model = None
            should_train_buy = True
        else:
            print("⚠️ 輸入無效，預設為使用現有模型 (跳過訓練)...")
            if os.path.exists(best_buy_path):
                buy_model = PPO.load(best_buy_path, device=device)
            else:
                buy_model = PPO.load(final_buy_path, device=device)
            should_train_buy = False

    if should_train_buy:
        if buy_model is None:
            print("🚀 開始全新訓練 Buy Agent (Split)...")
            buy_model = PPO("MlpPolicy", buy_env, **ppo_params)
        else:
            print("🚀 接續訓練 Buy Agent (Split)...")
            
        buy_model.learn(total_timesteps=TOTAL_TIMESTEPS_BUY, callback=callbacks, reset_num_timesteps=False)
        buy_model.save(os.path.join(MODELS_PATH, "ppo_buy_split_final"))
    
    # 6. 訓練 Sell Agent
    print("\n=== 檢查 Sell Agent 模型 (Split) ===")
    sell_checkpoint_callback = CheckpointCallback(save_freq=5000, save_path=MODELS_PATH, name_prefix="ppo_sell_split")
    sell_eval_callback = EvalCallback(eval_sell_env, best_model_save_path=os.path.join(MODELS_PATH, "best_sell_split"),
                                 log_path=SELL_RESULTS_PATH, eval_freq=2000, n_eval_episodes=100, deterministic=True, render=False)
    sell_callbacks = CallbackList([sell_checkpoint_callback, sell_eval_callback])
    
    final_sell_path = os.path.join(MODELS_PATH, "ppo_sell_split_final.zip")
    best_sell_path = os.path.join(MODELS_PATH, "best_sell_split", "best_model.zip")
    
    sell_model = None
    should_train_sell = True
    
    # Check for checkpoints
    sell_ckpt_files = glob.glob(os.path.join(MODELS_PATH, "ppo_sell_split_*_steps.zip"))
    has_sell_model = os.path.exists(final_sell_path) or os.path.exists(best_sell_path) or len(sell_ckpt_files) > 0
    
    if has_sell_model:
        print(f"\n⚠️ 偵測到已存在的 Sell Agent 模型！")
        if os.path.exists(best_sell_path):
             print(f"發現最佳模型: {best_sell_path}")
        if os.path.exists(final_sell_path):
             print(f"發現最終模型: {final_sell_path}")
        elif sell_ckpt_files:
             print(f"發現 {len(sell_ckpt_files)} 個 Checkpoints")

        print("請選擇操作:")
        print("1. 使用現有模型 (跳過訓練)")
        print("2. 接續先前中斷的訓練 (Resume)")
        print("3. 將模型刪除，重新開始訓練 (Restart)")
        
        choice = input("請輸入選項 (1/2/3): ").strip()
        
        if choice == '1':
            print("✅ 選擇使用現有模型，跳過訓練...")
            if os.path.exists(best_sell_path):
                sell_model = PPO.load(best_sell_path, device=device)
                print(f"已載入最佳模型: {best_sell_path}")
            elif os.path.exists(final_sell_path):
                sell_model = PPO.load(final_sell_path, device=device)
                print(f"已載入最終模型: {final_sell_path}")
            elif sell_ckpt_files:
                 latest_ckpt = max(sell_ckpt_files, key=lambda x: int(x.split('_')[-2]))
                 sell_model = PPO.load(latest_ckpt, device=device)
                 print(f"已載入最新 Checkpoint: {latest_ckpt}")
            should_train_sell = False
            
        elif choice == '2':
            print("🔄 選擇接續訓練...")
            latest_ckpt = None
            if sell_ckpt_files:
                latest_ckpt = max(sell_ckpt_files, key=lambda x: int(x.split('_')[-2]))
            
            if latest_ckpt:
                sell_model = PPO.load(latest_ckpt, env=sell_env, device=device)
                print(f"已載入最新存檔準備接續訓練: {latest_ckpt}")
            elif os.path.exists(best_sell_path):
                sell_model = PPO.load(best_sell_path, env=sell_env, device=device)
                print(f"⚠️ 找不到 checkpoint，已載入最佳模型: {best_sell_path}")
            elif os.path.exists(final_sell_path):
                sell_model = PPO.load(final_sell_path, env=sell_env, device=device)
                print(f"已載入最終模型準備接續訓練: {final_sell_path}")
            else:
                 print("⚠️ 找不到任何模型可供接續，將重新開始訓練。")
                 sell_model = None
            should_train_sell = True
            
        elif choice == '3':
            print("🗑️ 選擇刪除舊模型並重新訓練...")
            for p in [final_sell_path, best_sell_path]:
                if os.path.exists(p):
                    try:
                        os.remove(p)
                        print(f"已刪除: {p}")
                    except OSError as e:
                        print(f"無法刪除 {p}: {e}")
            
            for p in sell_ckpt_files:
                try:
                    os.remove(p)
                    print(f"已刪除 Checkpoint: {p}")
                except OSError as e:
                    print(f"無法刪除 Checkpoint {p}: {e}")

            sell_model = None
            should_train_sell = True
        else:
            print("⚠️ 輸入無效，預設為使用現有模型 (跳過訓練)...")
            if os.path.exists(best_sell_path):
                sell_model = PPO.load(best_sell_path, device=device)
            elif os.path.exists(final_sell_path):
                sell_model = PPO.load(final_sell_path, device=device)
            should_train_sell = False

    if should_train_sell:
        if sell_model is None:
            print("🚀 開始全新訓練 Sell Agent (Split)...")
            sell_model = PPO("MlpPolicy", sell_env, **ppo_params)
        else:
            print("🚀 接續訓練 Sell Agent (Split)...")
            
        sell_model.learn(total_timesteps=TOTAL_TIMESTEPS_SELL, callback=sell_callbacks, reset_num_timesteps=False)
        sell_model.save(os.path.join(MODELS_PATH, "ppo_sell_split_final"))
    
    # 7. 執行回測 (Test Set Only)
    print("\n=== 執行回測 (Test Set: 2012-2017) ===")
    
    # Load Best Models
    best_buy_path = os.path.join(MODELS_PATH, "best_buy_split", "best_model.zip")
    best_sell_path = os.path.join(MODELS_PATH, "best_sell_split", "best_model.zip")
    
    if os.path.exists(best_buy_path): buy_model = PPO.load(best_buy_path, device=device)
    if os.path.exists(best_sell_path): sell_model = PPO.load(best_sell_path, device=device)
    
    backtester = DetailedBacktesterPaper(test_data, buy_model, sell_model)
    daily_stats, trade_logs = backtester.run()
    
    if not daily_stats.empty:
        initial_val = daily_stats.iloc[0]['Total_Value']
        final_val = daily_stats.iloc[-1]['Total_Value']
        roi = (final_val - initial_val) / initial_val * 100
        print(f"Test Set ROI: {roi:.2f}%")
        
        # Benchmark Comparison (^TWII)
        bench_roi = 0.0
        bench_data = None
        if "^TWII" in raw_data_dict:
            bench_df = raw_data_dict["^TWII"]
            mask = (bench_df.index >= '2012-01-01') & (bench_df.index <= '2017-12-31')
            bench_data = bench_df.loc[mask].copy()
            # Fix: Drop NaN values to ensure valid start/end prices
            bench_data = bench_data.dropna(subset=['Close'])
            
            if not bench_data.empty:
                bench_start = bench_data.iloc[0]['Close']
                bench_end = bench_data.iloc[-1]['Close']
                bench_roi = (bench_end - bench_start) / bench_start * 100
                bench_data['Normalized_Value'] = (bench_data['Close'] / bench_start) * initial_val
                print(f"Benchmark ROI: {bench_roi:.2f}%")

        plt.figure(figsize=(12, 6))
        plt.plot(daily_stats.index, daily_stats['Total_Value'], label=f'AI (ROI: {roi:.2f}%)', color='red')
        if bench_data is not None:
            plt.plot(bench_data.index, bench_data['Normalized_Value'], label=f'^TWII (ROI: {bench_roi:.2f}%)', color='gray', linestyle='--')
        plt.title('Out-of-Sample Backtest (2012-2017)')
        plt.legend()
        plt.grid(True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(os.path.join(RESULTS_PATH, f"split_test_performance_{timestamp}.png"))
        print(f"Plot saved: split_test_performance_{timestamp}.png")
