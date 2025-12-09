# -*- coding: utf-8 -*-
"""
Pro Trader RL (Paper Strict Version)
嚴格遵照論文 "Pro Trader RL" 的邏輯、獎勵函數與參數設定。
資料集：台灣 50 (TW50)
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
    MODELS_PATH = os.path.join(PROJECT_PATH, 'models_paper') # 區隔模型資料夾
    RESULTS_PATH = os.path.join(PROJECT_PATH, 'results_paper')
    DATA_PATH = os.path.join(PROJECT_PATH, 'data')

    for path in [MODELS_PATH, RESULTS_PATH, DATA_PATH]:
        if not os.path.exists(path):
            os.makedirs(path)
            
    return PROJECT_PATH, MODELS_PATH, RESULTS_PATH, DATA_PATH, device

# --- 1. 資料下載 (與原版相同) ---
def fetch_tw50_data(data_path, start_date="2010-01-01"):
    tickers = [
        "0050.TW", "^TWII",
        "2330.TW", "2317.TW", "2454.TW", "2881.TW", "2382.TW", "2308.TW", "2882.TW", "2412.TW", "2891.TW", "3711.TW",
        "2886.TW", "2884.TW", "2303.TW", "2885.TW", "3231.TW", "3034.TW", "5880.TW", "2892.TW", "3008.TW", "2357.TW",
        "2002.TW", "2890.TW", "1101.TW", "2880.TW", "2883.TW", "2887.TW", "2345.TW", "3045.TW", "5871.TW", "3037.TW",
        "2912.TW", "1216.TW", "6505.TW", "4938.TW", "5876.TW", "1303.TW", "2395.TW", "2379.TW", "1301.TW", "3017.TW",
        "1326.TW", "2603.TW", "1590.TW", "3661.TW", "2327.TW", "4904.TW", "2801.TW", "1605.TW", "1504.TW", "2207.TW"
    ]
    print(f"開始下載 {len(tickers)} 檔標的資料...")
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
    
    ha_high = df[['High', 'Open', 'Close']].max(axis=1) # Simplified, strictly it's max(High, HA_Open, HA_Close)
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
    atr = atr.fillna(method='bfill') # Fix: Fill initial NaNs to allow recursion
    
    hl2 = (high + low) / 2
    basic_upperband = hl2 + (multiplier * atr)
    basic_lowerband = hl2 - (multiplier * atr)
    
    final_upperband = basic_upperband.copy()
    final_lowerband = basic_lowerband.copy()
    
    trend = np.zeros(len(df))
    
    # Loop to calculate SuperTrend
    # Note: This is a simplified iterative implementation
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
            
        if trend[i] == 1:
            pass # final_upperband.iloc[i] = np.nan # Optional: Hide upper band in uptrend
        else:
            pass # final_lowerband.iloc[i] = np.nan
            
    # Return the trend line (SuperTrend value)
    st = pd.Series(np.where(trend == 1, final_lowerband, final_upperband), index=df.index)
    return pd.DataFrame({'SUPERT_': st}) # Naming to match pandas_ta style roughly

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
    # ATR
    df['ATR'] = AverageTrueRange(df['High'], df['Low'], df['Close'], window=10).average_true_range()
    
    # RSI
    df['RSI'] = RSIIndicator(df['Close'], window=14).rsi()
    
    # MFI
    try:
        df['MFI'] = MFIIndicator(df['High'], df['Low'], df['Close'], df['Volume'], window=14).money_flow_index()
    except:
        df['MFI'] = 50.0
    
    # Heikin Ashi
    ha = calculate_heikin_ashi(df)
    df['HA_Open'] = ha['HA_open']
    df['HA_High'] = ha['HA_high']
    df['HA_Low'] = ha['HA_low']
    df['HA_Close'] = ha['HA_close']

    # SuperTrend
    st1 = calculate_supertrend(df, length=14, multiplier=2.0)
    st2 = calculate_supertrend(df, length=21, multiplier=1.0)
    df['SuperTrend_1'] = st1.iloc[:, 0]
    df['SuperTrend_2'] = st2.iloc[:, 0]

    # Normalization (Paper Logic)
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

    # Signals & Future Returns for Training
    df['Signal_Buy_Filter'] = (df['High'] > df['DC_Upper_10'])
    # Paper: "whether the return rate is 10% or more"
    # We calculate max return in next 20 days to check if 10% is achievable
    df['Next_20d_Max'] = df['High'].shift(-20).rolling(20).max() / df['Close'] - 1
    
    # Fix: Only drop rows where FEATURES are missing. Keep rows with missing labels (Next_20d_Max) for backtesting.
    feature_cols_to_check = [c for c in df.columns if c != 'Next_20d_Max']
    return df.dropna(subset=feature_cols_to_check)

# --- 3. RL 環境定義 (Strict Paper Logic) ---

class BuyEnvPaper(gym.Env):
    """
    Buy Knowledge RL (Paper Version)
    - Reward: Discrete (+1 if return >= 10%, else 0 or +1 for correct wait)
    - Threshold: 10%
    """
    def __init__(self, data_dict, is_training=True):
        super().__init__()
        self.samples = []
        self.pos_samples = [] # Success cases (>= 10%)
        self.neg_samples = [] # Fail cases (< 10%)
        
        # 檢查欄位
        sample_df = next(iter(data_dict.values()))
        missing = [c for c in FEATURE_COLS if c not in sample_df.columns]
        if missing: raise ValueError(f"Missing columns: {missing}")

        for t, df in data_dict.items():
            # Training requires labels, so we drop rows where Next_20d_Max is NaN
            df = df.dropna(subset=['Next_20d_Max'])
            signals = df[df['Signal_Buy_Filter'] == True]
            if len(signals) > 0:
                states = signals[FEATURE_COLS].values.astype(np.float32)
                future_rets = signals['Next_20d_Max'].values.astype(np.float32)
                for i in range(len(signals)):
                    sample = (states[i], future_rets[i])
                    self.samples.append(sample)
                    
                    # Class Balancing Logic
                    if future_rets[i] >= 0.10:
                        self.pos_samples.append(sample)
                    else:
                        self.neg_samples.append(sample)

        self.action_space = spaces.Discrete(2) # 0: Wait, 1: Buy
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(len(FEATURE_COLS),), dtype=np.float32)
        self.is_training = is_training
        self.idx = 0
        self.current_sample = None # To store the selected sample for the episode

    def reset(self, seed=None, options=None):
        if self.is_training:
            # Class Balancing: 50% chance to pick a positive sample
            # This forces the agent to see success cases 50% of the time,
            # preventing it from learning "Always Wait" due to low base rate.
            if np.random.rand() < 0.5 and len(self.pos_samples) > 0:
                idx = np.random.randint(0, len(self.pos_samples))
                self.current_sample = self.pos_samples[idx]
            elif len(self.neg_samples) > 0:
                idx = np.random.randint(0, len(self.neg_samples))
                self.current_sample = self.neg_samples[idx]
            else:
                # Fallback if one class is empty (unlikely)
                idx = np.random.randint(0, len(self.samples))
                self.current_sample = self.samples[idx]
        else:
            # Evaluation: Sequential (True Distribution)
            self.idx = (self.idx + 1) % len(self.samples)
            self.current_sample = self.samples[self.idx]
            
        return self.current_sample[0], {}

    def step(self, action):
        _, max_ret = self.current_sample
        reward = 0.0
        
        # Paper Logic: Threshold = 10% (0.10)
        is_success = (max_ret >= 0.10)

        if action == 1: # Buy (Predicting Success)
            if is_success:
                reward = 1.0 # Scenario 1: Predict Buy, Actual >= 10% -> +1
            else:
                reward = 0.0 # Scenario 2: Predict Buy, Actual < 10% -> 0
        else: # Wait (Predicting Fail)
            if not is_success:
                reward = 1.0 # Scenario 3: Predict Wait, Actual < 10% -> +1 (Avoided bad trade)
            else:
                reward = 0.0 # Scenario 4: Predict Wait, Actual >= 10% -> 0 (Missed opportunity)

        return self.samples[self.idx][0], reward, True, False, {}

class SellEnvPaper(gym.Env):
    """
    Sell Knowledge RL (Paper Version)
    - Observation: Features + [SellReturn] (Current/Buy_Price)
    - Reward: Relative Ranking (Max +2, Min +1) for profitable trades
    """
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
                    
                    # Pre-calculate returns for ranking logic
                    returns = episode_prices / buy_price
                    
                    self.episodes.append({
                        'features': episode_features,
                        'prices': episode_prices,
                        'returns': returns,
                        'buy_price': buy_price
                    })
                    
        self.action_space = spaces.Discrete(2) # 0: Hold, 1: Sell
        # Obs: Features + [SellReturn] (Paper mentions adding SellReturn to features)
        self.obs_dim = len(FEATURE_COLS) + 1 
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.obs_dim,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.current_episode = self.episodes[np.random.randint(0, len(self.episodes))]
        self.day = 0
        return self._get_observation(), {}

    def _get_observation(self):
        market_features = self.current_episode['features'][self.day]
        current_return = self.current_episode['returns'][self.day] # SellReturn
        
        # Paper: Input state includes SellReturn
        obs = np.concatenate([market_features, [current_return]])
        return obs.astype(np.float32)

    def step(self, action):
        current_return = self.current_episode['returns'][self.day]
        reward = 0.0
        done = False
        
        # Paper Logic: 10% Threshold
        is_profitable = (current_return >= 1.10)

        if action == 1: # Sell
            if is_profitable:
                # Ranking Logic (Simplified for efficiency)
                # Paper: Rank returns >= 10%. Max=+2, Min=+1.
                # We calculate rank relative to the max return in the 120-day window
                max_ret_in_window = np.max(self.current_episode['returns'])
                if max_ret_in_window < 1.10:
                    # If whole window never reached 10%, but we are here (shouldn't happen if logic is strict, but for safety)
                    reward = 1.0 
                else:
                    # Linear interpolation between 1.0 and 2.0 based on closeness to max
                    # If current == max, reward = 2.0
                    # If current == 1.10 (min threshold), reward = 1.0
                    if max_ret_in_window == 1.10:
                        reward = 1.0
                    else:
                        ratio = (current_return - 1.10) / (max_ret_in_window - 1.10)
                        reward = 1.0 + ratio # Maps to [1.0, 2.0]
            else:
                # Predict Sell, Actual < 10% -> -1 (Paper: "If the return rate is less than 10%, -1 point")
                reward = -1.0
            done = True
            
        elif self.day >= 119: # Time's up
            # Forced sell
            if is_profitable:
                # Same ranking logic
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
                # Predict Hold, Actual < 10% -> +0.5 (Paper: "If the return rate is less than 10%, +0.5 point")
                reward = 0.5
            else:
                # Predict Hold, Actual >= 10% -> 0 (Missed selling opportunity? Paper doesn't explicitly penalize holding profit, but implies selling is better)
                reward = 0.0
            self.day += 1

        next_obs = self._get_observation() if not done else self._get_observation()
        return next_obs, reward, done, False, {}

# --- 4. 回測類別 (Paper Strict Logic) ---
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
        self.dates = [d for d in self.dates if d >= pd.Timestamp('2021-01-01')]
        if not self.dates:
            print("⚠️ Warning: No dates found for backtesting after 2021-01-01!")
        self.trade_logs = [] # 新增：交易紀錄

    def run(self):
        print(f"執行回測 (Paper Logic)...")
        records = []
        for current_date in tqdm(self.dates):
            # 1. 賣出檢查 (Sell Knowledge + Stop Loss Rules)
            for ticker in list(self.inventory.keys()):
                info = self.inventory[ticker]
                df = self.data_dict[ticker]
                if current_date not in df.index: continue
                row = df.loc[current_date]
                price = row['Close']
                info['days_held'] += 1
                
                # Calculate Return
                current_return = price / info['cost_price']
                
                # --- Stop Loss Rules (Paper Strict) ---
                # Rule 1: Stop Loss on Dips (-10%)
                if current_return < 0.90:
                    self._sell(ticker, price, "StopLoss_Dip", current_date)
                    continue
                
                # Rule 2: Stop Loss on Sideways (20 days < 10%)
                # Note: Paper says "If return rate is 10% or less for 20 days... sell on 21st day"
                # We track max return in holding period or check daily returns? 
                # Paper implies "continuous" poor performance. We check if current return has been < 1.10 for 20 days.
                if info['days_held'] >= 20:
                    # 簡化實作：如果持有超過 20 天且當前報酬率 < 10%，則觸發
                    # (嚴格來說應該檢查過去連續 20 天，但這裡假設若第 20 天還沒 10% 就砍)
                    if current_return < 1.10:
                        self._sell(ticker, price, "StopLoss_Sideways", current_date)
                        continue

                # --- Sell Knowledge RL ---
                market_features = row[FEATURE_COLS].values.astype(np.float32)
                sell_state = np.concatenate([market_features, [current_return]])
                
                # Get probabilities
                # predict returns (action, state) if deterministic=True
                # We need probabilities for the threshold logic
                obs_tensor = torch.as_tensor(sell_state).unsqueeze(0).to(self.sell_model.device)
                with torch.no_grad():
                    distribution = self.sell_model.policy.get_distribution(obs_tensor)
                    probs = distribution.distribution.probs.cpu().numpy()[0] # [prob_hold, prob_sell]
                
                prob_hold = probs[0]
                prob_sell = probs[1]
                
                # Paper Threshold: (Sell - Hold) > 0.85
                if (prob_sell - prob_hold) > 0.85:
                    self._sell(ticker, price, "AI_Signal", current_date)

            # 2. 買入檢查 (Buy Knowledge)
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
                                # 紀錄買入
                                self.trade_logs.append({
                                    'Date': current_date,
                                    'Ticker': ticker,
                                    'Action': 'Buy',
                                    'Price': row['Close'],
                                    'Reason': 'AI_Signal'
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
                'Date': current_date,
                'Total_Value': self.cash + daily_equity,
                'Cash': self.cash,
                'Invested_Amount': daily_equity,
                'Holdings_Count': len(held_stocks)
            })
            
        if not records:
            print("⚠️ Warning: No backtest records generated.")
            return pd.DataFrame(columns=['Total_Value', 'Cash', 'Invested_Amount', 'Holdings_Count']), self.trade_logs

        return pd.DataFrame(records).set_index('Date'), self.trade_logs

    def _sell(self, ticker, price, reason, date):
        info = self.inventory[ticker]
        revenue = info['shares'] * price * (1 - 0.004)
        self.cash += revenue
        del self.inventory[ticker]
        # 紀錄賣出
        self.trade_logs.append({
            'Date': date,
            'Ticker': ticker,
            'Action': 'Sell',
            'Price': price,
            'Reason': reason
        })

# --- Main Execution ---
if __name__ == "__main__":
    # 1. 設定環境
    PROJECT_PATH, MODELS_PATH, RESULTS_PATH, DATA_PATH, device = setup_environment()
    
    # 2. 下載與處理資料
    raw_data_dict, market_index_df = fetch_tw50_data(DATA_PATH)
    
    print("正在計算特徵...")
    processed_data = {}
    bench_df = raw_data_dict.get("^TWII")
    for ticker, df in raw_data_dict.items():
        if ticker != "^TWII":
            try:
                processed_data[ticker] = calculate_features(df, bench_df)
            except Exception as e:
                print(f"Error processing {ticker}: {e}")
    
    # 3. 準備訓練環境 (Vectorized)
    train_data = {k: v for k, v in processed_data.items() if k != "0050.TW"}
    
    # Determine number of CPUs
    n_cpu = multiprocessing.cpu_count()
    # WinError 1455 Fix: Limit n_envs to avoid OOM on Windows
    # Each env copies data and loads PyTorch, consuming significant RAM.
    # Safe limit: 4 to 6.
    n_envs = min(14, max(1, n_cpu - 1)) 
    print(f"Detected {n_cpu} CPUs. Using {n_envs} environments for training to avoid memory issues.")
    
    # Create Vectorized Environments
    # Note: We pass the class and kwargs to make_vec_env
    buy_env = make_vec_env(BuyEnvPaper, n_envs=n_envs, seed=0, vec_env_cls=SubprocVecEnv, env_kwargs={'data_dict': train_data, 'is_training': True})
    # Sell Env usually doesn't need heavy parallelization if it's just for training the sell agent which is faster, 
    # but we can vectorize it too for consistency and speed.
    sell_env = make_vec_env(SellEnvPaper, n_envs=n_envs, seed=0, vec_env_cls=SubprocVecEnv, env_kwargs={'data_dict': train_data})
    
    # Eval env should remain single for accurate evaluation
    # Eval env should remain single for accurate evaluation, but use SubprocVecEnv to match training env type
    eval_env = make_vec_env(BuyEnvPaper, n_envs=1, seed=0, vec_env_cls=SubprocVecEnv, env_kwargs={'data_dict': train_data, 'is_training': True})
    eval_sell_env = make_vec_env(SellEnvPaper, n_envs=1, seed=0, vec_env_cls=SubprocVecEnv, env_kwargs={'data_dict': train_data})
    
    # 4. 設定 PPO 參數 (Optimized for Speed)
    # Paper: "3 hidden layers" -> net_arch=[64, 64, 64] (approx)
    # Optimization: Increased batch_size and n_steps for GPU efficiency
    ppo_params = {
        "learning_rate": 0.0001,
        "n_steps": 2048 // n_envs, # Adjust n_steps so total buffer size is similar or larger
        "batch_size": 512, # Increased from 64 to 512 for GPU efficiency
        "ent_coef": 0.01, # Strict Paper Value
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "vf_coef": 0.5,
        "verbose": 1,
        "device": device,
        "policy_kwargs": dict(net_arch=[64, 64, 64]) # 3 Hidden Layers
    }
    
    # Adjust n_steps to be at least some reasonable number
    if ppo_params["n_steps"] < 128:
        ppo_params["n_steps"] = 128
    
    # 5. 訓練 Buy Agent
    print("\n=== 檢查 Buy Agent 模型 ===")
    
    # ... (Buy Agent Params) ...
    TOTAL_TIMESTEPS_BUY = 2_000_000
    TOTAL_TIMESTEPS_SELL = 1_000_000
    print(f"訓練步數設定: Buy={TOTAL_TIMESTEPS_BUY}, Sell={TOTAL_TIMESTEPS_SELL}")
    print(f"(註：論文原始設定約為 Buy=12.2M, Sell=9.5M)")

    # Separate Log Paths
    BUY_RESULTS_PATH = os.path.join(RESULTS_PATH, "buy")
    SELL_RESULTS_PATH = os.path.join(RESULTS_PATH, "sell")
    for p in [BUY_RESULTS_PATH, SELL_RESULTS_PATH]:
        if not os.path.exists(p):
            os.makedirs(p)

    checkpoint_callback = CheckpointCallback(save_freq=5000, save_path=MODELS_PATH, name_prefix="ppo_buy_paper")
    eval_callback = EvalCallback(eval_env, best_model_save_path=os.path.join(MODELS_PATH, "best_buy_paper"),
                                 log_path=BUY_RESULTS_PATH, eval_freq=2000, n_eval_episodes=100, deterministic=True, render=False)
    callbacks = CallbackList([checkpoint_callback, eval_callback])
    
    # 檢查是否已有訓練好的模型
    final_buy_path = os.path.join(MODELS_PATH, "ppo_buy_paper_final.zip")
    best_buy_path = os.path.join(MODELS_PATH, "best_buy_paper", "best_model.zip")
    
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
            # Search for latest checkpoint
            ckpt_files = glob.glob(os.path.join(MODELS_PATH, "ppo_buy_paper_*_steps.zip"))
            latest_ckpt = None
            if ckpt_files:
                # Extract step count and find max
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
            # 嘗試刪除舊檔案
            for p in [final_buy_path, best_buy_path]:
                if os.path.exists(p):
                    try:
                        os.remove(p)
                        print(f"已刪除: {p}")
                    except OSError as e:
                        print(f"無法刪除 {p}: {e}")
            
            # Fix: Also delete intermediate checkpoints
            ckpt_pattern = os.path.join(MODELS_PATH, "ppo_buy_paper_*_steps.zip")
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
            print("🚀 開始全新訓練 Buy Agent (Paper Logic)...")
            buy_model = PPO("MlpPolicy", buy_env, **ppo_params)
        else:
            print("🚀 接續訓練 Buy Agent (Paper Logic)...")
            
        buy_model.learn(total_timesteps=TOTAL_TIMESTEPS_BUY, callback=callbacks, reset_num_timesteps=False)
        buy_model.save(os.path.join(MODELS_PATH, "ppo_buy_paper_final"))
    
    # 6. 訓練 Sell Agent
    print("\n=== 檢查 Sell Agent 模型 ===")
    final_sell_path = os.path.join(MODELS_PATH, "ppo_sell_paper_final.zip")
    best_sell_path = os.path.join(MODELS_PATH, "best_sell_paper", "best_model.zip")
    
    sell_model = None
    should_train_sell = True
    
    # Check for checkpoints
    sell_ckpt_files = glob.glob(os.path.join(MODELS_PATH, "ppo_sell_paper_*_steps.zip"))
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
            elif sell_ckpt_files:
                 latest_ckpt = max(sell_ckpt_files, key=lambda x: int(x.split('_')[-2]))
                 sell_model = PPO.load(latest_ckpt, device=device)
            should_train_sell = False

    if should_train_sell:
        if sell_model is None:
            print("🚀 開始全新訓練 Sell Agent (Paper Logic)...")
            sell_model = PPO("MlpPolicy", sell_env, **ppo_params)
        else:
            print("🚀 接續訓練 Sell Agent (Paper Logic)...")
        
        # Add CheckpointCallback for Sell Agent
        sell_checkpoint_callback = CheckpointCallback(save_freq=5000, save_path=MODELS_PATH, name_prefix="ppo_sell_paper")
        sell_eval_callback = EvalCallback(eval_sell_env, best_model_save_path=os.path.join(MODELS_PATH, "best_sell_paper"),
                                     log_path=SELL_RESULTS_PATH, eval_freq=2000, n_eval_episodes=100, deterministic=True, render=False)
        sell_callbacks = CallbackList([sell_checkpoint_callback, sell_eval_callback])
        
        sell_model.learn(total_timesteps=TOTAL_TIMESTEPS_SELL, callback=sell_callbacks, reset_num_timesteps=False)
        sell_model.save(os.path.join(MODELS_PATH, "ppo_sell_paper_final"))
    
    # 7. 執行回測
    print("\n=== 執行回測 (Paper Logic) ===")
    best_buy_path = os.path.join(MODELS_PATH, "best_buy_paper", "best_model.zip")
    if os.path.exists(best_buy_path):
        buy_model = PPO.load(best_buy_path, device=device)
    else:
        buy_model = PPO.load(os.path.join(MODELS_PATH, "ppo_buy_paper_final"), device=device)
    
    best_sell_path = os.path.join(MODELS_PATH, "best_sell_paper", "best_model.zip")
    if os.path.exists(best_sell_path):
        sell_model = PPO.load(best_sell_path, device=device)
        print(f"回測使用最佳 Sell Model: {best_sell_path}")
    else:
        sell_model = PPO.load(os.path.join(MODELS_PATH, "ppo_sell_paper_final"), device=device)
        print(f"回測使用最終 Sell Model")
    
    stock_data_only = {k: v for k, v in processed_data.items() if k != "^TWII" and k != "0050.TW"}
    analyzer = DetailedBacktesterPaper(stock_data_only, buy_model, sell_model)
    daily_stats, trade_logs = analyzer.run()
    
    # 8. 輸出結果與繪圖
    print("\n回測完成！")
    initial_val = daily_stats.iloc[0]['Total_Value']
    final_val = daily_stats.iloc[-1]['Total_Value']
    print(f"期初資金: {initial_val:.0f}")
    print(f"期末資金: {final_val:.0f}")

    # --- 列印交易紀錄 (新增功能) ---
    print("\n=== 交易紀錄明細 ===")
    print(f"{'日期':<12} {'代號':<10} {'動作':<6} {'價格':<10} {'原因'}")
    print("-" * 60)
    # 將 trade_logs 轉為 DataFrame 方便排序與顯示
    if trade_logs:
        logs_df = pd.DataFrame(trade_logs)
        logs_df['Date'] = pd.to_datetime(logs_df['Date'])
        logs_df = logs_df.sort_values(by='Date')
        
        for _, row in logs_df.iterrows():
            date_str = row['Date'].strftime('%Y-%m-%d')
            ticker = row['Ticker'].replace('.TW', '')
            action = "買入" if row['Action'] == 'Buy' else "賣出"
            price = f"{row['Price']:.2f}"
            reason = row['Reason']
            print(f"{date_str:<12} {ticker:<10} {action:<6} {price:<10} {reason}")
    else:
        print("無交易紀錄")
    print("-" * 60)
    
    # --- 準備 Benchmark (0050.TW) ---
    bench_df = processed_data.get("0050.TW")
    if bench_df is not None:
        # 截取回測期間的 Benchmark 資料
        bench_df = bench_df.loc[daily_stats.index[0]:daily_stats.index[-1]]
        # 正規化：讓 Benchmark 起始資金與 AI 相同
        bench_norm = (bench_df['Close'] / bench_df['Close'].iloc[0]) * initial_val
    else:
        bench_norm = None
        print("⚠️ 無法取得 0050.TW 資料作為 Benchmark")

    # --- 繪圖 ---
    plt.figure(figsize=(14, 8))
    
    # 1. 繪製 AI 績效
    plt.plot(daily_stats.index, daily_stats['Total_Value'], label='AI Portfolio', color='blue', linewidth=2)
    
    # 2. 繪製 Benchmark
    if bench_norm is not None:
        plt.plot(bench_norm.index, bench_norm, label='Benchmark (0050.TW)', color='gray', linestyle='--', alpha=0.8)
    
    # 3. 標註買賣點
    # 為了避免圖表過於雜亂，我們只標註點，並用不同顏色區分
    buy_logs = [log for log in trade_logs if log['Action'] == 'Buy']
    sell_logs = [log for log in trade_logs if log['Action'] == 'Sell']
    
    # 提取買入點的日期與對應的 Portfolio Value (為了標在曲線上)
    buy_dates = [log['Date'] for log in buy_logs]
    # 找到最接近的日期對應的 Value
    buy_values = [daily_stats.loc[d]['Total_Value'] for d in buy_dates if d in daily_stats.index]
    buy_dates_filtered = [d for d in buy_dates if d in daily_stats.index] # 確保對齊
    
    sell_dates = [log['Date'] for log in sell_logs]
    sell_values = [daily_stats.loc[d]['Total_Value'] for d in sell_dates if d in daily_stats.index]
    sell_dates_filtered = [d for d in sell_dates if d in daily_stats.index]

    # 修改顏色：買入=紅色 (Red), 賣出=綠色 (Green)
    plt.scatter(buy_dates_filtered, buy_values, marker='^', color='red', s=50, label='Buy Signal', zorder=5)
    plt.scatter(sell_dates_filtered, sell_values, marker='v', color='green', s=50, label='Sell Signal', zorder=5)

    # 4. 標註股票代號 (選填，避免重疊過多只標註前幾個或間隔標註)
    # 這裡示範標註：只標註買入點的股票代號
    for i, log in enumerate(buy_logs):
        if log['Date'] in daily_stats.index:
             # 簡單防止文字重疊：稍微錯開 y 軸
             y_val = daily_stats.loc[log['Date']]['Total_Value']
             plt.text(log['Date'], y_val * 1.02, log['Ticker'].replace('.TW', ''), fontsize=8, color='red', alpha=0.7)

    plt.title('Pro Trader RL (Paper Logic) vs 0050.TW Benchmark')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value (TWD)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(RESULTS_PATH, "paper_performance_enhanced.png")
    plt.savefig(save_path)
    print(f"增強版圖表已儲存至: {save_path}")
