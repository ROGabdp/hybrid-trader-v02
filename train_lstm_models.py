# -*- coding: utf-8 -*-
"""
長週期 LSTM 模型訓練腳本
訓練範圍：2000-01-01 ~ 2023-12-31

目的：
- 確保 Scaler 看過完整的歷史高低點
- 讓 RL 系統可以正確處理 2000 年以來的數據

執行方式：
  python train_lstm_models.py
"""

import os
import sys
import glob
import subprocess
from pathlib import Path

# Windows 終端機 UTF-8 編碼設定
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# =============================================================================
# 設定
# =============================================================================
TRAIN_START = "2000-01-01"
TRAIN_END = "2023-12-31"

PROJECT_PATH = Path(__file__).parent
CACHE_DIR = PROJECT_PATH / "data" / "processed"


def train_lstm_models():
    """訓練長週期 LSTM 模型"""
    
    print("\n" + "=" * 70)
    print("  🧠 Long-Period LSTM Model Training")
    print("=" * 70)
    print(f"  訓練範圍：{TRAIN_START} ~ {TRAIN_END}")
    print("=" * 70)
    
    # =========================================================================
    # 1. 訓練 T+1 模型 (Multivariate)
    # =========================================================================
    print("\n" + "-" * 60)
    print("📊 Training LSTM T+1 Model (Multivariate)")
    print("-" * 60)
    
    cmd_1d = [
        sys.executable,
        "twii_model_registry_multivariate.py",
        "train",
        "--start", TRAIN_START,
        "--end", TRAIN_END
    ]
    
    print(f"[Command] {' '.join(cmd_1d)}")
    result_1d = subprocess.run(cmd_1d, cwd=PROJECT_PATH)
    
    if result_1d.returncode != 0:
        print(f"[Error] T+1 model training failed with code {result_1d.returncode}")
    else:
        print("[Success] T+1 model training completed!")
    
    # =========================================================================
    # 2. 訓練 T+5 模型 (5-Day Direct)
    # =========================================================================
    print("\n" + "-" * 60)
    print("📈 Training LSTM T+5 Model (5-Day Direct)")
    print("-" * 60)
    
    cmd_5d = [
        sys.executable,
        "twii_model_registry_5d.py",
        "train",
        "--start", TRAIN_START,
        "--end", TRAIN_END
    ]
    
    print(f"[Command] {' '.join(cmd_5d)}")
    result_5d = subprocess.run(cmd_5d, cwd=PROJECT_PATH)
    
    if result_5d.returncode != 0:
        print(f"[Error] T+5 model training failed with code {result_5d.returncode}")
    else:
        print("[Success] T+5 model training completed!")
    
    # =========================================================================
    # 3. 清除快取
    # =========================================================================
    print("\n" + "-" * 60)
    print("🗑️ Clearing Feature Cache")
    print("-" * 60)
    
    cache_files = list(CACHE_DIR.glob("*.pkl"))
    
    if cache_files:
        print(f"[Cache] Found {len(cache_files)} cached files to delete:")
        for f in cache_files:
            try:
                f.unlink()
                print(f"  ✅ Deleted: {f.name}")
            except Exception as e:
                print(f"  ❌ Failed to delete {f.name}: {e}")
    else:
        print("[Cache] No cached files found.")
    
    # =========================================================================
    # 4. 回報結果
    # =========================================================================
    print("\n" + "=" * 70)
    print("📁 Training Results")
    print("=" * 70)
    
    # T+1 模型路徑
    models_1d_dir = PROJECT_PATH / "saved_models_multivariate"
    if models_1d_dir.exists():
        models_1d = list(models_1d_dir.glob(f"model_{TRAIN_START}_{TRAIN_END}.keras"))
        if models_1d:
            print(f"\n✅ T+1 Model Artifacts:")
            print(f"   {models_1d_dir / f'model_{TRAIN_START}_{TRAIN_END}.keras'}")
            print(f"   {models_1d_dir / f'feature_scaler_{TRAIN_START}_{TRAIN_END}.pkl'}")
            print(f"   {models_1d_dir / f'target_scaler_{TRAIN_START}_{TRAIN_END}.pkl'}")
            print(f"   {models_1d_dir / f'meta_{TRAIN_START}_{TRAIN_END}.json'}")
    
    # T+5 模型路徑
    models_5d_dir = PROJECT_PATH / "saved_models_5d"
    if models_5d_dir.exists():
        models_5d = list(models_5d_dir.glob(f"model_{TRAIN_START}_{TRAIN_END}.keras"))
        if models_5d:
            print(f"\n✅ T+5 Model Artifacts:")
            print(f"   {models_5d_dir / f'model_{TRAIN_START}_{TRAIN_END}.keras'}")
            print(f"   {models_5d_dir / f'feature_scaler_{TRAIN_START}_{TRAIN_END}.pkl'}")
            print(f"   {models_5d_dir / f'target_scaler_{TRAIN_START}_{TRAIN_END}.pkl'}")
            print(f"   {models_5d_dir / f'meta_{TRAIN_START}_{TRAIN_END}.json'}")
    
    print("\n" + "=" * 70)
    print("  ✅ Phase 2.5 Completed: Long-Period LSTM Models Trained")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    train_lstm_models()
