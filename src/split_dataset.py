"""
Partitions data/rocket_engine_anomaly_dataset.csv into data/train_data.csv and data/test_data.csv
using a leak-free Grouped Split by run_id (hot-fire test firing).
"""

import os
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

MAIN_CSV = os.path.join(DATA_DIR, "rocket_engine_anomaly_dataset.csv")
TRAIN_CSV = os.path.join(DATA_DIR, "train_data.csv")
TEST_CSV = os.path.join(DATA_DIR, "test_data.csv")

def split_dataset():
    print("=" * 65)
    print("PARTITIONING DATASET INTO data/train_data.csv AND data/test_data.csv")
    print("=" * 65)
    
    if not os.path.exists(MAIN_CSV):
        raise FileNotFoundError(f"{MAIN_CSV} not found! Run generate_dataset.py first.")
        
    df = pd.read_csv(MAIN_CSV)
    print(f"Loaded master dataset: {len(df):,} rows from {MAIN_CSV}")
    
    # 15 Hot-Fire Runs for Training (60%)
    train_runs = [
        "TEST_NOM_01", "TEST_NOM_02", "TEST_NOM_03", 
        "TEST_NOM_04", "TEST_NOM_05", "TEST_NOM_06", "TEST_NOM_07",
        "TEST_ANO_HPOTP_01",
        "TEST_ANO_TURB_01",
        "TEST_ANO_CAV_01",
        "TEST_ANO_COOL_01",
        "TEST_ANO_VALV_01",
        "TEST_ANO_INJ_01",
        "TEST_ANO_SENS_01",
        "TEST_ANO_INSTAB_01"
    ]
    
    # 10 Hot-Fire Runs for Testing (40%)
    test_runs = [
        "TEST_NOM_08", "TEST_NOM_09", "TEST_NOM_10",
        "TEST_ANO_HPOTP_02",
        "TEST_ANO_TURB_02",
        "TEST_ANO_CAV_02",
        "TEST_ANO_COOL_02",
        "TEST_ANO_VALV_02",
        "TEST_ANO_INJ_02",
        "TEST_ANO_SENS_02"
    ]
    
    assert set(train_runs).isdisjoint(set(test_runs)), "Error: Train and Test runs overlap!"
    assert len(train_runs) + len(test_runs) == df["run_id"].nunique(), "Error: Missing runs!"
    
    train_df = df[df["run_id"].isin(train_runs)].copy()
    test_df = df[df["run_id"].isin(test_runs)].copy()
    
    train_df.to_csv(TRAIN_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)
    
    print(f"[OK] Train dataset: {len(train_df):,} rows -> {TRAIN_CSV}")
    print(f"[OK] Test dataset:  {len(test_df):,} rows -> {TEST_CSV}")
    print("=" * 65)

if __name__ == "__main__":
    split_dataset()
