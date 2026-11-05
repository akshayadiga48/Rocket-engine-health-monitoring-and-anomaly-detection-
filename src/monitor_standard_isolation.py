"""
Method 2: Standard Isolation Forest (Unconditional Multi-Sensor Outlier Detection)
Calculates detection delay from actual fault onset, saves predictions and model,
and updates results/benchmark_metrics.json.
"""

import os
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

TRAIN_DATA_PATH = os.path.join(DATA_DIR, "train_data.csv")
TEST_DATA_PATH = os.path.join(DATA_DIR, "test_data.csv")
OUTPUT_CSV = os.path.join(RESULTS_DIR, "results_standard_isolation.csv")
MODEL_FILE = os.path.join(RESULTS_DIR, "model_standard_isolation.pkl")
METRICS_JSON = os.path.join(RESULTS_DIR, "benchmark_metrics.json")

FEATURE_COLS = [
    "p_mcc_bar", "m_ox_kg_s", "m_fuel_kg_s", "of_mixture_ratio",
    "p_hpotp_disch_bar", "t_hpotp_turb_k", "rpm_hpotp", "vib_hpotp_radial_g",
    "p_hpftp_disch_bar", "t_hpftp_turb_k", "rpm_hpftp", "vib_hpftp_radial_g",
    "p_inj_ox_dome_bar", "p_inj_fuel_man_bar", "coolant_dp_bar", "coolant_jacket_temp_k",
    "pos_fpov_pct", "pos_opov_pct"
]

def run_standard_isolation_monitor():
    print("=" * 65)
    print("METHOD 2: STANDARD ISOLATION FOREST (TRAIN / TEST PIPELINE)")
    print("=" * 65)
    
    start_time = time.time()
    if not os.path.exists(TRAIN_DATA_PATH) or not os.path.exists(TEST_DATA_PATH):
        raise FileNotFoundError("Train/Test datasets missing from data/. Please run split_dataset.py first.")
        
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)
    
    nominal_train = train_df[train_df["anomaly_flag"] == 0]
    print(f"Loaded Train Data: {len(train_df):,} total rows | {len(nominal_train):,} nominal baseline rows")
    print(f"Loaded Test Data:  {len(test_df):,} total rows (Unseen hot-fire runs)")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(nominal_train[FEATURE_COLS])
    
    print("Fitting global Isolation Forest on nominal training baseline...")
    iso_forest = IsolationForest(
        n_estimators=150,
        contamination=0.03,
        max_samples="auto",
        random_state=42,
        n_jobs=-1
    )
    iso_forest.fit(X_train_scaled)
    
    joblib.dump({"scaler": scaler, "model": iso_forest, "features": FEATURE_COLS}, MODEL_FILE)
    print(f"Trained model saved to: {MODEL_FILE}")
    
    train_scores = -iso_forest.decision_function(X_train_scaled)
    threshold = float(np.percentile(train_scores, 98.0))
    print(f"Calibrated Baseline Threshold: {threshold:.4f}")
    
    X_test_scaled = scaler.transform(test_df[FEATURE_COLS])
    test_scores = -iso_forest.decision_function(X_test_scaled)
    test_preds = (test_scores > threshold).astype(int)
    
    test_df["anomaly_score"] = np.round(test_scores, 4)
    test_df["pred_anomaly"] = test_preds
    
    exec_time = time.time() - start_time
    
    y_true = test_df["anomaly_flag"].values
    y_pred = test_df["pred_anomaly"].values
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    throttle_phases = ["Throttle_Down", "Cruising_65", "Throttle_Up"]
    throttle_nominal = test_df[(test_df["operational_phase"].isin(throttle_phases)) & (test_df["anomaly_flag"] == 0)]
    throttle_far = float((throttle_nominal["pred_anomaly"] == 1).mean()) if len(throttle_nominal) > 0 else 0.0
    
    # Detection Delay: Time from actual fault onset (anomaly_flag == 1) to first detection
    delays = []
    anomaly_runs = test_df[test_df["anomaly_flag"] == 1]["run_id"].unique()
    for run_id in anomaly_runs:
        run_df = test_df[test_df["run_id"] == run_id]
        onset_time = run_df[run_df["anomaly_flag"] == 1]["timestamp_sec"].iloc[0]
        detected = run_df[(run_df["pred_anomaly"] == 1) & (run_df["timestamp_sec"] >= onset_time)]
        if len(detected) > 0:
            first_detect_time = detected["timestamp_sec"].iloc[0]
            delays.append(round(first_detect_time - onset_time, 2))
        else:
            total_duration = run_df["timestamp_sec"].iloc[-1] - onset_time
            delays.append(round(total_duration, 2))
            
    mean_detection_delay = float(np.mean(delays)) if delays else 0.0
    
    print(f"\nExecution Time:       {exec_time:.3f} s")
    print(f"Accuracy:             {acc * 100:.2f}%")
    print(f"Precision:            {prec * 100:.2f}%")
    print(f"Recall:               {rec * 100:.2f}%")
    print(f"F1-Score:             {f1 * 100:.2f}%")
    print(f"Confusion:            TP={tp:,}, FP={fp:,}, TN={tn:,}, FN={fn:,}")
    print(f"Throttling FAR:       {throttle_far * 100:.2f}%")
    print(f"Mean Detection Delay: {mean_detection_delay:.2f} s")
    
    export_cols = [
        "run_id", "timestamp_sec", "operational_phase", "throttle_pct",
        "anomaly_flag", "anomaly_class", "redline_tripped",
        "pred_anomaly", "anomaly_score"
    ]
    test_df[export_cols].to_csv(OUTPUT_CSV, index=False)
    print(f"\nTest predictions saved to: {OUTPUT_CSV}")
    
    metrics_entry = {
        "method_name": "Standard Isolation Forest",
        "dataset_evaluated": "test_data.csv",
        "total_test_samples": len(test_df),
        "execution_time_sec": round(exec_time, 4),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "throttle_far": round(throttle_far, 4),
        "mean_detection_delay_sec": round(mean_detection_delay, 2)
    }
    
    all_metrics = {}
    if os.path.exists(METRICS_JSON):
        try:
            with open(METRICS_JSON, "r") as f:
                all_metrics = json.load(f)
        except Exception:
            all_metrics = {}
            
    all_metrics["standard_isolation"] = metrics_entry
    with open(METRICS_JSON, "w") as f:
        json.dump(all_metrics, f, indent=4)
    print(f"Metrics saved to:          {METRICS_JSON}")
    print("=" * 65)

if __name__ == "__main__":
    run_standard_isolation_monitor()
