"""
Method 1: Fixed Threshold Monitoring (Static Operating Envelope & Redlines)
Evaluates threshold breaches, calculates detection delay from actual fault onset,
saves predictions to results/results_fixed_threshold.csv, and logs metrics to results/benchmark_metrics.json.
"""

import os
import json
import time
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

TEST_DATA_PATH = os.path.join(DATA_DIR, "test_data.csv")
OUTPUT_CSV = os.path.join(RESULTS_DIR, "results_fixed_threshold.csv")
METRICS_JSON = os.path.join(RESULTS_DIR, "benchmark_metrics.json")

STATIC_THRESHOLDS = {
    "p_mcc_bar": {"min": 180.0, "max": 230.0},
    "vib_hpotp_radial_g": {"max": 8.5},
    "t_hpftp_turb_k": {"max": 1080.0},
    "t_hpotp_turb_k": {"max": 880.0},
    "coolant_jacket_temp_k": {"max": 640.0},
    "coolant_dp_bar": {"max": 65.0},
    "p_hpotp_disch_bar": {"min": 250.0},
    "p_hpftp_disch_bar": {"min": 350.0}
}

def run_fixed_threshold_monitor():
    print("=" * 65)
    print("METHOD 1: FIXED THRESHOLD MONITORING (TEST EVALUATION)")
    print("=" * 65)
    
    start_time = time.time()
    if not os.path.exists(TEST_DATA_PATH):
        raise FileNotFoundError(f"Test dataset not found at {TEST_DATA_PATH}. Please run split_dataset.py first.")
        
    test_df = pd.read_csv(TEST_DATA_PATH)
    print(f"Loaded test dataset: {len(test_df):,} rows from {TEST_DATA_PATH}")
    
    pred_anomalies = []
    anomaly_reasons = []
    anomaly_scores = []
    
    for _, row in test_df.iterrows():
        is_breached = False
        reasons = []
        max_ratio = 0.0
        
        for col, bounds in STATIC_THRESHOLDS.items():
            val = row[col]
            if "max" in bounds:
                max_val = bounds["max"]
                ratio = val / max_val
                if ratio > max_ratio:
                    max_ratio = ratio
                if val > max_val:
                    is_breached = True
                    reasons.append(f"{col} > {max_val} ({val:.1f})")
            if "min" in bounds:
                min_val = bounds["min"]
                ratio = min_val / max(0.01, val)
                if ratio > max_ratio:
                    max_ratio = ratio
                if val < min_val:
                    is_breached = True
                    reasons.append(f"{col} < {min_val} ({val:.1f})")
                    
        pred_anomalies.append(1 if is_breached else 0)
        anomaly_reasons.append("; ".join(reasons) if reasons else "None")
        anomaly_scores.append(round(max_ratio, 4))
        
    test_df["pred_anomaly"] = pred_anomalies
    test_df["anomaly_score"] = anomaly_scores
    test_df["violation_details"] = anomaly_reasons
    
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
        "pred_anomaly", "anomaly_score", "violation_details"
    ]
    test_df[export_cols].to_csv(OUTPUT_CSV, index=False)
    print(f"\nPredictions saved to: {OUTPUT_CSV}")
    
    metrics_entry = {
        "method_name": "Fixed Threshold Monitoring",
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
            
    all_metrics["fixed_threshold"] = metrics_entry
    with open(METRICS_JSON, "w") as f:
        json.dump(all_metrics, f, indent=4)
    print(f"Metrics saved to:     {METRICS_JSON}")
    print("=" * 65)

if __name__ == "__main__":
    run_fixed_threshold_monitor()
