"""
Generates the Liquid Rocket Engine Anomaly Detection Dataset (LRE-AD-25K) directly into data/
"""

import os
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

TARGET_CSV = os.path.join(DATA_DIR, "rocket_engine_anomaly_dataset.csv")

np.random.seed(42)

def generate_dataset():
    print("=" * 65)
    print("GENERATING MASTER ROCKET TELEMETRY DATASET INTO data/")
    print("=" * 65)
    
    run_configs = [
        {"run_id": f"TEST_NOM_{i:02d}", "anomaly": "NOMINAL", "throttle_profile": "standard"}
        for i in range(1, 11)
    ]
    
    anomaly_specs = [
        ("TEST_ANO_HPOTP_01", "HPOTP_BEARING_DEGRADATION", 42.0, 1.0),
        ("TEST_ANO_HPOTP_02", "HPOTP_BEARING_DEGRADATION", 38.0, 0.85),
        ("TEST_ANO_TURB_01",  "HPFTP_TURBINE_BLADE_EROSION", 35.0, 0.95),
        ("TEST_ANO_TURB_02",  "HPFTP_TURBINE_BLADE_EROSION", 40.0, 0.78),
        ("TEST_ANO_CAV_01",   "LOX_PUMP_CAVITATION", 45.0, 0.90),
        ("TEST_ANO_CAV_02",   "LOX_PUMP_CAVITATION", 50.0, 0.80),
        ("TEST_ANO_COOL_01",  "COOLANT_CHANNEL_RESTRICTION", 30.0, 0.92),
        ("TEST_ANO_COOL_02",  "COOLANT_CHANNEL_RESTRICTION", 36.0, 0.75),
        ("TEST_ANO_VALV_01",  "FPOV_ACTUATOR_DRIFT", 40.0, 0.88),
        ("TEST_ANO_VALV_02",  "FPOV_ACTUATOR_DRIFT", 48.0, 0.70),
        ("TEST_ANO_INJ_01",   "INJECTOR_EROSION_OR_CHOKING", 35.0, 0.85),
        ("TEST_ANO_INJ_02",   "INJECTOR_EROSION_OR_CHOKING", 44.0, 0.72),
        ("TEST_ANO_SENS_01",  "SENSOR_BIAS_DRIFT_P_MCC", 28.0, 0.80),
        ("TEST_ANO_SENS_02",  "SENSOR_BIAS_DRIFT_P_MCC", 32.0, 0.90),
        ("TEST_ANO_INSTAB_01","COMBUSTION_INSTABILITY_ROUGHNESS", 52.0, 0.96)
    ]
    
    for r_id, a_type, onset, max_sev in anomaly_specs:
        run_configs.append({
            "run_id": r_id,
            "anomaly": a_type,
            "onset_time": onset,
            "max_severity": max_sev,
            "throttle_profile": "standard"
        })
        
    records = []
    dt = 0.1
    time_series = np.arange(0.0, 100.0 + dt, dt)
    
    for config in run_configs:
        run_id = config["run_id"]
        a_type = config["anomaly"]
        onset = config.get("onset_time", 999.0)
        target_sev = config.get("max_severity", 0.0)
        
        cal_p_mcc = np.random.normal(0, 0.4)
        cal_t_hpftp = np.random.normal(0, 3.0)
        cal_t_hpotp = np.random.normal(0, 2.5)
        cal_vib_hpotp = np.random.uniform(0.95, 1.05)
        cal_vib_hpftp = np.random.uniform(0.95, 1.05)
        
        for t in time_series:
            t = round(float(t), 2)
            
            if t < 5.0:
                phase = "Ignition"
                power_pct = 20.0 + (t / 5.0) * 45.0
            elif t < 10.0:
                phase = "Ramp_Up"
                power_pct = 65.0 + ((t - 5.0) / 5.0) * 39.0
            elif t < 55.0:
                phase = "Mainstage_104"
                power_pct = 104.0
            elif t < 60.0:
                phase = "Throttle_Down"
                power_pct = 104.0 - ((t - 55.0) / 5.0) * 39.0
            elif t < 80.0:
                phase = "Cruising_65"
                power_pct = 65.0
            elif t < 85.0:
                phase = "Throttle_Up"
                power_pct = 65.0 + ((t - 80.0) / 5.0) * 35.0
            elif t < 92.0:
                phase = "Mainstage_100"
                power_pct = 100.0
            else:
                phase = "Shutdown"
                power_pct = max(0.0, 100.0 - ((t - 92.0) / 8.0) * 100.0)
                
            throttle_factor = power_pct / 100.0
            
            if t >= onset and a_type != "NOMINAL":
                if a_type in ["COMBUSTION_INSTABILITY_ROUGHNESS", "SENSOR_BIAS_DRIFT_P_MCC"]:
                    progress = min(1.0, (t - onset) / 3.0)
                else:
                    progress = min(1.0, (t - onset) / 12.0)
                severity = round(target_sev * progress, 4)
                anomaly_flag = 1
            else:
                severity = 0.0
                anomaly_flag = 0
                
            p_mcc_nom = 204.0 * throttle_factor + cal_p_mcc
            m_ox_nom = 410.0 * throttle_factor
            m_fuel_nom = 68.0 * throttle_factor
            p_hpotp_disch_nom = 305.0 * (throttle_factor ** 1.1)
            p_hpftp_disch_nom = 435.0 * (throttle_factor ** 1.1)
            rpm_hpotp_nom = 28100.0 * np.sqrt(max(0.01, throttle_factor))
            rpm_hpftp_nom = 35100.0 * np.sqrt(max(0.01, throttle_factor))
            t_hpotp_turb_nom = 780.0 + 35.0 * (throttle_factor - 1.0) + cal_t_hpotp
            t_hpftp_turb_nom = 985.0 + 50.0 * (throttle_factor - 1.0) + cal_t_hpftp
            vib_hpotp_nom = (3.2 * throttle_factor + 0.3) * cal_vib_hpotp
            vib_hpftp_nom = (4.1 * throttle_factor + 0.4) * cal_vib_hpftp
            p_inj_ox_nom = p_mcc_nom + 22.0 * (throttle_factor ** 1.2)
            p_inj_fuel_nom = p_mcc_nom + 31.0 * (throttle_factor ** 1.2)
            coolant_dp_nom = 46.0 * (throttle_factor ** 1.3)
            coolant_temp_nom = 515.0 + 30.0 * throttle_factor
            pos_fpov_nom = 54.0 * np.sqrt(throttle_factor)
            pos_opov_nom = 58.0 * np.sqrt(throttle_factor)
            
            p_mcc = p_mcc_nom
            m_ox = m_ox_nom
            m_fuel = m_fuel_nom
            p_hpotp_disch = p_hpotp_disch_nom
            p_hpftp_disch = p_hpftp_disch_nom
            rpm_hpotp = rpm_hpotp_nom
            rpm_hpftp = rpm_hpftp_nom
            t_hpotp_turb = t_hpotp_turb_nom
            t_hpftp_turb = t_hpftp_turb_nom
            vib_hpotp = vib_hpotp_nom
            vib_hpftp = vib_hpftp_nom
            p_inj_ox = p_inj_ox_nom
            p_inj_fuel = p_inj_fuel_nom
            coolant_dp = coolant_dp_nom
            coolant_temp = coolant_temp_nom
            pos_fpov = pos_fpov_nom
            pos_opov = pos_opov_nom
            noise_amp_p = 0.5
            
            if anomaly_flag:
                if a_type == "HPOTP_BEARING_DEGRADATION":
                    vib_hpotp += severity * 7.5
                    t_hpotp_turb += severity * 28.0
                    rpm_hpotp -= severity * 450.0
                elif a_type == "HPFTP_TURBINE_BLADE_EROSION":
                    t_hpftp_turb += severity * 135.0
                    vib_hpftp += severity * 3.8
                    pos_fpov += severity * 8.5
                elif a_type == "LOX_PUMP_CAVITATION":
                    p_hpotp_disch -= severity * 38.0
                    m_ox -= severity * 25.0
                    p_mcc -= severity * 14.0
                    vib_hpotp += severity * 4.2
                    noise_amp_p += severity * 3.5
                elif a_type == "COOLANT_CHANNEL_RESTRICTION":
                    coolant_dp += severity * 32.0
                    coolant_temp += severity * 165.0
                    p_hpftp_disch += severity * 18.0
                elif a_type == "FPOV_ACTUATOR_DRIFT":
                    pos_fpov -= severity * 14.0
                    t_hpftp_turb -= severity * 85.0
                    p_mcc -= severity * 16.0
                    m_fuel -= severity * 6.5
                elif a_type == "INJECTOR_EROSION_OR_CHOKING":
                    p_inj_fuel += severity * 29.0
                    p_inj_ox += severity * 22.0
                    p_mcc -= severity * 11.0
                    noise_amp_p += severity * 2.0
                elif a_type == "SENSOR_BIAS_DRIFT_P_MCC":
                    p_mcc += severity * 22.0
                elif a_type == "COMBUSTION_INSTABILITY_ROUGHNESS":
                    noise_amp_p += severity * 12.0
                    vib_hpftp += severity * 3.5
                    vib_hpotp += severity * 3.0
                    
            p_mcc += np.random.normal(0, noise_amp_p)
            m_ox += np.random.normal(0, 0.8)
            m_fuel += np.random.normal(0, 0.25)
            p_hpotp_disch += np.random.normal(0, 0.9)
            p_hpftp_disch += np.random.normal(0, 1.2)
            rpm_hpotp += np.random.normal(0, 45.0)
            rpm_hpftp += np.random.normal(0, 55.0)
            t_hpotp_turb += np.random.normal(0, 1.8)
            t_hpftp_turb += np.random.normal(0, 2.2)
            vib_hpotp += np.random.normal(0, 0.15)
            vib_hpftp += np.random.normal(0, 0.18)
            p_inj_ox += np.random.normal(0, 0.7)
            p_inj_fuel += np.random.normal(0, 0.8)
            coolant_dp += np.random.normal(0, 0.3)
            coolant_temp += np.random.normal(0, 1.0)
            pos_fpov += np.random.normal(0, 0.1)
            pos_opov += np.random.normal(0, 0.1)
            
            of_ratio = round(m_ox / max(0.1, m_fuel), 3)
            redline_tripped = (
                (t_hpftp_turb > 1080.0) or
                (t_hpotp_turb > 880.0) or
                (vib_hpotp > 8.5) or
                (coolant_temp > 640.0)
            )
            
            records.append({
                "run_id": run_id,
                "timestamp_sec": t,
                "operational_phase": phase,
                "throttle_pct": round(power_pct, 1),
                "p_mcc_bar": round(p_mcc, 2),
                "m_ox_kg_s": round(m_ox, 2),
                "m_fuel_kg_s": round(m_fuel, 2),
                "of_mixture_ratio": of_ratio,
                "p_hpotp_disch_bar": round(p_hpotp_disch, 2),
                "t_hpotp_turb_k": round(t_hpotp_turb, 1),
                "rpm_hpotp": round(rpm_hpotp, 0),
                "vib_hpotp_radial_g": round(max(0.0, vib_hpotp), 2),
                "p_hpftp_disch_bar": round(p_hpftp_disch, 2),
                "t_hpftp_turb_k": round(t_hpftp_turb, 1),
                "rpm_hpftp": round(rpm_hpftp, 0),
                "vib_hpftp_radial_g": round(max(0.0, vib_hpftp), 2),
                "p_inj_ox_dome_bar": round(p_inj_ox, 2),
                "p_inj_fuel_man_bar": round(p_inj_fuel, 2),
                "coolant_dp_bar": round(coolant_dp, 2),
                "coolant_jacket_temp_k": round(coolant_temp, 1),
                "pos_fpov_pct": round(pos_fpov, 2),
                "pos_opov_pct": round(pos_opov, 2),
                "anomaly_flag": anomaly_flag,
                "anomaly_class": a_type if anomaly_flag else "NOMINAL",
                "severity": severity,
                "redline_tripped": int(redline_tripped)
            })

    df = pd.DataFrame(records)
    df.to_csv(TARGET_CSV, index=False)
    print(f"Master dataset successfully created: {len(df):,} rows -> {TARGET_CSV}")

if __name__ == "__main__":
    generate_dataset()
