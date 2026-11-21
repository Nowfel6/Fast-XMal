import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from src.dataset import load_dataset

def main():
    print("=" * 80)
    print(" FAST-XMAL: STOCHASTIC & STRUCTURED BINDER MASKING AUDIT")
    print("=" * 80)

    _, _, X_hold_60, y_hold, top60_cols, _, _, _ = load_dataset()
    model = joblib.load("models/v7_fast_xmal_student.joblib")

    clean_pred = model.predict(X_hold_60)
    base_acc = accuracy_score(y_hold, clean_pred) * 100.0
    base_f1  = f1_score(y_hold, clean_pred, average="macro") * 100.0
    print(f"Clean Baseline (0% Masking): Acc = {base_acc:.2f}% | Macro F1 = {base_f1:.2f}%")

    # 1. Stochastic Masking (0% - 20%, 20 seeds)
    print("\n[+] Evaluating Stochastic Masking (20 trials per level)...")
    seeds = [42, 100, 2024, 7, 999, 123, 314, 2718, 2025, 8080, 11, 22, 33, 44, 55, 66, 77, 88, 101, 111]
    for rate in [0.05, 0.10, 0.15, 0.20]:
        accs, f1s = [], []
        for s in seeds:
            Xm = X_hold_60.copy()
            rng = np.random.default_rng(s)
            Xm[rng.random(size=Xm.shape) < rate] = 0.0
            pm = model.predict(Xm)
            accs.append(accuracy_score(y_hold, pm) * 100.0)
            f1s.append(f1_score(y_hold, pm, average="macro") * 100.0)
        m_acc, s_acc = np.mean(accs), np.std(accs, ddof=1)
        m_f1, s_f1   = np.mean(f1s), np.std(f1s, ddof=1)
        print(f"  {int(rate*100):2d}% Random Masking: Acc = {m_acc:.2f} ± {s_acc:.2f}% | F1 = {m_f1:.2f} ± {s_f1:.2f}% (Ret={m_f1/base_f1*100:.1f}%)")

    # 2. Structured 23-Binder Zero-Masking
    with open("configs/kernel_binder_partition.json", "r") as fh:
        k_b_info = json.load(fh)
    binder_indices = k_b_info["binder_indices"]
    print(f"\n[+] Evaluating Structured Zero-Masking on all {len(binder_indices)} Binder features...")
    X_m23 = X_hold_60.copy()
    X_m23[:, binder_indices] = 0.0
    p_m23 = model.predict(X_m23)
    acc_m23 = accuracy_score(y_hold, p_m23) * 100.0
    f1_m23  = f1_score(y_hold, p_m23, average="macro") * 100.0

    print(f"  Structured 23-Binder Masked (37 Kernel-Only):")
    print(f"    Holdout Accuracy : {acc_m23:.2f}% ({int((p_m23 == y_hold).sum())} / 2,320)")
    print(f"    Holdout Macro F1 : {f1_m23:.2f}% (Retention = {f1_m23/base_f1*100:.1f}%)")
    print("=" * 80)

if __name__ == "__main__":
    main()