import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from src.dataset import load_dataset

def main():
    print("=" * 80)
    print(" FAST-XMAL: CROSS-PARTITION DUPLICATE VECTOR SENSITIVITY AUDIT")
    print("=" * 80)

    _, _, X_hold_60, y_hold, _, X_raw, dev_idx, hold_idx = load_dataset()
    model = joblib.load("models/v7_fast_xmal_student.joblib")

    dev_set = set(map(tuple, X_raw.iloc[dev_idx].to_numpy()))
    hold_tuples = list(map(tuple, X_raw.iloc[hold_idx].to_numpy()))

    clean_mask = np.array([t not in dev_set for t in hold_tuples], dtype=bool)
    num_dupes = int((~clean_mask).sum())
    print(f"[✓] Total Cross-Partition Identical 470-D Vectors: {num_dupes} / 2,320 ({num_dupes/2320*100:.2f}%)")

    preds = model.predict(X_hold_60)
    acc_full  = accuracy_score(y_hold, preds) * 100.0
    f1_full   = f1_score(y_hold, preds, average="macro") * 100.0

    acc_clean = accuracy_score(y_hold[clean_mask], preds[clean_mask]) * 100.0
    f1_clean  = f1_score(y_hold[clean_mask], preds[clean_mask], average="macro") * 100.0

    print(f"  Full Holdout  (N = 2,320) : Acc = {acc_full:.2f}% | Macro F1 = {f1_full:.2f}%")
    print(f"  Clean Holdout (N = 2,300) : Acc = {acc_clean:.2f}% | Macro F1 = {f1_clean:.2f}%")
    print(f"  Impact of Excluding Dupes : Delta Acc = {acc_clean - acc_full:+.2f} pp | Delta F1 = {f1_clean - f1_full:+.2f} pp")
    print("=" * 80)

if __name__ == "__main__":
    main()