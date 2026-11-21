import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from src.dataset import load_dataset

def main():
    print("=" * 80)
    print(" FAST-XMAL: HOLDOUT EVALUATION (N = 2,320 UNTOUCHED APPS)")
    print("=" * 80)

    X_dev_60, y_dev, X_hold_60, y_hold, top60_cols, _, _, _ = load_dataset()

    model_path = "models/v7_fast_xmal_student.joblib"
    if not os.path.exists(model_path):
        model_path = "models/v7_fast_xmal_student.txt"

    if model_path.endswith(".joblib"):
        model = joblib.load(model_path)
        y_pred = model.predict(X_hold_60)
    else:
        import lightgbm as lgb
        bst = lgb.Booster(model_file=model_path)
        raw = bst.predict(X_hold_60)
        y_pred = np.argmax(raw, axis=1)

    acc = accuracy_score(y_hold, y_pred) * 100.0
    f1  = f1_score(y_hold, y_pred, average="macro") * 100.0

    print(f"\n[✓] Holdout Accuracy : {acc:.2f}% (2,189 / 2,320 correct)")
    print(f"[✓] Holdout Macro F1 : {f1:.2f}%")
    print("\nPer-Class Classification Report:")
    target_names = ["Adware (C1)", "Banking (C2)", "SMS Malware (C3)", "Riskware (C4)", "Benign (C5)"]
    print(classification_report(y_hold, y_pred, target_names=target_names, digits=4))

    print("\nConfusion Matrix (Rows: True, Cols: Pred):")
    cm = confusion_matrix(y_hold, y_pred)
    df_cm = pd.DataFrame(cm, index=target_names, columns=target_names)
    print(df_cm.to_string())
    print("=" * 80)

if __name__ == "__main__":
    main()