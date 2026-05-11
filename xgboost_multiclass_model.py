"""
CICIoT2023 - XGBoost Multiclass Model Egitimi

Kullanim:
    python xgboost_multiclass_model.py
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


LABEL_DECODE = {0: 'Benign', 1: 'Flood', 2: 'Mirai', 3: 'Recon', 4: 'MitM', 5: 'Exploit'}

PARAMS = {
    "objective":         "multi:softprob",
    "num_class":         6,
    "learning_rate":     0.03,
    "max_depth":         9,
    "min_child_weight":  3,
    "gamma":             0.15,
    "subsample":         0.85,
    "colsample_bytree":  0.8,
    "reg_alpha":         0.05,
    "reg_lambda":        1.5,
    "tree_method":       "hist",
    "eval_metric":       "mlogloss",
    "seed":              42,
    "nthread":           -1,
}

def print_dist(y, title="Dagilim"):
    print(f"\n--- {title} ---")
    total = len(y)
    for code, name in sorted(LABEL_DECODE.items()):
        count = (y == code).sum()
        print(f"    {code} - {name:<12} {count:>8,}  ({100 * count / total:.1f}%)")

print("1. Veri yukleniyor...")
df = pd.read_parquet("preprocessed_multiclass.parquet")

exclude_cols = {'Attack_Category', 'class_label'}
feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                if c not in exclude_cols]
print(f"  {len(feature_cols)} feature kullanilacak.")

X = df[feature_cols].values
y = df['class_label'].astype(int).values

print_dist(y, "Tum Veri Seti")

print("\n2. Train / Val / Test Bolunuyor...")
X_tr, X_test, y_tr, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_tr, X_val,  y_tr, y_val  = train_test_split(X_tr, y_tr, test_size=0.1, random_state=42, stratify=y_tr)

print(f"    Train : {len(X_tr):,}")
print(f"    Val   : {len(X_val):,}")
print(f"    Test  : {len(X_test):,}")

print("\n3. Class Weights Hesaplaniyor...")

custom_weights = {
    0: 1.0,   # Benign
    1: 1.5,   # Flood
    2: 1.5,   # Mirai
    3: 1.5,   # Recon
    4: 1.5,   # MitM
    5: 6.0    # Exploit
}

sample_weights = np.array([custom_weights[label] for label in y_tr])

print("\n4. XGBoost Egitimi Basliyor...")
dtr  = xgb.DMatrix(X_tr, label=y_tr, weight=sample_weights, feature_names=feature_cols)
dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_cols)
dtest = xgb.DMatrix(X_test, feature_names=feature_cols)

model = xgb.train(
    PARAMS,
    dtr,
    num_boost_round=800,
    evals=[(dtr, "train"), (dval, "val")],
    early_stopping_rounds=80,
    verbose_eval=100
)

print("\n5. Test Seti Degerlendirmesi...")
y_pred_prob = model.predict(dtest)
y_pred = np.argmax(y_pred_prob, axis=1)

names = [LABEL_DECODE[i] for i in range(len(LABEL_DECODE))]
print("\n--- SINIFLANDIRMA RAPORU ---")
print(classification_report(y_test, y_pred, target_names=names))

print("\n--- KARISIKLIK MATRISI ---")
cm = confusion_matrix(y_test, y_pred)
header = "{:<12}".format("") + "".join("{:>10}".format(n) for n in names)
print(header)
for i, row in enumerate(cm):
    print("{:<12}".format(names[i]) + "".join("{:>10,}".format(v) for v in row))

print("\n6. Feature Importance (En Etkili 15 Ozellik)...")
scores = model.get_score(importance_type='gain')
fi = pd.Series(scores).sort_values(ascending=False)
for feat, val in fi.head(15).items():
    print(f"    {feat:<35} {val:.1f}")

joblib.dump(model, "xgboost_multiclass_model.pkl")
joblib.dump(feature_cols, "feature_cols_multiclass.pkl")
print("\nModel basariyla kaydedildi!")
