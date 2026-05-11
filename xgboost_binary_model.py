"""
CICIoT2023 - XGBoost Binary Model Egitimi
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

print("1. Veri yukleniyor...")
df = pd.read_parquet("preprocessed_binary.parquet")
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

exclude_cols = {'Label', 'Attack_Category', 'binary_label'}
feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                if c not in exclude_cols]
print(f"  {len(feature_cols)} feature kullanilacak.")

X = df[feature_cols].values
y = df['binary_label'].astype(int).values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
)

print(f"\n    Train : {len(X_train):,}")
print(f"    Val   : {len(X_val):,}")
print(f"    Test  : {len(X_test):,}")

print("\n2. XGBoost Modeli egitiliyor...")
n_normal = (y_train == 0).sum()
n_attack = (y_train == 1).sum()
scale_pos_weight = n_normal / n_attack

params = {
    "objective":          "binary:logistic",
    "learning_rate":      0.03,
    "max_depth":          7,
    "min_child_weight":   3,
    "gamma":              0.1,
    "scale_pos_weight":   scale_pos_weight,
    "subsample":          0.85,
    "colsample_bytree":   0.8,
    "colsample_bylevel":  0.8,
    "reg_alpha":          0.05,
    "reg_lambda":         1.5,
    "tree_method":        "hist",
    "eval_metric":        "aucpr",
    "seed":               42,
    "nthread":            -1,
}

dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
dval   = xgb.DMatrix(X_val,   label=y_val,   feature_names=feature_cols)
dtest  = xgb.DMatrix(X_test,  feature_names=feature_cols)

xgbmodel = xgb.train(
    params,
    dtrain,
    num_boost_round=700,
    evals=[(dtrain, "train"), (dval, "validation")],
    early_stopping_rounds=30,
    verbose_eval=50,
)

print("\n3. Test seti degerlendirmesi...")
y_pred = (xgbmodel.predict(dtest) >= 0.40).astype(int)

print("\n--- SINIFLANDIRMA RAPORU ---")
print(classification_report(y_test, y_pred, target_names=['Normal (0)', 'Saldiri (1)']))

print("\n--- KARISIKLIK MATRISI ---")
cm = confusion_matrix(y_test, y_pred)
print("                  Tahmin: Normal | Tahmin: Saldiri")
print(f"Gercek Normal :   {cm[0][0]:>8,}      |   {cm[0][1]:>8,}")
print(f"Gercek Saldiri:   {cm[1][0]:>8,}      |   {cm[1][1]:>8,}")

print("\n4. Feature Importance (En Etkili 15 Ozellik)...")
scores = xgbmodel.get_score(importance_type='gain')
fi = pd.Series(scores).sort_values(ascending=False)
for feat, val in fi.head(15).items():
    print(f"    {feat:<35} {val:.1f}")

joblib.dump(xgbmodel, "xgboost_binary_model.pkl")
joblib.dump(feature_cols, "feature_cols_binary.pkl")
print(f"\nModel kaydedildi: xgboost_binary_model.pkl")
