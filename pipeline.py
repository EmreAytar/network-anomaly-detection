"""
CICIoT2023 - Saldiri Tespit Pipeline

Kullanim:
    python pipeline.py --csv   features.csv --binary
    python pipeline.py --csv   features.csv --multi
    python pipeline.py --pcap  trafik.pcap  --binary
    python pipeline.py --pcap  trafik.pcap  --multi
    python pipeline.py --csv   --multi   (otomatik test verisi)
    python pipeline.py --pcap  --multi   (otomatik test PCAP)
"""

import argparse
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb

from pcap_extractor import extract_features
from test_pcap_generator import generate_pcap, generate_test_csv


MULTICLASS_DECODE = {
    0: 'Benign',
    1: 'Flood',
    2: 'Mirai',
    3: 'Recon',
    4: 'MitM',
    5: 'Exploit'
}

BINARY_THRESHOLD = 0.40
DROP_COLS = {'Label', 'binary_label', 'Attack_Category', 'class_label', 'label', 'Number'}


def get_feature_names(model) -> list:
    """Model beklenen feature isimlerini dondurur."""
    if isinstance(model, xgb.Booster):
        return model.feature_names
    try:
        return list(model.feature_names_in_)
    except AttributeError:
        try:
            return model.get_booster().feature_names
        except Exception:
            return None


def prepare_X(df: pd.DataFrame, feature_names: list) -> xgb.DMatrix:
    """DataFrame'i model icin DMatrix'e donusturur."""
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    str_cols = df.select_dtypes(include=['object']).columns.tolist()
    if str_cols:
        df = df.drop(columns=str_cols)

    df = df.replace([np.inf, -np.inf], np.nan)

    for col in feature_names:
        if col not in df.columns:
            df[col] = 0.0

    nan_count = df[feature_names].isnull().sum().sum()
    if nan_count > 0:
        print("  Uyari: {} NaN deger 0 ile dolduruldu.".format(nan_count))

    X = df[feature_names].fillna(0)
    return xgb.DMatrix(X)


def predict_binary(model: xgb.Booster, dmatrix: xgb.DMatrix, threshold: float = BINARY_THRESHOLD):
    """Binary model: saldiri olasiliklarini dondurur."""
    proba = model.predict(dmatrix)
    labels = (proba >= threshold).astype(int)
    return labels, proba


def predict_multiclass(model: xgb.Booster, dmatrix: xgb.DMatrix):
    """Multiclass model: en yuksek olasilikli sinifi dondurur."""
    proba = model.predict(dmatrix)
    labels = np.argmax(proba, axis=1)
    confidence = np.max(proba, axis=1)
    return labels, confidence


DEFAULT_BINARY_MODEL = "xgboost_binary_model.pkl"
DEFAULT_MULTI_MODEL  = "xgboost_multiclass_model.pkl"


def run_pipeline(args):
    print("\n  CICIoT2023 Saldiri Tespit Pipeline")

    use_binary = args.binary
    use_multi  = args.multi

    if not use_binary and not use_multi:
        print("HATA: --binary veya --multi seceneklerinden birini belirtmelisiniz.")
        sys.exit(1)

    # 1. Model yukle
    print("\n[1/3] Model yukleniyor...")

    model = None
    model_type = None

    if use_binary:
        model_type = "binary"
        model_path = DEFAULT_BINARY_MODEL
        if not os.path.exists(model_path):
            print("HATA: Binary model bulunamadi: {}".format(model_path))
            sys.exit(1)
        model = joblib.load(model_path)
        if not isinstance(model, xgb.Booster):
            try:
                model = model.get_booster()
            except Exception:
                pass
        print("  Binary model yuklendi: {}".format(model_path))
        print("  Binary threshold: {}".format(BINARY_THRESHOLD))
    else:
        model_type = "multi"
        model_path = DEFAULT_MULTI_MODEL
        if not os.path.exists(model_path):
            print("HATA: Multiclass model bulunamadi: {}".format(model_path))
            sys.exit(1)
        model = joblib.load(model_path)
        if not isinstance(model, xgb.Booster):
            try:
                model = model.get_booster()
            except Exception:
                pass
        print("  Multiclass model yuklendi: {}".format(model_path))

    # 2. Veri yukle
    print("\n[2/3] Veri hazirlaniyor...")
    if args.pcap is not None:
        if args.pcap:
            if not os.path.exists(args.pcap):
                print("PCAP bulunamadi: {}".format(args.pcap))
                sys.exit(1)
            pcap_path = args.pcap
        else:
            pcap_path = generate_pcap()
        df = extract_features(pcap_path)
        if df is None or df.empty:
            print("PCAP'ten ozellik cikarilamadi.")
            sys.exit(1)
    elif args.csv is not None and args.csv:
        if not os.path.exists(args.csv):
            print("CSV bulunamadi: {}".format(args.csv))
            sys.exit(1)
        df = pd.read_csv(args.csv)
        print("  CSV yuklendi: {} satir, {} sutun".format(len(df), len(df.columns)))
    else:
        csv_path = generate_test_csv()
        df = pd.read_csv(csv_path)
        print("  Otomatik test verisi yuklendi: {} satir".format(len(df)))

    # 3. Tahmin
    print("\n[3/3] Tahmin yapiliyor...")
    feature_names = get_feature_names(model)
    dmatrix = prepare_X(df.copy(), feature_names)

    t0 = time.time()
    out = df.copy()

    if model_type == "binary":
        binary_labels, binary_proba = predict_binary(model, dmatrix)
        elapsed = time.time() - t0

        attacks = int(binary_labels.sum())
        normals = len(df) - attacks
        total = len(df)

        print("\n" + "=" * 55)
        print("           BINARY ANOMALI TESPIT SONUCLARI")
        print("=" * 55)
        print("  Analiz suresi        : {:.3f} saniye".format(elapsed))
        print("  Toplam incelenen     : {}".format(total))
        print("  Normal trafik   (0)  : {}  ({:.1f}%)".format(normals, 100 * normals / total))
        print("  Tespit edilen saldiri: {}  ({:.1f}%)".format(attacks, 100 * attacks / total))
        print("  Ortalama guven       : {:.1f}%".format(float(np.mean(binary_proba) * 100)))
        print("=" * 55)
        print("SALDIRI TESPIT EDILDI!" if attacks > 0 else "Ag trafigi temiz.")
        print("=" * 55 + "\n")

        out['Binary_Karar']    = binary_labels
        out['Binary_Olasilik'] = np.round(binary_proba, 4)
        out['Karar_Etiketi']   = ['SALDIRI' if p == 1 else 'NORMAL' for p in binary_labels]

    else:
        multi_labels, multi_conf = predict_multiclass(model, dmatrix)
        elapsed = time.time() - t0

        multi_names = np.array([MULTICLASS_DECODE.get(int(l), 'Bilinmeyen') for l in multi_labels])
        attack_count = int((multi_labels > 0).sum())
        total = len(df)

        print("\n" + "=" * 55)
        print("         MULTICLASS KATEGORI TESPIT SONUCLARI")
        print("=" * 55)
        print("  Analiz suresi        : {:.3f} saniye".format(elapsed))
        print("  Toplam incelenen     : {}".format(total))
        print("\n  Sinif Dagilimi:")
        for code, name in MULTICLASS_DECODE.items():
            mask  = multi_labels == code
            count = int(mask.sum())
            if count > 0:
                avg_c = float(multi_conf[mask].mean() * 100)
                print("    {:<12} {:>6} adet  ({:.1f}%)  (ort. guven: {:.1f}%)".format(
                    name, count, 100 * count / total, avg_c))
        print("\n" + "=" * 55)
        print("SALDIRI TESPIT EDILDI! ({} adet)".format(attack_count) if attack_count > 0 else "Ag trafigi temiz.")
        print("=" * 55 + "\n")

        out['Multi_Sinif']    = multi_labels
        out['Multi_Kategori'] = multi_names
        out['Multi_Guven']    = np.round(multi_conf, 4)

    out.to_csv(args.output, index=False)
    print("  Sonuclar kaydedildi: {}".format(args.output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CICIoT2023 Saldiri Tespit Pipeline")

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pcap", type=str, nargs='?', const='', default=None,
                        help="Giris PCAP dosyasi (dosya verilmezse otomatik test PCAP uretilir)")
    source.add_argument("--csv",  type=str, nargs='?', const='', default=None,
                        help="Giris CSV dosyasi (dosya verilmezse otomatik test verisi uretilir)")

    model_type = parser.add_mutually_exclusive_group(required=True)
    model_type.add_argument("--binary", action="store_true", help="Binary model kullan (Normal/Saldiri)")
    model_type.add_argument("--multi",  action="store_true", help="Multiclass model kullan (6 sinif)")

    parser.add_argument("--output", type=str, default="sonuc.csv", help="Cikti CSV (varsayilan: sonuc.csv)")

    run_pipeline(parser.parse_args())
