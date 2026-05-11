"""
CICIoT2023 - Veri Hazirlama (Multiclass)
Ham CSV dosyalarindan preprocessed parquet dosyasi olusturur.

Kullanim:
    python prepare_data_multiclass.py
"""

import os
import glob
import numpy as np
import pandas as pd

SKIP_FOLDERS = {'MERGED_CSV'}

CATEGORY_MAP = {
    'ddos':          'Flood',
    'dos':           'Flood',
    'mirai':         'Mirai',
    'recon':         'Recon',
    'vulnerability': 'Recon',
    'mitm':          'MitM',
    'spoofing':      'MitM',
    'bruteforce':    'Exploit',
    'dictionary':    'Exploit',
    'backdoor':      'Exploit',
    'malware':       'Exploit',
    'xss':           'Exploit',
    'sql':           'Exploit',
    'command':       'Exploit',
    'upload':        'Exploit',
    'browser':       'Exploit'
}

ALL_CATEGORIES = ['Benign', 'Flood', 'Mirai', 'Recon', 'MitM', 'Exploit']

LABEL_ENCODE = {
    'Benign':  0,
    'Flood':   1,
    'Mirai':   2,
    'Recon':   3,
    'MitM':    4,
    'Exploit': 5,
}

TARGET_PER_ATTACK_CATEGORY = 300_000

def get_category(folder_name):
    name = folder_name.lower()
    for key, cat in CATEGORY_MAP.items():
        if key in name:
            return cat
    return 'DROP'

def load_category_random(base_dir, folders, category, target):
    all_files = []
    file_lengths = []
    for folder in folders:
        files = glob.glob(os.path.join(base_dir, folder, "*.csv"))
        for f in files:
            try:
                with open(f, encoding='latin1') as fh:
                    n = sum(1 for _ in fh) - 1
                if n > 0:
                    all_files.append(f)
                    file_lengths.append(n)
            except Exception:
                pass

    if not all_files:
        return pd.DataFrame()

    total_available = sum(file_lengths)
    actual_target = min(target, total_available)

    print(f"    [{category}] Toplam {total_available:,} satir bulundu, {actual_target:,} alinacak.")

    dfs = []
    for f, n in zip(all_files, file_lengths):
        pay = max(1, min(round(actual_target * (n / total_available)), n))
        try:
            df = pd.read_csv(f, low_memory=False, encoding='latin1')
            if len(df) > pay:
                df = df.sample(n=pay, random_state=42)
            df['Attack_Category'] = category
            dfs.append(df)
        except Exception:
            pass

    if not dfs:
        return pd.DataFrame()

    df_cat = pd.concat(dfs, ignore_index=True)
    if len(df_cat) > actual_target:
        df_cat = df_cat.sample(n=actual_target, random_state=42)
    return df_cat

def load_benign(base_dir, benign_folder="Benign_Final"):
    """Benign klasorundan tum CSV dosyalarini yukler."""
    benign_path = os.path.join(base_dir, benign_folder)
    if not os.path.isdir(benign_path):
        print(f"  UYARI: Benign klasoru bulunamadi: {benign_path}")
        return pd.DataFrame()

    files = glob.glob(os.path.join(benign_path, "*.csv"))
    if not files:
        print(f"  UYARI: Benign klasorunde CSV bulunamadi.")
        return pd.DataFrame()

    dfs = []
    total_rows = 0
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False, encoding='latin1')
            df['Attack_Category'] = 'Benign'
            dfs.append(df)
            total_rows += len(df)
        except Exception as e:
            print(f"  Atlandi ({f}): {e}")

    if not dfs:
        return pd.DataFrame()

    df_benign = pd.concat(dfs, ignore_index=True)
    print(f"    [Benign] Toplam {total_rows:,} satir yuklendi.")
    return df_benign


def prepare_multiclass(base_dir, output_file="preprocessed_multiclass.parquet"):
    print("\n1. Benign veriler yukleniyor...")
    df_benign = load_benign(base_dir)

    all_folders = [
        f for f in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, f)) and f not in SKIP_FOLDERS
              and f != 'Benign_Final'
    ]

    attack_categories = [c for c in ALL_CATEGORIES if c != 'Benign']
    category_folders = {cat: [] for cat in attack_categories}
    for folder in all_folders:
        cat = get_category(folder)
        if cat != 'DROP':
            category_folders[cat].append(folder)

    print("\n2. Saldiri verileri yukleniyor ve orantili ornekleniyor...")
    category_dfs = {}
    for category, folders in category_folders.items():
        if not folders:
            continue
        df_cat = load_category_random(base_dir, folders, category, TARGET_PER_ATTACK_CATEGORY)
        if not df_cat.empty:
            category_dfs[category] = df_cat

    print("\n3. Birlestiriliyor ve temizleniyor...")
    all_dfs = []
    if not df_benign.empty:
        all_dfs.append(df_benign)
    all_dfs.extend(category_dfs.values())

    df_final = pd.concat(all_dfs, ignore_index=True)
    df_final.columns = df_final.columns.str.strip()
    df_final['class_label'] = df_final['Attack_Category'].map(LABEL_ENCODE)

    df_final.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_final.fillna(0, inplace=True)

    drop_cols = [c for c in ['Number', 'Label', 'binary_label'] if c in df_final.columns]
    if drop_cols:
        df_final.drop(columns=drop_cols, inplace=True)

    non_numeric = df_final.select_dtypes(include=['object']).columns.tolist()
    non_numeric = [c for c in non_numeric if c != 'Attack_Category']
    if non_numeric:
        print(f"  String sutunlar atiliyor: {non_numeric}")
        df_final.drop(columns=non_numeric, inplace=True)

    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

    total = len(df_final)
    print("\n  Sinif Dagilimi:")
    for cat, code in LABEL_ENCODE.items():
        count = (df_final['class_label'] == code).sum()
        print(f"    {code} - {cat:<12} {count:>8,}  ({100 * count / total:.1f}%)")

    df_final.to_parquet(output_file, index=False)
    print(f"\n  Kaydedildi: {output_file}")

if __name__ == "__main__":
    RAW_DATA_DIR = r"C:\Datasets\ciciot2023\archive\CIC_IOT_Dataset2023\CSV"
    prepare_multiclass(RAW_DATA_DIR)
