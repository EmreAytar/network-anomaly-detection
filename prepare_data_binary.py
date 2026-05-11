"""
CICIoT2023 - Veri Hazirlama (Binary)
Dagitim: %70 Benign / %30 Saldiri
"""

import os
import glob
import numpy as np
import pandas as pd

SKIP_FOLDERS = {'Benign_Final', 'MERGED_CSV'}

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

ATTACK_CATEGORIES = ['Flood', 'Mirai', 'Recon', 'MitM', 'Exploit']

CATEGORY_WEIGHTS = {
    'Flood':   1,
    'Mirai':   1,
    'Recon':   1,
    'MitM':    1,
    'Exploit': 1,
}


def get_category(folder_name: str) -> str:
    name = folder_name.lower()
    for key, cat in CATEGORY_MAP.items():
        if key in name:
            return cat
    return 'DROP'


def load_folder(base_dir: str, folder: str, category: str) -> pd.DataFrame:
    files = glob.glob(os.path.join(base_dir, folder, "*.csv"))
    if not files:
        return pd.DataFrame()
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f, low_memory=False, encoding='latin1'))
        except Exception as e:
            print("  Atlandi ({}): {}".format(f, e))
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    df['Label'] = folder
    df['Attack_Category'] = category
    return df


def prepare_dataset(base_dir: str, output_file: str = "preprocessed_binary.parquet"):

    print("1. Benign veriler yukleniyor...")
    benign_files = glob.glob(os.path.join(base_dir, "Benign_Final", "*.csv"))
    benign_dfs = []
    for f in benign_files:
        try:
            df = pd.read_csv(f, low_memory=False, encoding='latin1')
            df['Label'] = 'BENIGN'
            df['Attack_Category'] = 'Benign'
            benign_dfs.append(df)
        except Exception as e:
            print("  Atlandi ({}): {}".format(f, e))

    df_benign = pd.concat(benign_dfs, ignore_index=True)
    total_benign = len(df_benign)
    print("  Toplam benign satir: {:,}".format(total_benign))

    total_attack_target = int(total_benign * (30 / 70))
    total_weight = sum(CATEGORY_WEIGHTS.values())
    per_unit = total_attack_target // total_weight
    category_targets = {cat: per_unit * w for cat, w in CATEGORY_WEIGHTS.items()}

    print("\n2. Saldiri verileri yukleniyor...")
    print("  Toplam saldiri hedefi: {:,}".format(total_attack_target))
    print("  Kategori hedefleri:")
    for cat, tgt in category_targets.items():
        print("    {:<12} {:,}".format(cat, tgt))

    all_folders = [
        f for f in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, f)) and f not in SKIP_FOLDERS
    ]

    category_folders: dict[str, list] = {cat: [] for cat in ATTACK_CATEGORIES}
    for folder in all_folders:
        cat = get_category(folder)
        if cat != 'DROP':
            category_folders[cat].append(folder)

    attack_dfs = []
    for category, folders in category_folders.items():
        if not folders:
            print("  [{}] klasor bulunamadi, atlandi.".format(category))
            continue

        cat_dfs = []
        for folder in folders:
            df_f = load_folder(base_dir, folder, category)
            if not df_f.empty:
                cat_dfs.append(df_f)

        if not cat_dfs:
            continue

        df_cat = pd.concat(cat_dfs, ignore_index=True)
        target = category_targets[category]
        if len(df_cat) > target:
            df_cat = df_cat.sample(n=target, random_state=42)

        attack_dfs.append(df_cat)
        print("  [{}] {:,} satir alindi ({} klasor)".format(category, len(df_cat), len(folders)))

    df_attack = pd.concat(attack_dfs, ignore_index=True)
    print("\n  Toplam saldiri satiri: {:,}".format(len(df_attack)))

    print("\n3. Birlestiriliyor ve temizleniyor...")
    df_final = pd.concat([df_benign, df_attack], ignore_index=True)
    df_final.columns = df_final.columns.str.strip()

    df_final['binary_label'] = (df_final['Label'] != 'BENIGN').astype(int)

    df_final.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_final.fillna(0, inplace=True)

    drop_cols = [c for c in ['Number'] if c in df_final.columns]
    if drop_cols:
        df_final.drop(columns=drop_cols, inplace=True)

    non_numeric = df_final.select_dtypes(include=['object']).columns.tolist()
    non_numeric = [c for c in non_numeric if c not in ('Attack_Category', 'Label')]
    if non_numeric:
        print(f"  String sutunlar atiliyor: {non_numeric}")
        df_final.drop(columns=non_numeric, inplace=True)

    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

    total = len(df_final)
    n_benign = (df_final['binary_label'] == 0).sum()
    n_attack = (df_final['binary_label'] == 1).sum()

    print("\n  Toplam satir  : {:,}".format(total))
    print("  Benign  (0)   : {:,}  ({:.1f}%)".format(n_benign, 100 * n_benign / total))
    print("  Saldiri (1)   : {:,}  ({:.1f}%)".format(n_attack, 100 * n_attack / total))
    print("\n  Kategori dagilimi:")
    print(df_final['Attack_Category'].value_counts().to_string())

    df_final.to_parquet(output_file, index=False)
    print("\n  Kaydedildi: {}".format(output_file))


if __name__ == "__main__":
    RAW_DATA_DIR = r"C:\Datasets\ciciot2023\archive\CIC_IOT_Dataset2023\CSV"
    prepare_dataset(RAW_DATA_DIR)
