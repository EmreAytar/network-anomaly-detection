"""
CICIoT2023 - Test Veri Uretici
CSV modu: Gercek CICIoT2023 veri setinden rastgele ornekler cekilir.
PCAP modu: Test PCAP uretilir ve pcap_extractor ile CSV'ye donusturulur.

Kullanim:
    python test_pcap_generator.py                        # CSV: veri setinden ornekle
    python test_pcap_generator.py --pcap                 # PCAP: uret + extract
    python test_pcap_generator.py --count 500            # satir/paket sayisi
    python test_pcap_generator.py --output my_test.csv   # ozel dosya adi
"""

import argparse
import glob
import os
import random
import time
import logging

logging.getLogger("scapy.runtime").setLevel(logging.ERROR)

import numpy as np
import pandas as pd


DATASET_BASE = r"C:\Datasets\ciciot2023\archive\CIC_IOT_Dataset2023\CSV"

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

LABEL_ENCODE = {
    'Benign': 0, 'Flood': 1, 'Mirai': 2,
    'Recon': 3, 'MitM': 4, 'Exploit': 5,
}


def _get_category(folder_name):
    """Klasor adina gore kategori belirle."""
    lower = folder_name.lower()
    if 'benign' in lower:
        return 'Benign'
    for key, cat in CATEGORY_MAP.items():
        if key in lower:
            return cat
    return None


def generate_test_csv(output="test_generated.csv", total_rows=200, benign_ratio=0.6):
    """
    CICIoT2023 veri setindeki CSV dosyalarindan rastgele satirlar cekilir.
    Label sutunlari cikarilarak model test verisi olusturulur.
    """
    if not os.path.exists(DATASET_BASE):
        print("HATA: Veri seti bulunamadi: {}".format(DATASET_BASE))
        import sys; sys.exit(1)

    print("\n  CICIoT2023 Test CSV Uretici (veri setinden)")
    print("  Toplam satir: {}, Benign orani: %{:.0f}".format(total_rows, benign_ratio * 100))

    n_benign = int(total_rows * benign_ratio)
    n_attack = total_rows - n_benign

    category_files = {cat: [] for cat in LABEL_ENCODE}

    for folder in os.listdir(DATASET_BASE):
        folder_path = os.path.join(DATASET_BASE, folder)
        if not os.path.isdir(folder_path):
            continue
        cat = _get_category(folder)
        if cat is None:
            continue
        csvs = glob.glob(os.path.join(folder_path, "*.csv"))
        category_files[cat].extend(csvs)

    samples = []
    true_labels = []

    benign_csvs = category_files.get('Benign', [])
    if benign_csvs:
        bf = random.choice(benign_csvs)
        df_b = pd.read_csv(bf, encoding='latin1', nrows=max(n_benign * 3, 1000))
        if len(df_b) >= n_benign:
            df_b = df_b.sample(n=n_benign, random_state=random.randint(0, 9999))
        samples.append(df_b)
        true_labels.extend(['Benign'] * len(df_b))
        print("  Benign        {:>5} satir ({})".format(len(df_b), os.path.basename(bf)))

    attack_cats = [c for c in LABEL_ENCODE if c != 'Benign']
    per_cat = max(1, n_attack // len(attack_cats))

    for cat in attack_cats:
        cat_csvs = category_files.get(cat, [])
        if not cat_csvs:
            print("  {:<12} ATLANILDI (dosya bulunamadi)".format(cat))
            continue
        cf = random.choice(cat_csvs)
        df_c = pd.read_csv(cf, encoding='latin1', nrows=max(per_cat * 3, 500))
        n_sample = min(per_cat, len(df_c))
        if n_sample > 0:
            df_c = df_c.sample(n=n_sample, random_state=random.randint(0, 9999))
            samples.append(df_c)
            true_labels.extend([cat] * len(df_c))
            print("  {:<12} {:>5} satir ({})".format(cat, len(df_c), os.path.basename(cf)))

    if not samples:
        print("HATA: Hic veri cekilemedi!")
        import sys; sys.exit(1)

    df_test = pd.concat(samples, ignore_index=True)
    df_test.columns = df_test.columns.str.strip()

    idx = list(range(len(df_test)))
    random.shuffle(idx)
    df_test = df_test.iloc[idx].reset_index(drop=True)
    true_labels = [true_labels[i] for i in idx]

    true_df = pd.DataFrame({
        'true_category': true_labels,
        'class_label': [LABEL_ENCODE[c] for c in true_labels]
    })
    true_file = output.replace('.csv', '_labels.csv')
    true_df.to_csv(true_file, index=False)

    drop_cols = [c for c in ['Label', 'binary_label', 'label', 'Attack_Category', 'class_label'] if c in df_test.columns]
    if drop_cols:
        df_test.drop(columns=drop_cols, inplace=True)

    df_test.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_test.fillna(0, inplace=True)

    df_test.to_csv(output, index=False)

    print("\n  Test verisi: {} ({} satir, {} feature)".format(output, len(df_test), len(df_test.columns)))
    print("  Gercek etiketler: {}".format(true_file))

    return output


def generate_pcap(output="test.pcap", total_packets=1000):
    """CICIoT2023 uyumlu test PCAP dosyasi uretir."""
    from scapy.all import (
        Ether, IP, TCP, UDP, ICMP, ARP, DNS, DNSQR,
        Raw, wrpcap, RandMAC, conf
    )
    conf.verb = 0

    ETH = Ether(src="aa:bb:cc:00:00:01", dst="aa:bb:cc:00:00:02")
    PKTS = 10

    DEVICES = ["10.0.0." + str(i) for i in range(10, 18)]
    SERVERS = ["10.0.0.1", "10.0.0.2", "10.0.0.3"]
    DNS_SRV = "8.8.8.8"
    ATTACKERS = ["192.168.1.100", "192.168.1.101"]
    BOTS = [f"10.0.1.{i}" for i in range(1, 11)]

    # Benign: ~50% TCP + ~50% UDP, TTL~94, Rate~2849, IAT~0.004
    def _benign_tcp(ts):
        p = []; s, d = random.choice(DEVICES), random.choice(SERVERS)
        sp = random.randint(49152, 65535)
        dp = random.choice([443, 443, 80, 1883])
        ttl = random.choice([64, 128, 255])
        for f in ["S", "SA", "A", "PA", "PA", "PA", "PA", "PA", "FA", "A"]:
            ts += random.uniform(0.001, 0.008)
            if "P" in f:
                pkt = ETH/IP(src=s, dst=d, ttl=ttl)/TCP(sport=sp, dport=dp, flags=f)/Raw(load=bytes(random.randint(0, 255) for _ in range(random.randint(60, 1500))))
            else:
                pkt = ETH/IP(src=s, dst=d, ttl=ttl)/TCP(sport=sp, dport=dp, flags=f)
            pkt.time = ts; p.append(pkt)
        return p

    def _benign_udp(ts):
        p = []; s = random.choice(DEVICES)
        sp = random.randint(1024, 65535)
        dp = random.choice([53, 53, 123, 443])
        d = DNS_SRV if dp == 53 else random.choice(SERVERS)
        ttl = random.choice([64, 128, 255])
        for _ in range(PKTS):
            ts += random.uniform(0.001, 0.008)
            if dp == 53:
                pkt = ETH/IP(src=s, dst=d, ttl=ttl)/UDP(sport=sp, dport=dp)/DNS(rd=1, qd=DNSQR(qname=random.choice(["iot.local", "api.sensor.net"])))
            else:
                pkt = ETH/IP(src=s, dst=d, ttl=ttl)/UDP(sport=sp, dport=dp)/Raw(load=bytes(random.randint(0, 255) for _ in range(random.randint(40, 300))))
            pkt.time = ts; p.append(pkt)
        return p

    def _benign(ts):
        if random.random() < 0.5:
            return _benign_tcp(ts)
        return _benign_udp(ts)

    # Flood: Proto=6(TCP), syn_flag~0.98, Rate~18K
    def _flood(ts):
        p = []; a = f"{random.randint(1, 223)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"
        t = random.choice(SERVERS); sp = random.randint(1024, 65535); dp = random.choice([80, 443])
        for _ in range(PKTS):
            ts += random.uniform(0.00005, 0.0002)
            pkt = ETH/IP(src=a, dst=t, ttl=64)/TCP(sport=sp, dport=dp, flags="S")
            pkt.time = ts; p.append(pkt)
        return p

    # Mirai: Proto=47(GRE), TTL~64.5, buyuk paketler
    def _mirai(ts):
        p = []; b = random.choice(BOTS); t = f"10.0.0.{random.randint(1, 254)}"
        for _ in range(PKTS):
            ts += random.uniform(0.0001, 0.0005)
            payload = bytes(random.randint(0, 255) for _ in range(random.randint(500, 650)))
            pkt = ETH/IP(src=b, dst=t, ttl=64, proto=47)/Raw(load=payload)
            pkt.time = ts; p.append(pkt)
        return p

    # Recon: Proto~6, TTL~119, karisik flags
    def _recon(ts):
        p = []; s = random.choice(ATTACKERS); t = f"10.0.0.{random.randint(1, 254)}"
        sp = random.randint(1024, 65535); dp = random.randint(1, 1024)
        ttl = random.choice([128, 255, 128, 255])
        for _ in range(PKTS):
            ts += random.uniform(0.0001, 0.003)
            flags = random.choice(["S", "S", "SA", "A", "A", "A", "PA", "PA", "RA", "RA"])
            pkt = ETH/IP(src=s, dst=t, ttl=ttl)/TCP(sport=sp, dport=dp, flags=flags)
            pkt.time = ts; p.append(pkt)
        return p

    # MitM: Proto=17(UDP), DNS Spoofing
    def _mitm(ts):
        p = []; v = random.choice(DEVICES)
        sp = 53; dp = random.randint(1024, 65535)
        for _ in range(PKTS):
            ts += random.uniform(0.0001, 0.001)
            pkt = ETH/IP(src=DNS_SRV, dst=v, ttl=random.randint(40, 128))/UDP(sport=sp, dport=dp)/DNS(qr=1, aa=1, qd=DNSQR(qname="api.sensor.net"))
            pkt.time = ts; p.append(pkt)
        return p

    # Exploit: yavaş Rate~98, HTTP payloads
    def _exploit(ts):
        p = []; a = random.choice(ATTACKERS); t = random.choice(DEVICES)
        sp = random.randint(1024, 65535); dp = random.choice([80, 443, 22])
        ttl = random.choice([64, 128])
        payloads = [b"GET /login?user=admin'OR'1'='1 HTTP/1.1\r\n\r\n",
                    b"POST /api HTTP/1.1\r\n\r\nid=1;DROP TABLE users;--",
                    b"GET /search?q=<script>alert(1)</script>\r\n\r\n"]
        for i in range(PKTS):
            ts += random.uniform(0.005, 0.040)
            flags = random.choice(["PA", "PA", "PA", "PA", "PA", "S", "SA", "A", "FA", "A"])
            pkt = ETH/IP(src=a, dst=t, ttl=ttl)/TCP(sport=sp, dport=dp, flags=flags)/Raw(load=payloads[i % len(payloads)])
            pkt.time = ts; p.append(pkt)
        return p

    dist = {'Benign': (0.50, _benign), 'Flood': (0.10, _flood), 'Mirai': (0.10, _mirai),
            'Recon': (0.10, _recon), 'MitM': (0.10, _mitm), 'Exploit': (0.10, _exploit)}

    print("\n  CICIoT2023 Test PCAP Uretici")
    print("  Toplam paket (hedef): {}".format(total_packets))
    print("  Cikti: {}\n".format(output))

    all_pkts = []; base = time.time()
    for cat, (ratio, gen) in dist.items():
        n_flows = max(1, int(total_packets * ratio) // PKTS)
        pkts = []
        for _ in range(n_flows):
            pkts.extend(gen(base + random.uniform(0, 10)))
        all_pkts.extend(pkts)
        print("  {:<12} {:>3} flow x {} pkt = {:>5} paket".format(cat, n_flows, PKTS, len(pkts)))

    all_pkts.sort(key=lambda p: p.time)
    wrpcap(output, all_pkts)
    print("\n  Toplam {} paket -> {} kaydedildi.".format(len(all_pkts), output))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CICIoT2023 Test Veri Uretici")
    parser.add_argument("--pcap",   action="store_true", help="PCAP formatinda uret (varsayilan: CSV)")
    parser.add_argument("--output", type=str, default=None, help="Cikti dosyasi")
    parser.add_argument("--count",  type=int, default=200,  help="Satir/paket sayisi")
    args = parser.parse_args()

    if args.pcap:
        out = args.output or "test.pcap"
        generate_pcap(output=out, total_packets=args.count)
    else:
        out = args.output or "test_generated.csv"
        generate_test_csv(output=out, total_rows=args.count)
