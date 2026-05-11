"""
CICIoT2023 - PCAP Feature Extraction
PCAP dosyalarindan CICIoT2023 formatinda ozellik cikarimi.
"""

import socket
from collections import defaultdict

import dpkt
import numpy as np
import pandas as pd


COLUMNS = [
    'Header_Length', 'Protocol Type', 'Time_To_Live', 'Rate',
    'fin_flag_number', 'syn_flag_number', 'rst_flag_number', 'psh_flag_number',
    'ack_flag_number', 'ece_flag_number', 'cwr_flag_number',
    'ack_count', 'syn_count', 'fin_count', 'rst_count',
    'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC',
    'TCP', 'UDP', 'DHCP', 'ARP', 'ICMP', 'IGMP', 'IPv', 'LLC',
    'Tot sum', 'Min', 'Max', 'AVG', 'Std', 'Tot size', 'IAT', 'Number', 'Variance'
]

_HTTP   = {80, 8080}
_HTTPS  = {443, 8443}
_DNS    = {53}
_TELNET = {23}
_SMTP   = {25, 587, 465}
_SSH    = {22}
_IRC    = {194, 6667, 6668, 6669}
_DHCP   = {67, 68}

WINDOW_SIZE = 10


def extract_features(pcap_path: str) -> pd.DataFrame | None:
    """PCAP dosyasindan CICIoT2023 formatinda ozellik cikarimi yapar."""
    flows: dict[tuple, list] = defaultdict(list)
    pkt_num = 0

    print("PCAP okunuyor: {}".format(pcap_path))

    try:
        with open(pcap_path, 'rb') as fh:
            try:
                reader = dpkt.pcap.Reader(fh)
            except ValueError:
                fh.seek(0)
                reader = dpkt.pcapng.Reader(fh)

            for ts, raw in reader:
                pkt_num += 1
                if pkt_num % 50_000 == 0:
                    print("  {} paket okundu...".format(pkt_num))

                try:
                    eth = dpkt.ethernet.Ethernet(raw)
                except Exception:
                    continue

                if eth.type < 0x0600:
                    flows[('LLC', 'LLC', 0, 0, 0)].append({
                        'ts': ts, 'size': len(raw), 'is_arp': False, 'is_llc': True,
                        'ip_hl': 0, 'ip_ver': 0, 'ttl': 0, 'proto': 0,
                        'transport': 'LLC', 'src_port': 0, 'dst_port': 0, 'flags': {}
                    })
                    continue

                if isinstance(eth.data, dpkt.arp.ARP):
                    flows[('ARP', 'ARP', 0, 0, 0)].append({
                        'ts': ts, 'size': len(raw), 'is_arp': True, 'is_llc': False,
                        'ip_hl': 0, 'ip_ver': 0, 'ttl': 0, 'proto': 0,
                        'transport': 'ARP', 'src_port': 0, 'dst_port': 0, 'flags': {}
                    })
                    continue

                pkt = {
                    'ts': ts, 'size': len(raw), 'is_arp': False, 'is_llc': False,
                    'ip_hl': 0, 'ip_ver': 0, 'ttl': 0, 'proto': 0,
                    'transport': '', 'src_port': 0, 'dst_port': 0, 'flags': {}
                }

                if isinstance(eth.data, dpkt.ip.IP):
                    ip = eth.data
                    pkt.update(ip_ver=4, ip_hl=ip.hl * 4, ttl=ip.ttl, proto=ip.p)
                    src_ip = socket.inet_ntoa(ip.src)
                    dst_ip = socket.inet_ntoa(ip.dst)
                    _fill_transport(ip.data, pkt)
                elif isinstance(eth.data, dpkt.ip6.IP6):
                    ip6 = eth.data
                    pkt.update(ip_ver=6, ip_hl=40, ttl=ip6.hlim, proto=ip6.nxt)
                    src_ip = socket.inet_ntop(socket.AF_INET6, ip6.src)
                    dst_ip = socket.inet_ntop(socket.AF_INET6, ip6.dst)
                    _fill_transport(ip6.data, pkt)
                else:
                    continue

                key = (src_ip, dst_ip, pkt['src_port'], pkt['dst_port'], pkt['proto'])
                flows[key].append(pkt)

    except Exception as e:
        print("PCAP okuma hatasi: {}".format(e))
        return None

    print("  {} paket okundu, {} akis bulundu.".format(pkt_num, len(flows)))

    records = []
    for pkts in flows.values():
        pkts.sort(key=lambda p: p['ts'])
        for i in range(0, len(pkts), WINDOW_SIZE):
            window = pkts[i: i + WINDOW_SIZE]
            if len(window) < 2:
                continue
            records.append(_window_features(window))

    if not records:
        return None

    df = pd.DataFrame(records, columns=COLUMNS)
    print("  {} satir uretildi.".format(len(df)))
    return df


def _fill_transport(layer4, pkt: dict) -> None:
    if isinstance(layer4, dpkt.tcp.TCP):
        f = layer4.flags
        pkt['transport'] = 'TCP'
        pkt['src_port']  = layer4.sport
        pkt['dst_port']  = layer4.dport
        pkt['flags'] = {
            'FIN': bool(f & dpkt.tcp.TH_FIN),
            'SYN': bool(f & dpkt.tcp.TH_SYN),
            'RST': bool(f & dpkt.tcp.TH_RST),
            'PSH': bool(f & dpkt.tcp.TH_PUSH),
            'ACK': bool(f & dpkt.tcp.TH_ACK),
            'ECE': bool(f & dpkt.tcp.TH_ECE),
            'CWR': bool(f & dpkt.tcp.TH_CWR),
        }
    elif isinstance(layer4, dpkt.udp.UDP):
        pkt['transport'] = 'UDP'
        pkt['src_port']  = layer4.sport
        pkt['dst_port']  = layer4.dport
    elif isinstance(layer4, dpkt.icmp.ICMP):
        pkt['transport'] = 'ICMP'
    elif isinstance(layer4, dpkt.igmp.IGMP):
        pkt['transport'] = 'IGMP'


def _window_features(window: list) -> list:
    n   = len(window)
    eps = 1e-9

    sizes   = np.array([p['size'] for p in window], dtype=float)
    tot_sum = float(sizes.sum())
    pkt_min = float(sizes.min())
    pkt_max = float(sizes.max())
    pkt_avg = float(sizes.mean())
    pkt_std = float(sizes.std())
    pkt_var = float(sizes.var())

    ts      = np.array([p['ts'] for p in window])
    dur     = float(ts[-1] - ts[0])
    rate    = n / (dur + eps)
    avg_iat = float(np.diff(ts).mean()) if n > 1 else 0.0

    ip_hls = [p['ip_hl'] for p in window if p['ip_hl'] > 0]
    ttls   = [p['ttl']   for p in window if p['ttl']   > 0]
    avg_hl  = float(np.mean(ip_hls)) if ip_hls else 0.0
    avg_ttl = float(np.mean(ttls))   if ttls   else 0.0

    protos     = [p['proto'] for p in window]
    proto_type = int(max(set(protos), key=protos.count)) if protos else 0

    def fcnt(name): return sum(1 for p in window if p['flags'].get(name, False))
    fin = fcnt('FIN'); syn = fcnt('SYN'); rst = fcnt('RST')
    psh = fcnt('PSH'); ack = fcnt('ACK'); ece = fcnt('ECE'); cwr = fcnt('CWR')

    tcp  = sum(1 for p in window if p['transport'] == 'TCP')
    udp  = sum(1 for p in window if p['transport'] == 'UDP')
    icmp = sum(1 for p in window if p['transport'] == 'ICMP')
    igmp = sum(1 for p in window if p['transport'] == 'IGMP')
    arp  = sum(1 for p in window if p['is_arp'])
    llc_802 = sum(1 for p in window if p['is_llc'])
    ipv  = sum(1 for p in window if p['ip_ver'] in (4, 6))
    llc  = ipv + arp + llc_802 

    http = https = dns = telnet = smtp = ssh = irc = dhcp = 0
    for p in window:
        ports = {p['src_port'], p['dst_port']}
        if ports & _DHCP:   dhcp   += 1; continue
        if ports & _HTTP:   http   += 1
        if ports & _HTTPS:  https  += 1
        if ports & _DNS:    dns    += 1
        if ports & _TELNET: telnet += 1
        if ports & _SMTP:   smtp   += 1
        if ports & _SSH:    ssh    += 1
        if ports & _IRC:    irc    += 1

    r = lambda c: round(c / n, 10)

    return [
        avg_hl,    proto_type, avg_ttl,   rate,
        r(fin),    r(syn),     r(rst),    r(psh),
        r(ack),    r(ece),     r(cwr),
        ack,       syn,        fin,        rst,
        r(http),   r(https),   r(dns),    r(telnet),
        r(smtp),   r(ssh),     r(irc),
        r(tcp),    r(udp),     r(dhcp),   r(arp),
        r(icmp),   r(igmp),    r(ipv),    r(llc),
        tot_sum,   pkt_min,    pkt_max,   pkt_avg,
        pkt_std,   pkt_avg,    avg_iat,   n,
        pkt_var,
    ]
