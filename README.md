# CICIoT2023 Anomali Tespiti

CICIoT2023 veri seti ile XGBoost kullanarak ağ saldırısı tespiti yapan pipeline.

## Nasıl Çalıştırılır

Önce veriyi hazırla, sonra modeli eğit:
```bash
python prepare_data_multiclass.py
python xgboost_multiclass_model.py

python prepare_data_binary.py
python xgboost_binary_model.py
```

Tahmin için:
```bash
python pipeline.py --csv dosya.csv --multi
python pipeline.py --csv --multi          # otomatik test verisi üretir
python pipeline.py --pcap dosya.pcap --binary
```

## Dosyalar

- `pipeline.py` — ana pipeline, csv veya pcap alıp tahmin yapar
- `pcap_extractor.py` — pcap dosyasından feature çıkarır
- `test_pcap_generator.py` — test için csv/pcap üretir
- `prepare_data_multiclass.py` — ham veriyi parquet'e çevirir
- `xgboost_multiclass_model.py` — modeli eğitir

## Bilinen Sorunlar

**pcap_extractor ve test_pcap_generator tam düzgün çalışmıyor.** Veri setindeki CSV'ler CICFlowMeter çıktısı değil, kendilerinin yazdığı DPKT tabanlı scriptlerle üretilmiş ve bu scriptler açık kaynak olarak paylaşılmamış. Bu extractor bu orijinal feature değerlerini tam olarak üretemiyor. PCAP modunda bazı benign trafik yanlış sınıflandırılabiliyor. CSV moduyla veri setinden örnek çekerek test etmek daha güvenilir sonuç veriyor.
