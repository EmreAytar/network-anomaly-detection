```markdown
# 🛡️ CICIoT2023 Ağ Anomali ve Saldırı Tespiti

Bu proje, **CICIoT2023** veri setini kullanarak ağ trafiğindeki siber saldırıları ve anomalileri tespit etmek için **XGBoost** tabanlı bir makine öğrenmesi boru hattı (pipeline) sunmaktadır. Proje hem çok sınıflı (multiclass - saldırı türünü belirleme) hem de ikili (binary - normal/saldırı) sınıflandırma yeteneklerine sahiptir.

## 🚀 Kurulum ve Model Eğitimi

Modelleri sıfırdan eğitmek için öncelikle ham veriyi işleyip (`parquet` formatına çevirerek), ardından eğitim betiklerini çalıştırmalısınız.

### Çok Sınıflı (Multiclass) Model İçin:
```bash
python prepare_data_multiclass.py
python xgboost_multiclass_model.py

```

### İkili (Binary) Model İçin:

```bash
python prepare_data_binary.py
python xgboost_binary_model.py

```

## 🎯 Tahmin ve Test İşlemleri (Inference)

Ana `pipeline.py` dosyası üzerinden CSV veya PCAP formatındaki ağ trafiği verilerini test edebilirsiniz.

```bash
# Belirli bir CSV dosyası üzerinden çok sınıflı tahmin
python pipeline.py --csv dosya.csv --multi

# Otomatik test verisi üreterek çok sınıflı tahmin
python pipeline.py --csv --multi          

# Doğrudan PCAP dosyası üzerinden ikili (binary) tahmin
python pipeline.py --pcap dosya.pcap --binary

```

## 📂 Dosya Yapısı ve Modüller

* `pipeline.py`: Canlı tahminleme işlemlerini yürüten, CSV veya PCAP formatlarını kabul eden ana boru hattı.
* `pcap_extractor.py`: Ham PCAP (paket yakalama) dosyalarından modelin anlayabileceği özellikleri (feature) çıkaran modül.
* `test_pcap_generator.py`: Model testleri için sentetik veya örneklenmiş CSV/PCAP verileri üreten yardımcı betik.
* `prepare_data_multiclass.py` / `prepare_data_binary.py`: Ham veri setini modele uygun hale getirip optimize edilmiş parquet formatına dönüştüren veri ön işleme modülleri.
* `xgboost_multiclass_model.py` / `xgboost_binary_model.py`: XGBoost algoritmalarının hiperparametre ayarları ve eğitim süreçlerini içeren modüller.

## ⚠️ Bilinen Sorunlar ve Teknik Notlar

* **Özellik Çıkarımı (Feature Extraction):** CICIoT2023 veri setindeki orijinal CSV'ler, genel kabul gören *CICFlowMeter* aracı ile değil, araştırmacıların kendi yazdığı *DPKT tabanlı özel scriptler* ile üretilmiştir. Bu scriptler açık kaynaklı olarak paylaşılmadığı için, projedeki `pcap_extractor` modülü orijinal özellikleri tam olarak üretemeyebilir.
* **PCAP Modu Performansı:** `pcap_extractor` ve `test_pcap_generator` modülleri henüz tam stabil çalışmamaktadır. Özellik çıkarımındaki farklılıklardan dolayı, PCAP modunda çalışırken bazı *benign* (zararsız) trafik paketleri yanlış sınıflandırılabilmektedir.
* **Öneri:** Modelin gerçek performansını test etmek için PCAP modu yerine, **CSV moduyla** (`--csv`) doğrudan veri setinden örnekler çekerek test edilmesi çok daha güvenilir sonuçlar vermektedir.

```

```
