# Yapay Zeka Tabanlı Ağ Anomali Tespiti 🛡️

Bu proje, IoT (Nesnelerin İnterneti) ağlarında meydana gelen siber saldırıları makine öğrenmesi ve derin öğrenme teknikleri kullanarak tespit etmeyi amaçlayan bir bitirme projesidir.

## 📌 Proje Özeti
Geleneksel imza tabanlı güvenlik sistemlerinin (IDS/IPS) yetersiz kaldığı sıfır gün (zero-day) saldırılarını ve karmaşık anomali durumlarını tespit etmek için hiyerarşik bir sınıflandırma mimarisi geliştirilmiştir. Proje kapsamında ağ trafiği analiz edilerek zararlı aktiviteler yüksek doğruluk oranıyla sınıflandırılmaktadır.

## 🚀 Kullanılan Teknolojiler ve Araçlar
* **Diller:** Python
* **Veri İşleme ve Makine Öğrenmesi:** Pandas, Scikit-learn, SMOTE (Sınıf Dengesizliği için)
* **Ağ Analizi Araçları:** Wireshark, Suricata (Trafik incelemesi ve kural tabanlı analiz)
* **Veri Setleri:** CIC-IIoT-2023, UNSW-NB15

## ⚙️ Metodoloji
1. **Veri Ön İşleme:** Ağ paketlerinden elde edilen veriler temizlendi, ölçeklendirildi ve anlamsız özellikler çıkarıldı.
2. **SMOTE ile Veri Dengeleme:** Ağ trafiğindeki "Normal" veri yığınının, "Saldırı" verilerini baskılamasını önlemek için azınlık sınıfları (MitM, Flood vb.) sentetik olarak artırıldı.
3. **Model Mimarisi:** - **Aşama 1:** Gelen trafiğin "Normal" mi yoksa "Anomali" mi olduğunun tespiti.
   - **Aşama 2:** Tespit edilen anomalinin hangi saldırı ailesine (DDoS, Mirai, MitM vs.) ait olduğunun sınıflandırılması.

## 📊 Örnek Bulgular ve Başarı Metrikleri
* **Genel Doğruluk (Accuracy):** %94+
* **Mirai / Flood F1-Skoru:** 1.00
* **MitM F1-Skoru:** 0.93

## 💻 Kurulum ve Kullanım
Veri setleri boyutlarından dolayı bu depoya dahil edilmemiştir. Kodu kendi ortamınızda çalıştırmak için:
1. Repoyu klonlayın: `git clone https://github.com/kullaniciadiniz/network-anomaly-detection.git`
2. İlgili Kaggle veri setlerini indirip `/data` klasörü içine yerleştirin.
3. Gerekli kütüphaneleri yükleyin: `pip install -r requirements.txt`
4. Modeli eğitin: `python train.py` (veya jupyter notebook üzerinden hücreleri çalıştırın).
