# BelediyeBot

# 🏛️ BelediyeBot – Yapay Zeka Tabanlı Belediye Karar Destek Sistemi

BelediyeBot, belediye hizmetlerinde **su tüketimi tahmini, otopark doluluk analizi, çalışan güvenliği kontrolü, sağlık risk tahmini, kentsel dönüşüm öncelik analizi** ve **sohbet asistanı** sunan çok modüllü yapay zekâ destekli bir platformdur.  
Hem vatandaş hem de belediye çalışanları için kullanıcı dostu arayüzler sağlar.

---

## 🔍 Neler Var?

### Modüller
- **1_Su_Tahmini.py** → ANN tabanlı su tüketimi tahmini  
- **2_Otopark_Analizi.py** → OpenCV ile boş/dolu park yeri tespiti  
- **Calisan_ekipman_kontrol.py** → Kask & yelek güvenlik analizi  
- **Saglik_Risk_Tahmini.py** → Sağlık risk skorlaması  
- **Kentsel_Donusum_Oncelik.py** → Kentsel dönüşüm öncelik analizi  
- **4_Sohbet_Asistani.py** → Gemini API tabanlı sohbet yönlendirme  
- **Home.py** → Tüm modülleri bir araya getiren ana sayfa  

### Eğitim Scriptleri
- **ilkproje.py** → Su tahmini model eğitimi  
- **sprojesi.py** → Sağlık risk modeli eğitimi  
- **prom.py** → Kentsel dönüşüm modeli eğitimi  

### Modeller & Kaynaklar
- **modelim.keras, risk_model.keras, urban_model.keras** → Eğitilmiş yapay sinir ağı modelleri  
- **ParkAlanlari.pkl, rois.json** → Görüntü işleme için ROI verileri  
- **CSV/XLSX dosyaları** → Su tüketimi, sağlık riski, dönüşüm verileri  

---

## ⚙️ Kurulum

1️⃣ Ortam oluşturma (Python 3.10+)  
```bash
git clone https://github.com/kendi-repo/BelediyeBot.git
cd BelediyeBot
python -m venv .venv
.venv\Scripts\activate   # Windows
source .venv/bin/activate # Linux/Mac
2️⃣ Gerekli paketleri yükleme
pip install -r requirements.txt
________________________________________
🚀 Çalıştırma
•	Ana arayüz (tüm modüller)
streamlit run Home.py
•	Tek modül çalıştırma (örnek)
streamlit run 1_Su_Tahmini.py
•	OpenCV tabanlı analizler
python 2_Otopark_Analizi.py
python Calisan_ekipman_kontrol.py
•	Eğitim scriptleri (isteğe bağlı yeniden eğitim)
python ilkproje.py         # su modeli
python sprojesi.py         # sağlık modeli
python prom.py             # dönüşüm modeli
________________________________________
🗂️ Klasör Yapısı (Özet)
BelediyeBot/
│
├── Home.py
├── 1_Su_Tahmini.py
├── 2_Otopark_Analizi.py
├── 4_Sohbet_Asistani.py
├── Saglik_Risk_Tahmini.py
├── Kentsel_Donusum_Oncelik.py
├── Calisan_ekipman_kontrol.py
│
├── modeller/
│   ├── modelim.keras
│   ├── risk_model.keras
│   ├── urban_model.keras
│   ├── scaler.save, risk_scaler.save, urban_scaler.save
│   ├── columns.save, risk_columns.save, urban_columns.save
│
├── veriler/
│   ├── birlesik_su_tuketimi_3.xlsx
│   ├── sentetik_saglik_risk_verisi_ilce_enlem_boylam_kategorili.xlsx
│   ├── kentsel_donusum_oncelik_verisi_v2.xlsx
│
├── media/
│   ├── Kanalizasyon altyapı işleri.mp4
│   ├── ilk_kare.jpg
│   └── video.mp4
│
└── utils/
    ├── algilama.py
    ├── ou.py
    ├── otrt.py
    ├── ParkAlanlari.pkl
    └── rois.json
________________________________________
📊 Değerlendirme
•	Su Tüketimi → RMSE, MAE, R² metrikleriyle ölçüm
•	Sağlık Riski & Kentsel Dönüşüm → Risk kategorileri üzerinde doğruluk, F1-Score değerlendirmesi
•	Otopark & Güvenlik → ROI tabanlı manuel test sonuçları (OpenCV)

