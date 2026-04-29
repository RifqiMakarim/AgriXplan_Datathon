# 🌾 AgriXplan - Sistem Pemetaan Produktivitas Padi Indonesia

AgriXplan adalah aplikasi *dashboard* interaktif yang dikembangkan untuk **Microsoft Datathon**. Aplikasi ini menggunakan pemodelan **Geographically Weighted Regression (GWR)** untuk menganalisis dan memetakan faktor-faktor spasial yang memengaruhi produktivitas padi di 38 provinsi di Indonesia. 

Dengan AgriXplan, Anda dapat mengetahui secara spesifik faktor apa (seperti Penggunaan Pupuk NPK, Lahan OPT, Suhu, Curah Hujan, atau Alsintan) yang paling dominan di setiap wilayah.

## ✨ Fitur Utama
* **Peta Interaktif (Geospatial Analysis):** Peta sebaran produktivitas padi menggunakan Folium.
* **GWR & XAI (Explainable AI):** Mengekstrak dan memvisualisasikan pengaruh (*coefficient*) masing-masing variabel prediktor secara lokal per provinsi.
* **Integrasi Azure Machine Learning:** Mendukung penarikan data mentah (`datathon_agri_clean`) secara langsung dari *Azure ML Workspace* dengan dukungan *fallback* ke data CSV lokal.
* **Visualisasi Interaktif:** Scatter plot, horizontal bar chart, dan pemeringkatan produktivitas menggunakan Plotly.
* **Filter & Sorting Dinamis:** Eksplorasi dataset GWR lengkap dengan kemampuan pengurutan berdasarkan variabel aktual maupun koefisien regresi.

## 🛠️ Teknologi yang Digunakan
* **Frontend/Framework:** [Streamlit](https://streamlit.io/)
* **Geospatial & Modeling:** GeoPandas, `mgwr`, Folium
* **Data Processing & Vis:** Pandas, NumPy, Plotly
* **Cloud Platform:** Azure Machine Learning (`azure-ai-ml`, `azure-identity`)

## 🚀 Cara Menjalankan Aplikasi di Lokal

### 1. Persiapan Environment
Pastikan Anda sudah menginstal Python 3.9 atau lebih baru. Disarankan menggunakan *virtual environment*:
```bash
python -m venv myenv
# Pengguna Windows:
myenv\Scripts\activate
# Pengguna Mac/Linux:
source myenv/bin/activate
```

### 2. Instalasi Dependensi
Jalankan perintah berikut untuk menginstal semua *library* yang dibutuhkan:
```bash
pip install -r requirements.txt
```

### 3. Konfigurasi Azure ML (Opsional)
Aplikasi ini sudah dilengkapi dengan mekanisme *fallback* sehingga tetap bisa berjalan menggunakan data `AgriData.csv` secara lokal. Namun, jika Anda ingin menghubungkannya ke Azure ML:
1. Salin file `.env.example` dan ubah namanya menjadi `.env`.
2. Isi kredensial Azure Anda di dalam file `.env`:
   ```env
   AZURE_TENANT_ID="xxxx"
   AZURE_SUBSCRIPTION_ID="xxxx"
   AZURE_RESOURCE_GROUP="xxxxx"
   AZURE_WORKSPACE_NAME="xxxxx"
   ```
3. Lakukan otentikasi Azure CLI di terminal Anda:
   ```bash
   az login --tenant xxxxx
   ```

### 4. Jalankan Aplikasi
Ketikkan perintah berikut untuk meluncurkan *dashboard* Streamlit:
```bash
streamlit run app.py
```
Aplikasi akan otomatis terbuka di browser Anda pada alamat `http://localhost:8501`.

## 📂 Struktur File Utama
* `app.py`: File utama aplikasi Streamlit yang berisi alur pemrosesan data, pemodelan GWR, dan layout visual.
* `AgriData.csv`: Dataset lokal untuk fallback jika koneksi Azure gagal.
* `GeoIndonesia.json`: Berkas spasial (GeoJSON) yang mendefinisikan batas poligon dan letak wilayah provinsi di Indonesia.
* `requirements.txt`: Daftar package Python pendukung.

## 🤝 Lisensi & Kredit
Proyek ini dibuat dalam rangka partisipasi pada kompetisi **Microsoft Datathon 2026**. Hak cipta atas ide dan implementasi dipegang oleh pengembang.
