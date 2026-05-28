# 🫀 Prediksi Risiko Penyakit Jantung (Heart Disease Predictor)

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)

Aplikasi berbasis web menggunakan **Streamlit** dan **Machine Learning (Random Forest)** untuk memprediksi probabilitas dan risiko penyakit jantung pada individu berdasarkan faktor gaya hidup dan kondisi kesehatan.

---

## 🚀 Fitur Utama
* **Prediksi Interaktif:** Input data kesehatan personal secara langsung melalui UI Streamlit yang bersih.
* **Hasil Real-time:** Menampilkan probabilitas risiko penyakit jantung (Tinggi / Rendah) secara instan.
* **Pipeline Machine Learning Kuat:** Menggunakan `ColumnTransformer` untuk pre-processing data (StandardScaler untuk fitur numerik & OneHotEncoder untuk fitur kategorikal).
* **Evaluasi Komprehensif:** Menyertakan skrip all-in-one untuk Exploratory Data Analysis (EDA), pelatihan model, evaluasi performa (Classification Report, Confusion Matrix, ROC-AUC), dan penyimpanan model otomatis.

---

## 📁 Struktur Direktori
```text
heart_disease_predictor/
│
├── models/
│   └── heart_disease_model.pkl   # Model Random Forest yang dilatih (di-ignore dari git karena ukuran besar)
│
├── app.py                         # Aplikasi web Streamlit utama
├── eda_and_training.py            # Skrip EDA, preprocessing, training, dan evaluasi
├── data.csv                       # Dataset utama (berisi data faktor risiko)
├── requirements.txt               # Daftar dependensi Python
├── .gitignore                     # Konfigurasi file yang diabaikan oleh Git
│
# Output Visualisasi Hasil Pelatihan
├── output_bmi_distribution.png    # Visualisasi distribusi BMI
├── output_label_distribution.png  # Visualisasi distribusi target (Heart Disease)
├── output_confusion_matrix.png    # Heatmap Confusion Matrix evaluasi model
└── output_roc_curve.png           # Grafik kurva ROC dan skor AUC
```

---

## 🛠️ Instalasi & Persiapan

Ikuti langkah-langkah di bawah ini untuk menjalankan proyek ini secara lokal:

### 1. Kloning Repositori
```bash
git clone https://github.com/Iqbal0105/heart-disease-predictor.git
cd heart-disease-predictor
```

### 2. Buat & Aktifkan Virtual Environment
* **Windows (PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
* **macOS/Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Instal Dependensi
```bash
pip install -r requirements.txt
```

---

## 🏋️ Pelatihan Model & EDA

Jika Anda ingin melatih ulang model menggunakan dataset `data.csv`, jalankan skrip berikut:
```bash
python eda_and_training.py
```
Skrip ini akan melakukan:
1. Membaca data dan memproses missing values.
2. Menyimpan diagram distribusi target dan BMI.
3. Membagi data menjadi train-test set (80:20).
4. Melatih model **Random Forest Classifier** di dalam sebuah `Pipeline` terintegrasi.
5. Mencetak laporan klasifikasi serta menyimpan grafik Confusion Matrix dan ROC Curve.
6. Menyimpan model final yang siap digunakan ke `models/heart_disease_model.pkl`.

---

## 🖥️ Menjalankan Aplikasi Web (Streamlit)

Setelah model berhasil dilatih (atau menggunakan model yang sudah ada), Anda dapat menjalankan aplikasi web interaktif dengan perintah berikut:

```bash
streamlit run app.py
```

Setelah dijalankan, buka browser Anda dan akses tautan lokal yang ditampilkan (biasanya `http://localhost:8501`).

---

## 📊 Metrik Evaluasi Model
Model dilatih menggunakan algoritma **Random Forest Classifier** dan dievaluasi dengan beberapa metrik utama:
* **Confusion Matrix:** Menunjukkan akurasi prediksi untuk kasus Positif (memiliki penyakit jantung) dan Negatif (sehat).
* **ROC-AUC Score:** Mengukur seberapa baik model dapat membedakan kelas pasien sakit jantung dan yang sehat.
* Semua grafik evaluasi disimpan langsung sebagai berkas gambar di direktori utama sehingga memudahkan analisis hasil pelatihan secara visual.

---

## 📝 Fitur Input Data Pengguna
Aplikasi web Streamlit menerima input berikut:
* **Kategori Usia:** Rentang kelompok umur (misal: 18-24, 50-54, dsb).
* **BMI (Body Mass Index):** Angka indeks massa tubuh (dihitung otomatis berdasarkan berat/tinggi badan).
* **Kesehatan Umum (General Health):** Tingkat kesehatan yang dirasakan secara subjektif.
* **Status Merokok, Status Diabetes, Aktivitas Fisik, Riwayat Stroke, Kesulitan Berjalan, Penyakit Ginjal, dan Jenis Kelamin.**

---

*Dibuat dengan 💻 menggunakan Python, Streamlit, dan Scikit-Learn.*
