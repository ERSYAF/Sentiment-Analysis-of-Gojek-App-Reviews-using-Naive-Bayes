# 🤖 Sentiment Analysis of Gojek App Reviews using Naive Bayes
## 📌 Deskripsi Project
Project ini bertujuan untuk menganalisis sentimen pengguna terhadap aplikasi Gojek berdasarkan ulasan yang diberikan di Google Play Store menggunakan teknik Natural Language Processing (NLP) dan algoritma Naive Bayes.
Project ini bersifat **end-to-end**, mulai dari proses pengambilan data (crawling), preprocessing, pelatihan model, evaluasi, hingga deployment dalam bentuk aplikasi web menggunakan Streamlit.
---

## 🎯 Tujuan
* Mengklasifikasikan sentimen ulasan pengguna (Positif, Negatif, Netral)
* Menerapkan teknik preprocessing teks dalam NLP
* Membangun model machine learning menggunakan Naive Bayes
* Menyediakan aplikasi interaktif untuk prediksi sentimen

---

## 🧠 Teknologi yang Digunakan
* Python
* Pandas & NumPy
* Scikit-learn
* NLTK / Sastrawi (Preprocessing teks Bahasa Indonesia)
* Matplotlib / Seaborn (Visualisasi)
* Streamlit (Web App)
* Google Play Scraper (Crawling data)

---

## 📂 Struktur Project

```
Sentiment-Analysis-Gojek-Naive-Bayes/
│
├── data/
│   ├── dataset_gojek.csv
│   ├── reviews_clean.csv
│
├── notebooks/
│   ├── 1_crawling_data.ipynb
│   ├── 2_preprocessing.ipynb
│   ├── 3_labeling.ipynb
│   ├── 4_modeling.ipynb
│
├── app/
│   ├── app.py
│
├── requirements.txt
├── README.md
└── .gitignore
```
---

## 🔍 Metodologi
### 1. Crawling Data
Data ulasan diambil dari aplikasi Gojek di Google Play Store menggunakan Python (scraping).

### 2. Preprocessing Data
* Case Folding (mengubah huruf menjadi lowercase)
* Cleaning (menghapus simbol, angka, dll)
* Tokenization
* Stopword Removal
* Stemming (menggunakan Sastrawi)

### 3. Labeling Data
Data diklasifikasikan menjadi:
* Positif
* Negatif

### 4. Feature Extraction
Menggunakan:
* TF-IDF Vectorizer

### 5. Modeling
* Algoritma: Naive Bayes (MultinomialNB)
* Pembagian data: Train & Test

### 6. Evaluasi Model
Menggunakan:
* Accuracy
* Precision
* Recall
* Confusion Matrix

---
## 📊 Hasil Evaluasi Model
Model Naive Bayes diuji menggunakan beberapa nilai smoothing parameter (alpha) untuk mendapatkan performa terbaik.
### 🔹 Perbandingan Performa Model

| Alpha | Accuracy | Precision | Recall | F1-Score |
|------|----------|----------|--------|----------|
| 0.01 | 0.8493 | 0.8890 | 0.8771 | 0.8830 |
| 0.05 | 0.8543 | 0.8919 | 0.8823 | 0.8871 |
| 0.25 | 0.8597 | 0.8969 | 0.8853 | 0.8911 |
| 0.5  | 0.8600 | 0.8982 | 0.8843 | 0.8912 |
| 1.0  | **0.8630** | **0.9054** | 0.8807 | **0.8929** |

---

### 🏆 Model Terbaik
Model dengan performa terbaik diperoleh pada:
- **Alpha = 1.0**
- Accuracy: **86.3%**
- Precision: **90.54%**
- Recall: **88.07%**
- F1-Score: **89.29%**
---

### 📌 Insight
- Nilai alpha yang lebih besar memberikan performa yang lebih stabil
- Model menunjukkan performa yang baik dalam mengklasifikasikan sentimen positif dan negatif
- Precision yang tinggi menunjukkan model cukup akurat dalam prediksi sentimen
---

## 💻 Aplikasi Streamlit
Project ini dilengkapi dengan aplikasi web interaktif menggunakan Streamlit.

### Fitur:
* Input teks ulasan
* Prediksi sentimen secara real-time
* Menampilkan hasil klasifikasi
* Visualisasi data (opsional)

### Cara Menjalankan:
```bash id="runapp1"
cd app
streamlit run app.py
```
---

## ⚙️ Cara Instalasi
### 1. Clone Repository
```bash id="clone1"
git clone https://github.com/username/nama-repo-kamu.git
cd nama-repo-kamu
```

### 2. Buat Virtual Environment (Opsional)
```bash id="venv1"
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash id="install1"
pip install -r requirements.txt
```

---


