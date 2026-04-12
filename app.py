import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import re
import string
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from wordcloud import WordCloud
import io
from collections import Counter

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentimen Analisis Gojek",
    page_icon="🛵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Space+Mono:wght@400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    min-height: 100vh;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.04) !important;
    border-right: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(20px);
}
[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}

/* Headers */
h1, h2, h3, h4 {
    color: #ffffff !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 16px !important;
    padding: 1rem !important;
    backdrop-filter: blur(10px);
}
[data-testid="metric-container"] label {
    color: #94a3b8 !important;
    font-size: 0.8rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
}

/* Cards */
.card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 20px;
    padding: 1.5rem;
    backdrop-filter: blur(12px);
    margin-bottom: 1rem;
}

/* Hero title */
.hero-title {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #60efff, #0061ff, #ff6bff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 0.5rem;
}
.hero-sub {
    color: #94a3b8;
    font-size: 1rem;
    margin-bottom: 2rem;
}

/* Sentiment badge */
.badge-pos {
    background: linear-gradient(135deg, #10b981, #059669);
    color: white;
    padding: 0.4rem 1rem;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.95rem;
    display: inline-block;
}
.badge-neg {
    background: linear-gradient(135deg, #ef4444, #dc2626);
    color: white;
    padding: 0.4rem 1rem;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.95rem;
    display: inline-block;
}

/* Tab styling */
[data-baseweb="tab"] {
    color: #94a3b8 !important;
    font-weight: 600;
}
[aria-selected="true"] {
    color: #ffffff !important;
    border-bottom-color: #60efff !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #0061ff, #60efff) !important;
    color: white !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.6rem 2rem !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    transition: all 0.3s ease;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0,97,255,0.4) !important;
}

/* Text area */
.stTextArea textarea {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    color: #ffffff !important;
    border-radius: 12px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

/* Divider */
hr {
    border-color: rgba(255,255,255,0.1) !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}

/* Expander */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 12px !important;
}
summary {
    color: #e2e8f0 !important;
}

/* Select box */
.stSelectbox > div > div {
    background: rgba(255,255,255,0.07) !important;
    border-color: rgba(255,255,255,0.15) !important;
    color: white !important;
}

/* Slider */
.stSlider [data-baseweb="slider"] {
    color: #60efff;
}

p, li, span, label, div {
    color: #cbd5e1;
}
</style>
""", unsafe_allow_html=True)


# ─── HELPER FUNCTIONS ─────────────────────────────────────────────────────────

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.strip()
    return text

@st.cache_data
def load_and_train(uploaded_file):
    df = pd.read_csv(uploaded_file)
    # normalize column names
    df.columns = [c.strip().lower() for c in df.columns]
    
    # detect text and label columns
    text_col = None
    label_col = None
    for c in df.columns:
        if 'content' in c or 'review' in c or 'text' in c or 'ulasan' in c:
            text_col = c
        if 'sentiment' in c or 'label' in c or 'class' in c:
            label_col = c
    
    if text_col is None:
        text_col = df.columns[0]
    if label_col is None:
        label_col = df.columns[-1]
    
    df = df[[text_col, label_col]].dropna()
    df.columns = ['content', 'sentiment']
    df['sentiment'] = df['sentiment'].astype(int)
    
    return df, text_col, label_col

@st.cache_resource
def train_model(df, alpha=1.0):
    cv = CountVectorizer()
    x = cv.fit_transform(df['content'])
    y = df['sentiment']
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.70, random_state=42)
    
    model = MultinomialNB(alpha=alpha)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'cm': confusion_matrix(y_test, y_pred),
        'y_test': y_test,
        'y_pred': y_pred,
    }
    return model, cv, metrics

def make_wordcloud_fig(text, colormap, bg):
    wc = WordCloud(
        max_font_size=140,
        background_color=bg,
        colormap=colormap,
        width=700, height=380,
        margin=5,
        max_words=200,
    ).generate(text)
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_alpha(0)
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    plt.tight_layout(pad=0)
    return fig

def dark_fig():
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor('#1e1b4b')
    ax.set_facecolor('#1e1b4b')
    return fig, ax

def apply_dark_theme(ax, fig=None):
    ax.tick_params(colors='#94a3b8')
    ax.xaxis.label.set_color('#94a3b8')
    ax.yaxis.label.set_color('#94a3b8')
    ax.title.set_color('#ffffff')
    for spine in ax.spines.values():
        spine.set_edgecolor((1, 1, 1, 0.1))
    if fig:
        fig.patch.set_facecolor('#1e1b4b')
    ax.set_facecolor('#1e1b4b')


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛵 Sentimen Gojek")
    st.markdown("---")
    
    st.markdown("### 📂 Upload Dataset")
    uploaded = st.file_uploader(
        "Upload file CSV ulasan",
        type=["csv"],
        help="Format: kolom 'content' dan 'sentiment' (0=negatif, 1=positif)"
    )
    
    st.markdown("---")
    st.markdown("### ⚙️ Konfigurasi Model")
    alpha_val = st.select_slider(
        "Alpha (Naive Bayes)",
        options=[0.01, 0.05, 0.25, 0.5, 1.0],
        value=1.0
    )
    
    st.markdown("---")
    st.markdown("### 📊 Menu")
    menu = st.radio(
        "Pilih Halaman",
        ["🏠 Overview", "📊 Visualisasi", "🤖 Model & Evaluasi", "🔍 Prediksi Teks"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown(
        "<div style='color:#64748b;font-size:0.75rem;'>Dibuat dengan ❤️ menggunakan Streamlit & Scikit-learn</div>",
        unsafe_allow_html=True
    )


# ─── MAIN AREA ────────────────────────────────────────────────────────────────

# Hero header
st.markdown("""
<div class='hero-title'>Analisis Sentimen Ulasan Gojek</div>
<div class='hero-sub'>Naive Bayes Classifier · Count Vectorizer · NLP Bahasa Indonesia</div>
""", unsafe_allow_html=True)

# Load data state
if uploaded is None:
    # Use demo/sample data if no upload
    st.info("💡 Upload file CSV di sidebar untuk memulai, atau lihat demo di bawah ini.")
    
    # Generate sample data for demo
    np.random.seed(42)
    n = 500
    positive_samples = [
        "aplikasi gojek sangat bagus dan mudah digunakan",
        "driver ramah dan tepat waktu sangat memuaskan",
        "layanan gojek terbaik sangat membantu keseharian",
        "harga terjangkau kualitas terjamin terimakasih gojek",
        "go food enak dan cepat sampai makanan masih hangat",
        "gojek selalu diandalkan setiap hari sangat puas",
        "pelayanan memuaskan driver baik dan sopan sekali",
        "fitur lengkap dan mudah dipahami aplikasinya bagus",
    ] * (n // 8)
    negative_samples = [
        "aplikasi sering error dan tidak bisa dibuka",
        "driver tidak sesuai foto dan tidak ramah",
        "pesanan sering telat dan makanan sudah dingin",
        "tarif terlalu mahal tidak sesuai dengan pelayanan",
        "customer service tidak membantu sama sekali",
        "sering dibatalkan sepihak tanpa konfirmasi dulu",
        "aplikasi lambat dan sering crash sangat mengecewakan",
        "kualitas menurun tidak seperti dulu sudah buruk",
    ] * (n // 8)
    
    pos_labels = [1] * len(positive_samples)
    neg_labels = [0] * len(negative_samples)
    
    demo_df = pd.DataFrame({
        'content': positive_samples + negative_samples,
        'sentiment': pos_labels + neg_labels
    }).sample(frac=1, random_state=42).reset_index(drop=True)
    
    df = demo_df
    st.caption("📌 Data di bawah adalah **data demo**. Upload CSV Anda untuk analisis nyata.")
else:
    try:
        df, text_col, label_col = load_and_train(uploaded)
        st.success(f"✅ Dataset berhasil dimuat: **{len(df):,}** baris data")
    except Exception as e:
        st.error(f"❌ Gagal memuat file: {e}")
        st.stop()

# Train model
model, cv, metrics = train_model(df, alpha=alpha_val)

df_pos = df[df['sentiment'] == 1]
df_neg = df[df['sentiment'] == 0]

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
if menu == "🏠 Overview":
    
    # KPI Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📝 Total Ulasan", f"{len(df):,}")
    with col2:
        st.metric("😊 Positif", f"{len(df_pos):,}", delta=f"{len(df_pos)/len(df)*100:.1f}%")
    with col3:
        st.metric("😞 Negatif", f"{len(df_neg):,}", delta=f"-{len(df_neg)/len(df)*100:.1f}%", delta_color="inverse")
    with col4:
        st.metric("🎯 Akurasi Model", f"{metrics['accuracy']*100:.2f}%")
    
    st.markdown("---")
    
    col_a, col_b = st.columns([1, 1])
    
    with col_a:
        st.markdown("### 📊 Distribusi Sentimen")
        fig, ax = plt.subplots(figsize=(5, 5))
        fig.patch.set_facecolor('#1e1b4b')
        ax.set_facecolor('#1e1b4b')
        
        sizes = [len(df_pos), len(df_neg)]
        colors = ['#10b981', '#ef4444']
        explode = (0.05, 0.05)
        wedges, texts, autotexts = ax.pie(
            sizes, explode=explode, colors=colors,
            autopct='%1.1f%%', startangle=90,
            wedgeprops=dict(width=0.6, edgecolor='#1e1b4b', linewidth=2),
            textprops={'color': 'white', 'fontsize': 12, 'fontweight': 'bold'}
        )
        ax.text(0, 0, f"{len(df):,}\nUlasan", ha='center', va='center',
                color='white', fontsize=13, fontweight='bold', linespacing=1.6)
        legend = ax.legend(
            ['😊 Positif', '😞 Negatif'],
            loc='lower center', ncol=2,
            bbox_to_anchor=(0.5, -0.07),
            frameon=False,
            labelcolor='white',
            fontsize=11
        )
        st.pyplot(fig, use_container_width=True)
        plt.close()
    
    with col_b:
        st.markdown("### 📈 Distribusi Panjang Teks")
        df['text_len'] = df['content'].apply(lambda x: len(str(x).split()))
        
        fig, ax = plt.subplots(figsize=(5, 5))
        fig.patch.set_facecolor('#1e1b4b')
        ax.set_facecolor('#1e1b4b')
        
        ax.hist(df[df['sentiment']==1]['text_len'], bins=30, color='#10b981', alpha=0.7, label='Positif', edgecolor='none')
        ax.hist(df[df['sentiment']==0]['text_len'], bins=30, color='#ef4444', alpha=0.7, label='Negatif', edgecolor='none')
        ax.set_xlabel('Jumlah Kata', color='#94a3b8')
        ax.set_ylabel('Frekuensi', color='#94a3b8')
        ax.tick_params(colors='#94a3b8')
        ax.legend(frameon=False, labelcolor='white')
        for spine in ax.spines.values():
            spine.set_edgecolor((1, 1, 1, 0.08))
        ax.set_title('Distribusi Panjang Ulasan', color='white', fontsize=13, fontweight='bold', pad=12)
        st.pyplot(fig, use_container_width=True)
        plt.close()
    
    st.markdown("---")
    st.markdown("### 🗂️ Sampel Data")
    
    tab1, tab2 = st.tabs(["😊 Ulasan Positif", "😞 Ulasan Negatif"])
    with tab1:
        st.dataframe(
            df_pos[['content', 'sentiment']].head(10).rename(columns={'content':'Ulasan','sentiment':'Label'}),
            use_container_width=True, hide_index=True
        )
    with tab2:
        st.dataframe(
            df_neg[['content', 'sentiment']].head(10).rename(columns={'content':'Ulasan','sentiment':'Label'}),
            use_container_width=True, hide_index=True
        )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: VISUALISASI
# ─────────────────────────────────────────────────────────────────────────────
elif menu == "📊 Visualisasi":
    st.markdown("### ☁️ Word Cloud")
    
    positive_text = " ".join(df_pos['content'].astype(str).tolist()).lower()
    negative_text = " ".join(df_neg['content'].astype(str).tolist()).lower()
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""<div style='text-align:center;margin-bottom:0.5rem;'>
            <span class='badge-pos'>😊 Ulasan Positif</span></div>""", unsafe_allow_html=True)
        fig = make_wordcloud_fig(positive_text, 'YlGn', '#0d1b2a')
        st.pyplot(fig, use_container_width=True)
        plt.close()
    
    with col2:
        st.markdown("""<div style='text-align:center;margin-bottom:0.5rem;'>
            <span class='badge-neg'>😞 Ulasan Negatif</span></div>""", unsafe_allow_html=True)
        fig = make_wordcloud_fig(negative_text, 'Reds', '#0d1b2a')
        st.pyplot(fig, use_container_width=True)
        plt.close()
    
    st.markdown("---")
    st.markdown("### 🔝 Top Kata Terbanyak")
    
    n_words = st.slider("Tampilkan top N kata", min_value=5, max_value=30, value=15)
    
    def top_words(text, n):
        words = re.sub(r'[^a-zA-Z\s]', '', text.lower()).split()
        stopwords_id = {'yang','dan','di','ke','dari','untuk','dengan','ini','itu',
                        'tidak','ada','juga','sudah','saya','kami','anda','nya',
                        'bisa','lebih','sangat','app','nan','yg','gak','ga','banget',
                        'kalo','kalau','pas','lagi','jadi','buat','mau','dah','udah',
                        'ya','iya','si','tau','tapi','tp','ok','oke','waktu','karena'}
        words = [w for w in words if len(w) > 2 and w not in stopwords_id]
        return Counter(words).most_common(n)
    
    col1, col2 = st.columns(2)
    
    with col1:
        top_pos = top_words(positive_text, n_words)
        words_p, freqs_p = zip(*top_pos)
        fig, ax = plt.subplots(figsize=(5, 6))
        fig.patch.set_facecolor('#1e1b4b')
        ax.set_facecolor('#1e1b4b')
        bars = ax.barh(words_p[::-1], freqs_p[::-1], color='#10b981', alpha=0.85, height=0.7)
        ax.set_title('Top Kata — Positif', color='white', fontsize=12, fontweight='bold', pad=10)
        ax.tick_params(colors='#94a3b8', labelsize=9)
        for spine in ax.spines.values(): spine.set_visible(False)
        for bar, val in zip(bars, freqs_p[::-1]):
            ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                    str(val), va='center', color='#10b981', fontsize=8, fontweight='bold')
        st.pyplot(fig, use_container_width=True)
        plt.close()
    
    with col2:
        top_neg = top_words(negative_text, n_words)
        words_n, freqs_n = zip(*top_neg)
        fig, ax = plt.subplots(figsize=(5, 6))
        fig.patch.set_facecolor('#1e1b4b')
        ax.set_facecolor('#1e1b4b')
        bars = ax.barh(words_n[::-1], freqs_n[::-1], color='#ef4444', alpha=0.85, height=0.7)
        ax.set_title('Top Kata — Negatif', color='white', fontsize=12, fontweight='bold', pad=10)
        ax.tick_params(colors='#94a3b8', labelsize=9)
        for spine in ax.spines.values(): spine.set_visible(False)
        for bar, val in zip(bars, freqs_n[::-1]):
            ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                    str(val), va='center', color='#ef4444', fontsize=8, fontweight='bold')
        st.pyplot(fig, use_container_width=True)
        plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: MODEL & EVALUASI
# ─────────────────────────────────────────────────────────────────────────────
elif menu == "🤖 Model & Evaluasi":
    st.markdown("### 🤖 Kinerja Model Naive Bayes")
    st.markdown(f"*Alpha yang digunakan: **{alpha_val}***")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🎯 Akurasi", f"{metrics['accuracy']*100:.2f}%")
    with col2:
        st.metric("🔬 Presisi", f"{metrics['precision']*100:.2f}%")
    with col3:
        st.metric("📡 Recall", f"{metrics['recall']*100:.2f}%")
    with col4:
        st.metric("⚖️ F1-Score", f"{metrics['f1']*100:.2f}%")
    
    st.markdown("---")
    
    col_cm, col_alpha = st.columns([1, 1])
    
    with col_cm:
        st.markdown("### 🔷 Confusion Matrix")
        cm = metrics['cm']
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor('#1e1b4b')
        ax.set_facecolor('#1e1b4b')
        
        sns.heatmap(
            cm, annot=True, fmt='d',
            cmap='Blues', linewidths=2,
            linecolor='#1e1b4b',
            ax=ax,
            annot_kws={"size": 14, "weight": "bold", "color": "white"},
            cbar_kws={"shrink": 0.8}
        )
        ax.set_xlabel('Prediksi', color='#94a3b8', fontsize=11)
        ax.set_ylabel('Aktual', color='#94a3b8', fontsize=11)
        ax.set_xticklabels(['Negatif', 'Positif'], color='#94a3b8')
        ax.set_yticklabels(['Negatif', 'Positif'], color='#94a3b8', rotation=0)
        ax.set_title('Confusion Matrix', color='white', fontsize=13, fontweight='bold', pad=12)
        st.pyplot(fig, use_container_width=True)
        plt.close()
    
    with col_alpha:
        st.markdown("### 📉 Perbandingan Alpha")
        alphas = [0.01, 0.05, 0.25, 0.5, 1.0]
        accs = []
        f1s = []
        
        x_all = cv.transform(df['content'])
        y_all = df['sentiment']
        _, x_test_a, _, y_test_a = train_test_split(x_all, y_all, train_size=0.7, random_state=42)
        
        x_train_a, _, y_train_a, _ = train_test_split(x_all, y_all, train_size=0.7, random_state=42)
        
        for a in alphas:
            m = MultinomialNB(alpha=a)
            m.fit(x_train_a, y_train_a)
            yp = m.predict(x_test_a)
            accs.append(accuracy_score(y_test_a, yp) * 100)
            f1s.append(f1_score(y_test_a, yp) * 100)
        
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor('#1e1b4b')
        ax.set_facecolor('#1e1b4b')
        
        x_pos = range(len(alphas))
        ax.plot([str(a) for a in alphas], accs, 'o-', color='#60efff', linewidth=2.5,
                markersize=7, label='Akurasi', markerfacecolor='white')
        ax.plot([str(a) for a in alphas], f1s, 's--', color='#ff6bff', linewidth=2.5,
                markersize=7, label='F1-Score', markerfacecolor='white')
        ax.set_xlabel('Alpha', color='#94a3b8', fontsize=11)
        ax.set_ylabel('Score (%)', color='#94a3b8', fontsize=11)
        ax.tick_params(colors='#94a3b8')
        ax.legend(frameon=False, labelcolor='white')
        for spine in ax.spines.values():
            spine.set_edgecolor((1, 1, 1, 0.08))
        ax.set_title('Akurasi & F1 vs Alpha', color='white', fontsize=13, fontweight='bold', pad=12)
        ax.grid(axis='y', color=(1, 1, 1, 0.06), linestyle='--')
        st.pyplot(fig, use_container_width=True)
        plt.close()
    
    st.markdown("---")
    st.markdown("### 📋 Classification Report")
    
    from sklearn.metrics import classification_report
    report = classification_report(
        metrics['y_test'], metrics['y_pred'],
        target_names=['Negatif (0)', 'Positif (1)'],
        output_dict=True
    )
    report_df = pd.DataFrame(report).T
    report_df = report_df.round(4)
    st.dataframe(report_df, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: PREDIKSI TEKS
# ─────────────────────────────────────────────────────────────────────────────
elif menu == "🔍 Prediksi Teks":
    st.markdown("### 🔍 Prediksi Sentimen Teks Baru")
    st.markdown("Masukkan ulasan Gojek, dan model akan memprediksi sentimennya secara langsung.")
    
    user_input = st.text_area(
        "Tulis ulasan di sini...",
        placeholder="Contoh: Gojek sangat membantu saya sehari-hari, driver selalu tepat waktu dan ramah.",
        height=120
    )
    
    col_btn, _ = st.columns([1, 4])
    with col_btn:
        predict_btn = st.button("🚀 Prediksi Sekarang", use_container_width=True)
    
    if predict_btn:
        if user_input.strip() == "":
            st.warning("⚠️ Mohon masukkan teks ulasan terlebih dahulu.")
        else:
            vec = cv.transform([user_input])
            pred = model.predict(vec)[0]
            proba = model.predict_proba(vec)[0]
            
            conf = max(proba) * 100
            
            st.markdown("---")
            
            if pred == 1:
                st.markdown(f"""
                <div style='background:linear-gradient(135deg,rgba(16,185,129,0.15),rgba(5,150,105,0.1));
                    border:1px solid rgba(16,185,129,0.4);border-radius:20px;padding:2rem;text-align:center;'>
                    <div style='font-size:3.5rem;margin-bottom:0.5rem;'>😊</div>
                    <div style='color:#10b981;font-size:1.8rem;font-weight:800;'>POSITIF</div>
                    <div style='color:#94a3b8;margin-top:0.5rem;'>Kepercayaan Model: <span style='color:#10b981;font-weight:700;'>{conf:.1f}%</span></div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style='background:linear-gradient(135deg,rgba(239,68,68,0.15),rgba(220,38,38,0.1));
                    border:1px solid rgba(239,68,68,0.4);border-radius:20px;padding:2rem;text-align:center;'>
                    <div style='font-size:3.5rem;margin-bottom:0.5rem;'>😞</div>
                    <div style='color:#ef4444;font-size:1.8rem;font-weight:800;'>NEGATIF</div>
                    <div style='color:#94a3b8;margin-top:0.5rem;'>Kepercayaan Model: <span style='color:#ef4444;font-weight:700;'>{conf:.1f}%</span></div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Probability bar
            st.markdown("#### Probabilitas Tiap Kelas")
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(figsize=(4, 1.8))
                fig.patch.set_facecolor('#1e1b4b')
                ax.set_facecolor('#1e1b4b')
                colors = ['#ef4444', '#10b981']
                bars = ax.barh(['Negatif', 'Positif'], proba * 100,
                               color=colors, height=0.5)
                ax.set_xlim(0, 100)
                ax.tick_params(colors='#94a3b8', labelsize=9)
                for spine in ax.spines.values(): spine.set_visible(False)
                for bar, val in zip(bars, proba * 100):
                    ax.text(val + 1, bar.get_y() + bar.get_height()/2,
                            f'{val:.1f}%', va='center', color='white', fontsize=9)
                st.pyplot(fig, use_container_width=True)
                plt.close()
    
    st.markdown("---")
    st.markdown("### 📦 Prediksi Batch")
    st.markdown("Upload CSV dengan kolom `content` untuk prediksi massal.")
    
    batch_file = st.file_uploader("Upload CSV untuk batch prediksi", type=["csv"], key="batch")
    
    if batch_file:
        try:
            batch_df = pd.read_csv(batch_file)
            batch_df.columns = [c.lower().strip() for c in batch_df.columns]
            
            text_col = 'content' if 'content' in batch_df.columns else batch_df.columns[0]
            batch_texts = batch_df[text_col].astype(str).tolist()
            
            batch_vec = cv.transform(batch_texts)
            batch_pred = model.predict(batch_vec)
            batch_proba = model.predict_proba(batch_vec)
            
            batch_df['prediksi'] = ['Positif 😊' if p == 1 else 'Negatif 😞' for p in batch_pred]
            batch_df['kepercayaan (%)'] = (np.max(batch_proba, axis=1) * 100).round(2)
            
            st.success(f"✅ Berhasil memprediksi **{len(batch_df):,}** ulasan!")
            st.dataframe(batch_df[[text_col, 'prediksi', 'kepercayaan (%)']],
                         use_container_width=True, hide_index=True)
            
            csv_out = batch_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇️ Download Hasil Prediksi",
                data=csv_out,
                file_name="hasil_prediksi_sentimen.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Error: {e}")