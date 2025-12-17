import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# =============================
# CONFIG
# =============================
st.set_page_config(
    page_title="Dashboard Klasterisasi Banjir DKI Jakarta",
    layout="wide"
)

# =============================
# LOAD DATA & MODEL
# =============================
@st.cache_data
def load_data():
    df = pd.read_csv('data_banjir_clustered.csv')
    agregasi_kel = pd.read_csv('agregasi_kelurahan.csv')
    return df, agregasi_kel

@st.cache_resource
def load_model():
    with open('kmeans_model.pkl', 'rb') as f:
        kmeans = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return kmeans, scaler

# Load objects

df, agregasi_kelurahan = load_data()
kmeans, scaler = load_model()

fitur = [
    'ketinggian_air_cm',
    'jumlah_terdampak_rt',
    'jumlah_terdampak_kk',
    'jumlah_terdampak_jiwa',
    'jumlah_pengungsi_tertinggi'
]

# =============================
# SIDEBAR NAVIGATION
# =============================
st.sidebar.title("Navigasi")
menu = st.sidebar.radio(
    "Pilih Menu",
    ["📊 Overview", "🗺️ Analisis Wilayah", "🔮 Prediksi Banjir"]
)

# =============================
# OVERVIEW
# =============================
if menu == "📊 Overview":
    st.title("Dashboard Klasterisasi Banjir DKI Jakarta")
    st.markdown("---")

    # Distribusi & Scatter
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribusi Kategori Banjir")
        fig, ax = plt.subplots()
        df['kategori_banjir'].value_counts().plot(kind='bar', ax=ax)
        ax.set_xlabel("Kategori")
        ax.set_ylabel("Jumlah Kejadian")
        st.pyplot(fig)

    with col2:
        st.subheader("Scatter Plot Klaster")
        fig, ax = plt.subplots()
        sns.scatterplot(
            data=df,
            x='ketinggian_air_cm',
            y='jumlah_terdampak_jiwa',
            hue='kategori_banjir',
            ax=ax
        )
        ax.set_xlabel("Ketinggian Air (cm)")
        ax.set_ylabel("Jumlah Jiwa Terdampak")
        st.pyplot(fig)

    st.markdown("---")

    # Perbandingan Klaster
    st.subheader("📈 Perbandingan Karakteristik Klaster")
    col3, col4 = st.columns(2)

    with col3:
        fig, ax = plt.subplots()
        sns.boxplot(
            data=df,
            x='kategori_banjir',
            y='ketinggian_air_cm',
            ax=ax
        )
        ax.set_title("Ketinggian Air per Klaster")
        ax.set_xlabel("Kategori Banjir")
        ax.set_ylabel("Ketinggian Air (cm)")
        st.pyplot(fig)

    with col4:
        fig, ax = plt.subplots()
        sns.boxplot(
            data=df,
            x='kategori_banjir',
            y='jumlah_terdampak_jiwa',
            ax=ax
        )
        ax.set_title("Jiwa Terdampak per Klaster")
        ax.set_xlabel("Kategori Banjir")
        ax.set_ylabel("Jumlah Jiwa")
        st.pyplot(fig)

    # Heatmap ringkasan
    st.markdown("### 🔥 Ringkasan Rata-rata Variabel per Klaster")
    cluster_mean = df.groupby('kategori_banjir')[fitur].mean()
    fig, ax = plt.subplots(figsize=(8,4))
    sns.heatmap(cluster_mean, annot=True, fmt='.1f', ax=ax)
    st.pyplot(fig)

# =============================
# ANALISIS WILAYAH
# =============================
elif menu == "🗺️ Analisis Wilayah":
    st.title("Analisis Banjir per Kelurahan")
    st.markdown("---")

    top_n = st.slider("Tampilkan Top N Kelurahan", 5, 20, 10)

    top_kel = agregasi_kelurahan.sort_values(
        'persentase_banjir_berat', ascending=False
    ).head(top_n)

    st.subheader("Tabel Agregasi Kelurahan")
    st.dataframe(top_kel)

    st.subheader("Persentase Banjir Berat")
    fig, ax = plt.subplots(figsize=(8,5))
    ax.barh(top_kel['kelurahan'], top_kel['persentase_banjir_berat'])
    ax.set_xlabel("Persentase (%)")
    ax.invert_yaxis()
    st.pyplot(fig)

# =============================
# PREDIKSI BANJIR
# =============================
elif menu == "🔮 Prediksi Banjir":
    st.title("Prediksi Kategori Banjir")
    st.markdown("Masukkan parameter kejadian banjir:")

    col1, col2 = st.columns(2)

    with col1:
        ketinggian = st.number_input("Ketinggian Air (cm)", 0, 300, 50)
        rt = st.number_input("Jumlah RT Terdampak", 0, 100, 1)
        kk = st.number_input("Jumlah KK Terdampak", 0, 5000, 10)

    with col2:
        jiwa = st.number_input("Jumlah Jiwa Terdampak", 0, 10000, 50)
        pengungsi = st.number_input("Jumlah Pengungsi Tertinggi", 0, 1000, 0)

    if st.button("Prediksi"):
        input_df = pd.DataFrame(
            [[ketinggian, rt, kk, jiwa, pengungsi]],
            columns=fitur
        )

        input_scaled = scaler.transform(input_df)
        pred = kmeans.predict(input_scaled)[0]

        hasil = "Banjir Berat" if pred == 0 else "Banjir Ringan"

        st.subheader("Hasil Prediksi")
        if hasil == "Banjir Berat":
            st.error("⚠️ Kategori: BANJIR BERAT")
        else:
            st.success("✅ Kategori: BANJIR RINGAN")

# =============================
# FOOTER
# =============================
st.markdown("---")
st.caption("Aplikasi Klasterisasi Banjir - K-Means | Data Publik DKI Jakarta")


