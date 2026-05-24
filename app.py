# app.py


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt

st.set_page_config(page_title="Forecasting Stok", layout="wide")

st.title("📦 K-Means Clustering + Double Exponential Smoothing")
st.write("Aplikasi forecasting pergerakan stok menggunakan K-Means dan Double Exponential Smoothing")

uploaded_file = st.file_uploader("Upload file Excel", type=["xlsx"])

if uploaded_file is not None:

    # =========================
    # LOAD DATA
    # =========================
    df = pd.read_excel(uploaded_file)

    st.write(df.columns)

    st.subheader("Data Awal")
    st.dataframe(df.head())

    # =========================
    # PREPROCESSING
    # =========================
    df['tgl_input'] = pd.to_datetime(df['tgl_input'])

    # total pergerakan stok
    df['total_pergerakan'] = df['keluar']

    # =========================
    # FITUR CLUSTERING
    # =========================
    fitur_cluster = df.groupby('id_produk').agg({
        'keluar': 'sum',
        'total_pergerakan': 'sum'
    }).reset_index()

    # scaling
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(
        fitur_cluster[['keluar', 'total_pergerakan']]
    )

    # =========================
    # K-MEANS CLUSTERING
    # =========================
    jumlah_cluster = st.slider("Jumlah Cluster", 2, 10, 3)

    kmeans = KMeans(n_clusters=jumlah_cluster, random_state=42)
    fitur_cluster['cluster'] = kmeans.fit_predict(scaled_data)

    st.subheader("Hasil Clustering")
    st.dataframe(fitur_cluster.head())

    # visualisasi cluster
    fig, ax = plt.subplots(figsize=(8, 5))

    scatter = ax.scatter(
        fitur_cluster['masuk'],
        fitur_cluster['keluar'],
        c=fitur_cluster['cluster']
    )

    ax.set_xlabel('Total Masuk')
    ax.set_ylabel('Total Keluar')
    ax.set_title('Visualisasi K-Means Clustering')

    st.pyplot(fig)

    # =========================
    # PILIH PRODUK UNTUK FORECAST
    # =========================
    st.subheader("Forecasting Produk")

    daftar_produk = df['id_produk'].unique()
    selected_produk = st.selectbox(
        "Pilih Produk",
        daftar_produk
    )

    # filter data produk
    produk_df = df[df['id_produk'] == selected_produk].copy()

    # agregasi bulanan
    bulanan = produk_df.groupby(
        pd.Grouper(key='tgl_input', freq='M')
    )['keluar'].sum().reset_index()

    bulanan.columns = ['tanggal', 'permintaan']

    st.write("Data Bulanan")
    st.dataframe(bulanan)

    # =========================
    # DOUBLE EXPONENTIAL SMOOTHING
    # =========================
    if len(bulanan) >= 3:

        model = ExponentialSmoothing(
            bulanan['permintaan'],
            trend='add',
            seasonal=None
        ).fit()

        future_periods = st.number_input(
            "Jumlah periode forecast",
            min_value=1,
            max_value=24,
            value=6
        )

        forecast = model.forecast(future_periods)

        # ubah nilai minus jadi 0
        forecast = np.where(forecast < 0, 0, forecast)

        # tanggal masa depan
        future_dates = pd.date_range(
            start=bulanan['tanggal'].max() + pd.offsets.MonthEnd(1),
            periods=future_periods,
            freq='M'
        )

        forecast_df = pd.DataFrame({
            'tanggal': future_dates,
            'forecast': forecast
        })

        st.subheader("Hasil Forecast")
        st.dataframe(forecast_df)

        # =========================
        # VISUALISASI
        # =========================
        fig2, ax2 = plt.subplots(figsize=(10, 5))

        ax2.plot(
            bulanan['tanggal'],
            bulanan['permintaan'],
            label='Data Aktual'
        )

        ax2.plot(
            forecast_df['tanggal'],
            forecast_df['forecast'],
            label='Forecast'
        )

        ax2.set_title(f'Forecast Produk {selected_produk}')
        ax2.set_xlabel('Tanggal')
        ax2.set_ylabel('Permintaan')
        ax2.legend()

        st.pyplot(fig2)

    else:
        st.warning("Data produk terlalu sedikit untuk forecasting")
