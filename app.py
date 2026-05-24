# ====================================================
# CLUSTERING + FORECAST HOLT WINTERS (PYTHON)
# ====================================================

# ====================================================
# 1. IMPORT LIBRARY
# ====================================================

import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from statsmodels.tsa.holtwinters import ExponentialSmoothing

import matplotlib.pyplot as plt

# ====================================================
# 2. IMPORT DATA
# ====================================================

data_barang = pd.read_excel(
    "PKL UNNES DATA BARANG KELUAR.xlsx"
)

# ====================================================
# 3. FORMAT TANGGAL
# ====================================================

data_barang["tgl_input"] = pd.to_datetime(
    data_barang["tgl_input"]
)

# ====================================================
# 4. BUAT BULAN
# ====================================================

data_barang["bulan"] = (
    data_barang["tgl_input"]
    .dt.to_period("M")
    .astype(str)
)

# ====================================================
# 5. TOTAL PENJUALAN
#    PER BULAN PER PRODUK
# ====================================================

data_bulanan_produk = (
    data_barang
    .groupby(
        ["bulan", "id_produk"]
    )["keluar"]
    .sum()
    .reset_index()
)

data_bulanan_produk.rename(
    columns={
        "keluar": "total_keluar"
    },
    inplace=True
)

print(data_bulanan_produk.head())

# ====================================================
# 6. BUAT CLUSTER MUSIM
# ====================================================
# Cluster 1 = Jan-Jun
# Cluster 2 = Jul-Des

data_bulanan_produk["bulan_angka"] = pd.to_datetime(
    data_bulanan_produk["bulan"]
).dt.month

data_bulanan_produk["cluster"] = np.where(
    data_bulanan_produk["bulan_angka"].isin(
        [1,2,3,4,5,6]
    ),
    "Cluster 1",
    "Cluster 2"
)

print(data_bulanan_produk.head())

# ====================================================
# 7. CLUSTERING K-MEANS
# ====================================================

rekap_produk = (
    data_bulanan_produk
    .groupby("id_produk")
    .agg({
        "total_keluar": [
            "sum",
            "mean"
        ]
    })
)

rekap_produk.columns = [
    "total_penjualan",
    "rata_penjualan"
]

rekap_produk = rekap_produk.reset_index()

# scaling
scaler = StandardScaler()

X = scaler.fit_transform(
    rekap_produk[
        [
            "total_penjualan",
            "rata_penjualan"
        ]
    ]
)

# KMeans
kmeans = KMeans(
    n_clusters=2,
    random_state=123
)

rekap_produk["hasil_cluster"] = (
    kmeans.fit_predict(X)
)

print(rekap_produk.head())

# ====================================================
# 8. FILTER CLUSTER 1
# ====================================================

produk_cluster1 = rekap_produk[
    rekap_produk["hasil_cluster"] == 0
]["id_produk"]

data_cluster1 = data_bulanan_produk[
    data_bulanan_produk["id_produk"]
    .isin(produk_cluster1)
]

print(data_cluster1.head())

# ====================================================
# 9. FORECAST HOLT WINTERS
# ====================================================

hasil_forecast = []

for produk in data_cluster1["id_produk"].unique():

    df_produk = data_cluster1[
        data_cluster1["id_produk"] == produk
    ].copy()

    df_produk = df_produk.sort_values(
        "bulan"
    )

    # minimal 6 bulan
    if len(df_produk) < 6:
        continue

    ts_data = df_produk["total_keluar"]

    try:

        # model Holt Winters
        model = ExponentialSmoothing(
            ts_data,
            trend="add",
            seasonal=None
        ).fit()

        # forecast 6 bulan
        forecast = model.forecast(6)

        # hilangkan negatif
        forecast = np.ceil(
            np.maximum(
                0,
                forecast
            )
        )

        hasil_forecast.append({

            "id_produk": produk,

            "forecast_6_bulan":
                forecast.sum()
        })

    except:
        continue

# ====================================================
# 10. HASIL FORECAST
# ====================================================

hasil_forecast = pd.DataFrame(
    hasil_forecast
)

hasil_forecast = (
    hasil_forecast
    .sort_values(
        "forecast_6_bulan",
        ascending=False
    )
)

print(hasil_forecast)

# ====================================================
# 11. TOP 10 PRODUK
# ====================================================

top10 = hasil_forecast.head(10)

# ====================================================
# 12. GRAFIK
# ====================================================

plt.figure(figsize=(10,6))

plt.barh(
    top10["id_produk"],
    top10["forecast_6_bulan"]
)

plt.xlabel("Forecast")

plt.ylabel("Produk")

plt.title(
    "Forecast Produk Cluster 1"
)

plt.gca().invert_yaxis()

plt.show()
