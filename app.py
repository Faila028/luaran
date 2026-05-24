#====================================================
# CLUSTERING + FORECAST HOLT WINTERS
#====================================================

#====================================================
# 1. LOAD LIBRARY
#====================================================
library(readxl)
library(dplyr)
library(lubridate)
library(tidyr)
library(cluster)
library(factoextra)
library(forecast)
library(purrr)
library(ggplot2)

#====================================================
# 2. IMPORT DATA
#====================================================
data_barang <- read_excel(
  "PKL UNNES DATA BARANG KELUAR(1).xlsx"
)

#====================================================
# 3. FORMAT TANGGAL
#====================================================
data_barang$tgl_input <- as.Date(
  data_barang$tgl_input
)

#====================================================
# 4. BUAT BULAN
#====================================================
data_barang$bulan <- format(
  data_barang$tgl_input,
  "%Y-%m"
)

#====================================================
# 5. TOTAL KELUAR
#    PER BULAN PER PRODUK
#====================================================
data_bulanan_produk <- data_barang %>%
  
  group_by(
    bulan,
    id_produk
  ) %>%
  
  summarise(
    total_keluar =
      sum(keluar, na.rm = TRUE),
    
    .groups = "drop"
  )

View(data_bulanan_produk)

#====================================================
# 6. BUAT CLUSTER MUSIM
#====================================================
# Cluster 1 = Jan-Jun
# Cluster 2 = Jul-Des

data_bulanan_produk <- data_bulanan_produk %>%
  
  mutate(
    
    bulan_angka =
      month(
        as.Date(
          paste0(bulan, "-01")
        )
      ),
    
    cluster =
      case_when(
        
        bulan_angka %in% c(1,2,3,4,5,6)
        ~ "Cluster 1",
        
        bulan_angka %in% c(7,8,9,10,11,12)
        ~ "Cluster 2"
      )
  )

View(data_bulanan_produk)

#====================================================
# 7. LIHAT TOTAL CLUSTER
#====================================================
rekap_cluster <- data_bulanan_produk %>%
  
  group_by(cluster) %>%
  
  summarise(
    total_keluar =
      sum(total_keluar)
  )

View(rekap_cluster)

#====================================================
# 8. FILTER CLUSTER 1
#====================================================
data_cluster1 <- data_bulanan_produk %>%
  
  filter(cluster == "Cluster 1")

View(data_cluster1)

#====================================================
# 9. SPLIT PER PRODUK
#====================================================
list_produk <- split(
  data_cluster1,
  data_cluster1$id_produk
)

#====================================================
# 10. FORECAST HOLT WINTERS
#====================================================
hasil_forecast <- map_df(
  
  list_produk,
  
  function(df_produk){
    
    # urutkan bulan
    df_produk <- df_produk %>%
      arrange(bulan)
    
    # minimal 6 data
    if(nrow(df_produk) < 6)
      return(NULL)
    
    # time series
    ts_data <- ts(
      df_produk$total_keluar,
      frequency = 6
    )
    
    # MODEL HOLT WINTERS
    model_hw <- HoltWinters(
      ts_data,
      gamma = FALSE
    )
    
    # FORECAST 6 BULAN
    fc <- forecast(
      model_hw,
      h = 6
    )
    
    # HASIL
    data.frame(
      
      id_produk =
        unique(df_produk$id_produk),
      
      bulan_ke = 1:6,
      
      prediksi =
        ceiling(
          pmax(
            0,
            as.numeric(fc$mean)
          )
        )
    )
  }
)

#====================================================
# 11. LIHAT HASIL
#====================================================
View(hasil_forecast)

#====================================================
# 12. TOTAL FORECAST
#====================================================
total_forecast <- hasil_forecast %>%
  
  group_by(id_produk) %>%
  
  summarise(
    total_prediksi =
      sum(prediksi)
  ) %>%
  
  arrange(
    desc(total_prediksi)
  )

View(total_forecast)

#====================================================
# 13. TOP 10 PRODUK
#====================================================
top10 <- total_forecast %>%
  
  slice(1:10)

View(top10)

#====================================================
# 14. GRAFIK
#====================================================
ggplot(
  top10,
  
  aes(
    x = reorder(
      id_produk,
      total_prediksi
    ),
    
    y = total_prediksi
  )
) +
  
  geom_col() +
  
  coord_flip() +
  
  labs(
    title =
      "Forecast Produk Cluster 1",
    
    x = "Produk",
    
    y = "Forecast"
  ) +
  
  theme_minimal()
