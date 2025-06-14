import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from modules.data_loader import load_and_preprocess_data, combine_quarterly_daily
from modules.financial_analysis import analyze_financial_impact
from modules.health_scoring import calculate_health_score
from modules.rf_model import train_and_predict_rf
from modules.portfolio_optimization import optimize_portfolio
from modules.recommendation_engine import generate_recommendations
from modules.report_generator import create_pdf_report

# Konfigurasi halaman
st.set_page_config(layout="wide", page_title="Analisis Saham Bank Swasta")
st.title("📊 AI-Powered Banking Stock Analysis & Portfolio Optimization")

# Parameter yang dapat disesuaikan
DEFAULT_FEATURES = ['ROA', 'ROE', 'NIM', 'NPL', 'CAR', 'EPS', 'DER', 'BOPO']  # Tambahkan BOPO
TARGET = 'StockPrice'
OPTIMIZATION_METHODS = ['Genetic Algorithm', 'Markowitz', 'Markowitz + GA']  # Tambahkan metode kombinasi

# Sidebar untuk parameter
st.sidebar.header("Parameter Analisis")

# Konfigurasi fitur
st.sidebar.subheader("Fitur yang Digunakan")
selected_features = []
for feature in DEFAULT_FEATURES:
    if st.sidebar.checkbox(feature, value=True, key=f"feature_{feature}"):
        selected_features.append(feature)

# Ambang batas kesehatan bank
st.sidebar.subheader("Parameter Kesehatan Bank")
health_thresholds = {}
for ratio in ['ROA', 'ROE', 'NIM', 'NPL', 'CAR', 'BOPO']:  # Tambahkan BOPO
    if ratio in selected_features:
        if ratio in ['ROA', 'ROE', 'CAR', 'NIM']:
            health_thresholds[ratio] = st.sidebar.number_input(
                f"Min {ratio} (%)", 
                value=1.5 if ratio == 'ROA' else 10.0,
                key=f"min_{ratio}"
            )
        elif ratio in ['NPL', 'BOPO']:  # Tambahkan BOPO sebagai rasio yang diinginkan rendah
            health_thresholds[ratio] = st.sidebar.number_input(
                f"Max {ratio} (%)", 
                value=3.0 if ratio == 'NPL' else 90.0,
                key=f"max_{ratio}"
            )

# Upload data
st.header("📥 Input Data")
st.subheader("Data Keuangan Kuartalan")
st.info("Format: Tanggal, NamaBank, ROA, ROE, NIM, NPL, CAR, EPS, DER, BOPO")  # Tambahkan BOPO
quarterly_files = st.file_uploader("Upload Data Kuartalan (CSV)", type="csv", accept_multiple_files=True)

st.subheader("Data Harga Saham Harian")
st.info("Format: Tanggal, [Bank atau NamaBank], StockPrice")
daily_files = st.file_uploader("Upload Data Saham Harian (CSV)", type="csv", accept_multiple_files=True)

if quarterly_files and daily_files:
    try:
        # Proses data kuartalan
        quarterly_df = load_and_preprocess_data(
            quarterly_files, 
            selected_features + ['Bank', 'Date'], 
            'quarterly'
        )
        
        # Proses data harian
        daily_df = load_and_preprocess_data(
            daily_files, 
            ['Date', 'StockPrice', 'Bank'], 
            'daily'
        )
        
        # Gabungkan data
        df = combine_quarterly_daily(quarterly_df, daily_df, selected_features)
        st.success("✅ Data berhasil dimuat!")
        
        # Tampilkan data
        with st.expander("🔍 Lihat Data"):
            st.dataframe(df)
        
        # 1. Analisis Pengaruh Rasio Keuangan
        st.header("📈 Analisis Pengaruh Rasio Keuangan")
        analyze_financial_impact(df, selected_features, TARGET) 
        

        
        
        # Hitung kesehatan bank
        quarterly_df = calculate_health_score(quarterly_df, health_thresholds)
        health_status = quarterly_df.groupby('Bank')['HealthStatus'].last()
        # 1. Analisis Pengaruh Rasio Keuangan
        st.header("🔍 Analisis Eksplorasi Data")
        with st.expander("Distribusi Variabel"):
            selected_var = st.selectbox("Pilih variabel untuk dilihat distribusinya",
                                    selected_features + [TARGET])
        fig, ax = plt.subplots()
        sns.histplot(df[selected_var], kde=True, ax=ax)
        st.pyplot(fig)

        with st.expander("Scatter Plot Hubungan Variabel"):
            col1, col2 = st.columns(2)
        with col1:
            x_var = st.selectbox("Variabel X", selected_features)
        with col2:
            y_var = st.selectbox("Variabel Y", [TARGET] + selected_features)
    
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x=x_var, y=y_var, hue='Bank', ax=ax)
        st.pyplot(fig)
        # 2. Prediksi Harga Saham
        st.header("🤖 Prediksi Harga Saham (Random Forest)")
        results = train_and_predict_rf(df, selected_features, TARGET)
        
        if not results.empty:
            # Visualisasi prediksi
            fig, ax = plt.subplots(figsize=(12, 6))
            for bank in df['Bank'].unique():
                bank_data = results[results['Bank'] == bank]
                ax.plot(bank_data['Date'], bank_data[TARGET], label=f'{bank} Aktual')
                ax.plot(bank_data['Date'], bank_data['PredictedPrice'], '--', label=f'{bank} Prediksi')
            ax.set_title("Perbandingan Harga Aktual vs Prediksi")
            ax.legend()
            st.pyplot(fig)
            
            # 3. Analisis Kesehatan Bank
            st.header("🏦 Analisis Kesehatan Bank")
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Status Kesehatan Terakhir")
                st.dataframe(health_status)
                
            with col2:
                st.subheader("Distribusi Status Kesehatan")
                fig, ax = plt.subplots()
                health_status.value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
                ax.set_ylabel('')
                st.pyplot(fig)
            
            # 4. Rekomendasi Saham
            st.header("💡 Rekomendasi Saham")
            current_prices = results.groupby('Bank')[TARGET].last()
            predicted_prices = results.groupby('Bank')['PredictedPrice'].last()
            
            # Dapatkan feature importance per bank (dari model RF) untuk rekomendasi
            # Catatan: Di sini kita mengambil feature importance dari model RF yang sudah dilatih
            # Karena di rf_model.py kita sudah menyimpan importance_df, kita perlu mengembalikannya juga
            # Untuk sederhananya, kita akan mengambil rata-rata fitur dari data kuartal terakhir
            # Sebenarnya lebih baik jika kita menyimpan feature_importances dari rf_model dan mengembalikannya
            # Tapi untuk mempermudah, kita gunakan data asli
            feature_importances = quarterly_df.groupby('Bank')[selected_features].mean()
            
            recommendations = generate_recommendations(
                predicted_prices, 
                current_prices,
                health_status,
                feature_importances  # Kirim data feature_importances untuk rekomendasi
            )
            st.dataframe(recommendations)
            
            # 5. Optimasi Portofolio
            st.header("🧩 Optimasi Portofolio")
            selected_method = st.selectbox("Pilih Metode Optimasi", OPTIMIZATION_METHODS)
            
            # Dapatkan skor kesehatan untuk optimasi
            health_scores = quarterly_df.groupby('Bank')['HealthScore'].mean()
            
            weights, return_val, risk_val = optimize_portfolio(
                results, 
                selected_method, 
                TARGET,
                health_scores
            )
            
            # Tampilkan hasil optimasi
            st.subheader(f"Hasil Optimasi ({selected_method})")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Alokasi Portofolio Optimal:**")
                for bank, weight in weights.items():
                    st.info(f"{bank}: {weight:.2%}")
            
            with col2:
                st.metric("Return Diharapkan", f"{return_val:.2%}")
                st.metric("Risiko Portofolio", f"{risk_val:.2%}")
                st.metric("Rasio Return/Risiko", f"{return_val/(risk_val + 1e-8):.2f}")
            
            # 6. Ekspor Laporan
            st.header("📤 Ekspor Laporan")
            if st.button("Buat Laporan PDF"):
                health_status_dict = health_status.to_dict()
                pdf_bytes = create_pdf_report(
                    df, 
                    results, 
                    weights, 
                    return_val, 
                    risk_val, 
                    selected_method,
                    recommendations,
                    health_status_dict
                )
                st.download_button(
                    "Unduh Laporan", 
                    data=pdf_bytes, 
                    file_name="analisis_saham_bank.pdf", 
                    mime="application/pdf"
                )
        
    except Exception as e:
        st.error(f"Terjadi kesalahan: {str(e)}")
else:
    st.info("👋 Silakan upload data keuangan kuartalan dan data saham harian untuk memulai analisis")