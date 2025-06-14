import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

def train_and_predict_rf(df, features, target):
    """Latih model RF dan lakukan prediksi"""
    if df.empty or len(features) == 0:
        st.warning("⚠️ Tidak ada data yang cukup untuk pelatihan model")
        return pd.DataFrame()
    
    # Bersihkan data secara global
    df_clean = df.copy()
    for col in features + [target]:
        # Ganti Inf dengan NaN
        df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)
        # Drop baris dengan NaN
        df_clean = df_clean.dropna(subset=[col])
    
    df = df_clean
    
    results = []
    all_importances = []
    evaluation_results = []
    
    for bank in df['Bank'].unique():
        bank_df = df[df['Bank'] == bank].copy()
        if len(bank_df) < 10:
            continue
            
        # Bersihkan data per bank
        bank_df_clean = bank_df.dropna(subset=features + [target])
        if len(bank_df_clean) < 10:
            st.warning(f"⚠️ Bank {bank} tidak memiliki cukup data setelah pembersihan")
            continue
            
        X = bank_df_clean[features].values
        y = bank_df_clean[target].values
        
        # Penanganan nilai yang hilang
        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(X)
        
        # Normalisasi fitur
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Latih model Random Forest
        rf_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        rf_mse = mean_squared_error(y_test, rf_pred)
        rf_r2 = r2_score(y_test, rf_pred)
        
        # Latih model Linear Regression untuk perbandingan
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        lr_pred = lr_model.predict(X_test)
        lr_mse = mean_squared_error(y_test, lr_pred)
        lr_r2 = r2_score(y_test, lr_pred)
        
        evaluation_results.append({
            'Bank': bank,
            'RF_MSE': rf_mse,
            'RF_R2': rf_r2,
            'LR_MSE': lr_mse,
            'LR_R2': lr_r2
        })
        
        # Prediksi dengan seluruh data untuk plotting
        y_pred = rf_model.predict(X_scaled)
        
        # Simpan hasil
        bank_df_clean['PredictedPrice'] = y_pred
        bank_df_clean['Error'] = bank_df_clean[target] - bank_df_clean['PredictedPrice']
        results.append(bank_df_clean)
        
        # Simpan importance
        importances = pd.Series(rf_model.feature_importances_, index=features)
        all_importances.append(importances)
    
    # Gabungkan hasil semua bank
    if not results:
        return pd.DataFrame()
    
    result_df = pd.concat(results)
    
    # Tampilkan feature importance
    if all_importances:
        st.subheader("Analisis Kepentingan Fitur")
        importance_df = pd.concat(all_importances, axis=1).T
        importance_df['Bank'] = df['Bank'].unique()[:len(importance_df)]
        
        # Rata-rata kepentingan fitur
        avg_importance = importance_df.groupby('Bank').mean().mean(axis=0)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        avg_importance.sort_values().plot.barh(ax=ax)
        ax.set_title("Rata-rata Kepentingan Fitur")
        ax.set_xlabel("Tingkat Kepentingan")
        st.pyplot(fig)
        
        # Analisis hipotesis berdasarkan feature importance
        st.subheader("Analisis Hipotesis Berdasarkan Feature Importance")
        hypothesis_features = ['ROA', 'ROE', 'NIM', 'NPL', 'BOPO', 'CAR']
        filtered_importance = avg_importance[avg_importance.index.isin(hypothesis_features)]
        
        if not filtered_importance.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            filtered_importance.sort_values().plot.barh(ax=ax)
            ax.set_title("Kepentingan Fitur untuk Hipotesis")
            ax.set_xlabel("Tingkat Kepentingan")
            st.pyplot(fig)
            
            # Interpretasi
            st.info("Interpretasi:")
            st.write("- Fitur dengan kepentingan tinggi mendukung hipotesis terkait")
            st.write("- Fitur dengan kepentingan rendah (<0.05) menunjukkan pengaruh kecil terhadap prediksi harga saham")
            
            # Identifikasi fitur yang tidak berpengaruh
            insignificant_features = filtered_importance[filtered_importance < 0.05].index.tolist()
            if insignificant_features:
                st.warning(f"⚠️ Fitur berikut memiliki pengaruh kecil terhadap prediksi: {', '.join(insignificant_features)}")
    
    # Tampilkan evaluasi model
    st.subheader("Evaluasi Model (H₇)")
    if evaluation_results:
        eval_df = pd.DataFrame(evaluation_results)
        st.write("**Perbandingan Akurasi Model:**")
        st.dataframe(eval_df)
        
        # Kesimpulan
        avg_rf_mse = eval_df['RF_MSE'].mean()
        avg_lr_mse = eval_df['LR_MSE'].mean()
        st.success(f"Random Forest lebih akurat (MSE lebih rendah) daripada Linear Regression: {'Ya' if avg_rf_mse < avg_lr_mse else 'Tidak'}")
    
    return result_df