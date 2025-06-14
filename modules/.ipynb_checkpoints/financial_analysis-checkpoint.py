import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.diagnostic import linear_harvey_collier
from linearmodels.panel import compare
import numpy as np
from scipy.stats import pearsonr, f  # Menambahkan modul f untuk uji Chow
from linearmodels.panel import compare, PanelOLS, RandomEffects  # Menambahkan PanelOLS dan RandomEffects


def perform_regression_analysis(df, features, target):
    """Melakukan analisis regresi lengkap dengan berbagai uji statistik"""
    st.subheader("📊 Analisis Regresi Lengkap")
    
    # Persiapan data
    X = df[features]
    y = df[target]
    X = sm.add_constant(X)  # Tambahkan intercept
    
    # Fit model OLS
    model = sm.OLS(y, X).fit()
    
    # Tampilkan summary regresi
    st.text(str(model.summary()))
    
    # Visualisasi residual
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(model.resid, kde=True, ax=axes[0])
    axes[0].set_title('Distribusi Residual')
    sm.qqplot(model.resid, line='s', ax=axes[1])
    axes[1].set_title('Q-Q Plot Residual')
    st.pyplot(fig)
    
    # Uji Multikolinearitas (VIF)
    st.subheader("Uji Multikolinearitas (VIF)")
    vif_data = pd.DataFrame()
    vif_data["Variabel"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    st.dataframe(vif_data.style.highlight_between(subset="VIF", left=5, right=10, color="orange"))
    
    # Uji Heteroskedastisitas
    st.subheader("Uji Heteroskedastisitas")
    labels = ['LM Statistic', 'LM-Test p-value', 'F-Statistic', 'F-Test p-value']
    
    # Breusch-Pagan test
    bp_test = het_breuschpagan(model.resid, model.model.exog)
    bp_result = pd.DataFrame([bp_test], columns=labels, index=['Breusch-Pagan'])
    
    # White test
    white_test = het_white(model.resid, model.model.exog)
    white_result = pd.DataFrame([white_test], columns=labels, index=['White'])
    
    st.dataframe(pd.concat([bp_result, white_result]))
    
    # Uji Chow untuk stabilitas parameter
    st.subheader("Uji Chow untuk Stabilitas Parameter")
    try:
        chow_result = linear_harvey_collier(model)
        st.write(f"Chow Test Statistic: {chow_result[0]:.4f}")
        st.write(f"P-value: {chow_result[1]:.4f}")
        st.write("Interpretasi: Model stabil" if chow_result[1] > 0.05 else "Interpretasi: Model tidak stabil")
    except Exception as e:
        st.warning(f"Tidak dapat melakukan uji Chow: {str(e)}")
    
    return model

#def analyze_financial_impact(df, features, target):
#    """Analisis pengaruh rasio keuangan terhadap harga saham"""
#    if df.empty or not features:
#        st.warning("⚠️ Tidak ada data yang cukup untuk analisis")
#        return
    
    # Analisis regresi lengkap
#    regression_model = perform_regression_analysis(df, features, target)
    
    # Analisis korelasi (existing code)
#    st.subheader("Analisis Korelasi Rasio Keuangan")
#    corr_matrix = df[features + [target]].corr()
    
#    fig, ax = plt.subplots(figsize=(10, 8))
#    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
#    st.pyplot(fig)



def analyze_financial_impact(df, features, target):
    """Analisis pengaruh rasio keuangan terhadap harga saham"""
    if df.empty or not features:
        st.warning("⚠️ Tidak ada data yang cukup untuk analisis")
        return


    df_clean = df.copy()
    for col in features + [target]:
        # Ganti Inf dengan NaN
        df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)
        # Drop baris dengan NaN
        df_clean = df_clean.dropna(subset=[col])
    
    df = df_clean
    
    # Analisis regresi lengkap
  #  regression_model = perform_regression_analysis(df, features, target)

    
    # 1. Analisis Korelasi
    st.subheader("Analisis Korelasi Rasio Keuangan")
    corr_matrix = df[features + [target]].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
    st.pyplot(fig)
    
    # 2. Analisis Regresi Linier Berganda
    st.subheader("Analisis Regresi Linier Berganda")
    
    # Persiapkan data
    X = df[features]
    y = df[target]
    X = sm.add_constant(X)  # Tambahkan konstanta
    
    # Hilangkan missing values
    data = X.join(y).dropna()
    if len(data) < 2:
        st.warning("⚠️ Tidak cukup data untuk analisis regresi")
        return
    
    X_clean = data[X.columns]
    y_clean = data[target]
    
    # Latih model
    model = sm.OLS(y_clean, X_clean)
    results = model.fit()
    
    # Tampilkan hasil regresi
    st.write(results.summary())
    
    # Buat tabel hasil regresi yang lebih rapi
    st.subheader("Ringkasan Hasil Regresi")
    regression_data = {
        'Variabel': X_clean.columns,
        'Koefisien': results.params,
        'Std Error': results.bse,
        't-stat': results.tvalues,
        'p-value': results.pvalues
    }
    regression_df = pd.DataFrame(regression_data)
    st.table(regression_df)
    
    # Tampilkan R-squared
    col1, col2 = st.columns(2)
    col1.metric("R-squared", f"{results.rsquared:.4f}")
    col2.metric("Adjusted R-squared", f"{results.rsquared_adj:.4f}")
    
    # 3. Uji Asumsi Klasik
    st.subheader("Uji Asumsi Klasik Regresi")
    
    # Uji Multikolinearitas (VIF)
    st.write("**Uji Multikolinearitas (VIF):**")
    vif_data = []
    for i in range(1, len(features) + 1):  # Lewati konstanta
        vif = variance_inflation_factor(X_clean.values, i)
        vif_data.append({'Variabel': features[i-1], 'VIF': vif})
    
    vif_df = pd.DataFrame(vif_data)
    st.table(vif_df)
    st.caption("VIF > 10 menunjukkan multikolinearitas tinggi")
    
    # Uji Heteroskedastisitas
    st.write("**Uji Heteroskedastisitas:**")
    # Breusch-Pagan
    bp_test = het_breuschpagan(results.resid, results.model.exog)
    bp_df = pd.DataFrame({
        'Statistik': ['LM-statistic', 'p-value', 'F-statistic', 'F p-value'],
        'Nilai': bp_test
    })
    st.table(bp_df)
    st.caption("H0: Homoskedastisitas (p-value < 0.05 berarti heteroskedastisitas)")
    
    # White test
    white_test = het_white(results.resid, results.model.exog)
    white_df = pd.DataFrame({
        'Statistik': ['LM-statistic', 'p-value', 'F-statistic', 'F p-value'],
        'Nilai': white_test
    })
    st.table(white_df)
    
    # 4. Uji Hipotesis
    st.subheader("Uji Hipotesis")
    
    # Uji t (parsial)
    st.write("**Uji Hipotesis Parsial (Uji t):**")
    for i, feature in enumerate(features):
        p_value = results.pvalues[feature]
        sign = "signifikan" if p_value < 0.05 else "tidak signifikan"
        effect = "positif" if results.params[feature] > 0 else "negatif"
        st.write(f"- {feature}: Pengaruh {effect} {sign} (p={p_value:.4f})")
    
    # Uji F (simultan)
    st.write(f"**Uji Hipotesis Simultan (Uji F):**")
    st.write(f"F-statistic: {results.fvalue:.4f}, p-value: {results.f_pvalue:.4f}")
    st.caption("H0: Semua koefisien = 0 (tidak ada pengaruh)")
    
    # 5. Uji Chow untuk stabilitas parameter (Perbaikan 3: Gunakan modul f yang sudah diimport)
    try:
        st.write("**Uji Chow untuk Stabilitas Parameter:**")
        mid_point = len(X_clean) // 2
        X1, y1 = X_clean.iloc[:mid_point], y_clean.iloc[:mid_point]
        X2, y2 = X_clean.iloc[mid_point:], y_clean.iloc[mid_point:]
        
        model1 = sm.OLS(y1, X1).fit()
        model2 = sm.OLS(y2, X2).fit()
        
        k = len(features) + 1
        rss_pooled = results.ssr
        rss1 = model1.ssr
        rss2 = model2.ssr
        
        chow_stat = ((rss_pooled - (rss1 + rss2)) / k / ((rss1 + rss2) / (len(X_clean) - 2*k)))
        p_value = 1 - f.cdf(chow_stat, k, len(X_clean) - 2*k)  # Perbaikan di sini
        
        st.write(f"Chow Statistic: {chow_stat:.4f}, p-value: {p_value:.4f}")
        st.caption("H0: Parameter stabil (tidak ada perubahan struktural)")
    except Exception as e:
        st.warning(f"Tidak dapat melakukan uji Chow: {str(e)}")
    
    # 6. Uji Hausman (Perbaikan 4: PanelOLS dan RandomEffects sudah diimport)
    if 'Bank' in df.columns and 'Date' in df.columns:
        try:
            st.write("**Uji Hausman untuk Pemilihan Model Panel:**")
            panel_df = df.copy().set_index(['Bank', 'Date'])
            
            # Pastikan variabel dependen dan independen tidak mengandung NaN
            panel_data = panel_df[[target] + features].dropna()
            
            if not panel_data.empty:
                y_panel = panel_data[target]
                X_panel = panel_data[features]
                
                fe_model = PanelOLS(y_panel, X_panel, entity_effects=True).fit()
                re_model = RandomEffects(y_panel, X_panel).fit()
                
                hausman_result = compare({"FE": fe_model, "RE": re_model})
                st.write(hausman_result)
                st.caption("H0: Random effects lebih efisien")
            else:
                st.warning("Tidak cukup data untuk uji Hausman setelah pembersihan")
        except Exception as e:
            st.warning(f"Tidak dapat melakukan uji Hausman: {str(e)}")
    
    # 7. Analisis per Rasio (kode asli)
    st.subheader("Analisis Pengaruh Per Rasio")
    selected_ratio = st.selectbox("Pilih Rasio Keuangan", features)
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Korelasi dengan {target}:**")
        corr, p_value = pearsonr(df[selected_ratio], df[target])
        st.metric("Koefisien Korelasi", f"{corr:.2f}")
        st.metric("Signifikansi Statistik", "Signifikan" if p_value < 0.05 else "Tidak Signifikan")
    
    with col2:
        fig, ax = plt.subplots()
        sns.regplot(x=selected_ratio, y=target, data=df, ax=ax)
        ax.set_title(f"Hubungan {selected_ratio} vs {target}")
        st.pyplot(fig)

        
    
#    st.subheader("Uji Hipotesis")
#    hypotheses = [
#        ("ROA", "positif", "H₁: ROA berpengaruh positif terhadap harga saham"),
#        ("ROE", "negatif", "H₂: ROE berpengaruh negatif terhadap harga saham"),
#        ("NIM", "positif", "H₃: NIM berpengaruh positif terhadap harga saham"),
#        ("NPL", "negatif", "H₄: NPL berpengaruh negatif terhadap harga saham"),
#        ("BOPO", "negatif", "H₅: BOPO berpengaruh negatif terhadap harga saham"),
#        ("CAR", "positif", "H₆: CAR berpengaruh positif terhadap harga saham")
#    ]
    
#    results = []
#    for feature, expected, hypothesis in hypotheses:
#        if feature in df.columns:
#            corr, p_value = pearsonr(df[feature], df[target])
            
#            # Tentukan signifikansi
#            significance = "Signifikan" if p_value < 0.05 else "Tidak signifikan"
            
            # Tentukan arah korelasi
#            direction = "positif" if corr > 0 else "negatif"
            
            # Tentukan apakah hipotesis didukung
#            if p_value >= 0.05:
#                supported = "Tidak dapat disimpulkan (tidak signifikan)"
#            else:
#                supported = "Ya" if direction == expected else "Tidak"
            
#            results.append({
#                "Hipotesis": hypothesis,
#                "Korelasi": f"{corr:.4f}",
#                "Arah": direction,
#                "Signifikansi": significance,
#                "Didukung": supported
#           })
    
#    st.table(pd.DataFrame(results))
    
    st.subheader("Analisis Tren Per Bank")
    selected_bank = st.selectbox("Pilih Bank", df['Bank'].unique())
    bank_df = df[df['Bank'] == selected_bank]
    
    if not bank_df.empty:
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Plot harga saham
        color = 'tab:blue'
        ax1.set_xlabel('Tanggal')
        ax1.set_ylabel('Harga Saham', color=color)
        ax1.plot(bank_df['Date'], bank_df[target], color=color, label='Harga Saham')
        ax1.tick_params(axis='y', labelcolor=color)
        
        # Plot rasio keuangan
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Rasio Keuangan', color=color)
        for feature in features:
            ax2.plot(bank_df['Date'], bank_df[feature], '--', label=feature)
        ax2.tick_params(axis='y', labelcolor=color)
        
        fig.tight_layout()
        fig.legend(loc='upper left')
        st.pyplot(fig)