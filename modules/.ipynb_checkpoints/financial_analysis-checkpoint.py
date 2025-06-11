import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def analyze_financial_impact(df, features, target):
    """Analisis pengaruh rasio keuangan terhadap harga saham"""
    if df.empty or not features:
        st.warning("⚠️ Tidak ada data yang cukup untuk analisis")
        return
    
    st.subheader("Analisis Korelasi Rasio Keuangan")
    corr_matrix = df[features + [target]].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    st.subheader("Uji Hipotesis")
    hypotheses = [
        ("ROA", "positif", "H₁: ROA berpengaruh positif terhadap harga saham"),
        ("ROE", "negatif", "H₂: ROE berpengaruh negatif terhadap harga saham"),
        ("NIM", "positif", "H₃: NIM berpengaruh positif terhadap harga saham"),
        ("NPL", "negatif", "H₄: NPL berpengaruh negatif terhadap harga saham"),
        ("BOPO", "negatif", "H₅: BOPO berpengaruh negatif terhadap harga saham"),
        ("CAR", "positif", "H₆: CAR berpengaruh positif terhadap harga saham")
    ]
    
    results = []
    for feature, expected, hypothesis in hypotheses:
        if feature in df.columns:
            corr, p_value = pearsonr(df[feature], df[target])
            
            # Tentukan signifikansi
            significance = "Signifikan" if p_value < 0.05 else "Tidak signifikan"
            
            # Tentukan arah korelasi
            direction = "positif" if corr > 0 else "negatif"
            
            # Tentukan apakah hipotesis didukung
            if p_value >= 0.05:
                supported = "Tidak dapat disimpulkan (tidak signifikan)"
            else:
                supported = "Ya" if direction == expected else "Tidak"
            
            results.append({
                "Hipotesis": hypothesis,
                "Korelasi": f"{corr:.4f}",
                "Arah": direction,
                "Signifikansi": significance,
                "Didukung": supported
            })
    
    st.table(pd.DataFrame(results))


    
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
