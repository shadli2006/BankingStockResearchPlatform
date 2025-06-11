from fpdf import FPDF
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64

def create_pdf_report(raw_df, results_df, weights, return_val, risk_val, method, recommendations, health_status):
    """Buat laporan PDF dari hasil analisis"""
    pdf = FPDF()
    pdf.add_page()
    
    # Header
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Laporan Analisis Saham Bank Swasta", ln=True, align='C')
    pdf.ln(10)
    
    # 1. Ringkasan Eksekutif
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Ringkasan Eksekutif", ln=True)
    pdf.set_font("Arial", '', 12)
    pdf.multi_cell(0, 8, """
    Laporan ini menyajikan analisis komprehensif saham bank swasta yang terdaftar di BEI 
    menggunakan model AI dan optimasi portofolio. Analisis meliputi:
    - Pengaruh rasio keuangan terhadap harga saham
    - Prediksi harga saham dengan Random Forest
    - Optimasi portofolio menggunakan {}
    - Rekomendasi alokasi investasi
    - Analisis kesehatan bank dan rekomendasi saham
    """.format(method))
    pdf.ln(5)
    
    # 2. Hasil Prediksi
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Hasil Prediksi Harga Saham", ln=True)
    
    # Grafik prediksi
    fig, ax = plt.subplots(figsize=(10, 6))
    for bank in results_df['Bank'].unique():
        bank_data = results_df[results_df['Bank'] == bank]
        ax.plot(bank_data['Date'], bank_data['StockPrice'], label=f'{bank} Aktual')
        ax.plot(bank_data['Date'], bank_data['PredictedPrice'], '--', label=f'{bank} Prediksi')
    ax.legend()
    ax.set_title("Perbandingan Harga Aktual vs Prediksi")
    
    img_bytes = plot_to_bytes(fig)
    pdf.image(img_bytes, x=10, y=30, w=190)
    pdf.ln(120)
    
    # 3. Optimasi Portofolio
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Optimasi Portofolio", ln=True)
    pdf.set_font("Arial", '', 12)
    pdf.cell(200, 10, txt=f"Metode: {method}", ln=True)
    pdf.cell(200, 10, txt=f"Return Diharapkan: {return_val:.2%}", ln=True)
    pdf.cell(200, 10, txt=f"Risiko Portofolio: {risk_val:.2%}", ln=True)
    pdf.ln(5)
    
    # Tabel bobot portofolio
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Alokasi Portofolio Optimal:", ln=True)
    pdf.set_font("Arial", '', 12)
    for bank, weight in weights.items():
        pdf.cell(200, 10, txt=f"- {bank}: {weight:.2%}", ln=True)
    pdf.ln(10)
    
    # Grafik alokasi
    fig, ax = plt.subplots()
    ax.pie(weights.values(), labels=weights.keys(), autopct='%1.1f%%')
    ax.set_title("Alokasi Portofolio")
    
    img_bytes = plot_to_bytes(fig)
    pdf.image(img_bytes, x=60, y=100, w=90)
    
    # 4. Analisis Kesehatan Bank
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Analisis Kesehatan Bank", ln=True)
    pdf.set_font("Arial", '', 12)
    
    pdf.cell(200, 10, txt="Status Kesehatan Bank:", ln=True)
    for bank, status in health_status.items():
        pdf.cell(200, 10, txt=f"- {bank}: {status}", ln=True)
    pdf.ln(10)
    
    # 5. Rekomendasi Saham
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Rekomendasi Saham", ln=True)
    pdf.set_font("Arial", '', 12)
    
    for _, row in recommendations.iterrows():
        pdf.multi_cell(0, 8, 
                      f"- {row['Bank']}: {row['Rekomendasi']} (Harga Saat Ini: {row['Harga Saat Ini']:.2f}, Prediksi: {row['Harga Prediksi']:.2f}, Perubahan: {row['Perubahan Prediksi (%)']:.2f}%) - {row['Alasan']}")
    pdf.ln(5)
    
    # 6. Uji Hipotesis
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Hasil Uji Hipotesis", ln=True)
    pdf.set_font("Arial", '', 12)
    
    hipotesis = [
        "H₁: ROA berpengaruh positif terhadap fluktuasi harga saham",
        "H₂: ROE berpengaruh negatif terhadap fluktuasi harga saham",
        "H₃: NIM berpengaruh positif terhadap fluktuasi harga saham",
        "H₄: NPL berpengaruh negatif terhadap fluktuasi harga saham",
        "H₅: BOPO berpengaruh negatif terhadap fluktuasi harga saham",
        "H₆: CAR berpengaruh positif terhadap fluktuasi harga saham",
        "H₇: Model Random Forest lebih akurat dalam prediksi harga saham",
        "H₈: Kombinasi Markowitz + Genetic Algorithm menghasilkan return lebih tinggi"
    ]
    
    for i, h in enumerate(hipotesis):
        pdf.cell(200, 10, txt=f"{i+1}. {h}", ln=True)
    
    # 7. Interpretasi Hasil Hipotesis
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Interpretasi Hasil Hipotesis", ln=True)
    pdf.set_font("Arial", '', 12)
    pdf.multi_cell(0, 8, """
    1. Hipotesis yang didukung: Rasio keuangan tersebut memiliki pengaruh signifikan terhadap harga saham
    2. Hipotesis yang tidak didukung: Hubungan yang dihipotesiskan tidak terbukti dalam data
    3. Tidak signifikan: Tidak ada bukti statistik yang cukup untuk menyimpulkan hubungan
    """)
    
    # Simpan PDF
    return pdf.output(dest='S').encode('latin1')

def plot_to_bytes(fig):
    """Konversi matplotlib figure ke bytes"""
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return buf