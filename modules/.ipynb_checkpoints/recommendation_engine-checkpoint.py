import pandas as pd

def generate_recommendations(predicted_prices, current_prices, health_status, feature_importances=None):
    """
    Hasilkan rekomendasi saham berdasarkan prediksi, kesehatan, dan kepentingan fitur
    
    Parameters:
        predicted_prices (Series): Harga prediksi
        current_prices (Series): Harga saat ini
        health_status (Series): Status kesehatan bank
        feature_importances (DataFrame): Kepentingan fitur per bank
        
    Returns:
        DataFrame: Rekomendasi untuk setiap bank
    """
    recommendations = []
    for bank in current_prices.index:
        pred = predicted_prices.get(bank, 0)
        current = current_prices[bank]
        health = health_status.get(bank, 'Tidak Sehat')
        
        # Hitung perubahan harga yang diprediksi
        price_change = (pred - current) / current if current > 0 else 0
        
        # Logika rekomendasi dasar
        if health == 'Sehat':
            if price_change > 0.05:  # Prediksi naik >5%
                rec = 'BELI'
                reason = "Bank sehat dengan potensi kenaikan harga >5%"
            elif price_change < -0.05:  # Prediksi turun >5%
                rec = 'JUAL'
                reason = "Bank sehat tapi prediksi penurunan harga >5%"
            else:
                rec = 'TAHAN'
                reason = "Bank sehat dengan harga stabil"
        else:  # Bank tidak sehat
            if price_change > 0.03:
                rec = 'TAHAN'
                reason = "Bank tidak sehat tapi prediksi kenaikan harga >3%"
            elif price_change > -0.03:
                rec = 'TAHAN'
                reason = "Bank tidak sehat dengan perubahan harga kecil"
            else:
                rec = 'JUAL'
                reason = "Bank tidak sehat dengan prediksi penurunan harga"
        
        # Penyesuaian berdasarkan fitur yang tidak berpengaruh
        if feature_importances is not None and bank in feature_importances.index:
            bank_importances = feature_importances.loc[bank]
            # Ambil fitur yang kepentingannya di bawah 0.05
            insignificant_features = [feat for feat, imp in bank_importances.items() if imp < 0.05]
            
            if insignificant_features:
                reason += f". Perhatian: fitur {', '.join(insignificant_features)} memiliki pengaruh kecil"
                if rec == 'BELI':
                    rec = 'HATI-HATI BELI'
                elif rec == 'TAHAN':
                    rec = 'AWASI'
        
        recommendations.append({
            'Bank': bank,
            'Harga Saat Ini': current,
            'Harga Prediksi': pred,
            'Perubahan Prediksi (%)': price_change * 100,
            'Status Kesehatan': health,
            'Rekomendasi': rec,
            'Alasan': reason
        })
    
    return pd.DataFrame(recommendations)