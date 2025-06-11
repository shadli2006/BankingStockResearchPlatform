import pandas as pd

def calculate_health_score(df, thresholds):
    """
    Hitung skor kesehatan bank berdasarkan parameter yang ditentukan
    
    Parameters:
        df (DataFrame): Data rasio keuangan
        thresholds (dict): Ambang batas untuk setiap rasio
        
    Returns:
        DataFrame: Data dengan kolom tambahan HealthScore dan HealthStatus
    """

    # Konversi kolom rasio ke tipe numerik
    for ratio in thresholds.keys():
        if ratio in df.columns:
            df[ratio] = pd.to_numeric(df[ratio], errors='coerce')
   
    scored_df = df.copy()
    scored_df['HealthScore'] = 0
    
    # Hitung skor berdasarkan threshold
    for ratio, threshold in thresholds.items():
        if ratio in scored_df.columns:
            # Untuk rasio yang diinginkan tinggi (ROA, ROE, CAR, NIM)
            if ratio in ['ROA', 'ROE', 'CAR', 'NIM']:
                scored_df[ratio + '_Status'] = scored_df[ratio] >= threshold
                scored_df['HealthScore'] += scored_df[ratio + '_Status'].astype(int)
            # Untuk rasio yang diinginkan rendah (NPL, BOPO)
            elif ratio in ['NPL', 'BOPO']:
                scored_df[ratio + '_Status'] = scored_df[ratio] <= threshold
                scored_df['HealthScore'] += scored_df[ratio + '_Status'].astype(int)
    
    # Hitung persentase kepatuhan
    total_ratios = len(thresholds)
    scored_df['ComplianceRate'] = scored_df['HealthScore'] / total_ratios
    
    # Tentukan status kesehatan
    scored_df['HealthStatus'] = scored_df['ComplianceRate'].apply(
        lambda x: 'Sehat' if x >= 0.7 else 'Tidak Sehat'
    )
    
    return scored_df