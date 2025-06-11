import pandas as pd
import re

def load_and_preprocess_data(uploaded_files, required_columns, data_type):
    """Muat dan proses data dari file CSV"""
    dfs = []
    for file in uploaded_files:
        try:
            df = pd.read_csv(file)
            
            # Hanya tambahkan kolom Bank dari nama file jika belum ada
            if 'Bank' not in df.columns and 'NamaBank' not in df.columns:
                bank_name = file.name.split('.')[0]
                df['Bank'] = bank_name
            
            # Konversi tanggal dengan format berbeda-beda
            if 'Date' in df.columns:
                # Fungsi untuk mengonversi format tanggal 2-digit tahun ke 4-digit
                def convert_date(date_str):
                    if pd.isna(date_str) or not isinstance(date_str, str):
                        return date_str
                    
                    # Tangani format seperti "02/01/25" -> "02/01/2025"
                    if re.match(r'\d{1,2}/\d{1,2}/\d{2}$', date_str):
                        day, month, year = date_str.split('/')
                        year = f"20{year}" if int(year) < 50 else f"19{year}"
                        return f"{day}/{month}/{year}"
                    
                    return date_str
                
                # Normalisasi format tanggal
                df['Date'] = df['Date'].astype(str).apply(convert_date)
                
                # Coba berbagai format tanggal
                for fmt in ['%d/%m/%Y', '%Y-%m-%d', '%m/%d/%Y']:
                    try:
                        df['Date'] = pd.to_datetime(df['Date'], format=fmt, errors='coerce')
                        # Periksa jika ada nilai yang masih null
                        if not df['Date'].isnull().all():
                            break
                    except:
                        continue
            
            # Konversi kolom numerik
            numeric_cols = [col for col in required_columns if col not in ['Date', 'Bank']]
            for col in numeric_cols:
                if col in df.columns:
                    # Ganti koma dengan titik dan konversi ke float
                    df[col] = pd.to_numeric(
                        df[col].astype(str).str.replace(',', '.'), 
                        errors='coerce'
                    )
            
            # Standarisasi nama kolom
            if 'NamaBank' in df.columns:
                df['Bank'] = df['NamaBank']
                df = df.drop(columns=['NamaBank'])
            
            dfs.append(df)
        except Exception as e:
            raise ValueError(f"Error processing {file.name}: {str(e)}")
    
    if not dfs:
        return pd.DataFrame()
    
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Validasi kolom
    for col in required_columns:
        if col not in combined_df.columns:
            raise ValueError(f"Kolom {col} tidak ditemukan dalam data {data_type}")
    
    return combined_df

def combine_quarterly_daily(quarterly_df, daily_df, features):
    """Gabungkan data kuartalan dengan data harian (rata-rata per kuartal)"""
    combined_data = []
    
    # Pastikan kolom Bank ada di kedua dataframe
    if 'Bank' not in daily_df.columns or 'Bank' not in quarterly_df.columns:
        raise ValueError("Kolom 'Bank' harus ada di kedua dataset")
    
    # Pastikan tanggal sudah datetime
    quarterly_df['Date'] = pd.to_datetime(quarterly_df['Date'])
    daily_df['Date'] = pd.to_datetime(daily_df['Date'])
    
    # Dapatkan bank unik
    unique_banks = set(quarterly_df['Bank']).intersection(set(daily_df['Bank']))
    
    for bank in unique_banks:
        bank_quarterly = quarterly_df[quarterly_df['Bank'] == bank]
        bank_daily = daily_df[daily_df['Bank'] == bank]
        
        for _, q_row in bank_quarterly.iterrows():
            quarter_date = q_row['Date']
            year = quarter_date.year
            quarter = (quarter_date.month - 1) // 3 + 1
            
            # Tentukan rentang kuartal
            if quarter == 1:  # Jan-Mar
                start_date = pd.Timestamp(year, 1, 1)
                end_date = pd.Timestamp(year, 3, 31)
            elif quarter == 2:  # Apr-Jun
                start_date = pd.Timestamp(year, 4, 1)
                end_date = pd.Timestamp(year, 6, 30)
            elif quarter == 3:  # Jul-Sep
                start_date = pd.Timestamp(year, 7, 1)
                end_date = pd.Timestamp(year, 9, 30)
            else:  # Okt-Dec
                start_date = pd.Timestamp(year, 10, 1)
                end_date = pd.Timestamp(year, 12, 31)
            
            # Filter data harian dalam rentang kuartal
            mask = (
                (bank_daily['Date'] >= start_date) &
                (bank_daily['Date'] <= end_date))
            
            daily_subset = bank_daily.loc[mask].copy()
            
            if not daily_subset.empty:
                # Hitung rata-rata harga saham per kuartal
                avg_price = daily_subset['StockPrice'].mean()
                
                # Buat baris baru dengan rata-rata harga
                new_row = q_row.copy()
                new_row['StockPrice'] = avg_price
                combined_data.append(new_row)
    
    return pd.DataFrame(combined_data) if combined_data else pd.DataFrame()