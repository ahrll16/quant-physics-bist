import os
import pandas as pd
import numpy as np

def compute_rsi(series, window=14):
    """Göreceli Güç Endeksi (RSI) Hesaplama"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def build_bist_features(input_dir="data", output_dir="data"):
    print("🧠 BIST 100 Özellik Mühendisliği Motoru Çalıştırılıyor...")
    
    returns_path = os.path.join(input_dir, "bist100_log_returns.csv")
    if not os.path.exists(returns_path):
        raise FileNotFoundError("Önce 1_bist_data_loader.py çalıştırılmalıdır!")
        
    log_returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)
    
    # 1. Target Labeling (Piyasa Rejimi: BIST Genel Gerçekleşen Volatilitesi)
    # 92 hissenin günlük ortalama volatilitesi üzerinden piyasa risk rejimi belirlenir
    market_vol_21d = log_returns.std(axis=1).rolling(window=21).std() * np.sqrt(252)
    threshold = market_vol_21d.median()
    
    # Rejim 1: Yüksek Risk / Kriz | Rejim 0: Sakin Piyasa
    target_labels = (market_vol_21d > threshold).astype(int)
    
    feature_dict = {}
    
    print(f"📊 {log_returns.shape[1]} hisse için istatistiksel ve fiziksel momentler türetiliyor...")
    
    for col in log_returns.columns:
        ret = log_returns[col]
        
        # 5 günlük & 21 günlük Gerçekleşen Volatilite (Realized Volatility)
        vol_5d = ret.rolling(window=5).std() * np.sqrt(252)
        vol_21d = ret.rolling(window=21).std() * np.sqrt(252)
        
        # Momentum (RSI)
        rsi_14d = compute_rsi(ret, window=14)
        
        # Yüksek Dereceli İstatistiksel Momentler (Physics-to-Finance)
        skew_21d = ret.rolling(window=21).skew() # Çarpıklık
        kurt_21d = ret.rolling(window=21).kurt() # Basıklık / Kuyruk Riski
        
        df_feat = pd.DataFrame({
            f"{col}_ret": ret,
            f"{col}_vol5d": vol_5d,
            f"{col}_vol21d": vol_21d,
            f"{col}_rsi": rsi_14d,
            f"{col}_skew": skew_21d,
            f"{col}_kurt": kurt_21d
        })
        
        feature_dict[col] = df_feat
        
    # Tüm hisse özelliklerini tek bir DataFrame'de birleştir
    all_features = pd.concat(feature_dict.values(), axis=1)
    
    # Hedef etiketi ekle
    all_features['TARGET_REGIME'] = target_labels
    
    # NaN değerleri temizle (rolling window dolayısıyla oluşan ilk 21 gün)
    all_features = all_features.dropna()
    
    output_path = os.path.join(output_dir, "bist100_features.csv")
    all_features.to_csv(output_path)
    
    print(f"✅ BIST 100 Özellik Seti Başarıyla Oluşturuldu: {output_path}")
    print(f"📈 Temizlenmiş Veri Boyutu: {all_features.shape[0]} Gün x {all_features.shape[1]} Metrik")

if __name__ == "__main__":
    build_bist_features()

