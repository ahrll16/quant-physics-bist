import os
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import joblib

BIST100_TICKERS = [
    'GARAN.IS', 'AKBNK.IS', 'YKBNK.IS', 'ISCTR.IS', 'HALKB.IS', 'VAKBN.IS', 'TSKB.IS',
    'KCHOL.IS', 'SAHOL.IS', 'SISE.IS', 'DOHOL.IS', 'AGHOL.IS', 'TKFEN.IS', 'ENKAI.IS', 'ALARK.IS',
    'THYAO.IS', 'PGSUS.IS', 'TAVHL.IS', 'FROTO.IS', 'TOASO.IS', 'ARCLK.IS', 'VESBE.IS', 'TUPRS.IS',
    'PETKM.IS', 'EREGL.IS', 'KRDMD.IS', 'BRISA.IS', 'DOAS.IS', 'ASELS.IS', 'KONTR.IS', 'MIATK.IS',
    'REEDR.IS', 'SDTTR.IS', 'OTKAR.IS', 'BIMAS.IS', 'CCOLA.IS', 'AEFES.IS', 'SOKM.IS', 'MGROS.IS',
    'ULKER.IS', 'ASTOR.IS', 'EUPWR.IS', 'GESAN.IS', 'SASA.IS', 'HEKTS.IS', 'ENJSA.IS', 'AKSEN.IS',
    'CWENE.IS', 'EKGYO.IS', 'TCELL.IS', 'TTKOM.IS', 'ECILC.IS'
]

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def run_pipeline(data_dir="data", model_dir="models"):
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Geliştirme 1: 2018'den İtibaren Geniş Veri Aralığı
    print("🚀 1. BIST Verileri Yükleniyor (2018 - 2026 Genişletilmiş Tarihsel Veri)...")
    df = yf.download(BIST100_TICKERS, start="2018-01-01", end="2026-08-01")['Close']
    df = df.dropna(how='all', axis=1).ffill().bfill()
    log_returns = np.log(df / df.shift(1)).dropna()
    
    print("🧠 2. Özellik Mühendisliği (Momentum & Fiziksel Momentler)...")
    market_vol_21d = log_returns.std(axis=1).rolling(window=21).std() * np.sqrt(252)
    threshold = market_vol_21d.median()
    target_labels = (market_vol_21d > threshold).astype(int)
    
    feature_dict = {}
    for col in log_returns.columns:
        ret = log_returns[col]
        vol_5d = ret.rolling(window=5).std() * np.sqrt(252)
        vol_21d = ret.rolling(window=21).std() * np.sqrt(252)
        rsi_14d = compute_rsi(ret, window=14)
        skew_21d = ret.rolling(window=21).skew()
        kurt_21d = ret.rolling(window=21).kurt()
        
        feature_dict[col] = pd.DataFrame({
            f"{col}_ret": ret, f"{col}_vol5d": vol_5d, f"{col}_vol21d": vol_21d,
            f"{col}_rsi": rsi_14d, f"{col}_skew": skew_21d, f"{col}_kurt": kurt_21d
        })
        
    all_features = pd.concat(feature_dict.values(), axis=1)
    all_features['TARGET_REGIME'] = target_labels
    all_features = all_features.dropna()
    
    all_features.to_csv(os.path.join(data_dir, "bist_features.csv"))
    log_returns.to_csv(os.path.join(data_dir, "bist_returns.csv"))
    
    print("🌲 3. Gelişmiş XGBoost Risk Modeli Eğitiliyor...")
    X = all_features.drop(columns=['TARGET_REGIME']).values
    y = all_features['TARGET_REGIME'].values
    
    split_idx = int(len(X) * 0.80)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Overfitting'i önlemek için ilave regularizasyon parametreleri
    model = xgb.XGBClassifier(
        n_estimators=150, 
        learning_rate=0.03, 
        max_depth=4, 
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    preds_proba = model.predict_proba(X_test_scaled)[:, 1]
    auc = roc_auc_score(y_test, preds_proba)
    print(f"✅ BIST Risk Modeli Test ROC-AUC Skoru: {auc:.4f}")
    
    joblib.dump(scaler, os.path.join(model_dir, "bist_scaler.pkl"))
    joblib.dump(model, os.path.join(model_dir, "bist_xgb_model.pkl"))
    
    test_dates = all_features.index[split_idx:]
    df_signals = pd.DataFrame({
        'Date': test_dates, 
        'Actual_Regime': y_test,
        'Risk_Probability': preds_proba
    }).set_index('Date')
    
    df_signals.to_csv(os.path.join(data_dir, "bist_signals.csv"))
    print("✅ Pipeline Başarıyla Tamamlandı!")

if __name__ == "__main__":
    run_pipeline()
