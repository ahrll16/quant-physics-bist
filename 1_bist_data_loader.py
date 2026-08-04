import os
import yfinance as yf
import pandas as pd
import numpy as np

# Tam Teşekküllü Popüler 100 BIST Hissesi
BIST100_TICKERS = [
    # Bankacılık & Finans
    'GARAN.IS', 'AKBNK.IS', 'YKBNK.IS', 'ISCTR.IS', 'HALKB.IS', 'VAKBN.IS', 'TSKB.IS', 'SKBNK.IS', 'ALBRK.IS', 'ANHYT.IS',
    # Holdingler & İnşaat
    'KCHOL.IS', 'SAHOL.IS', 'SISE.IS', 'DOHOL.IS', 'AGHOL.IS', 'TKFEN.IS', 'ENKAI.IS', 'ALARK.IS', 'BERA.IS', 'GSDHO.IS',
    # Havacılık & Lojistik
    'THYAO.IS', 'PGSUS.IS', 'TAVHL.IS', 'CLEBI.IS',
    # Sanayi, Otomotiv & Dayanıklı Tüketim
    'FROTO.IS', 'TOASO.IS', 'ARCLK.IS', 'VESBE.IS', 'TUPRS.IS', 'PETKM.IS', 'EREGL.IS', 'KRDMD.IS', 'BRISA.IS', 'DOAS.IS', 'OYAKC.IS', 'BUCIM.IS',
    # Savunma & Teknoloji
    'ASELS.IS', 'KONTR.IS', 'MIATK.IS', 'REEDR.IS', 'SDTTR.IS', 'OTKAR.IS', 'LOGO.IS', 'FONET.IS', 'KFEIN.IS', 'NETAS.IS', 'PENTA.IS',
    # Perakende, Gıda & Mağazacılık
    'BIMAS.IS', 'CCOLA.IS', 'AEFES.IS', 'SOKM.IS', 'MGROS.IS', 'ULKER.IS', 'TABGD.IS', 'MAVI.IS', 'SUWEN.IS', 'SOKE.IS',
    # Enerji & Elektrik
    'ASTOR.IS', 'EUPWR.IS', 'GESAN.IS', 'SASA.IS', 'HEKTS.IS', 'ENJSA.IS', 'AKSEN.IS', 'CWENE.IS', 'SMTRT.IS', 'ALFAS.IS', 'GWIND.IS', 'ODAS.IS', 'ZOREN.IS', 'CANTE.IS',
    # Gayrimenkul (GYO)
    'EKGYO.IS', 'TRGYO.IS', 'ISGYO.IS', 'SNGYO.IS', 'KLGYO.IS', 'PSGYO.IS',
    # İletişim, Sağlık & Diğer Popüler Hisseler
    'TCELL.IS', 'TTKOM.IS', 'ECILC.IS', 'GENIL.IS', 'MPARK.IS', 'KCAER.IS', 'EGEEN.IS', 'BFREN.IS', 'KONTR.IS', 'GUBRF.IS',
    'ISMEN.IS', 'INFO.IS', 'TURSG.IS', 'ANSGR.IS', 'HEKTS.IS', 'KLSER.IS', 'FORTE.IS', 'TETMT.IS', 'PASEU.IS', 'BMTKS.IS'
]

def download_bist100_data(start_date="2021-01-01", end_date="2026-08-01", output_dir="data"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Tekrarlayan sembolleri temizle
    unique_tickers = list(dict.fromkeys(BIST100_TICKERS))
    print(f"🚀 {len(unique_tickers)} adet BIST hissesi için veri indiriliyor...")

    # Yfinance ile veri indirme
    df = yf.download(unique_tickers, start=start_date, end=end_date)['Close']
    
    # İnmeyen veya tamamen boş hisse kolonlarını temizle
    df = df.dropna(how='all', axis=1)
    
    # Eksik verileri doldur (Forward/Backward Fill)
    df = df.ffill().bfill()

    # Logaritmik Getiriler (Stationarity)
    log_returns = np.log(df / df.shift(1)).dropna()

    prices_path = os.path.join(output_dir, "bist100_prices.csv")
    returns_path = os.path.join(output_dir, "bist100_log_returns.csv")

    df.to_csv(prices_path)
    log_returns.to_csv(returns_path)

    print(f"\n✅ BIST 100 Kapanış Fiyatları: {prices_path}")
    print(f"✅ BIST 100 Log Getirileri: {returns_path}")
    print(f"📊 Başarıyla Yüklenen Veri Boyutu: {log_returns.shape[0]} Gün x {log_returns.shape[1]} Hisse")

if __name__ == "__main__":
    download_bist100_data()
