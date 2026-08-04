import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import yfinance as yf
import pandas as pd
import numpy as np
import joblib

# -------------------------------------------------------------
# E-POSTA / GMAIL KONFİGÜRASYONU (Kendi Bilgilerini Yaz)
# -------------------------------------------------------------
SENDER_EMAIL = "alibrsharmanli@gmail.com"
SENDER_PASSWORD = "nkfr phfx znco kmzp"
RECEIVER_EMAIL = "alibrsharmanli@gmail.com"
# -------------------------------------------------------------

BIST100_TICKERS = [
    'GARAN.IS', 'AKBNK.IS', 'YKBNK.IS', 'ISCTR.IS', 'HALKB.IS', 'VAKBN.IS', 'TSKB.IS',
    'KCHOL.IS', 'SAHOL.IS', 'SISE.IS', 'DOHOL.IS', 'AGHOL.IS', 'TKFEN.IS', 'ENKAI.IS', 'ALARK.IS',
    'THYAO.IS', 'PGSUS.IS', 'TAVHL.IS', 'FROTO.IS', 'TOASO.IS', 'ARCLK.IS', 'VESBE.IS', 'TUPRS.IS',
    'PETKM.IS', 'EREGL.IS', 'KRDMD.IS', 'BRISA.IS', 'DOAS.IS', 'ASELS.IS', 'KONTR.IS', 'MIATK.IS',
    'REEDR.IS', 'SDTTR.IS', 'OTKAR.IS', 'BIMAS.IS', 'CCOLA.IS', 'AEFES.IS', 'SOKM.IS', 'MGROS.IS',
    'ULKER.IS', 'ASTOR.IS', 'EUPWR.IS', 'GESAN.IS', 'SASA.IS', 'HEKTS.IS', 'ENJSA.IS', 'AKSEN.IS',
    'CWENE.IS', 'EKGYO.IS', 'TCELL.IS', 'TTKOM.IS', 'ECILC.IS'
]

GLOBAL_TICKERS = {'^GSPC': 'SP500', '^VIX': 'VIX', 'DX-Y.NYB': 'DXY', 'BZ=F': 'BRENT', 'GC=F': 'GOLD'}

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def send_email_report(subject, body_html):
    if SENDER_EMAIL == "kendi_gmail_adresin@gmail.com":
        print("⚠️ E-posta bilgileri girilmediği için mail atılmadı.")
        return

    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECEIVER_EMAIL
    msg.attach(MIMEText(body_html, 'html'))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())
        print("📧 Ertesi günün detaylı risk ve hisse analiz raporu Gmail adresinize gönderildi!")
    except Exception as e:
        print(f"❌ E-Posta Gönderme Hatası: {e}")

def generate_daily_intelligence(model_dir="models"):
    print("🔍 Ertesi Günün BIST & Dünya Trend Analizi Taranıyor...")
    
    scaler = joblib.load(os.path.join(model_dir, "bist_scaler.pkl"))
    model = joblib.load(os.path.join(model_dir, "bist_xgb_model.pkl"))
    
    # 1. BIST Verilerini Çek
    df_bist = yf.download(BIST100_TICKERS, period="60d")['Close'].ffill().bfill()
    log_returns = np.log(df_bist / df_bist.shift(1)).dropna()
    
    feature_dict = {}
    stock_analysis = []
    
    for col in log_returns.columns:
        ret = log_returns[col]
        vol_5d = ret.rolling(5).std() * np.sqrt(252)
        vol_21d = ret.rolling(21).std() * np.sqrt(252)
        rsi_14d = compute_rsi(ret, window=14)
        skew_21d = ret.rolling(21).skew()
        kurt_21d = ret.rolling(21).kurt()
        
        feature_dict[col] = pd.DataFrame({
            f"{col}_ret": ret, f"{col}_vol5d": vol_5d, f"{col}_vol21d": vol_21d,
            f"{col}_rsi": rsi_14d, f"{col}_skew": skew_21d, f"{col}_kurt": kurt_21d
        })
        
        # Hisse Bazlı Ertesi Gün Trend & Beklenen Volatilite Bandı
        last_rsi = rsi_14d.iloc[-1]
        last_vol = vol_5d.iloc[-1] / np.sqrt(252) * 100 # Günlük tahmini % oynaklık
        clean_symbol = col.replace('.IS', '')
        
        if last_rsi < 35:
            signal = "AL / Aşırı Satım"
            badge = "#10b981"
        elif last_rsi > 70:
            signal = "SAT / Aşırı Alım"
            badge = "#ef4444"
        else:
            signal = "TUT / Nötr"
            badge = "#6b7280"
            
        stock_analysis.append({
            'Hisse': clean_symbol,
            'Son_Fiyat': df_bist[col].iloc[-1],
            'RSI': round(last_rsi, 1),
            'Sinyal': signal,
            'Badge': badge,
            'Beklenen_Bant': f"±%{last_vol:.2f}"
        })
        
    bist_feats = pd.concat(feature_dict.values(), axis=1).dropna()
    
    # 2. Makro Dünya Verilerini Çek
    df_macro = yf.download(list(GLOBAL_TICKERS.keys()), period="60d")['Close'].rename(columns=GLOBAL_TICKERS).ffill().bfill()
    macro_feats = pd.DataFrame(index=df_macro.index)
    for col in df_macro.columns:
        macro_feats[f"{col}_ret"] = np.log(df_macro[col] / df_macro[col].shift(1))
        macro_feats[f"{col}_vol21d"] = macro_feats[f"{col}_ret"].rolling(21).std() * np.sqrt(252)
    macro_feats = macro_feats.dropna()
    
    # Tahmin İçin Birleştir
    common_idx = bist_feats.index.intersection(macro_feats.index)
    latest_row = pd.concat([bist_feats.loc[common_idx], macro_feats.loc[common_idx]], axis=1).iloc[-1:].values
    
    latest_scaled = scaler.transform(latest_row)
    risk_prob = model.predict_proba(latest_scaled)[0, 1]
    
    stock_weight = (1.0 - risk_prob) * 100
    cash_weight = risk_prob * 100
    last_date = common_idx[-1].strftime('%Y-%m-%d')
    
    # Öne Çıkan Hisse Sinyalleri (İlk 5 Al ve İlk 5 Sat Adayı)
    df_stocks = pd.DataFrame(stock_analysis)
    top_buys = df_stocks[df_stocks['Sinyal'].str.contains("AL")].sort_values(by='RSI').head(5)
    
    # HTML E-Posta Raporu
    html_rows = ""
    for _, row in top_buys.iterrows():
        html_rows += f"""
        <tr>
            <td style="padding: 8px; border: 1px solid #cbd5e1; font-weight: bold;">{row['Hisse']}</td>
            <td style="padding: 8px; border: 1px solid #cbd5e1;">{row['Son_Fiyat']:.2f} TL</td>
            <td style="padding: 8px; border: 1px solid #cbd5e1;">{row['RSI']}</td>
            <td style="padding: 8px; border: 1px solid #cbd5e1; color: {row['Badge']}; font-weight: bold;">{row['Sinyal']}</td>
            <td style="padding: 8px; border: 1px solid #cbd5e1;">{row['Beklenen_Bant']}</td>
        </tr>
        """
        
    subject = f"📊 BIST 100 & Dünya Trend Ertesi Gün Raporu - {last_date}"
    email_html = f"""
    <html>
      <body style="font-family: Arial, sans-serif; color: #333; line-height: 1.6;">
        <div style="max-width: 650px; margin: 0 auto; border: 1px solid #e0e0e0; border-radius: 8px; padding: 20px;">
          <h2 style="color: #0b132b; text-align: center; border-bottom: 2px solid #2563eb; padding-bottom: 10px;">
            🌍 BIST 100 & Makro Ertesi Gün Tahmin Raporu
          </h2>
          <p><strong>Analiz Tarihi:</strong> {last_date}</p>

          <div style="background-color: #f8fafc; padding: 15px; border-radius: 6px; margin: 15px 0;">
            <p style="margin: 5px 0;"><strong>BIST Genel Risk Olasılığı:</strong> %{risk_prob*100:.2f}</p>
            <p style="margin: 5px 0;"><strong>Önerilen Hisse Ağırlığı:</strong> %{stock_weight:.2f}</p>
            <p style="margin: 5px 0;"><strong>Önerilen Nakit / Repo:</strong> %{cash_weight:.2f}</p>
          </div>

          <h3 style="color: #2563eb;">🔥 Ertesi Gün Öne Çıkan Fırsat Hisseleri (Teknik Adaylar)</h3>
          <table style="width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 14px;">
            <tr style="background-color: #f1f5f9;">
              <th style="padding: 8px; border: 1px solid #cbd5e1; text-align: left;">Hisse</th>
              <th style="padding: 8px; border: 1px solid #cbd5e1;">Fiyat</th>
              <th style="padding: 8px; border: 1px solid #cbd5e1;">RSI</th>
              <th style="padding: 8px; border: 1px solid #cbd5e1;">Sinyal</th>
              <th style="padding: 8px; border: 1px solid #cbd5e1;">Beklenen Günlük % Bant</th>
            </tr>
            {html_rows if len(html_rows) > 0 else "<tr><td colspan='5' style='padding:10px; text-align:center;'>Şu an aşırı satımda öne çıkan hisse bulunmuyor.</td></tr>"}
          </table>

          <p style="font-size: 11px; color: #64748b; margin-top: 25px; text-align: center;">
            Bu rapor Quant Physics BIST Multi-Modal Risk Engine tarafından otonom üretilmiştir.
          </p>
        </div>
      </body>
    </html>
    """
    
    send_email_report(subject, email_html)

if __name__ == "__main__":
    generate_daily_intelligence()
