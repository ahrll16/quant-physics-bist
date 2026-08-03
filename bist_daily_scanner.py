import os
import smtplib
import requests
import xml.etree.ElementTree as ET
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import yfinance as yf
import pandas as pd
import numpy as np
import joblib

# -------------------------------------------------------------
# E-POSTA / GMAIL KONFİGÜRASYONU
# -------------------------------------------------------------
SENDER_EMAIL = "alibrsharmanli@gmail.com"
SENDER_PASSWORD = "zsrkhnqiauzyotxy"  # 16 Haneli Google Uygulama Şifren
RECEIVER_EMAIL = "alibrsharmanli@gmail.com"
# -------------------------------------------------------------

STOCK_CATEGORIES = {
    'BANKACILIK & FİNANS': ['AKBNK.IS', 'GARAN.IS', 'HALKB.IS', 'ISCTR.IS', 'TSKB.IS', 'VAKBN.IS', 'YKBNK.IS'],
    'HOLDİNG & YATIRIM': ['AGHOL.IS', 'ALARK.IS', 'DOHOL.IS', 'KCHOL.IS', 'SAHOL.IS', 'SISE.IS', 'TKFEN.IS'],
    'HAVACILIK & LOJİSTİK': ['ENKAI.IS', 'PGSUS.IS', 'TAVHL.IS', 'THYAO.IS'],
    'OTOMOTİV & SANAYİ': ['ARCLK.IS', 'BRISA.IS', 'DOAS.IS', 'FROTO.IS', 'OTKAR.IS', 'TOASO.IS', 'VESBE.IS'],
    'ENERJİ & MADENCİLİK': ['AKSEN.IS', 'ASTOR.IS', 'CWENE.IS', 'ENJSA.IS', 'EUPWR.IS', 'GESAN.IS', 'PETKM.IS', 'TUPRS.IS'],
    'SAVUNMA & TEKNOLOJİ': ['ASELS.IS', 'KONTR.IS', 'MIATK.IS', 'REEDR.IS', 'SDTTR.IS'],
    'PERAKENDE & GIDA': ['AEFES.IS', 'BIMAS.IS', 'CCOLA.IS', 'MGROS.IS', 'SOKM.IS', 'ULKER.IS'],
    'KİMYA & GAYRİMENKUL': ['ECILC.IS', 'EKGYO.IS', 'EREGL.IS', 'HEKTS.IS', 'KRDMD.IS', 'SASA.IS'],
    'İLETİŞİM': ['TCELL.IS', 'TTKOM.IS']
}

ALL_TICKERS = sorted([ticker for group in STOCK_CATEGORIES.values() for ticker in group])

GLOBAL_TICKERS = {
    '^GSPC': 'SP500', 
    '^VIX': 'VIX', 
    'DX-Y.NYB': 'DXY', 
    'BZ=F': 'BRENT', 
    'GC=F': 'GOLD',
    'USDJPY=X': 'USDJPY'
}

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def fetch_global_and_institutional_sentiment():
    """
    1. Türkiye Kredi Notu Artırımları (Moody's, S&P, Fitch, Görünüm)
    2. BofA / Yabancı Akış Söylentileri & Satış/Alım Haberleri
    3. Küresel Merkez Bankaları, Faiz, Savaş ve Devalüasyon Akışı
    """
    print("🌍 Kredi Notu Artırımları, BofA Akışları & Küresel Makro Haberler Taranıyor...")
    
    # Risk Artıran / Negatif Anahtar Kelimeler
    keywords_high_risk = [
        'bofa', 'bank of america', 'jp morgan', 'bofa satış', 'yabancı satışı', 'rumor', 'söylenti',
        'yabancı çıkışı', 'devaluation', 'devalüasyon', 'war', 'savaş', 'conflict', 'çatışma',
        'inflation', 'enflasyon', 'rate hike', 'faiz artışı', 'crisis', 'kriz', 'sanction', 'yaptırım',
        'not indirimi', 'downgrade', 'not düşürdü'
    ]
    
    # Risk Azaltan / Pozitif (Kredi Notu Artırımı vb.) Anahtar Kelimeler
    keywords_low_risk = [
        'kredi notu artırımı', 'not artırdı', 'notu yükseltti', 'upgrade', 'fitch', 'moody\'s', 's&p', 
        'görünüm pozitif', 'not artışı', 'yabancı alımı', 'bofa alım', 'rate cut', 'faiz indirimi', 
        'peace', 'barış', 'growth', 'büyüme', 'anlaşma'
    ]
    
    rss_urls = [
        "https://news.google.com/rss/search?q=türkiye+kredi+notu+artırımı+fitch+moodys+s&p&hl=tr&gl=TR&ceid=TR:tr",
        "https://news.google.com/rss/search?q=BofA+BIST+borsa+istanbul+yabancı&hl=tr&gl=TR&ceid=TR:tr",
        "https://news.google.com/rss/search?q=economy+global+market+liquidity+fed+war&hl=en-US&gl=US&ceid=US:en"
    ]
    
    risk_points = 0
    top_headlines = []
    
    for url in rss_urls:
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                root = ET.fromstring(resp.content)
                for item in root.findall('.//item')[:8]:
                    title = item.find('title').text
                    top_headlines.append(title)
                    title_lower = title.lower()
                    
                    for kw in keywords_high_risk:
                        if kw in title_lower:
                            risk_points += 2.0
                    for kw in keywords_low_risk:
                        if kw in title_lower:
                            risk_points -= 3.0  # Kredi notu artışı gibi pozitif haberler riski ciddi oranda kırar
        except Exception as e:
            print(f"⚠️ Haber akışı çekilirken bağlantı uyarısı: {e}")
            
    sentiment_risk_score = min(max(50 + (risk_points * 2.5), 5), 95)
    return sentiment_risk_score, top_headlines[:6]

def send_email_report(subject, body_html):
    if SENDER_EMAIL == "alibrsharmanli@gmail.com" and SENDER_PASSWORD == "kopyaladigin_16_haneli_sifre":
        print("⚠️ E-posta bilgileri güncellenmediği için mail atılmadı.")
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
        print("📧 Kredi Notu & BofA Entegreli Rapor Gmail Adresinize Gönderildi!")
    except Exception as e:
        print(f"❌ E-Posta Gönderme Hatası: {e}")

def generate_full_stock_signals(model_dir="models", data_dir="data"):
    news_risk_score, top_news = fetch_global_and_institutional_sentiment()
    
    print("🔍 BIST 100 & Kredi Notu/Makro Model Tahminleri Hesaplanıyor...")
    scaler = joblib.load(os.path.join(model_dir, "bist_scaler.pkl"))
    model = joblib.load(os.path.join(model_dir, "bist_xgb_model.pkl"))
    
    features_df_train = pd.read_csv(os.path.join(data_dir, "bist_features.csv"), index_col=0, nrows=2)
    expected_cols = [c for c in features_df_train.columns if c != 'TARGET_REGIME']
    
    df_bist = yf.download(ALL_TICKERS, period="500d")['Close'].ffill().bfill()
    log_returns = np.log(df_bist / df_bist.shift(1)).dropna()
    
    feature_dict = {}
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

    bist_feats = pd.concat(feature_dict.values(), axis=1).dropna()
    
    df_macro = yf.download(list(GLOBAL_TICKERS.keys()), period="60d")['Close'].rename(columns=GLOBAL_TICKERS).ffill().bfill()
    macro_feats = pd.DataFrame(index=df_macro.index)
    for col in df_macro.columns:
        macro_feats[f"{col}_ret"] = np.log(df_macro[col] / df_macro[col].shift(1))
        macro_feats[f"{col}_vol21d"] = macro_feats[f"{col}_ret"].rolling(21).std() * np.sqrt(252)
    macro_feats = macro_feats.dropna()
    
    common_idx = bist_feats.index.intersection(macro_feats.index)
    latest_df = pd.concat([bist_feats.loc[common_idx], macro_feats.loc[common_idx]], axis=1)
    latest_df_aligned = latest_df.reindex(columns=expected_cols, fill_value=0)
    latest_row = latest_df_aligned.iloc[-1:].values
    
    latest_scaled = scaler.transform(latest_row)
    model_risk_prob = model.predict_proba(latest_scaled)[0, 1] * 100
    
    combined_risk_prob = (model_risk_prob * 0.55) + (news_risk_score * 0.45)
    stock_weight = (100.0 - combined_risk_prob)
    cash_weight = combined_risk_prob
    last_date = common_idx[-1].strftime('%Y-%m-%d')
    
    timeframes = {
        'Günlük': 1, 'Haftalık': 5, 'Aylık': 21,
        '3 Aylık': 63, '6 Aylık': 126, '1 Yıllık': 252,
        '2 Yıllık': 504, '3 Yıllık': 756
    }
    
    html_sections = ""
    
    for category_name in sorted(STOCK_CATEGORIES.keys()):
        tickers_in_cat = sorted(STOCK_CATEGORIES[category_name])
        
        html_sections += f"""
        <div style="margin-top: 35px; border-bottom: 2px solid #2563eb; padding-bottom: 5px;">
            <h3 style="color: #1e3a8a; margin: 0;">🏢 {category_name}</h3>
        </div>
        """
        
        for ticker in tickers_in_cat:
            clean_symbol = ticker.replace('.IS', '')
            current_price = df_bist[ticker].iloc[-1]
            hist_ret = log_returns[ticker]
            
            ann_vol = hist_ret.iloc[-126:].std() * np.sqrt(252)
            daily_drift = hist_ret.iloc[-126:].mean()
            rsi_val = compute_rsi(hist_ret).iloc[-1]
            
            macro_sentiment_factor = 1.0 - ((news_risk_score - 50) / 100.0)
            adjusted_daily_drift = daily_drift * macro_sentiment_factor
            
            tf_rows = ""
            for tf_name, days in timeframes.items():
                if days <= 252:
                    expected_drift = adjusted_daily_drift * days
                else:
                    years = days / 252
                    bank_cagr = 0.35 * macro_sentiment_factor
                    expected_drift = np.log((1 + bank_cagr) ** years)
                    
                vol_period = ann_vol * np.sqrt(days / 252) * 100
                target_price = current_price * np.exp(expected_drift)
                pct_change = ((target_price - current_price) / current_price) * 100
                
                if pct_change > 4 and news_risk_score < 65 and rsi_val < 65:
                    sig = "AL / Yüksek Potansiyel"
                    color = "#10b981"
                elif pct_change < -3 or news_risk_score > 70 or rsi_val > 70:
                    sig = "SAT / Riskli & Nakite Geç"
                    color = "#ef4444"
                else:
                    sig = "TUT / Dengeli Pozisyon"
                    color = "#6b7280"
                    
                tf_rows += f"""
                <tr>
                    <td style="padding: 6px; border: 1px solid #e2e8f0; font-weight: bold;">{tf_name}</td>
                    <td style="padding: 6px; border: 1px solid #e2e8f0; text-align: right;">%{vol_period:.2f}</td>
                    <td style="padding: 6px; border: 1px solid #e2e8f0; text-align: right; font-weight: bold;">{target_price:.2f} TL ({pct_change:+.2f}%)</td>
                    <td style="padding: 6px; border: 1px solid #e2e8f0; text-align: center; color: {color}; font-weight: bold;">{sig}</td>
                    <td style="padding: 6px; border: 1px solid #e2e8f0; font-size: 11px;">Kredi Notu & BofA Akış Düzeltmeli</td>
                </tr>
                """
                
            html_sections += f"""
            <div style="background-color: #ffffff; border: 1px solid #cbd5e1; border-radius: 6px; padding: 12px; margin-top: 15px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-size: 16px; font-weight: bold; color: #0f172a;">📌 {clean_symbol}</span>
                    <span style="font-size: 13px; color: #475569;">Fiyat: <strong>{current_price:.2f} TL</strong> | RSI: <strong>{rsi_val:.1f}</strong></span>
                </div>
                <table style="width: 100%; border-collapse: collapse; font-size: 12px;">
                    <thead>
                        <tr style="background-color: #f8fafc; color: #334155;">
                            <th style="padding: 6px; border: 1px solid #e2e8f0; text-align: left;">Zaman Dilimi</th>
                            <th style="padding: 6px; border: 1px solid #e2e8f0; text-align: right;">Beklenen Oynaklık</th>
                            <th style="padding: 6px; border: 1px solid #e2e8f0; text-align: right;">Tahmini Fiyat (% Hedef)</th>
                            <th style="padding: 6px; border: 1px solid #e2e8f0; text-align: center;">Tazelenmiş Sinyal</th>
                            <th style="padding: 6px; border: 1px solid #e2e8f0; text-align: left;">Kredi Notu & Akış Notu</th>
                        </tr>
                    </thead>
                    <tbody>
                        {tf_rows}
                    </tbody>
                </table>
            </div>
            """

    news_items_html = "".join([f"<li style='margin-bottom:4px;'>{n}</li>" for n in top_news])

    subject = f"📈 Kredi Notu, BofA & Makro BIST 100 Raporu - {last_date}"
    email_html = f"""
    <html>
      <body style="font-family: Arial, sans-serif; color: #1e293b; background-color: #f1f5f9; padding: 20px;">
        <div style="max-width: 950px; margin: 0 auto; background-color: #ffffff; border-radius: 10px; padding: 25px; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
          
          <h2 style="color: #0f172a; text-align: center; border-bottom: 3px solid #2563eb; padding-bottom: 12px; margin-top: 0;">
            📈 BIST 100 Credit Rating & Institutional Flow Raporu
          </h2>
          <p style="text-align: center; color: #64748b; margin-bottom: 20px;">
            Rapor Tarihi: <strong>{last_date}</strong> | Kredi Notu Artırımları & BofA Akışları Entegreli
          </p>

          <div style="background-color: #eff6ff; border: 1px solid #bfdbfe; border-radius: 8px; padding: 15px; margin-bottom: 20px;">
            <h4 style="margin: 0 0 8px 0; color: #1e40af;">🧠 Hibrit Risk & Varlık Tahsisi Değerlendirmesi</h4>
            <p style="margin: 4px 0;"><strong>BIST Teknik Model Riski: %{model_risk_prob:.2f}</strong></p>
            <p style="margin: 4px 0;"><strong>Kredi Notu / BofA Akış Duygu Skoru: %{news_risk_score:.2f}</strong></p>
            <p style="margin: 4px 0; font-size: 15px; color: #1e3a8a;"><strong>Önerilen Portföy Dağılımı:</strong> <strong>%{stock_weight:.2f} Hisse / %{cash_weight:.2f} Nakit</strong></p>
          </div>

          <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; margin-bottom: 25px; font-size: 12px;">
            <strong style="color: #334155;">📰 Taranan Kredi Notu, BofA & Makro Haber Başlıkları:</strong>
            <ul style="margin: 6px 0 0 18px; padding: 0; color: #475569;">
                {news_items_html}
            </ul>
          </div>

          {html_sections}

          <div style="margin-top: 30px; padding-top: 15px; border-top: 1px solid #e2e8f0; text-align: center; font-size: 11px; color: #94a3b8;">
            Bu rapor Quant Physics BIST Engine tarafından otonom üretilmektedir.
          </div>
        </div>
      </body>
    </html>
    """
    
    send_email_report(subject, email_html)

if __name__ == "__main__":
    generate_full_stock_signals()
