import os
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import xml.etree.ElementTree as ET

try:
    from google import genai
except ImportError:
    genai = None

# Streamlit Konfigürasyonu
st.set_page_config(
    page_title="BIST Terminal",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Kurumsal Finans Terminali CSS
st.markdown("""
<style>
    .main { background-color: #0b0f19; color: #f1f5f9; }
    .stMetric { background-color: #1e293b; padding: 15px; border-radius: 8px; border: 1px solid #334155; }
    .disclaimer-box { background-color: #1e1b4b; border: 1px solid #4338ca; border-radius: 8px; padding: 12px; margin-bottom: 20px; color: #cbd5e1; font-size: 13px; }
    .news-box { background-color: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 18px; margin-bottom: 20px; font-size: 14px; line-height: 1.6; }
</style>
""", unsafe_allow_html=True)

# Yasal Uyarı Metni
st.markdown("""
<div class="disclaimer-box">
    <strong>⚖️ YASAL UYARI:</strong> Bu web sitesinde sunulan duygu analizleri, makine öğrenimi tahminleri ve kuantitatif veriler 
    <strong>yatırım danışmanlığı kapsamında değildir.</strong> Yalnızca akademik, istatistiksel ve eğitim amaçlı kuantitatif modeller içerir.
</div>
""", unsafe_allow_html=True)

st.title("📊 BIST Terminali")

# Sektörler ve Hisseler (A-Z Alfabetik Düzenleme)
RAW_STOCK_CATEGORIES = {
    'BANKACILIK & FİNANS': ['AKBNK.IS', 'GARAN.IS', 'HALKB.IS', 'ISCTR.IS', 'TSKB.IS', 'VAKBN.IS', 'YKBNK.IS'],
    'ENERJİ & MADENCİLİK': ['AKSEN.IS', 'ASTOR.IS', 'CWENE.IS', 'ENJSA.IS', 'EUPWR.IS', 'GESAN.IS', 'PETKM.IS', 'TUPRS.IS'],
    'HAVACILIK & LOJİSTİK': ['ENKAI.IS', 'PGSUS.IS', 'TAVHL.IS', 'THYAO.IS'],
    'HOLDİNG & YATIRIM': ['AGHOL.IS', 'ALARK.IS', 'DOHOL.IS', 'KCHOL.IS', 'SAHOL.IS', 'SISE.IS', 'TKFEN.IS'],
    'İLETİŞİM': ['TCELL.IS', 'TTKOM.IS'],
    'KİMYA & GAYRİMENKUL': ['ECILC.IS', 'EKGYO.IS', 'EREGL.IS', 'HEKTS.IS', 'KRDMD.IS', 'SASA.IS'],
    'OTOMOTİV & SANAYİ': ['ARCLK.IS', 'BRISA.IS', 'DOAS.IS', 'FROTO.IS', 'OTKAR.IS', 'TOASO.IS', 'VESBE.IS'],
    'PERAKENDE & GIDA': ['AEFES.IS', 'BIMAS.IS', 'CCOLA.IS', 'MGROS.IS', 'SOKM.IS', 'ULKER.IS'],
    'SAVUNMA & TEKNOLOJİ': ['ASELS.IS', 'KONTR.IS', 'MIATK.IS', 'REEDR.IS', 'SDTTR.IS']
}

# A-Z Alfabetik Sıralanmış Sektörler ve Hisseler
STOCK_CATEGORIES = {
    cat: sorted(RAW_STOCK_CATEGORIES[cat]) 
    for cat in sorted(RAW_STOCK_CATEGORIES.keys())
}

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

@st.cache_data(ttl=1800)
@st.cache_data(ttl=1800)
@st.cache_data(ttl=1800)
@st.cache_data(ttl=1800) # 30 dakika boyunca önbellekte tutar, ücretsiz API kotasını korur
def fetch_and_analyze_news_live():
    rss_urls = [
        "https://news.google.com/rss/headlines/section/topic/BUSINESS?hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=global+markets+OR+Federal+Reserve+OR+oil+prices+when:1d&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=BIST+100+OR+Borsa+Istanbul+OR+Merkez+Bankasi+when:1d&hl=tr&gl=TR&ceid=TR:tr"
    ]

    raw_titles = []
    for url in rss_urls:
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                root = ET.fromstring(resp.content)
                for item in root.findall('.//item')[:8]:
                    t_text = item.find('title').text
                    if t_text and t_text not in raw_titles:
                        raw_titles.append(t_text)
        except Exception:
            pass

    api_key = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", "")
    if not api_key:
        return 50.0, "<p>⚠️ GEMINI_API_KEY tanımlanmadı.</p>"

    all_titles_text = "\n".join([f"- {t}" for t in raw_titles])

    prompt = f"""
    Sen BIST 100 ve Küresel Piyasalar Baş Analistisin.
    Aşağıda son haber başlıkları bulunmaktadır:
    {all_titles_text}

    GÖREVİN:
    1. BIST 100 üzerindeki etkisi en yüksek olan 5-8 kritik haberi seç.
    2. Türkçe teknik özet ile [POZİTİF 🟢 / NEGATİF 🔴 / NÖTR 🟡] etiketli HTML liste özetini üret (<ul> ve <li> kullanarak).
    3. Sayfanın en son satırına 'RISK_SCORE: [sayı]' yaz (0-100 arası).
    """

    # Yalnızca %100 Ücretsiz Katman Destekli Hızlı Flash Modeli
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        output = response.text.strip()
        score = 50.0
        if "RISK_SCORE:" in output:
            parts = output.rsplit("RISK_SCORE:", 1)
            output_text = parts[0].strip()
            try:
                score = float(parts[1].strip().split()[0])
            except ValueError:
                score = 50.0
            return score, output_text
        return 50.0, output
    except Exception:
        # Ücretsiz kota anlık aşılırsa çökme yapmaz, canlı haber listesini basar
        fallback_html = "<ul>" + "".join([f"<li>🟡 {t}</li>" for t in raw_titles[:6]]) + "</ul>"
        return 50.0, f"<p><em>⚠️ Canlı Akış Haber Başlıkları:</em></p>{fallback_html}"

# Yan Menü
st.sidebar.header("⚙️ Terminal Kontrolü")
if st.sidebar.button("🔄 Piyasayı & Haberleri Yenile"):
    st.cache_data.clear()

news_risk, news_analysis_html = fetch_and_analyze_news_live()

# Üst Metrik Kartları
col1, col2, col3 = st.columns(3)
col1.metric("🧠 Makro Haber Risk Skoru", f"%{news_risk:.1f}")
col2.metric("📊 Önerilen Hisse Ağırlığı", f"%{100.0 - news_risk:.1f}")
col3.metric("💵 Önerilen Nakit Ağırlığı", f"%{news_risk:.1f}")

st.markdown("---")

# Haber Analizi Sekmesi
st.subheader("🌍 Canlı Küresel & Yerel Piyasa Haber Analizi")
st.markdown(f'<div class="news-box">{news_analysis_html}</div>', unsafe_allow_html=True)

st.markdown("---")
st.subheader("🏢 Sektörel Hisse Analizleri ve Çoklu Zaman Dilimi Hedefleri")

selected_cat = st.selectbox("İncelemek İstediğiniz Sektörü Seçin (A-Z):", list(STOCK_CATEGORIES.keys()))

tickers = STOCK_CATEGORIES[selected_cat]
df_bist = yf.download(tickers, period="300d")['Close'].ffill().bfill()
log_returns = np.log(df_bist / df_bist.shift(1)).dropna()

timeframes = {
    'Günlük': 1, 'Haftalık': 5, 'Aylık': 21,
    '3 Aylık': 63, '6 Aylık': 126, '1 Yıllık': 252,
    '2 Yıllık': 504, '3 Yıllık': 756
}

for ticker in tickers:
    clean_symbol = ticker.replace('.IS', '')
    current_price = df_bist[ticker].iloc[-1]
    hist_ret = log_returns[ticker]

    ann_vol = hist_ret.iloc[-126:].std() * np.sqrt(252)
    daily_drift = hist_ret.iloc[-126:].mean()
    rsi_val = compute_rsi(hist_ret).iloc[-1]

    macro_sentiment_factor = 1.0 - ((news_risk - 50) / 100.0)
    adjusted_daily_drift = daily_drift * macro_sentiment_factor

    tf_data = []
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

        if pct_change > 4 and news_risk < 65 and rsi_val < 65:
            sig = "AL / Yüksek Potansiyel 🟢"
        elif pct_change < -3 or news_risk > 70 or rsi_val > 70:
            sig = "SAT / Riskli & Nakit 🔴"
        else:
            sig = "TUT / Dengeli Pozisyon 🟡"

        tf_data.append({
            "Zaman Dilimi": tf_name,
            "Beklenen Oynaklık": f"%{vol_period:.2f}",
            "Tahmini Fiyat (% Hedef)": f"{target_price:.2f} TL ({pct_change:+.2f}%)",
            "Tazelenmiş Sinyal": sig,
            "Sistem Notu": "Akış Düzeltmeli"
        })

    with st.expander(f"📌 {clean_symbol} | Fiyat: {current_price:.2f} TL | RSI: {rsi_val:.1f}", expanded=True):
        st.table(pd.DataFrame(tf_data))
