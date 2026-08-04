import os
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import requests
import xml.etree.ElementTree as ET

try:
    from google import genai
except ImportError:
    genai = None

# Streamlit Sayfa Konfigürasyonu
st.set_page_config(
    page_title="Quant BIST & AI Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Koyu Tema CSS Stili
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1e293b; padding: 15px; border-radius: 10px; border: 1px solid #334155; }
    .disclaimer-box { background-color: #1e1b4b; border: 1px solid #4338ca; border-radius: 8px; padding: 12px; margin-bottom: 20px; color: #cbd5e1; font-size: 13px; }
</style>
""", unsafe_allow_html=True)

# Yasal Uyarı Paneli (En Üstte Sabit)
st.markdown("""
<div class="disclaimer-box">
    <strong>⚖️ YASAL UYARI:</strong> Bu web sitesinde sunulan yapay zeka duygu analizleri, makine öğrenimi tahminleri ve kuantitatif veriler 
    <strong>yatırım danışmanlığı kapsamında değildir.</strong> Yalnızca istatistiksel ve eğitim amaçlıdır.
</div>
""", unsafe_allow_html=True)

st.title("📈 BIST 100 Quant Engine & Gemini AI Terminal")
st.caption("Küresel Piyasalar, Asya/Avrupa Açılışları ve BIST 100 Canlı Yapay Zeka Analiz Paneli")

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

@st.cache_data(ttl=1800) # 30 dakikada bir veya yenile butonunda haber çeker
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

    api_key = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
    if not api_key or not genai:
        return 50.0, "Yapay Zeka API Anahtarı tanımlanmadı."

    all_titles_text = "\n".join([f"- {t}" for t in raw_titles])

    prompt = f"""
    Sen BIST 100 ve Küresel Piyasalar Baş Analistisin.
    Aşağıda son haber başlıkları bulunmaktadır:
    {all_titles_text}

    BIST 100 üzerindeki etkisi en yüksek olan 5-8 kritik haberi seç ve Türkçe özeti ile [POZİTİF 🟢 / NEGATİF 🔴 / NÖTR 🟡] olarak değerlendir.
    En son satıra 'RISK_SCORE: [sayı]' yaz (0-100 arası).
    """

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
    except Exception as e:
        return 50.0, f"Yapay Zeka Analiz Hatası: {e}"

# Yan Menü & Canlı Yenileme
st.sidebar.header("⚙️ Terminal Paneli")
if st.sidebar.button("🔄 Piyasayı & Haberleri Anlık Yenile"):
    st.cache_data.clear()

news_risk, news_analysis = fetch_and_analyze_news_live()

# Üst Özet Metrikleri
col1, col2, col3 = st.columns(3)
col1.metric("🧠 Yapay Zeka Haber Risk Skoru", f"%{news_risk:.1f}")
col2.metric("📊 Önerilen Hisse Ağırlığı", f"%{100.0 - news_risk:.1f}")
col3.metric("💵 Önerilen Nakit Ağırlığı", f"%{news_risk:.1f}")

st.markdown("---")

# Gemini AI Haber Süzgeci Sekmesi
with st.expander("🌍 Gemini AI Süzgecinden Geçen Canlı Küresel & Yerel Haber Analizi", expanded=True):
    st.markdown(news_analysis)

st.markdown("---")
st.subheader("🏢 Sektörel Hisse Analizleri ve Kuantitatif Sinyaller")

selected_cat = st.selectbox("İncelemek İstediğiniz Sektörü Seçin:", list(STOCK_CATEGORIES.keys()))

tickers = STOCK_CATEGORIES[selected_cat]
df_data = yf.download(tickers, period="60d")['Close'].ffill().bfill()

stock_summary = []
for t in tickers:
    clean_t = t.replace('.IS', '')
    last_price = df_data[t].iloc[-1]
    ret_21d = (df_data[t].iloc[-1] - df_data[t].iloc[-21]) / df_data[t].iloc[-21] * 100
    stock_summary.append({
        "Hisse": clean_t,
        "Son Fiyat (TL)": f"{last_price:.2f}",
        "21 Günlük Değişim": f"%{ret_21d:+.2f}",
        "Sistem Sinyali": "DENGELİ / TUT" if news_risk < 60 else "KORUMALI / NAKİT"
    })

st.table(pd.DataFrame(stock_summary))
