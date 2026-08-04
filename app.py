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

# Streamlit Sayfa Konfigürasyonu
st.set_page_config(
    page_title="Quant Physics BIST Terminal",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Mobil Uyumlu ve Sabit Koyu Tema CSS
st.markdown("""
<style>
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0f172a !important;
        color: #f8fafc !important;
    }
    
    [data-testid="stSidebar"] {
        background-color: #020617 !important;
    }
    
    .stMetric { 
        background-color: #1e293b !important; 
        padding: 12px; 
        border-radius: 8px; 
        border: 1px solid #334155 !important; 
        margin-bottom: 10px;
    }
    
    .disclaimer-box { 
        background-color: #1e1b4b !important; 
        border: 1px solid #4338ca !important; 
        border-radius: 8px; 
        padding: 12px; 
        margin-bottom: 15px; 
        color: #cbd5e1 !important; 
        font-size: 12px; 
        line-height: 1.4;
    }
    
    .news-box { 
        background-color: #1e293b !important; 
        border: 1px solid #334155 !important; 
        border-radius: 8px; 
        padding: 15px; 
        margin-bottom: 15px; 
        font-size: 13px; 
        line-height: 1.5; 
        color: #f8fafc !important;
        overflow-x: auto;
    }

    .fund-badge {
        background-color: #0f172a;
        border: 1px solid #334155;
        border-radius: 6px;
        padding: 6px 12px;
        margin-right: 8px;
        margin-bottom: 8px;
        display: inline-block;
        font-size: 12px;
    }

    @media (max-width: 768px) {
        .stMetric { padding: 10px; }
        .stMetric [data-testid="stMetricValue"] { font-size: 1.3rem !important; }
        h1 { font-size: 1.5rem !important; }
        h2 { font-size: 1.2rem !important; }
        h3 { font-size: 1.1rem !important; }
    }
</style>
""", unsafe_allow_html=True)

# Yasal Uyarı
st.markdown("""
<div class="disclaimer-box">
    <strong>⚖️ YASAL UYARI:</strong> Bu web sitesinde sunulan duygu analizleri, makine öğrenimi tahminleri ve kuantitatif veriler 
    <strong>yatırım danışmanlığı kapsamında değildir.</strong> Yalnızca akademik, istatistiksel ve eğitim amaçlı kuantitatif modeller içerir.
</div>
""", unsafe_allow_html=True)

st.title("BIST Terminali")

# Standart Sektör Havuzu
RAW_STOCK_CATEGORIES = {
    'BANKACILIK & FİNANS': ['AKBNK.IS', 'GARAN.IS', 'HALKB.IS', 'ISCTR.IS', 'TSKB.IS', 'VAKBN.IS', 'YKBNK.IS'],
    'ENERJİ & MADENCİLİK': ['AKSEN.IS', 'ASTOR.IS', 'CWENE.IS', 'ENJSA.IS', 'EUPWR.IS', 'GESAN.IS', 'PETKM.IS', 'TUPRS.IS', 'YEOTK.IS'],
    'HAVACILIK & LOJİSTİK': ['ENKAI.IS', 'PGSUS.IS', 'TAVHL.IS', 'THYAO.IS'],
    'HOLDİNG & YATIRIM': ['AGHOL.IS', 'ALARK.IS', 'DOHOL.IS', 'KCHOL.IS', 'SAHOL.IS', 'SISE.IS', 'TKFEN.IS'],
    'İLETİŞİM': ['TCELL.IS', 'TTKOM.IS'],
    'KİMYA & GAYRİMENKUL': ['ECILC.IS', 'EKGYO.IS', 'EREGL.IS', 'HEKTS.IS', 'KRDMD.IS', 'SASA.IS'],
    'OTOMOTİV & SANAYİ': ['ARCLK.IS', 'BRISA.IS', 'DOAS.IS', 'FROTO.IS', 'OTKAR.IS', 'TOASO.IS', 'VESBE.IS'],
    'PERAKENDE & GIDA': ['AEFES.IS', 'BIMAS.IS', 'CCOLA.IS', 'MGROS.IS', 'SOKM.IS', 'ULKER.IS'],
    'SAVUNMA & TEKNOLOJİ': ['ASELS.IS', 'KONTR.IS', 'MIATK.IS', 'REEDR.IS', 'SDTTR.IS']
}

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

@st.cache_data(ttl=1800)
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
        return 50.0, "<p>⚠️ GEMINI_API_KEY bulunamadı.</p>"

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
        fallback_html = "<ul>" + "".join([f"<li>🟡 {t}</li>" for t in raw_titles[:6]]) + "</ul>"
        return 50.0, f"<p><em>⚠️ Canlı Akış Haber Başlıkları:</em></p>{fallback_html}"

# 6 AY - 1 YIL PEAK ODAKLI OTONOM AI TARAMA MOTORU
@st.cache_data(ttl=1800)
def scan_peak_outlier_stocks(news_risk_score):
    search_universe = []
    for cat, t_list in RAW_STOCK_CATEGORIES.items():
        search_universe.extend(t_list)
    
    extra_candidates = [
        'YEOTK.IS', 'ANSGR.IS', 'ASTOR.IS', 'BVSAN.IS', 'CWENE.IS', 'GWIND.IS', 
        'INDES.IS', 'KAREL.IS', 'MIATK.IS', 'TKNSA.IS', 'EKGYO.IS', 'SKBNK.IS', 
        'TSKB.IS', 'TRGYO.IS', 'ODAS.IS', 'SNTRA.IS', 'KONTR.IS', 'ISGYO.IS', 
        'KRDMD.IS', 'ALBRK.IS', 'AKENR.IS', 'DOHOL.IS', 'MTRKS.IS', 'KLRHO.IS', 
        'EUPWR.IS', 'GESAN.IS', 'REEDR.IS', 'SDTTR.IS', 'HUNER.IS', 'MAGEN.IS'
    ]
    search_universe = list(set(search_universe + extra_candidates))
    all_symbols = search_universe + ['XU100.IS']

    try:
        df_data = yf.download(all_symbols, period="300d")
        df_c = df_data['Close'].ffill().bfill()
        df_v = df_data['Volume'].ffill().bfill()
    except Exception:
        return ['YEOTK.IS', 'INDES.IS', 'KAREL.IS', 'TSKB.IS', 'EKGYO.IS', 'DOHOL.IS', 'AKENR.IS', 'TRGYO.IS', 'SKBNK.IS', 'MTRKS.IS']

    scored_candidates = []
    macro_factor = 1.0 - ((news_risk_score - 50) / 100.0)

    for t in search_universe:
        if t not in df_c.columns or t == 'XU100.IS':
            continue
        try:
            c = df_c[t]
            v = df_v[t]
            curr_p = c.iloc[-1]
            
            # KURAL 1: Fiyat kesinlikle 50 TL ve altında olmalı
            if curr_p <= 50.0:
                log_ret = np.log(c / c.shift(1)).dropna()
                
                # 6 Ay ve 1 Yıl Getiri Drift İvmesi
                daily_drift = log_ret.iloc[-126:].mean()
                adjusted_drift = daily_drift * macro_factor
                
                # 1 Yıllık Tahmini Getiri Oranı (1Y Peak Potansiyeli)
                target_1y = curr_p * np.exp(adjusted_drift * 252)
                pct_1y = ((target_1y - curr_p) / curr_p) * 100.0

                stock_21d_ret = (c.iloc[-1] - c.iloc[-21]) / c.iloc[-21]
                bist_21d_ret = (df_c['XU100.IS'].iloc[-1] - df_c['XU100.IS'].iloc[-21]) / df_c['XU100.IS'].iloc[-21]
                relative_strength = stock_21d_ret - bist_21d_ret

                v_recent = v.iloc[-1]
                v_avg20 = v.iloc[-21:-1].mean()
                rvol = v_recent / (v_avg20 + 1e-9)
                
                # Peak Skoru: 1 Yıllık Yüzde Hedefi + İvme + Hacim
                peak_score = (pct_1y * 3.0) + (relative_strength * 100.0) + (rvol * 5.0)
                
                # Yükseliş yönlü ivmede olan şirketler
                if daily_drift > 0:
                    scored_candidates.append((t, peak_score, pct_1y))
        except Exception:
            continue

    # 1 Yıllık Peak potansiyeli ve ivmesi en yüksek olan 10 hisse
    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    selected_dark_horses = [x[0] for x in scored_candidates[:10]]

    # YEOTK'nin listede olmasını ve 10 taneye tamamlanmasını garantiye alan yapı
    if 'YEOTK.IS' not in selected_dark_horses:
        selected_dark_horses.insert(0, 'YEOTK.IS')
        selected_dark_horses = selected_dark_horses[:10]

    if len(selected_dark_horses) < 10:
        fallback_list = ['YEOTK.IS', 'INDES.IS', 'KAREL.IS', 'TSKB.IS', 'EKGYO.IS', 'DOHOL.IS', 'AKENR.IS', 'TRGYO.IS', 'SKBNK.IS', 'MTRKS.IS']
        for fb in fallback_list:
            if fb not in selected_dark_horses and len(selected_dark_horses) < 10:
                selected_dark_horses.append(fb)

    return sorted(selected_dark_horses)

# Yan Menü Kontrolü
st.sidebar.header("⚙️ Terminal Kontrolü")
if st.sidebar.button("🔄 Piyasayı & Haberleri Yenile"):
    st.cache_data.clear()

news_risk, news_analysis_html = fetch_and_analyze_news_live()

# Otonom Peak Tarama Motorunu Çalıştır
dynamic_dark_horses = scan_peak_outlier_stocks(news_risk)

# Sektör Menüsü
STOCK_CATEGORIES = {
    '🐎 DARK HORSE (PEAK & COMPOUNDER HİSSELER - ≤ 50 TL)': dynamic_dark_horses
}
for cat in sorted(RAW_STOCK_CATEGORIES.keys()):
    STOCK_CATEGORIES[cat] = sorted(RAW_STOCK_CATEGORIES[cat])

# Üst Metrik Kartları
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("🧠 Makro Risk Skoru", f"%{news_risk:.1f}")
    st.caption("Piyasa haberlerinin hesaplanan risk seviyesi.")
with col2:
    st.metric("📊 Hisse Ağırlığı", f"%{100.0 - news_risk:.1f}")
    st.caption("Portföyde tutulması önerilen hisse oranı.")
with col3:
    st.metric("💵 Nakit Ağırlığı", f"%{news_risk:.1f}")
    st.caption("Piyasa riskine karşı önerilen nakit oranı.")

st.markdown("---")

# Haber Analizi Sekmesi
st.subheader("🌍 Canlı Küresel & Yerel Piyasa Haber Analizi")
st.markdown(f'<div class="news-box">{news_analysis_html}</div>', unsafe_allow_html=True)

st.markdown("---")
st.subheader("🏢 Sektörel Hisse Analizleri, Bilanço Rasyoları ve Momentum")

selected_cat = st.selectbox("İncelemek İstediğiniz Sektörü Seçin (A-Z):", list(STOCK_CATEGORIES.keys()))

tickers = STOCK_CATEGORIES[selected_cat]
all_symbols = tickers + ['XU100.IS']
df_download = yf.download(all_symbols, period="300d")

df_close = df_download['Close'].ffill().bfill()
df_volume = df_download['Volume'].ffill().bfill()

log_returns = np.log(df_close / df_close.shift(1)).dropna()

timeframes = {
    'Günlük': 1, 'Haftalık': 5, 'Aylık': 21,
    '3 Aylık': 63, '6 Aylık': 126, '1 Yıllık': 252,
    '2 Yıllık': 504, '3 Yıllık': 756
}

for ticker in tickers:
    clean_symbol = ticker.replace('.IS', '')
    current_price = df_close[ticker].iloc[-1]
    hist_ret = log_returns[ticker]
    
    # RVOL & Relative Strength Hesaplamaları
    recent_volume = df_volume[ticker].iloc[-1]
    avg_volume_20d = df_volume[ticker].iloc[-21:-1].mean()
    rvol = recent_volume / (avg_volume_20d + 1e-9)

    stock_21d_ret = (df_close[ticker].iloc[-1] - df_close[ticker].iloc[-21]) / df_close[ticker].iloc[-21]
    bist_21d_ret = (df_close['XU100.IS'].iloc[-1] - df_close['XU100.IS'].iloc[-21]) / df_close['XU100.IS'].iloc[-21]
    relative_strength = stock_21d_ret - bist_21d_ret

    # Bilanço Metrikleri
    try:
        t_info = yf.Ticker(ticker).info
        pe_ratio = t_info.get('trailingPE', 'N/A')
        pb_ratio = t_info.get('priceToBook', 'N/A')
        profit_margin = t_info.get('profitMargins', 'N/A')
        
        pe_str = f"{pe_ratio:.2f}" if isinstance(pe_ratio, (int, float)) else "N/A"
        pb_str = f"{pb_ratio:.2f}" if isinstance(pb_ratio, (int, float)) else "N/A"
        pm_str = f"%{profit_margin*100:.1f}" if isinstance(profit_margin, (int, float)) else "N/A"
    except Exception:
        pe_str, pb_str, pm_str = "N/A", "N/A", "N/A"

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

        if pct_change > 4 and news_risk < 65 and rsi_val < 65 and rvol > 1.0 and relative_strength > 0:
            sig = "GÜÇLÜ AL 🟢🟢"
        elif pct_change > 3 and news_risk < 65 and rsi_val < 65:
            sig = "AL 🟢"
        elif pct_change < -3 or news_risk > 70 or rsi_val > 70 or (rvol < 0.7 and relative_strength < -0.05):
            sig = "SAT 🔴"
        else:
            sig = "TUT 🟡"

        tf_data.append({
            "Zaman Dilimi": tf_name,
            "Beklenen Oynaklık": f"%{vol_period:.2f}",
            "Tahmini Fiyat (% Hedef)": f"{target_price:.2f} TL ({pct_change:+.2f}%)",
            "Sinyal": sig
        })

    rvol_status = f"🟢 Yüksek Hacim (RVOL: {rvol:.2f})" if rvol >= 1.25 else f"🟡 Normal Hacim (RVOL: {rvol:.2f})"
    rs_status = f"🟢 BIST 100'den Güçlü (%{relative_strength*100:+.1f})" if relative_strength > 0 else f"🔴 BIST 100'den Zayıf (%{relative_strength*100:+.1f})"

    with st.expander(f"📌 {clean_symbol} | Fiyat: {current_price:.2f} TL | RSI: {rsi_val:.1f}", expanded=False):
        st.markdown(f"""
        <div style="margin-bottom:12px;">
            <span class="fund-badge">📑 F/K: <strong>{pe_str}</strong></span>
            <span class="fund-badge">📑 PD/DD: <strong>{pb_str}</strong></span>
            <span class="fund-badge">💰 Kâr Marjı: <strong>{pm_str}</strong></span>
            <span class="fund-badge">📊 {rvol_status}</span>
            <span class="fund-badge">⚖️ {rs_status}</span>
        </div>
        """, unsafe_allow_html=True)

        st.dataframe(
            pd.DataFrame(tf_data), 
            use_container_width=True, 
            hide_index=True
        )
