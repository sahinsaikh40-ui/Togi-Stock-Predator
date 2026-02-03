import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from SmartApi import SmartConnect 
import pyotp
import joblib
import os
import time
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from textblob import TextBlob 
from sklearn.ensemble import RandomForestRegressor

# ==========================================
# 🔐 0. LOGIN SYSTEM (NEW ADDITION)
# ==========================================
st.set_page_config(page_title="TOGI ANALYTICS PRO", page_icon="🦁", layout="wide", initial_sidebar_state="collapsed")

# ইউজারনেম এবং পাসওয়ার্ড লিস্ট (আপনি চাইলে আরও বাড়াতে পারেন)
USERS = {
    "admin": "admin123",    # আপনার জন্য
    "user": "12345",        # কাস্টমারের জন্য
    "demo": "demo"          # ডেমো ইউজার
}

def login_screen():
    st.markdown("""
        <style>
        .stApp {background-color: #000000;}
        .login-box {
            width: 350px; margin: auto; padding: 30px;
            border: 2px solid #00ff41; border-radius: 15px;
            background-color: #0e0e0e; text-align: center;
            box-shadow: 0 0 20px rgba(0, 255, 65, 0.2); margin-top: 100px;
        }
        .login-title { color: #00ff41; font-size: 24px; font-weight: bold; margin-bottom: 20px; }
        .stTextInput input { color: white !important; }
        </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        st.markdown("<div class='login-box'><div class='login-title'>🦁 TOGI ACCESS</div>", unsafe_allow_html=True)
        username = st.text_input("Username", placeholder="Enter ID")
        password = st.text_input("Password", type="password", placeholder="Enter Password")
        
        if st.button("SECURE LOGIN", type="primary", use_container_width=True):
            if username in USERS and USERS[username] == password:
                st.session_state['authenticated'] = True
                st.session_state['user'] = username
                st.success("Access Granted! Loading Engine...")
                time.sleep(1)
                st.rerun()
            else:
                st.error("❌ Invalid Access Credentials")
        st.markdown("</div>", unsafe_allow_html=True)

# সেশন চেক
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

if not st.session_state['authenticated']:
    login_screen()
    st.stop()  # লগিন না করলে কোড এখানে থেমে যাবে, নিচের কিছু দেখাবে না

# ==========================================
# 🚀 MAIN APP STARTS HERE (আপনার আগের কোড)
# ==========================================

# Sidebar Logout Button
with st.sidebar:
    st.write(f"👤 User: **{st.session_state['user']}**")
    if st.button("LOGOUT", type="secondary"):
        st.session_state['authenticated'] = False
        st.rerun()

st.markdown("""
    <style>
    .stApp { background-color: #000000; color: #00ff41; font-family: 'Consolas', monospace; }
    div[data-testid="stMetricValue"] { font-size: 24px !important; color: #00ff41 !important; text-shadow: 0 0 10px rgba(0,255,65,0.6); }
    .titan-card { border: 1px solid #333; padding: 20px; border-radius: 12px; background: #0b0b0b; text-align: center; margin-bottom: 20px; }
    .buy-glow { border: 2px solid #00ff00; box-shadow: 0 0 30px rgba(0, 255, 0, 0.4); }
    .sell-glow { border: 2px solid #ff0000; box-shadow: 0 0 30px rgba(255, 0, 0, 0.4); }
    .news-box { background: #111; padding: 10px; border-left: 3px solid #00e676; margin-top: 5px; font-size: 12px; color: #ccc; }
    .legal-warning { font-size: 10px; color: #666; text-align: center; margin-top: 50px; }
    section[data-testid="stSidebar"] { background-color: #050505; border-right: 1px solid #222; }
    </style>
""", unsafe_allow_html=True)

try:
    from tensorflow.keras.models import load_model
except: pass

# ... [বাকি সব ফাংশন এবং লজিক যা আপনার ছিল সব এখানে থাকবে] ...
# (কোড বড় না করার জন্য আমি নিচের অংশটুকু রিপিট করলাম না, কিন্তু আপনি আপনার আগের কোডের 
# "2. FULL STOCK UNIVERSE" থেকে শুরু করে শেষ পর্যন্ত এই লগিন পার্টের নিচে বসিয়ে দেবেন।)

# ==========================================
# 2. FULL STOCK UNIVERSE (Copy your previous code from here downwards)
# ==========================================
@st.cache_data
def get_full_stock_list():
    return [
        "NIFTY 50", "BANK NIFTY", "FINNIFTY",
        "AARTIIND", "ABB", "ABBOTINDIA", "ABCAPITAL", "ABFRL", "ACC", "ADANIENSOL", "ADANIENT", 
        "ADANIPORTS", "ALKEM", "AMBUJACEM", "APOLLOHOSP", "APOLLOTYRE", "ASHOKLEY", "ASIANPAINT", 
        "ASTRAL", "ATUL", "AUBANK", "AUROPHARMA", "AXISBANK", "BAJAJ-AUTO", "BAJAJFINSV", 
        "BAJFINANCE", "BALKRISIND", "BALRAMCHIN", "BANDHANBNK", "BANKBARODA", "BATAINDIA", "BEL", 
        "BERGEPAINT", "BHARATFORG", "BHARTIARTL", "BHEL", "BIOCON", "BOSCHLTD", "BPCL", "BRITANNIA", 
        "BSOFT", "CANBK", "CANFINHOME", "CHAMBLFERT", "CHOLAFIN", "CIPLA", "COALINDIA", "COFORGE", 
        "COLPAL", "CONCOR", "COROMANDEL", "CROMPTON", "CUB", "CUMMINSIND", "DABUR", "DALBHARAT", 
        "DEEPAKNTR", "DIVISLAB", "DIXON", "DLF", "DRREDDY", "EICHERMOT", "ESCORTS", "EXIDEIND", 
        "FEDERALBNK", "GAIL", "GLENMARK", "GMRINFRA", "GNFC", "GODREJCP", "GODREJPROP", "GRANULES", 
        "GRASIM", "GUJGASLTD", "HAL", "HAVELLS", "HCLTECH", "HDFCAMC", "HDFCBANK", "HDFCLIFE", 
        "HEROMOTOCO", "HINDALCO", "HINDCOPPER", "HINDPETRO", "HINDUNILVR", "ICICIBANK", "ICICIGI", 
        "ICICIPRULI", "IDEA", "IDFC", "IDFCFIRSTB", "IEX", "IGL", "INDHOTEL", "INDIACEM", 
        "INDIAMART", "INDIGO", "INDUSINDBK", "INDUSTOWER", "INFY", "IOC", "IPCALAB", "IRCTC", 
        "ITC", "JINDALSTEL", "JKCEMENT", "JSWSTEEL", "JUBLFOOD", "KOTAKBANK", "L&T", "LALPATHLAB", 
        "LAURUSLAB", "LICHSGFIN", "LT", "LTIM", "LTTS", "LUPIN", "M&M", "M&MFIN", "MARICO", 
        "MARUTI", "MCDOWELL-N", "MCX", "METROPOLIS", "MFSL", "MGL", "MOTHERSON", "MPHASIS", "MRF", 
        "MUTHOOTFIN", "NATIONALUM", "NAUKRI", "NAVINFLUOR", "NESTLEIND", "NMDC", "NTPC", 
        "OBEROIRLTY", "OFSS", "ONGC", "PAGEIND", "PEL", "PERSISTENT", "PETRONET", "PFC", 
        "PIDILITIND", "PIIND", "PNB", "POLYCAB", "POWERGRID", "PVRINOX", "RAMCOCEM", "RBLBANK", 
        "RECLTD", "RELIANCE", "SAIL", "SBICARD", "SBILIFE", "SBIN", "SHREECEM", "SHRIRAMFIN", 
        "SIEMENS", "SRF", "SUNPHARMA", "SUNTV", "SYNGENE", "TATACHEM", "TATACOMM", "TATACONSUM", 
        "TATAMOTORS", "TATAPOWER", "TATASTEEL", "TCS", "TECHM", "TITAN", "TORNTPHARM", "TRENT", 
        "TVSMOTOR", "UBL", "ULTRATECH", "UPL", "VEDL", "VOLTAS", "WIPRO", "ZEEL", "ZYDUSLIFE"
    ]

# ==========================================
# 3. SIDEBAR & CREDENTIALS
# ==========================================
with st.sidebar:
    st.title("🦁 SYSTEM ACCESS")
    # --- CHANGE: Use st.secrets for safety online ---
    if 'ANGEL_API_KEY' in st.secrets:
         ANGEL_API_KEY = st.secrets["ANGEL_API_KEY"]
         CLIENT_ID = st.secrets["CLIENT_ID"]
         PIN = st.secrets["PIN"]
         TOTP_SECRET = st.secrets["TOTP_SECRET"]
    else:
         ANGEL_API_KEY = st.text_input("API Key", "5tFV625l", type="password")
         CLIENT_ID = st.text_input("Client ID", "S52295271")
         PIN = st.text_input("PIN", "0303", type="password")
         TOTP_SECRET = st.text_input("TOTP", "F3XXND3SY2NOCDULEBP7SBXGQU", type="password")
    
    st.divider()
    st.markdown("### 🛠️ ANALYTICS MODULES")
    use_rf = st.checkbox("AI Projection", True)
    use_mc = st.checkbox("Monte Carlo Sim", True)
    
    st.info("ℹ️ This tool provides technical analysis data only. Not investment advice.")

@st.cache_resource
def init_connection(api_key, client, pin, totp):
    try:
        api = SmartConnect(api_key=api_key)
        t = pyotp.TOTP(totp).now()
        d = api.generateSession(client, pin, t)
        if d['status']: return api, True
    except: pass
    return None, False

smart_api, is_online = init_connection(ANGEL_API_KEY, CLIENT_ID, PIN, TOTP_SECRET)

# ==========================================
# 4. SMART DATA ENGINE
# ==========================================
def get_smart_period(interval):
    if interval in ["1m", "2m"]: return "5d"
    elif interval in ["5m", "15m", "30m"]: return "1mo"
    elif interval == "1h": return "6mo"
    else: return "2y"

def get_data(ticker, interval):
    if ticker == "NIFTY 50": ticker = "^NSEI"
    elif ticker == "BANK NIFTY": ticker = "^NSEBANK"
    elif ticker == "FINNIFTY": ticker = "NIFTY_FIN_SERVICE.NS"
    elif not ticker.endswith(".NS") and not ticker.startswith("^"): ticker += ".NS"
    
    period = get_smart_period(interval)
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    
    nifty = yf.download("^NSEI", period=period, interval=interval, progress=False)
    if isinstance(nifty.columns, pd.MultiIndex): nifty.columns = nifty.columns.get_level_values(0)
    
    vix = 0
    try:
        v_df = yf.download("^INDIAVIX", period="5d", progress=False)
        if not v_df.empty: vix = v_df['Close'].iloc[-1]
    except: pass
    
    return df, nifty, float(vix)

# ==========================================
# 5. ALL INDICATORS (NO LOGIC CHANGE)
# ==========================================
def add_indicators(df, nifty_df):
    if df.empty: return df
    
    # --- SCALPING ---
    df['EMA9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    
    delta = df['Close'].diff()
    df['RSI'] = 100 - (100 / (1 + (delta.where(delta>0,0).rolling(14).mean() / -delta.where(delta<0,0).rolling(14).mean())))
    min_rsi = df['RSI'].rolling(14).min(); max_rsi = df['RSI'].rolling(14).max()
    df['StochRSI'] = (df['RSI'] - min_rsi) / (max_rsi - min_rsi)
    df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()

    # --- INVESTING ---
    df['TR'] = np.maximum(df['High'] - df['Low'], np.abs(df['High'] - df['Close'].shift(1)))
    df['ATR'] = df['TR'].rolling(10).mean()
    hl2 = (df['High'] + df['Low']) / 2
    df['Up'] = hl2 + (3 * df['ATR']); df['Dn'] = hl2 - (3 * df['ATR'])
    st = df['Up'].copy(); trend = np.ones(len(df))
    for i in range(1, len(df)):
        if df['Close'].iloc[i] > st.iloc[i-1]:
            trend[i] = 1; st.iloc[i] = max(df['Dn'].iloc[i], st.iloc[i-1]) if trend[i-1] == 1 else df['Dn'].iloc[i]
        else:
            trend[i] = -1; st.iloc[i] = min(df['Up'].iloc[i], st.iloc[i-1]) if trend[i-1] == -1 else df['Up'].iloc[i]
    df['SuperTrend'] = st
    df['Trend'] = np.where(df['Close'] > df['SuperTrend'], "BULLISH", "BEARISH")

    x = np.arange(len(df)); y = df['Close'].values
    slope, intercept = np.polyfit(x, y, 1)
    df['LinReg'] = slope * x + intercept
    std_dev = df['Close'].std()
    df['LinReg_Up'] = df['LinReg'] + (2 * std_dev); df['LinReg_Low'] = df['LinReg'] - (2 * std_dev)

    if not nifty_df.empty:
        nifty_aligned = nifty_df['Close'].reindex(df.index).fillna(method='ffill')
        df['RS_Alpha'] = np.where(df['Close'].pct_change(14) > nifty_aligned.pct_change(14), "STRONG", "WEAK")
    else: df['RS_Alpha'] = "NEUTRAL"

    price_bins = pd.cut(df['Close'], bins=50)
    df['POC'] = df.groupby(price_bins)['Volume'].sum().idxmax().mid

    # Smart Money (Renamed for Compliance)
    df['PriceChange'] = df['Close'].pct_change()
    df['VolChange'] = df['Volume'].pct_change()
    
    conditions = [
        (df['PriceChange'] > 0) & (df['VolChange'] > 0), 
        (df['PriceChange'] < 0) & (df['VolChange'] > 0), 
        (df['PriceChange'] < 0) & (df['VolChange'] < 0), 
        (df['PriceChange'] > 0) & (df['VolChange'] < 0)
    ]
    # LEGAL SAFE TERMS
    choices = ["Accumulation (Bullish)", "Distribution (Bearish)", "Long Unwinding", "Short Covering"]
    df['SmartMoney'] = np.select(conditions, choices, default="Neutral")

    df['BB_Mid'] = df['Close'].rolling(20).mean(); std = df['Close'].rolling(20).std()
    df['BB_Up'] = df['BB_Mid'] + (2*std); df['BB_Low'] = df['BB_Mid'] - (2*std)
    df['Squeeze'] = np.where((df['BB_Up']-df['BB_Low'])/df['BB_Mid']*100 < 5, "ON", "OFF")
    
    return df

# ==========================================
# 6. INTELLIGENCE ENGINE
# ==========================================
@st.cache_resource
def load_models():
    if os.path.exists('togi_super_brain.h5'):
        try: return load_model('togi_super_brain.h5'), joblib.load('togi_super_scaler.pkl')
        except: pass
    return None, None

lstm_model, scaler = load_models()

def get_ai_prediction(df):
    try:
        data = df.copy().dropna()
        data['Target'] = data['Close'].shift(-1)
        features = ['Close', 'Open', 'High', 'Low', 'Volume', 'RSI']
        data = data.dropna()
        X = data[features][:-1]; y = data['Target'][:-1]
        rf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
        rf.fit(X, y)
        return rf.predict(data[features].iloc[[-1]])[0]
    except: return 0

def run_monte_carlo(df):
    returns = df['Close'].pct_change().dropna()
    mu = returns.mean(); sigma = returns.std()
    last = df['Close'].iloc[-1]
    sim_df = pd.DataFrame()
    for x in range(100):
        p = [last]
        for d in range(10): p.append(p[-1] * (1 + np.random.normal(mu, sigma)))
        sim_df[x] = p
    return sim_df, sim_df.iloc[-1].mean()

def analyze_news(ticker):
    if not ticker.endswith(".NS"): ticker += ".NS"
    score = 0; headlines = []
    try:
        t = yf.Ticker(ticker); news = t.news
        if news:
            for item in news[:3]:
                title = item.get('title', '')
                headlines.append(title); blob = TextBlob(title)
                if blob.sentiment.polarity > 0.1: score += 1
                elif blob.sentiment.polarity < -0.1: score -= 1
            return (1 if score > 0 else -1 if score < 0 else 0), headlines
    except: pass
    return 0, headlines

# ==========================================
# 7. DASHBOARD UI (LEGAL COMPLIANT)
# ==========================================
st.title("🦁 TOGI ANALYTICS <span style='color:#00ff41'>[PRO]</span>", anchor=False)

if is_online: st.success("🟢 DATA SOURCE: ANGEL ONE (LIVE)")
else: st.warning("🟠 DATA SOURCE: YAHOO FINANCE (DELAYED)")

tabs = st.tabs(["📊 TERMINAL", "⚡ MOMENTUM", "📡 SCREENER", "💰 CALCULATOR"])

# --- TAB 1: TERMINAL ---
with tabs[0]:
    c1, c2, c3, c4 = st.columns([1.5, 1.5, 0.8, 0.8])
    stock_list = get_full_stock_list()
    target = c2.text_input("SEARCH", "").upper() or c1.selectbox("ASSET", stock_list)
    
    tf = c3.selectbox("TIME", ["1m", "2m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo"])
    btn = c4.button("ANALYZE", type="primary")

    if btn:
        symbol = "^NSEI" if target=="NIFTY 50" else "^NSEBANK" if target=="BANK NIFTY" else target
        with st.spinner(f"🚀 Processing Data for {target}..."):
            df, nifty, vix = get_data(symbol, tf)
            
            if not df.empty:
                mc_df = None; mc_avg = 0
                df = add_indicators(df, nifty)
                ltp = df['Close'].iloc[-1]
                atr = df['ATR'].iloc[-1]
                smart_money = df['SmartMoney'].iloc[-1]
                ai_price = get_ai_prediction(df)
                news_score, headlines = analyze_news(symbol)
                
                if use_mc and tf not in ["1m", "2m", "5m"]: mc_df, mc_avg = run_monte_carlo(df)

                # --- SCORING LOGIC ---
                score = 50; reasons = []
                
                if "Accumulation" in smart_money: score += 10; reasons.append("Vol Analysis: Positive")
                elif "Distribution" in smart_money: score -= 10; reasons.append("Vol Analysis: Negative")
                
                if tf in ["1m", "2m", "5m", "15m", "30m"]:
                    if df['EMA9'].iloc[-1] > df['EMA20'].iloc[-1]: score += 10; reasons.append("Trend: Upward")
                    else: score -= 10
                    if ltp > df['VWAP'].iloc[-1]: score += 10; reasons.append("Price > VWAP")
                    else: score -= 10
                else:
                    if df['Trend'].iloc[-1] == "BULLISH": score += 10; reasons.append("SuperTrend: Positive")
                    else: score -= 10
                    if ltp < df['LinReg_Low'].iloc[-1]: score += 15; reasons.append("Statistically Undervalued")
                    elif ltp > df['LinReg_Up'].iloc[-1]: score -= 15; reasons.append("Statistically Overvalued")

                if ltp > df['POC'].iloc[-1]: score += 5; reasons.append("Above High Vol Node")
                if ai_price > ltp: score += 10
                if news_score == 1: score += 10

                # Target Adjust
                if score >= 60 and ai_price < ltp: ai_price = ltp + (atr * 2)
                elif score <= 40 and ai_price > ltp: ai_price = ltp - (atr * 2)

                # UI - LEGAL TERMS USED HERE
                st.divider()
                sc, opt = st.columns([1.5, 1])
                with sc:
                    if score >= 70:
                        st.markdown(f"<div class='titan-card buy-glow'><h1>🚀 BULLISH MOMENTUM</h1><h3>Proj. Res: ₹{ai_price:.2f}</h3><p>{' | '.join(reasons)}</p></div>", unsafe_allow_html=True)
                    elif score <= 30:
                        st.markdown(f"<div class='titan-card sell-glow'><h1>🔻 BEARISH MOMENTUM</h1><h3>Proj. Sup: ₹{ai_price:.2f}</h3><p>{' | '.join(reasons)}</p></div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div class='titan-card'><h1>⏳ NEUTRAL VIEW</h1><p>Score: {score}</p></div>", unsafe_allow_html=True)
                    if headlines:
                        for h in headlines: st.markdown(f"<div class='news-box'>{h}</div>", unsafe_allow_html=True)
                
                with opt:
                    st.markdown("### 🏦 VOLUME ANALYSIS")
                    st.info(f"Sentiment: {smart_money}")
                    if tf in ["1m", "2m", "5m", "15m", "30m"]:
                        st.markdown("---")
                        step = 50 if "NIFTY" in target else 100
                        atm = round(ltp / step) * step
                        st.markdown("#### ATM STRIKES (DATA)")
                        st.success(f"Call Strike: {int(atm)} | Sup: {int(ltp - atr)}")
                        st.error(f"Put Strike: {int(atm)} | Res: {int(ltp + atr)}")
                    else:
                        st.metric("TECH SCORE", f"{score}/100")

                # Charts
                tab1, tab2 = st.tabs(["📈 TECH CHART", "🎲 PROJECTION"])
                with tab1:
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
                    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']), row=1, col=1)
                    if tf in ["1m", "2m", "5m", "15m", "30m"]: 
                        fig.add_trace(go.Scatter(x=df.index, y=df['EMA9'], line=dict(color='yellow', width=1), name="EMA 9"), row=1, col=1)
                        fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], line=dict(color='magenta', width=1), name="EMA 20"), row=1, col=1)
                        fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], line=dict(color='cyan', dash='dot'), name="VWAP"), row=1, col=1)
                    else: 
                        fig.add_trace(go.Scatter(x=df.index, y=df['LinReg'], line=dict(color='yellow'), name="Fair Value"), row=1, col=1)
                        fig.add_trace(go.Scatter(x=df.index, y=df['SuperTrend'], line=dict(color='orange'), name="SuperTrend"), row=1, col=1)
                    fig.add_hline(y=df['POC'].iloc[-1], line_dash="dash", line_color="white", annotation_text="POC")
                    fig.add_trace(go.Bar(x=df.index, y=df['Volume']), row=2, col=1)
                    fig.update_layout(template="plotly_dark", height=600, plot_bgcolor='#000', paper_bgcolor='#000')
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    if use_mc and mc_df is not None:
                        fig_mc = go.Figure()
                        for c in mc_df.columns[:30]: fig_mc.add_trace(go.Scatter(y=mc_df[c], mode='lines', line=dict(width=1, color='rgba(0,255,65,0.1)'), showlegend=False))
                        fig_mc.update_layout(template="plotly_dark", height=400, title="Monte Carlo Projection")
                        st.plotly_chart(fig_mc, use_container_width=True)
                    else: st.info("Projection requires >15m timeframe")

# --- TAB 2: MOMENTUM ---
with tabs[1]:
    st.header("⚡ MOMENTUM SCANNER (5M)")
    if st.button("SCAN MARKET"):
        s_list = get_full_stock_list()[:10] 
        res = []
        bar = st.progress(0)
        for i, s in enumerate(s_list):
            d, n, _ = get_data(s, "5m")
            if not d.empty:
                d = add_indicators(d, n)
                act = "NEUTRAL"
                if d['EMA9'].iloc[-1] > d['EMA20'].iloc[-1] and d['Close'].iloc[-1] > d['VWAP'].iloc[-1]: act = "BULLISH"
                elif d['EMA9'].iloc[-1] < d['EMA20'].iloc[-1] and d['Close'].iloc[-1] < d['VWAP'].iloc[-1]: act = "BEARISH"
                res.append({"Stock": s, "View": act, "Vol Analysis": d['SmartMoney'].iloc[-1]})
            bar.progress((i+1)/len(s_list))
        st.dataframe(pd.DataFrame(res).style.applymap(lambda x: 'color:#0f0' if 'BULLISH' in x else 'color:#f00' if 'BEARISH' in x else 'color:#555', subset=['View']), use_container_width=True)

# --- TAB 3: SCREENER ---
with tabs[2]:
    st.header("📡 QUANT SCREENER (1 DAY)")
    if st.button("RUN SCAN"):
        s_list = get_full_stock_list()[:10]
        res = []
        bar = st.progress(0)
        for i, s in enumerate(s_list):
            d, n, _ = get_data(s, "1d")
            if not d.empty:
                d = add_indicators(d, n)
                act = "NEUTRAL"
                if d['Trend'].iloc[-1]=="BULLISH" and d['RS_Alpha'].iloc[-1]=="STRONG": act = "POSITIVE"
                res.append({"Stock": s, "Trend": d['Trend'].iloc[-1], "Alpha": d['RS_Alpha'].iloc[-1], "View": act})
            bar.progress((i+1)/len(s_list))
        st.dataframe(pd.DataFrame(res), use_container_width=True)

# --- TAB 4: CALCULATOR ---
with tabs[3]:
    st.header("💰 POSITION SIZING")
    c1, c2 = st.columns(2)
    cap = c1.number_input("Capital", 50000); risk = c2.number_input("Risk Amount", 1000); sl = st.number_input("Stop Loss Pts", 10.0)
    if st.button("CALCULATE"): st.metric("SUGGESTED QTY", f"{int(risk/sl)}")

# DISCLAIMER FOOTER
st.markdown("---")
st.markdown("<div class='legal-warning'>DISCLAIMER: This application is an Algorithmic Analysis Tool for educational purposes only. We are NOT SEBI Registered. All levels (Support/Resistance) are generated by automated mathematical formulas and AI. Please consult your financial advisor before trading. We are not responsible for any profit or loss.</div>", unsafe_allow_html=True)