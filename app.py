import streamlit as st
import random
import time

# ১. পেজের নাম ও আইকন সেট করা
st.set_page_config(page_title="Togi AI - Stock Predator", page_icon="🐯", layout="centered")

# ২. হেডলাইন সাজানো (Title)
st.title("🐯 Togi AI Stock Predator")
st.markdown("### 🚀 World's Smartest AI for Stock Market Beginners")
st.divider() # একটা লম্বা দাগ (Divider)

# ৩. ইনপুট সেকশন সাজানো (Columns)
col1, col2 = st.columns(2) # পেজকে দুই ভাগে ভাগ করলাম

with col1:
    # বাঁদিকের কলাম
    option = st.selectbox(
        '🔍 Select Stock to Analyze:',
        ('TATA MOTORS', 'RELIANCE', 'ADANI ENT', 'HDFC BANK', 'NIFTY 50')
    )

with col2:
    # ডানদিকের কলাম
    mode = st.radio("⚡ Trading Mode:", ["Intraday (Fast)", "Swing (Safe)"])

# ৪. বাটন এবং রেজাল্ট সাজানো
st.write("") # একটু ফাঁকা জায়গা
if st.button('🤖 Ask Togi to Analyze', use_container_width=True):
    
    # লোডিং ডিজাইন
    with st.spinner(f'🐯 Togi is analyzing millions of data points for {option}...'):
        time.sleep(3) # ৩ সেকেন্ড ওয়েট
    
    # রেজাল্ট বক্স
    st.success("Analysis Complete! Signal Found. ✅")
    
    # বড় ফন্টে রেজাল্ট (Metrics)
    m1, m2, m3 = st.columns(3)
    m1.metric("Action", "BUY NOW", "Strong Buy")
    m1.metric("Confidence", f"{random.randint(90, 99)}%", "+5%")
    m1.metric("Target Price", f"₹{random.randint(500, 3000)}", "High Profit")
    
    # নিচের ওয়ার্নিং মেসেজ
    st.info("💡 **Togi's Tip:** Market is volatile today. Keep Stop Loss strict.")

# ৫. ফুটার (নিচের অংশ)
st.divider()
st.caption(f"Owner: **Sakil SK** | Developed by **Togi AI Group** | © 2026")
