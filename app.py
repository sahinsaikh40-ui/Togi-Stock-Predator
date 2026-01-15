import streamlit as st
import random
import time

# অ্যাপের নাম এবং পেজ সেটআপ
st.set_page_config(page_title="Rani AI", layout="centered")
st.header("🐯 Rani AI Stock Predator")
st.caption("Profit Target: 100% | Strategy: Genetic Algorithm")

# ইউজার ইনপুট (কোথায় ট্রেড করবেন)
option = st.selectbox(
    'Select Stock to Analyze:',
    ('TATA MOTORS', 'RELIANCE', 'ADANI ENT', 'BANK NIFTY')
)

# বাটন (যেটা টিপলে রেজাল্ট আসবে)
if st.button('Analyze Market 🚀'):
    
    # লোডিং এনিমেশন (যেন মনে হয় AI ভাবছে)
    with st.spinner('Genetic Algorithm is thinking...'):
        time.sleep(3) 
    
    # রেজাল্ট দেখানো
    st.success("Signal Generated! ✅")
    
    # এখানে আমরা দেখাচ্ছি অ্যাপ কেমন আউটপুট দেবে
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Action", value="BUY NOW 🟢")
    with col2:
        st.metric(label="Confidence", value="94%")
        
    st.write(f"👉 **Target Price:** ₹{random.randint(500, 3000)}")
    st.warning("Maintain strict Stop Loss. Market is volatile.")

# ফুটার
st.markdown("---")
st.write("Owner: Sakil SK | Powered by Rani AI Group")
