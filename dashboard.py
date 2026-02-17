import streamlit as st
from advanced_stock_monitor import crew

st.set_page_config(page_title="Stock Portfolio Monitor", layout="wide")

st.title("Autonomous Stock Portfolio Monitor Dashboard")
st.markdown("**Multi-Agent AI System • CrewAI + Real Market Data + LSTM Forecasting**")

portfolio = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA", "NVDA"]

col1, col2 = st.columns([1, 3])

with col1:
    if st.button("Generate Latest Report", type="primary", use_container_width=True):
        with st.spinner("Running multi-agent analysis..."):
            result = crew.kickoff()
            st.session_state.report = str(result)

with col2:
    st.markdown("### Portfolio")
    st.write(" • ".join(portfolio))

if "report" in st.session_state:
    st.markdown("### 📊 Daily AI-Generated Report")
    st.markdown(st.session_state.report.replace("\n", "  \n"))

st.caption("Built with **3 specialized AI agents** • Real yfinance data • LSTM forecasting • Deployed on Streamlit Cloud")
