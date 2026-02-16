import streamlit as st
import plotly.graph_objects as go
import yfinance as yf
import pandas as pd
from advanced_stock_monitor import crew

st.set_page_config(page_title="Stock Portfolio Monitor", layout="wide")
st.title("🚀 Autonomous Stock Portfolio Monitor Dashboard")
st.markdown("Built with CrewAI + Grok-4 + PyTorch LSTM Forecasting")

portfolio = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA", "NVDA"]  # Add from beprevious script

col1, col2 = st.columns([1, 3])
with col1:
    if st.button("Generate Latest Report", type="primary"):
        with st.spinner("AI agents are analyzing..."):
            result = crew.kickoff()
        st.session_state.report = str(result)

with col2:
    st.markdown("### Select Stock for Chart")
    selected_ticker = st.selectbox("Stock", portfolio)

if "report" in st.session_state:
    st.markdown("### 📊 Daily AI-Generated Report")
    st.markdown(st.session_state.report.replace("\n", "  \n"))

st.markdown(f"### 📈 {selected_ticker} - Price & Moving Averages (2Y)")
data = yf.download(selected_ticker, period="2y", progress=False)
data["SMA50"] = data["Close"].rolling(50).mean()
data["SMA200"] = data["Close"].rolling(200).mean()

fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data["Close"], mode='lines', name='Close Price'))
fig.add_trace(go.Scatter(x=data.index, y=data["SMA50"], mode='lines', name='50-day SMA', line=dict(dash='dash')))
fig.add_trace(go.Scatter(x=data.index, y=data["SMA200"], mode='lines', name='200-day SMA', line=dict(dash='dot')))
fig.update_layout(height=600, xaxis_title="Date", yaxis_title="Price ($)", template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)

st.markdown("### Tips")
st.info("Deploy this for free on Streamlit Community Cloud → GitHub repo → Connect. Add to resume: 'Deployed interactive multi-agent stock analysis dashboard with real-time charts and LSTM forecasting.'")