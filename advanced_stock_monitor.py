import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
import yfinance as yf
import requests
from xml.etree import ElementTree as ET
import streamlit as st

from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool

PORTFOLIO = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA", "NVDA"]

llm = LLM(
    model="openrouter/meta-llama/llama-3.3-70b-instruct",
    base_url="https://openrouter.ai/api/v1",
    api_key=st.secrets["OPENROUTER_API_KEY"],
    temperature=0.0,
)

@tool("Stock Cross Checker")
def stock_cross_checker() -> str:
    """Detects golden or death crosses."""
    alerts = []
    for ticker in PORTFOLIO:
        try:
            data = yf.download(ticker, period="730d", progress=False)
            if len(data) < 200: continue
            data["SMA50"] = data["Close"].rolling(50).mean()
            data["SMA200"] = data["Close"].rolling(200).mean()
            data = data.dropna()
            if len(data) < 2: continue
            prev, latest = data.iloc[-2], data.iloc[-1]
            date = latest.name.strftime("%Y-%m-%d")
            if prev["SMA50"] <= prev["SMA200"] and latest["SMA50"] > latest["SMA200"]:
                alerts.append(f"Golden Cross: {ticker} on {date} (bullish)")
            elif prev["SMA50"] >= prev["SMA200"] and latest["SMA50"] < latest["SMA200"]:
                alerts.append(f"Death Cross: {ticker} on {date} (bearish)")
        except:
            pass
    return "\n".join(alerts) if alerts else "No crosses detected today."


@tool("Backtester")
def backtester() -> str:
    """5-year SMA crossover backtest."""
    results = []
    for ticker in PORTFOLIO:
        try:
            data = yf.Ticker(ticker).history(period="5y")
            if len(data) < 250:
                results.append(f"{ticker}: Insufficient data")
                continue
            data["SMA50"] = data["Close"].rolling(50).mean()
            data["SMA200"] = data["Close"].rolling(200).mean()
            data = data.dropna()
            data["position"] = np.where(data["SMA50"] > data["SMA200"], 1, 0)
            data["strategy_ret"] = data["position"].shift(1) * data["Close"].pct_change()
            strategy_return = (1 + data["strategy_ret"].dropna()).prod() - 1
            bh_return = data["Close"].iloc[-1] / data["Close"].iloc[0] - 1
            results.append(f"{ticker}: Strategy {strategy_return*100:+.2f}% | Buy&Hold {bh_return*100:+.2f}%")
        except:
            results.append(f"{ticker}: Calculation error")
    return "\n".join(results) if results else "No backtest results."


@tool("News Fetcher")
def news_fetcher() -> str:
    """Fetches recent news from Google News RSS using built-in libs."""
    news_items = []
    for t in PORTFOLIO:
        try:
            rss_url = f"https://news.google.com/rss/search?q={t}+stock&hl=en-US&gl=US&ceid=US:en"
            response = requests.get(rss_url, timeout=5)
            if response.status_code != 200:
                continue
            root = ET.fromstring(response.content)
            for item in root.findall(".//item")[:3]:  # Top 3 per ticker
                title_elem = item.find("title")
                source_elem = item.find("source")
                link_elem = item.find("link")
                title = title_elem.text if title_elem is not None else "Untitled"
                pub = source_elem.text if source_elem is not None else "Unknown"
                link = link_elem.text if link_elem is not None else ""
                news_items.append(f"{t}: [{title} ({pub})]({link})")
        except:
            pass
    return "\n\n".join(news_items) if news_items else "No recent news found."


@tool("LSTM Price Forecaster")
def lstm_price_forecaster() -> str:
    """Generates 5-day LSTM forecasts."""
    forecasts = []
    for ticker in PORTFOLIO:
        try:
            df = yf.download(ticker, period="3y", progress=False)["Close"]
            if len(df) < 200:
                forecasts.append(f"{ticker}: Insufficient data")
                continue
            data = df.values.reshape(-1, 1)
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data)

            seq_length = 60
            x = np.array([scaled_data[i-seq_length:i, 0] for i in range(seq_length, len(scaled_data))]).reshape(-1, seq_length, 1)
            y = scaled_data[seq_length:]

            x_train = torch.from_numpy(x).float()
            y_train = torch.from_numpy(y).float()

            class LSTMModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.lstm = nn.LSTM(1, 50, 1, batch_first=True)
                    self.fc = nn.Linear(50, 1)
                def forward(self, x):
                    out, _ = self.lstm(x)
                    return self.fc(out[:, -1, :])

            model = LSTMModel()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.MSELoss()

            model.train()
            for _ in range(15):  # Reduced for speed
                outputs = model(x_train)
                loss = criterion(outputs, y_train)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            predictions = []
            current_seq = scaled_data[-seq_length:].copy()
            current_tensor = torch.from_numpy(current_seq.reshape(1, seq_length, 1)).float()

            model.eval()
            for _ in range(5):
                with torch.no_grad():
                    next_pred = model(current_tensor)
                pred_val = next_pred.item()
                predictions.append(pred_val)
                current_seq = np.append(current_seq[1:], [[pred_val]], axis=0)
                current_tensor = torch.from_numpy(current_seq.reshape(1, seq_length, 1)).float()

            predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
            last_date = df.index[-1]
            future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=5, freq='B')

            pred_lines = [f"{date.strftime('%Y-%m-%d')}: ${price:.2f}" for date, price in zip(future_dates, predictions)]
            forecasts.append(f"{ticker} 5-Day LSTM Forecast:\n" + "\n".join(pred_lines))
        except:
            forecasts.append(f"{ticker}: Forecast failed")
    return "\n\n".join(forecasts) if forecasts else "No forecasts."


technical_analyst = Agent(
    role="Technical Analyst",
    goal="Run cross checker and backtester, return ONLY their exact raw outputs in plain text.",
    backstory="You are a silent tool runner. Do not summarize, add text, or use JSON/dictionaries. Output plain text only from the tools.",
    tools=[stock_cross_checker, backtester],
    llm=llm,
    allow_delegation=False,
    verbose=True
)

news_researcher = Agent(
    role="Financial News Researcher",
    goal="Run news fetcher, return ONLY its exact raw output in plain text.",
    backstory="You are a silent tool runner. Do not summarize, add text, or use JSON. Output plain text only from the tool.",
    tools=[news_fetcher],
    llm=llm,
    allow_delegation=False,
    verbose=True
)

forecast_agent = Agent(
    role="ML Price Forecaster",
    goal="Run LSTM forecaster and assemble the full report from all previous outputs. Paste everything exactly in plain text—no changes.",
    backstory="You are a silent assembler. Run your tool, then copy-paste all raw outputs into sections verbatim. No summaries or extra formatting.",
    tools=[lstm_price_forecaster],
    llm=llm,
    allow_delegation=False,
    verbose=True
)

technical_task = Task(
    description="""Run Stock Cross Checker then Backtester.
Output EXACTLY this plain text structure:
Cross Alerts
[exact plain text output from Stock Cross Checker]

5-Year Backtest Summary
[exact plain text output from Backtester]

Do not add any other text, JSON, summaries, or extra lines. Paste raw tool outputs only.""",
    expected_output="Raw plain text outputs in sections",
    agent=technical_analyst
)

news_task = Task(
    description="""Run News Fetcher.
Output EXACTLY this plain text structure:
Recent News
[exact plain text output from News Fetcher]

Do not add any other text, summaries, or extra lines. Paste raw tool output only.""",
    expected_output="Raw plain text output in section",
    agent=news_researcher
)

forecast_task = Task(
    description="""Run LSTM Price Forecaster.
Then assemble the FULL report by pasting EXACTLY the plain text outputs from ALL previous tasks + your own:

Cross Alerts
[exact plain text output from technical_task's Cross Alerts section]

5-Year Backtest Summary
[exact plain text output from technical_task's Backtest section]

Recent News
[exact plain text output from news_task]

5-Day LSTM Forecasts
[exact plain text output from LSTM Price Forecaster]

Do not add, change, summarize, or include any extra text, JSON, or lines. Paste verbatim plain text only.""",
    expected_output="Full combined plain text report with raw sections",
    agent=forecast_agent,
    context=[technical_task, news_task]
)

crew = Crew(
    agents=[technical_analyst, news_researcher, forecast_agent],
    tasks=[technical_task, news_task, forecast_task],
    verbose=True,
    memory=False
)