import os
import smtplib
from email.mime.text import MIMEText
from datetime import datetime

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim

from crewai import Agent, Task, Crew
from crewai_tools import tool

os.environ["OPENAI_API_KEY"] = "your_xai_api_key_here"
os.environ["OPENAI_API_BASE"] = "https://api.x.ai/v1"
os.environ["OPENAI_MODEL_NAME"] = "grok-4"

EMAIL_USER = "your_email@gmail.com"
EMAIL_PASS = "your_app_password"
EMAIL_TO = "your_email@gmail.com"

@tool("Stock Cross Checker")
def stock_cross_checker() -> str:
    portfolio = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA", "NVDA"]  # Add custom stock tickers here

    alerts = []
    crossed_tickers = []

    for ticker in portfolio:
        try:
            data = yf.download(ticker, period="730d", progress=False)
            if len(data) < 200:
                continue

            data["SMA50"] = data["Close"].rolling(50).mean()
            data["SMA200"] = data["Close"].rolling(200).mean()
            data = data.dropna()

            if len(data) < 2:
                continue

            prev, latest = data.iloc[-2], data.iloc[-1]
            date = latest.name.strftime("%Y-%m-%d")

            if prev["SMA50"] <= prev["SMA200"] and latest["SMA50"] > latest["SMA200"]:
                alerts.append(f"Golden Cross: {ticker} on {date} (bullish)")
                crossed_tickers.append(ticker)
            elif prev["SMA50"] >= prev["SMA200"] and latest["SMA50"] < latest["SMA200"]:
                alerts.append(f"Death Cross: {ticker} on {date} (bearish)")
                crossed_tickers.append(ticker)
        except Exception as e:
            alerts.append(f"{ticker}: Error - {str(e)}")

    alert_str = "\n".join(alerts) if alerts else "No crosses detected today."
    tickers_str = ",".join(crossed_tickers) if crossed_tickers else "None"
    return f"{alert_str}\n\nCrossed tickers: {tickers_str}"

@tool("Backtester")
def backtester() -> str:
    portfolio = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA", "NVDA"]  # Edit based on custom tickers added before

    results = []
    for ticker in portfolio:
        try:
            data = yf.download(ticker, period="5y", progress=False)
            data["SMA50"] = data["Close"].rolling(50).mean()
            data["SMA200"] = data["Close"].rolling(200).mean()
            data = data.dropna()

            data["golden"] = (data["SMA50"].shift(1) <= data["SMA200"].shift(1)) & (data["SMA50"] > data["SMA200"])
            data["death"] = (data["SMA50"].shift(1) >= data["SMA200"].shift(1)) & (data["SMA50"] < data["SMA200"])

            in_market = False
            position = []
            for g, d in zip(data["golden"], data["death"]):
                if g:
                    in_market = True
                elif d:
                    in_market = False
                position.append(1 if in_market else 0)
            data["position"] = position

            data["market_ret"] = data["Close"].pct_change()
            data["strategy_ret"] = data["position"].shift(1) * data["market_ret"]
            strategy_return = (1 + data["strategy_ret"].dropna()).prod() - 1
            bh_return = data["Close"].iloc[-1] / data["Close"].iloc[0] - 1

            results.append(f"{ticker}: Strategy {strategy_return*100:.2f}% | Buy & Hold {bh_return*100:.2f}%")
        except:
            results.append(f"{ticker}: Backtest failed")

    return "Backtest Results (5Y):\n" + "\n".join(results)

@tool("News Fetcher")
def news_fetcher(crossed_tickers: str) -> str:
    """Fetch top 3 recent news items for comma-separated tickers."""
    if not crossed_tickers or crossed_tickers.strip() == "None":
        return "No crossed stocks to research."

    tickers = [t.strip() for t in crossed_tickers.split(",")]
    news_items = []

    for t in tickers:
        try:
            ticker_obj = yf.Ticker(t)
            news = ticker_obj.news[:3]
            for item in news:
                title = item.get("title", "No title")
                pub = item.get("publisher", "Unknown")
                link = item.get("link", "")
                news_items.append(f"{t}: {title} ({pub}) → {link}")
        except:
            news_items.append(f"{t}: Failed to fetch news")

    return "Recent News for Crossed Stocks:\n" + ("\n".join(news_items) if news_items else "None found")

@tool("LSTM Price Forecaster")
def lstm_price_forecaster(crossed_tickers: str) -> str:
    """Quick LSTM forecast (next 5 trading days) for crossed stocks using PyTorch."""
    if not crossed_tickers or crossed_tickers.strip() == "None":
        return "No crossed stocks for forecasting."

    tickers = [t.strip() for t in crossed_tickers.split(",")]
    forecasts = []

    for ticker in tickers:
        try:
            df = yf.download(ticker, period="3y", progress=False)["Close"]
            if len(df) < 200:
                forecasts.append(f"{ticker}: Insufficient data")
                continue

            data = df.values.reshape(-1, 1)
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data)

            seq_length = 60
            x_list = []
            for i in range(seq_length, len(scaled_data)):
                x_list.append(scaled_data[i-seq_length:i, 0])
            x = np.array(x_list)
            x = np.reshape(x, (x.shape[0], x.shape[1], 1))

            y = scaled_data[seq_length:]

            x_train = torch.from_numpy(x).float()
            y_train = torch.from_numpy(y).float()

            class LSTMModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
                    self.fc = nn.Linear(50, 1)

                def forward(self, x):
                    out, _ = self.lstm(x)
                    return self.fc(out[:, -1, :])

            model = LSTMModel()
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            model.train()
            for _ in range(25):
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
                new_pred = np.array([[pred_val]])
                current_seq = np.append(current_seq[1:], new_pred, axis=0)
                current_tensor = torch.from_numpy(current_seq.reshape(1, seq_length, 1)).float()

            predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
            last_date = df.index[-1]
            future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=5, freq='B')

            pred_lines = [f"{date.strftime('%Y-%m-%d')}: ${price:.2f}" for date, price in zip(future_dates, predictions)]
            forecasts.append(f"{ticker} 5-Day Forecast:\n" + "\n".join(pred_lines))
        except Exception as e:
            forecasts.append(f"{ticker}: Forecast failed - {str(e)}")

    return "LSTM Price Forecasts\n" + "\n\n".join(forecasts)

technical_analyst = Agent(
    role="Technical Analyst",
    goal="Detect crosses, run backtests, and delegate news/forecast research when needed",
    backstory="Expert quant with deep knowledge of technical indicators and ML forecasting.",
    tools=[stock_cross_checker, backtester],
    allow_delegation=True,
    verbose=True
)

news_researcher = Agent(
    role="Financial News Researcher",
    goal="Provide relevant news context",
    backstory="Skilled at finding timely market insights.",
    tools=[news_fetcher],
    allow_delegation=False,
    verbose=True
)

forecast_agent = Agent(
    role="ML Price Forecaster",
    goal="Generate 5-day price forecasts using LSTM for stocks with signals",
    backstory="Machine learning specialist trained on time-series forecasting.",
    tools=[lstm_price_forecaster],
    allow_delegation=False,
    verbose=True
)

report_task = Task(
    description="""Generate a comprehensive daily report:
    - Cross alerts with dates
    - 5-year backtest summary
    - Recent news for crossed stocks (delegate if needed)
    - 5-day LSTM price forecasts for crossed stocks (delegate if needed)
    Format professionally with clear sections.""",
    expected_output="Structured report with sections.",
    agent=technical_analyst
)

crew = Crew(
    agents=[technical_analyst, news_researcher, forecast_agent],
    tasks=[report_task],
    verbose=2
)

def send_email_if_cross(report: str):
    if "golden cross" in report.lower() or "death cross" in report.lower():
        msg = MIMEText(f"Daily Stock Monitor Report ({datetime.now().strftime('%Y-%m-%d')}):\n\n{report}")
        msg["Subject"] = "🚨 Stock Cross Detected"
        msg["From"] = EMAIL_USER
        msg["To"] = EMAIL_TO

        try:
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(EMAIL_USER, EMAIL_PASS)
                server.sendmail(EMAIL_USER, EMAIL_TO, msg.as_string())
            print("Email alert sent!")
        except Exception as e:
            print(f"Email failed: {e}")
    else:
        print("No crosses → no email sent.")

if __name__ == "__main__":
    print("Running advanced daily stock monitor...\n")
    result = crew.kickoff()
    full_report = str(result)
    print("\n=== DAILY REPORT ===\n")
    print(full_report)
    send_email_if_cross(full_report)