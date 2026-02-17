# Autonomous Stock Portfolio Monitor Dashboard
A multi-agent AI-powered dashboard built with Streamlit and CrewAI that autonomously generates daily stock reports for a portfolio of tech giants (AAPL, MSFT, GOOG, AMZN, TSLA, NVDA). It combines real-time market data from yfinance, SMA golden/death cross detection, 5-year backtesting, recent news from Google News RSS, and 5-day LSTM price forecasts using PyTorch.

## Features

Multi-Agent System: 3 specialized CrewAI agents (Technical Analyst, News Researcher, ML Forecaster) collaborate to generate reports.
Technical Analysis: Detects golden/death crosses using 50/200-day SMAs.
Backtesting: Compares SMA crossover strategy vs. buy-and-hold over 5 years.
News Aggregation: Fetches latest headlines with clickable links from Google News RSS (top 3 per stock).
LSTM Forecasting: Trains PyTorch LSTM models for 5-day price predictions per stock.
Streamlit Dashboard: One-click report generation with markdown rendering for clean output.
Optimizations: Caching for yfinance data, reduced LSTM epochs for speed, secrets management for API keys.
Deployment-Ready: Securely deployable on Streamlit Cloud with no hardcoded keys.

## Demo
Live app: [stockportfolio-agent.streamlit.app](https://stockportfolio-agent.streamlit.app)

## Example Report Output:

#### Cross Alerts

No crosses detected today.

#### 5-Year Backtest Summary

AAPL: Strategy -14.40% | Buy&Hold +62.92%

MSFT: Strategy +20.23% | Buy&Hold +24.21%
... (full per-stock results)


#### Recent News
AAPL: [Nancy Pelosi Sold Apple and Bought This 1 Stock Instead. Here’s Why (24/7 Wall St.)](https://link-to-article)
... (clickable headlines with publishers)

#### 5-Day LSTM Forecasts

AAPL 5-Day LSTM Forecast:

2026-02-18: $232.73
... (per-stock predictions)

## Tech Stack

Frontend: Streamlit

AI Agents: CrewAI

LLM: Llama 3.3 70B via OpenRouter

Data: yfinance for market data, Google News RSS for news

ML: PyTorch for LSTM forecasting, NumPy/Pandas for data processing

Other: Requests/XML for RSS parsing, scikit-learn for scaling
