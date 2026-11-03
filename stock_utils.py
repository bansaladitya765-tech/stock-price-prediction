import pandas as pd
import yfinance as yf

STOCK_MAP = {
    "sbi": "SBIN.NS", "state bank of india": "SBIN.NS", "tcs": "TCS.NS",
    "tata consultancy services": "TCS.NS", "reliance": "RELIANCE.NS",
    "reliance industries": "RELIANCE.NS", "infosys": "INFY.NS",
    "hdfc bank": "HDFCBANK.NS", "icici bank": "ICICIBANK.NS", "itc": "ITC.NS",
    "tata motors": "TATAMOTORS.NS", "wipro": "WIPRO.NS", "axis bank": "AXISBANK.NS",
    "bharti airtel": "BHARTIARTL.NS", "hindustan unilever": "HINDUNILVR.NS",
}

def resolve_ticker(stock_name):
    value = stock_name.strip()
    if not value:
        raise ValueError("Please enter a company name or stock ticker.")
    ticker = STOCK_MAP.get(value.lower())
    return ticker if ticker else (value.upper() if "." in value else f"{value.upper()}.NS")

def download_stock(ticker, period="5y"):
    data = yf.download(ticker, period=period, interval="1d", auto_adjust=False, progress=False)
    if data.empty:
        raise ValueError(f"No price data found for {ticker}. Try SBI, TCS, Reliance, or SBIN.NS.")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data[[column for column in ["Open", "High", "Low", "Close", "Volume"] if column in data.columns]].dropna()
    if len(data) < 100:
        raise ValueError("Not enough historical market data was received.")
    return data
