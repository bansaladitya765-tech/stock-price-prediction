# Stock AI Analyzer

Enter a stock name such as SBI, TCS, Reliance, Infosys, or HDFC Bank.
The app resolves common Indian company names to NSE Yahoo Finance tickers,
downloads historical data, calculates technical indicators, trains a Random Forest
model, and displays a next-trading-day closing-price estimate.

## Setup
1. Extract the ZIP and open the folder in VS Code.
2. Open Terminal.
3. Create environment:
```bash
python -m venv venv
```
4. Windows PowerShell:
```powershell
.\venv\Scripts\Activate.ps1
```
5. Install:
```bash
pip install -r requirements.txt
```
6. Run:
```bash
streamlit run app.py
```

Examples: SBI, TCS, Reliance, Infosys, HDFC Bank, ICICI Bank, ITC, SBIN.NS.

The app shows latest price, predicted next close, expected percentage change,
trend, RSI, MACD, moving averages, 52-week high/low, volume, model metrics,
historical charts, test-set predictions, and a downloadable CSV.

This is an educational ML project. Predictions are uncertain and are not financial advice.
