import pandas as pd
import streamlit as st

from model import train_and_predict
from stock_utils import download_stock, resolve_ticker


st.set_page_config(page_title="Stock AI Analyzer", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

if "watchlist" not in st.session_state:
    st.session_state.watchlist = []

st.markdown("""
<style>
    .stApp {
        background: radial-gradient(circle at 80% -10%, #173d58 0, transparent 34%),
                    radial-gradient(circle at 16% 18%, #192c51 0, transparent 24%), #0a101c;
    }
    .block-container {padding-top: 2.4rem; padding-bottom: 3.5rem; max-width: 1360px;}
    [data-testid="stSidebar"] {background: linear-gradient(180deg, #111b2e 0%, #0b1322 100%); border-right: 1px solid rgba(135, 206, 250, .16);}
    [data-testid="stSidebar"] > div:first-child {padding-top: 1.8rem;}
    [data-testid="stSidebar"] h2 {letter-spacing: -.02em;}
    .hero {position: relative; overflow: hidden; padding: 2rem 2.2rem; border: 1px solid rgba(131, 205, 255, .25); border-radius: 22px;
           background: linear-gradient(122deg, rgba(26, 52, 91, .96), rgba(13, 92, 108, .72));
           box-shadow: 0 22px 50px rgba(0, 0, 0, .24); margin-bottom: 1.55rem;}
    .hero:after {content: ""; position: absolute; width: 270px; height: 270px; right: -80px; top: -130px; border-radius: 50%; background: rgba(103, 232, 249, .15); filter: blur(2px);}
    .hero h1 {position: relative; z-index: 1; margin: 0; font-size: clamp(2.15rem, 4vw, 3.1rem); letter-spacing: -.055em;}
    .hero p {position: relative; z-index: 1; margin: .5rem 0 0; color: #d5e9fb; font-size: 1.05rem;}
    .eyebrow {position: relative; z-index: 1; margin-bottom: .5rem; color: #83dcff; font-size: .76rem; font-weight: 750; letter-spacing: .12em; text-transform: uppercase;}
    .market-mascots {position: absolute; z-index: 2; right: 2rem; bottom: 1.5rem; display: flex; gap: .6rem;}
    .mascot {width: 54px; height: 54px; display: grid; place-items: center; border-radius: 16px; font-size: 2rem;
              background: rgba(8, 18, 35, .35); border: 1px solid rgba(255, 255, 255, .18); backdrop-filter: blur(7px);}
    .bull-mascot {box-shadow: inset 0 0 20px rgba(16, 185, 129, .17);}
    .bear-mascot {box-shadow: inset 0 0 20px rgba(248, 113, 113, .15);}
    .sentiment {display: flex; align-items: center; gap: .85rem; padding: .9rem 1.1rem; margin: .35rem 0 1.25rem;
                border-radius: 14px; border: 1px solid;}
    .sentiment-icon {font-size: 2rem; line-height: 1;}
    .sentiment-title {font-weight: 750; letter-spacing: .01em;}
    .sentiment-copy {font-size: .86rem; opacity: .86; margin-top: .1rem;}
    .sentiment-bull {background: rgba(6, 78, 59, .32); border-color: rgba(52, 211, 153, .32); color: #bbf7d0;}
    .sentiment-bear {background: rgba(104, 25, 42, .30); border-color: rgba(251, 113, 133, .32); color: #fecdd3;}
    .note {padding: .95rem 1.1rem; background: rgba(16, 56, 82, .62); border: 1px solid rgba(56, 189, 248, .25); border-left: 4px solid #38bdf8;
           border-radius: 12px; color: #d7efff; margin: .8rem 0 1.4rem;}
    div[data-testid="stMetric"] {background: linear-gradient(145deg, rgba(25, 38, 60, .95), rgba(18, 28, 46, .88)); border: 1px solid rgba(139, 190, 234, .18); border-radius: 16px; padding: 1.1rem 1rem; box-shadow: 0 10px 24px rgba(0, 0, 0, .12);}
    div[data-testid="stMetric"] label {color: #aebfd6 !important; font-size: .82rem !important; font-weight: 650 !important;}
    div[data-testid="stMetricValue"] {letter-spacing: -.035em;}
    div[data-testid="stTabs"] [data-baseweb="tab-list"] {gap: .4rem; border-bottom: 1px solid rgba(135, 166, 198, .16);}
    div[data-testid="stTabs"] button {font-weight: 650; border-radius: 9px 9px 0 0; padding: .7rem 1rem;}
    div[data-testid="stTabs"] button[aria-selected="true"] {background: rgba(44, 151, 190, .14); color: #7dd3fc;}
    div.stButton > button {border: 0; border-radius: 11px; font-weight: 700; padding: .66rem 1rem; background: linear-gradient(105deg, #0ea5e9, #14b8a6); box-shadow: 0 8px 18px rgba(14, 165, 233, .20); transition: transform .16s ease, box-shadow .16s ease;}
    div.stButton > button:hover {transform: translateY(-1px); box-shadow: 0 12px 24px rgba(14, 165, 233, .28);}
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {border-radius: 10px !important;}
    [data-testid="stDataFrame"] {border: 1px solid rgba(139, 190, 234, .16); border-radius: 12px; overflow: hidden;}
    h2, h3 {letter-spacing: -.028em;}
</style>
""", unsafe_allow_html=True)


QUICK_STOCKS = {
    "Custom company / ticker": "",
    "SBI": "SBI",
    "TCS": "TCS",
    "Reliance": "Reliance",
    "Infosys": "Infosys",
    "HDFC Bank": "HDFC Bank",
    "ICICI Bank": "ICICI Bank",
    "Tata Motors": "Tata Motors",
}
PERIODS = {"1 year": "1y", "2 years": "2y", "5 years (recommended)": "5y"}


def prepare_chart_data(data):
    chart = data.copy()
    chart["MA 20"] = chart["Close"].rolling(20).mean()
    chart["MA 50"] = chart["Close"].rolling(50).mean()
    return chart


with st.sidebar:
    st.markdown("## 📊 Stock Explorer")
    st.caption("Analyze NSE stocks with market data and machine learning.")
    st.divider()
    quick_stock = st.selectbox("Popular stocks", list(QUICK_STOCKS))
    manual_stock = st.text_input("Company name or NSE ticker", value="", placeholder="Example: TCS or SBIN.NS")
    period_label = st.select_slider("Historical data", options=list(PERIODS), value="5 years (recommended)")
    stock_input = manual_stock.strip() or QUICK_STOCKS[quick_stock]
    analyze = st.button("✨ Analyze stock", type="primary", use_container_width=True)
    st.divider()
    st.caption("For educational use only. This is not investment advice.")

st.markdown("""
<div class="hero">
  <div class="eyebrow">Smart market dashboard</div>
  <h1>📈 Stock AI Analyzer</h1>
  <p>Explore price trends, technical signals, and a next-day ML estimate in one elegant workspace.</p>
  <div class="market-mascots" title="Bull markets rise; bear markets fall.">
    <div class="mascot bull-mascot">🐂</div>
    <div class="mascot bear-mascot">🐻</div>
  </div>
</div>
""", unsafe_allow_html=True)

if analyze:
    try:
        if not stock_input:
            raise ValueError("Choose a popular stock or enter a company name/ticker first.")
        with st.spinner("Downloading market history and validating models..."):
            ticker = resolve_ticker(stock_input)
            data = download_stock(ticker, period=PERIODS[period_label])
            result = train_and_predict(data)
        st.session_state.analysis = {"ticker": ticker, "data": data, "result": result, "period": period_label}
    except Exception as error:
        st.error(f"Could not analyze this stock: {error}")

if "analysis" not in st.session_state:
    st.markdown("""
    <div class="note">Choose a company in the sidebar, then select <b>Analyze stock</b> to begin.</div>
    """, unsafe_allow_html=True)
    st.subheader("What you can explore")
    first, second, third = st.columns(3)
    first.info("**Price trends**\n\nCompare closing prices with 20- and 50-day averages.")
    second.info("**Technical signals**\n\nReview RSI, MACD, volatility, and volume activity.")
    third.info("**Model validation**\n\nSee model direction accuracy and export price history.")
    st.stop()

analysis = st.session_state.analysis
ticker, data, result = analysis["ticker"], analysis["data"], analysis["result"]
chart_data = prepare_chart_data(data)
latest_close = float(data["Close"].iloc[-1])
day_change = float(data["Close"].pct_change().iloc[-1] * 100)
month_change = float((data["Close"].iloc[-1] / data["Close"].iloc[-min(22, len(data))] - 1) * 100)
week_change = float((data["Close"].iloc[-1] / data["Close"].iloc[-min(6, len(data))] - 1) * 100)
year_data = data.tail(min(252, len(data)))
year_high = float(year_data["High"].max())
year_low = float(year_data["Low"].min())
from_year_high = (latest_close / year_high - 1) * 100

st.success(f"{ticker} analyzed using {analysis['period'].lower()} of daily market data.")
is_bullish = result["predicted_return"] >= 0
mood_class = "sentiment-bull" if is_bullish else "sentiment-bear"
mood_icon = "🐂" if is_bullish else "🐻"
mood_title = "Bullish model signal" if is_bullish else "Bearish model signal"
mood_copy = "The model estimates an upward next-day move." if is_bullish else "The model estimates a downward next-day move."
st.markdown(f"""
<div class="sentiment {mood_class}">
  <div class="sentiment-icon">{mood_icon}</div>
  <div><div class="sentiment-title">{mood_title}</div><div class="sentiment-copy">{mood_copy} This is an educational estimate, not investment advice.</div></div>
</div>
""", unsafe_allow_html=True)
overview_tab, technical_tab, insights_tab, model_tab, tools_tab, history_tab = st.tabs(["📌 Overview", "📊 Technical signals", "💡 Insights", "🧠 Model quality", "🧰 Tools", "🗂️ Price history"])

with overview_tab:
    one, two, three, four = st.columns(4)
    one.metric("Latest close", f"₹{latest_close:,.2f}", f"{day_change:+.2f}% today")
    two.metric("AI next-close estimate", f"₹{result['predicted']:,.2f}", f"{result['predicted_return']:+.2f}%")
    three.metric("1-month movement", f"{month_change:+.2f}%")
    four.metric("Direction accuracy", f"{result['metrics']['direction_accuracy']:.1f}%", help="How often the model correctly classified historical next-day direction on its held-out test period.")

    st.subheader("Closing price and moving averages")
    st.caption("Moving averages help show the short- and medium-term trend.")
    st.line_chart(chart_data[["Close", "MA 20", "MA 50"]].dropna(), height=420)

    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Recent daily closes")
        recent = data[["Close", "Volume"]].tail(10).sort_index(ascending=False).copy()
        recent["Daily change %"] = data["Close"].pct_change().tail(10).sort_index(ascending=False) * 100
        st.dataframe(recent.style.format({"Close": "₹{:,.2f}", "Volume": "{:,.0f}", "Daily change %": "{:+.2f}%"}), use_container_width=True)
    with right:
        st.subheader("Quick reading")
        trend = "above" if latest_close >= result["latest_indicators"]["MA_20"] else "below"
        momentum = "positive" if result["predicted_return"] >= 0 else "negative"
        st.write(f"The latest close is **{trend}** its 20-day average.")
        st.write(f"The model's next-day estimate has **{momentum}** direction.")
        st.caption("These are descriptive indicators, not buy or sell recommendations.")

with technical_tab:
    one, two, three, four = st.columns(4)
    rsi = result["latest_indicators"]["RSI"]
    rsi_label = "Overbought" if rsi >= 70 else "Oversold" if rsi <= 30 else "Neutral"
    one.metric("RSI (14)", f"{rsi:.1f}", rsi_label)
    two.metric("MACD", f"{result['latest_indicators']['MACD']:.2f}")
    three.metric("20-day average", f"₹{result['latest_indicators']['MA_20']:,.2f}")
    four.metric("20-day volatility", f"{result['latest_indicators']['Volatility'] * 100:.2f}%")
    st.subheader("Volume trend")
    st.bar_chart(data[["Volume"]].tail(60), height=300)
    st.markdown("""
    <div class="note"><b>How to read these:</b> RSI near 70 can indicate strong recent upward momentum; RSI near 30 can indicate weak recent momentum. MACD and moving averages are trend indicators. None predicts the market with certainty.</div>
    """, unsafe_allow_html=True)

with insights_tab:
    st.subheader("Market snapshot")
    one, two, three, four = st.columns(4)
    one.metric("1-week return", f"{week_change:+.2f}%")
    two.metric("1-month return", f"{month_change:+.2f}%")
    three.metric("52-week high", f"₹{year_high:,.2f}", f"{from_year_high:.1f}% from high")
    four.metric("52-week low", f"₹{year_low:,.2f}")

    st.subheader("Automated reading")
    price_above_ma = latest_close >= result["latest_indicators"]["MA_20"]
    rsi_state = "neutral" if 30 < rsi < 70 else ("strong / overbought" if rsi >= 70 else "weak / oversold")
    volume_20 = float(data["Volume"].tail(20).mean())
    volume_state = "above" if float(data["Volume"].iloc[-1]) >= volume_20 else "below"
    first, second, third = st.columns(3)
    first.success(f"**Trend**\n\nPrice is {'above' if price_above_ma else 'below'} the 20-day average.")
    second.info(f"**Momentum**\n\nRSI is {rsi_state} at {rsi:.1f}.")
    third.warning(f"**Activity**\n\nLatest volume is {volume_state} its 20-day average.")
    st.caption("These observations describe historical signals. They do not constitute a recommendation to buy or sell.")

with model_tab:
    st.subheader("Validated model results")
    st.caption(f"The app tested Extra Trees and Random Forest using time-ordered validation, then selected **{result['model_name']}**.")
    one, two, three = st.columns(3)
    one.metric("Direction accuracy", f"{result['metrics']['direction_accuracy']:.1f}%")
    two.metric("Average price error", f"₹{result['metrics']['MAE']:,.2f}")
    three.metric("RMSE", f"₹{result['metrics']['RMSE']:,.2f}")
    st.subheader("Most influential inputs")
    st.bar_chart(result["feature_importance"].head(12).set_index("Feature"), height=350)
    with st.expander("What does direction accuracy mean?"):
        st.write("It measures how often the model correctly predicted whether the next closing price moved up or down in a held-out historical period. Past validation does not guarantee future results.")

with tools_tab:
    st.subheader("Investment impact calculator")
    st.caption("A simple illustration using the model's next-day estimate; it does not include brokerage, taxes, or slippage.")
    investment = st.number_input("Investment amount (₹)", min_value=1000.0, value=10000.0, step=1000.0)
    units = investment / latest_close
    estimated_value = units * result["predicted"]
    projected_change = estimated_value - investment
    one, two, three = st.columns(3)
    one.metric("Estimated units", f"{units:.3f}")
    two.metric("Estimated next-day value", f"₹{estimated_value:,.2f}")
    three.metric("Illustrative change", f"₹{projected_change:+,.2f}")

    st.divider()
    left, right = st.columns(2)
    with left:
        st.subheader("Session watchlist")
        if st.button(f"⭐ Add {ticker} to watchlist") and ticker not in st.session_state.watchlist:
            st.session_state.watchlist.append(ticker)
            st.success(f"{ticker} added to this browser-session watchlist.")
        if st.session_state.watchlist:
            st.write(" • ".join(st.session_state.watchlist))
            remove_ticker = st.selectbox("Remove a stock", st.session_state.watchlist)
            if st.button("Remove selected stock"):
                st.session_state.watchlist.remove(remove_ticker)
                st.rerun()
        else:
            st.caption("Your watchlist is empty. Add the current stock to keep it handy during this session.")
    with right:
        st.subheader("Export analysis summary")
        summary = pd.DataFrame({
            "Metric": ["Ticker", "Latest close", "Predicted next close", "Predicted return", "Direction accuracy", "Selected model", "RSI"],
            "Value": [ticker, latest_close, result["predicted"], result["predicted_return"] / 100, result["metrics"]["direction_accuracy"] / 100, result["model_name"], rsi],
        })
        st.download_button("⬇️ Download analysis summary", summary.to_csv(index=False).encode("utf-8"), file_name=f"{ticker}_analysis_summary.csv", mime="text/csv")

with history_tab:
    st.subheader("Download and inspect market data")
    start_date, end_date = st.date_input("Date range", value=(data.index.min().date(), data.index.max().date()))
    filtered = data.loc[(data.index.date >= start_date) & (data.index.date <= end_date)]
    st.dataframe(filtered.sort_index(ascending=False).style.format({"Open": "₹{:,.2f}", "High": "₹{:,.2f}", "Low": "₹{:,.2f}", "Close": "₹{:,.2f}", "Volume": "{:,.0f}"}), use_container_width=True, height=420)
    st.download_button("⬇️ Download CSV", filtered.to_csv().encode("utf-8"), file_name=f"{ticker}_price_history.csv", mime="text/csv")
