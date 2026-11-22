import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Stock Insight (Educational)", layout="wide")

DISCLAIMER = (
    "Educational analysis tool. Not investment advice or a recommendation to buy or sell. "
    "Indicators describe past price behaviour and do not predict future returns. "
    "Do your own research or consult a SEBI-registered adviser."
)


@st.cache_data(ttl=600)
def load(symbol: str, period: str) -> pd.DataFrame:
    return yf.Ticker(symbol).history(period=period, interval="1d")


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"] = 100 - 100 / (1 + gain / loss)
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACDsig"] = df["MACD"].ewm(span=9, adjust=False).mean()
    return df


def observations(df: pd.DataFrame) -> list[str]:
    last = df.iloc[-1]
    notes = []
    if pd.notna(last["SMA50"]):
        side = "above" if last["Close"] > last["SMA50"] else "below"
        notes.append(f"Price is {side} its 50-day average.")
        cross = "above" if last["SMA20"] > last["SMA50"] else "below"
        notes.append(f"The 20-day average is {cross} the 50-day average.")
    if pd.notna(last["RSI"]):
        r = last["RSI"]
        zone = "in the historically overbought zone (above 70)" if r > 70 else (
            "in the historically oversold zone (below 30)" if r < 30 else "in the neutral zone")
        notes.append(f"RSI is {r:.0f}, {zone}.")
    if pd.notna(last["MACD"]):
        m = "above" if last["MACD"] > last["MACDsig"] else "below"
        notes.append(f"MACD is {m} its signal line.")
    if len(df) > 22:
        chg = (df["Close"].iloc[-1] / df["Close"].iloc[-22] - 1) * 100
        notes.append(f"Change over about 1 month: {chg:+.1f}%.")
    return notes


def ai_summary(api_key: str, symbol: str, notes: list[str]) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    prompt = (
        f"Stock: {symbol}\nTechnical readings:\n- " + "\n- ".join(notes) +
        "\n\nWrite a short, neutral, educational summary (max 5 sentences) of what these "
        "readings describe. Do NOT say buy, sell, hold, give price targets, or predict "
        "future prices. Mention that indicators are historical and can be wrong."
    )
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


st.title("📈 Stock Insight")
st.caption(DISCLAIMER)

with st.sidebar:
    symbol = st.text_input("Symbol (NSE: add .NS, BSE: add .BO)", "RELIANCE.NS").strip().upper()
    period = st.selectbox("History", ["3mo", "6mo", "1y", "2y"], index=2)
    watch = st.text_area("Watchlist (one per line)", "TCS.NS\nINFY.NS\nHDFCBANK.NS\nSBIN.NS")
    api_key = st.text_input("Anthropic API key (optional, for AI summary)", type="password")

try:
    data = load(symbol, period)
except Exception as e:
    st.error(f"Could not load data: {e}")
    st.stop()

if data.empty:
    st.warning("No data found. Check the symbol (e.g. TCS.NS).")
    st.stop()

df = add_indicators(data)
last, prev = df["Close"].iloc[-1], df["Close"].iloc[-2]
c1, c2, c3 = st.columns(3)
c1.metric("Last close", f"₹{last:,.2f}", f"{(last / prev - 1) * 100:+.2f}%")
c2.metric("52-period high", f"₹{df['High'].max():,.2f}")
c3.metric("52-period low", f"₹{df['Low'].min():,.2f}")

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.04)
fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"],
                             close=df["Close"], name="Price"), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["SMA20"], name="SMA20"), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["SMA50"], name="SMA50"), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI"), row=2, col=1)
fig.add_hline(y=70, line_dash="dot", row=2, col=1)
fig.add_hline(y=30, line_dash="dot", row=2, col=1)
fig.update_layout(height=600, xaxis_rangeslider_visible=False, margin=dict(t=20))
st.plotly_chart(fig, use_container_width=True)

st.subheader("What the indicators show")
notes = observations(df)
for n in notes:
    st.write("•", n)

if api_key and st.button("Generate AI summary"):
    with st.spinner("Thinking..."):
        try:
            st.info(ai_summary(api_key, symbol, notes))
        except Exception as e:
            st.error(f"AI summary failed: {e}")

st.subheader("Watchlist")
rows = []
for s in [x.strip().upper() for x in watch.splitlines() if x.strip()]:
    try:
        d = load(s, "1mo")
        if len(d) >= 2:
            rows.append({"Symbol": s, "Close (₹)": round(d["Close"].iloc[-1], 2),
                         "Day %": round((d["Close"].iloc[-1] / d["Close"].iloc[-2] - 1) * 100, 2)})
    except Exception:
        pass
if rows:
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.divider()
st.caption(DISCLAIMER)