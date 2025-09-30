import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

st.set_page_config(page_title="Champions App – GeoPAK10 & Buffett", page_icon="🏆", layout="wide")

# ----------------------------
# Hilfsfunktionen
# ----------------------------

def compute_geo_pak(price_series, window=260):
    """Berechne GeoPAK10 (geometrisches Mittel der letzten 10 Jahre, als %)."""
    if len(price_series.dropna()) < window:
        return np.nan
    returns = []
    for i in range(1, 11):
        try:
            past = price_series.iloc[-(i * window)]
            now = price_series.iloc[-1]
            r = (now / past) ** (1 / i) - 1
            returns.append(r)
        except Exception:
            continue
    if not returns:
        return np.nan
    geo = np.prod([(1 + r) for r in returns]) ** (1 / len(returns)) - 1
    return geo * 100  # Prozent


def compute_verlust_ratio(price_series):
    """Berechne Verlust-Ratio (max Drawdown / max Gewinn)."""
    if price_series.empty:
        return np.nan
    roll_max = price_series.cummax()
    dd = (price_series / roll_max - 1).min()
    up = (price_series / price_series.cummin() - 1).max()
    if up == 0:
        return np.nan
    return abs(dd) / up  # positiv


def compute_gewinn_konstanz(price_series):
    """Berechne Gewinnkonstanz = % positiver Jahre."""
    if price_series.empty:
        return np.nan
    yearly = price_series.resample("Y").last().pct_change().dropna()
    if yearly.empty:
        return np.nan
    return (yearly > 0).mean() * 100


def compute_sicherheitsscore(geo, vr, gk):
    """Berechne Sicherheitsscore nach Formel."""
    try:
        return (geo ** 0.8) * (vr ** -1.2) * ((gk / 100) ** 1.5)
    except Exception:
        return np.nan


def fetch_prices(tickers, start="2010-01-01"):
    """Lade Preisdaten via yfinance."""
    try:
        data = yf.download(tickers, start=start, auto_adjust=True, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data = data["Close"]
        return data
    except Exception as e:
        st.error(f"Fehler beim Laden der Kursdaten: {e}")
        return pd.DataFrame()


# ----------------------------
# Streamlit Oberfläche
# ----------------------------

st.title("🏆 Champions Analyse – GeoPAK10 & Buffett-Kriterien")

uploaded = st.file_uploader("📂 CSV mit Champions (Ticker, Name)", type=["csv"])

if uploaded:
    try:
        df_in = pd.read_csv(uploaded)
        if "Ticker" not in df_in.columns:
            st.error("Die CSV muss mindestens eine Spalte 'Ticker' enthalten.")
            st.stop()
    except Exception as e:
        st.error(f"CSV konnte nicht gelesen werden: {e}")
        st.stop()

    tickers = df_in["Ticker"].dropna().astype(str).tolist()
    names = dict(zip(df_in["Ticker"], df_in.get("Name", df_in["Ticker"])))

    st.info(f"{len(tickers)}/100 Champions ausgelesen.")

    with st.spinner("Lade Kursdaten von Yahoo Finance …"):
        prices = fetch_prices(tickers, start="2010-01-01")

    results = []
    for t in tickers:
        if t not in prices.columns:
            continue
        s = prices[t].dropna()
        if s.empty:
            continue

        geo = compute_geo_pak(s)
        vr = compute_verlust_ratio(s)
        gk = compute_gewinn_konstanz(s)
        score = compute_sicherheitsscore(geo, vr, gk)

        results.append({
            "Ticker": t,
            "Name": names.get(t, t),
            "Kurs aktuell": round(s.iloc[-1], 2),
            "GeoPAK10 (%)": round(geo, 2) if pd.notna(geo) else np.nan,
            "Verlust-Ratio": round(vr, 2) if pd.notna(vr) else np.nan,
            "Gewinnkonstanz (%)": round(gk, 1) if pd.notna(gk) else np.nan,
            "Sicherheitsscore": round(score, 3) if pd.notna(score) else np.nan,
        })

    df = pd.DataFrame(results)

    if df.empty:
        st.warning("Keine Kennzahlen berechnet – evtl. keine Kursdaten?")
        st.stop()

    df = df.sort_values("Sicherheitsscore", ascending=False).reset_index(drop=True)
    df["Rank"] = df.index + 1

    st.subheader("📊 Ergebnisse")
    st.dataframe(df, use_container_width=True)

    st.download_button("📥 Ergebnisse als CSV", df.to_csv(index=False).encode("utf-8"), "champions_scores.csv", "text/csv")

else:
    st.info("Bitte eine CSV-Datei mit den Champions hochladen.")
