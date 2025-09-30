# champions_app.py
# Streamlit-App zur Berechnung der Champions-Scores (GeoPAK10 + Buffett-Kriterien)

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import time
from datetime import datetime, timedelta

st.set_page_config(page_title="Champions Auswahl", page_icon="🏆", layout="wide")

# ------------------------------------------------------
# Hilfsfunktionen
# ------------------------------------------------------

def fetch_ohlc_safe(ticker_list, start, end, batch_size=20, pause=1):
    """Lädt Kursdaten in Batches, um Rate Limits von yfinance zu vermeiden."""
    all_prices = []
    failed = []

    for i in range(0, len(ticker_list), batch_size):
        batch = ticker_list[i:i+batch_size]
        try:
            data = yf.download(
                tickers=" ".join(batch),
                start=start,
                end=end,
                group_by="ticker",
                auto_adjust=False,
                progress=False,
                threads=True,
            )
            if isinstance(data.columns, pd.MultiIndex):
                for t in batch:
                    try:
                        closes = data[t]["Adj Close"].rename(t)
                        all_prices.append(closes)
                    except Exception:
                        failed.append(t)
            else:
                closes = data["Adj Close"].rename(batch[0])
                all_prices.append(closes)
        except Exception:
            failed.extend(batch)

        time.sleep(pause)

    if all_prices:
        prices = pd.concat(all_prices, axis=1).dropna(how="all")
    else:
        prices = pd.DataFrame()

    return prices, failed


def compute_geo_pak(price_series, window=260):
    """Berechne GeoPAK10 (geometrisches Mittel der letzten 10 Jahresrenditen)."""
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
    return np.prod([(1 + r) for r in returns]) ** (1 / len(returns)) - 1


def compute_verlust_ratio(price_series):
    """Berechne Verlust-Ratio (max Drawdown / max Gewinn)."""
    if price_series.empty:
        return np.nan
    roll_max = price_series.cummax()
    dd = (price_series / roll_max - 1).min()
    up = (price_series / price_series.cummin() - 1).max()
    if up == 0:
        return np.nan
    return dd / up


def compute_gewinn_konstanz(price_series, window=260):
    """Berechne Gewinnkonstanz (Prozentsatz positiver Jahre)."""
    if len(price_series) < window:
        return np.nan
    years = []
    for i in range(1, 11):
        try:
            start = price_series.iloc[-(i * window)]
            end = price_series.iloc[-((i - 1) * window + 1)]
            years.append(1 if end > start else 0)
        except Exception:
            continue
    if not years:
        return np.nan
    return np.mean(years) * 100


def compute_sicherheitsscore(geo, vr, gk):
    """Berechne Sicherheitsscore nach Formel."""
    try:
        return (geo ** 0.8) * (vr ** -1.2) * ((gk / 100) ** 1.5)
    except Exception:
        return np.nan


# ------------------------------------------------------
# UI
# ------------------------------------------------------

st.title("🏆 Champions Auswahl – GeoPAK10 + Buffett-Kriterien")

uploaded = st.file_uploader("CSV mit Champions (Ticker, Name)", type=["csv"])
if uploaded is None:
    st.info("Bitte CSV mit mindestens den Spalten **Ticker** und **Name** hochladen.")
    st.stop()

try:
    df_in = pd.read_csv(uploaded)
    tickers = df_in["Ticker"].astype(str).str.upper().tolist()
    name_map = dict(zip(df_in["Ticker"], df_in["Name"]))
    st.success(f"{len(tickers)} Champions eingelesen.")
except Exception as e:
    st.error(f"Fehler beim Einlesen der CSV: {e}")
    st.stop()

start_date = datetime.today() - timedelta(days=365*10)
end_date = datetime.today()

with st.spinner("Lade Kursdaten (Batchweise mit Pausen)…"):
    prices, failed = fetch_ohlc_safe(tickers, start_date, end_date)

if prices.empty:
    st.error("Keine Kursdaten geladen.")
    st.stop()

if failed:
    st.warning(f"{len(failed)} Ticker konnten nicht geladen werden: {failed}")

# ------------------------------------------------------
# Berechnungen
# ------------------------------------------------------

results = []
for t in prices.columns:
    s = prices[t].dropna()
    if s.empty:
        continue
    geo = compute_geo_pak(s)
    vr = compute_verlust_ratio(s)
    gk = compute_gewinn_konstanz(s)
    score = compute_sicherheitsscore(geo, vr, gk)

    # Buffett-Kriterien
    unterbewertet = "Ja" if score is not None and score >= 6 else "Nein"
    buffett_signal = "Ja" if unterbewertet == "Ja" else "Nein"

    results.append({
        "Ticker": t,
        "Name": name_map.get(t, t),
        "GeoPAK10": round(geo, 3) if geo is not None else np.nan,
        "Verlust-Ratio": round(vr, 3) if vr is not None else np.nan,
        "Gewinnkonstanz": round(gk, 1) if gk is not None else np.nan,
        "Sicherheitsscore": round(score, 3) if score is not None else np.nan,
        "Unterbewertung": unterbewertet,
        "Buffett-Signal": buffett_signal,
    })

df = pd.DataFrame(results)
if df.empty:
    st.warning("Keine Kennzahlen berechnet.")
    st.stop()

df = df.sort_values("Sicherheitsscore", ascending=False).reset_index(drop=True)
df["Rank"] = np.arange(1, len(df) + 1)

# ------------------------------------------------------
# Ausgabe
# ------------------------------------------------------

st.subheader("Ranking – Champions")
st.dataframe(df, use_container_width=True)

st.download_button("📥 Ergebnisse als CSV", df.to_csv(index=False).encode("utf-8"),
                   "champions_scores.csv", "text/csv")

st.caption("Nur Informations- und Ausbildungszwecke. Keine Anlageempfehlung.")
