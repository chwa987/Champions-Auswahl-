# champions_app.py
# Version 2.1 – Champions: GeoPAK10 + Verlust-Ratio (gewichtet 1..120) + Gewinn-Konstanz (7260 Teilperioden) + Buffett-Signale

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import time
from datetime import datetime, timedelta

st.set_page_config(page_title="🏆 Champions – GeoPAK10 & Sicherheit", page_icon="🏆", layout="wide")

# =========================
# CSV robust einlesen
# =========================
def read_champions_csv(uploaded):
    try:
        df = pd.read_csv(uploaded)
    except Exception:
        uploaded.seek(0)
        df = pd.read_csv(uploaded, sep=";")
    cols = {c.lower().strip(): c for c in df.columns}
    if "ticker" not in cols:
        # erste Spalte als Ticker interpretieren
        df = df.rename(columns={df.columns[0]:"Ticker"})
    if "Name" not in df.columns and "name" not in cols:
        df["Name"] = df.iloc[:,0]
    df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()
    df["Name"]   = df["Name"].astype(str).str.strip()
    df = df[df["Ticker"]!=""].drop_duplicates(subset=["Ticker"])
    return df[["Ticker","Name"]]

# =========================
# Daten laden (Batch)
# =========================
@st.cache_data(show_spinner=False, ttl=60*60)
def fetch_prices_batch(tickers, start, end, batch_size=20, pause=1.0):
    all_closes = []
    failed = []
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        try:
            data = yf.download(
                tickers=" ".join(batch),
                start=start,
                end=end,
                auto_adjust=True,
                group_by="ticker",
                progress=False,
                threads=True,
            )
            if isinstance(data.columns, pd.MultiIndex):
                for t in batch:
                    try:
                        closes = data[t]["Close"].rename(t)
                        all_closes.append(closes)
                    except Exception:
                        failed.append(t)
            else:
                closes = data["Close"].rename(batch[0])
                all_closes.append(closes)
        except Exception:
            failed.extend(batch)
        time.sleep(pause)
    close_df = pd.concat(all_closes, axis=1).sort_index().dropna(how="all") if all_closes else pd.DataFrame()
    return close_df, failed

# =========================
# Monatsreihe (genau 120)
# =========================
def last_120_monthly(series: pd.Series) -> pd.Series:
    m = series.resample("M").last().dropna()
    if len(m) < 120:
        return pd.Series(dtype=float)  # zu wenig Historie
    return m.tail(120)

# =========================
# GeoPAK10 (%)
# =========================
def geopak10_from_months(monthly_120: pd.Series) -> float:
    if monthly_120.empty or len(monthly_120) < 120:
        return np.nan
    start = monthly_120.iloc[0]
    end   = monthly_120.iloc[-1]
    if start <= 0:
        return np.nan
    cagr = (end / start) ** (1/10) - 1  # 10 Jahre
    return float(cagr * 100.0)  # als %

# =========================
# Verlust-Ratio (gewichtet 1..120)
# =========================
def verlust_ratio_weighted(monthly_120: pd.Series) -> float:
    if monthly_120.empty or len(monthly_120) < 120:
        return np.nan
    rets = monthly_120.pct_change().dropna()  # 119 Renditen
    if rets.empty:
        return np.nan
    # Gewichte: ältester=1 ... neuester=120 (an Monate, nicht an Renditen – also 119)
    # Wir gewichten die Renditen 1..119 proportional 2..120. Vereinfachung: wir bauen ein passendes Gewicht-Fenster.
    # Sauberer: Gewichte auf Monats-Returns so definieren, dass Return(t) (Monat i->i+1) das Gewicht des neueren Monats trägt.
    w_full = pd.Series(np.arange(1, len(monthly_120)+1), index=monthly_120.index)  # 1..120
    w_rets = w_full.iloc[1:]  # 2..120, ausgerichtet auf rets.index

    gains   = rets[rets > 0]
    losses  = rets[rets < 0]
    if gains.empty or losses.empty:
        return np.nan

    wg = w_rets.loc[gains.index].astype(float).values
    wl = w_rets.loc[losses.index].astype(float).values
    gains_w_mean  = np.average(gains.values,  weights=wg)
    losses_w_mean = np.average(np.abs(losses.values), weights=wl)

    if gains_w_mean <= 0:
        return np.nan
    return float(losses_w_mean / gains_w_mean)

# =========================
# Gewinn-Konstanz (7260 Teilperioden)
# =========================
def gewinn_konstanz_7260(monthly_120: pd.Series) -> float:
    """Anteil positiver Renditen über ALLE zusammenhängenden Teilperioden (1..120 Monate)."""
    if monthly_120.empty or len(monthly_120) < 120:
        return np.nan
    vals = monthly_120.values.astype(float)
    n = len(vals)  # 120
    positives = 0
    total = n*(n-1)//2  # 7260
    # Effizienter: kumulative Logpreise vermeiden Rundungsfehler
    # aber hier reicht direkt Verhältnis:
    for i in range(n-1):
        vi = vals[i]
        if vi <= 0:
            # defensiv
            continue
        # Vektorisiert über j>i
        rets = vals[i+1:] / vi - 1.0
        positives += int((rets > 0).sum())
    if total == 0:
        return np.nan
    return float(positives / total * 100.0)

# =========================
# Sicherheits-Score
# =========================
def sicherheits_score(geo_pct: float, vr: float, gk_pct: float) -> float:
    # GeoPAK10 als Prozent (z.B. 12.3), GK als Prozent
    if pd.isna(geo_pct) or pd.isna(vr) or pd.isna(gk_pct) or vr <= 0 or geo_pct <= -100:
        return np.nan
    try:
        return float((geo_pct ** 0.8) * (vr ** -1.2) * ((gk_pct / 100.0) ** 1.5))
    except Exception:
        return np.nan

# =========================
# Buffett-Signale (heuristisch, informativ)
# =========================
@st.cache_data(show_spinner=False, ttl=24*60*60)
def get_info_cached(ticker: str) -> dict:
    try:
        t = yf.Ticker(ticker)
        try:
            return t.get_info()
        except Exception:
            return t.info
    except Exception:
        return {}

def buffett_signale(ticker: str, monthly_120: pd.Series) -> dict:
    info = get_info_cached(ticker)
    # Unterbewertung (sehr simple Heuristik):
    pe = info.get("trailingPE") or info.get("forwardPE")
    pb = info.get("priceToBook")
    underval = None
    try:
        underval = (pe is not None and pe > 0 and pe < 15) and (pb is not None and pb > 0 and pb < 3)
    except Exception:
        pass

    # Qualität: ROE, operative Marge, FCF > 0, moderates Leverage
    roe    = info.get("returnOnEquity")         # dezimal
    margin = info.get("operatingMargins")
    fcf    = info.get("freeCashflow")
    dte    = info.get("debtToEquity")
    total_debt = info.get("totalDebt")

    qual = None
    try:
        roe_ok = (roe is not None and roe >= 0.15)
        mar_ok = (margin is not None and margin >= 0.12)
        fcf_ok = (fcf is not None and fcf > 0)
        lev_ok = ((dte is not None and dte < 1.5) or
                  (total_debt is not None and fcf is not None and fcf > 0 and total_debt < 3*fcf))
        qual = roe_ok and mar_ok and fcf_ok and lev_ok
    except Exception:
        pass

    # RSI/Volumen-Proxy aus Monatsreihe (optional, sehr grob):
    rsi_ok = None
    try:
        # 14-Perioden RSI auf Monatsbasis: sehr kurz – hier nur Richtungssignal
        rets = monthly_120.pct_change().dropna()
        if len(rets) >= 15:
            delta = rets.diff()
            up = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
            down = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
            rs = up / down.replace(0, np.nan)
            rsi = 100 - 100/(1+rs)
            rsi_ok = 45 <= float(rsi.iloc[-1]) <= 70
    except Exception:
        pass

    def lab(x):
        if x is True: return "Ja"
        if x is False:return "Nein"
        return "—"

    return {
        "Unterbewertung": lab(underval),
        "Qualität": lab(qual),
        "RSI/Volumen": "OK" if (rsi_ok is True) else ("Warnung" if rsi_ok is False else "—"),
        "Buffett-Signal": lab((underval is True) and (qual is True))
    }

# =========================
# UI
# =========================
st.title("🏆 Champions – GeoPAK10 • Verlust-Ratio (gew.) • Gewinn-Konstanz • Buffett-Signale")

uploaded = st.file_uploader("CSV mit Champions (Spalten: Ticker, Name)", type=["csv"])
if not uploaded:
    st.info("Bitte CSV hochladen. Die App berechnet alles weitere live.")
    st.stop()

try:
    df_in = read_champions_csv(uploaded)
except Exception as e:
    st.error(f"CSV konnte nicht gelesen werden: {e}")
    st.stop()

tickers = df_in["Ticker"].tolist()
name_map = dict(zip(df_in["Ticker"], df_in["Name"]))
st.markdown(f"**{len(tickers)}/{len(tickers)} ausgelesen.**")

# ~12 Jahre laden, um 120 Monate sicher zu haben
hist_end = pd.to_datetime(datetime.today().date())
hist_start = hist_end - pd.DateOffset(years=12)

with st.spinner("Lade Kursdaten (Batch-Modus gegen Rate-Limits)…"):
    close_df, failed = fetch_prices_batch(tickers, hist_start, hist_end, batch_size=20, pause=1.0)

if failed:
    st.warning(f"{len(failed)} Ticker ohne Daten: {failed}")

if close_df.empty:
    st.error("Keine Kursdaten geladen.")
    st.stop()

rows = []
ok = 0
for t in tickers:
    if t not in close_df.columns:
        rows.append({
            "Ticker": t, "Name": name_map.get(t,t),
            "Sicherheitsscore": np.nan, "GeoPAK10 (%)": np.nan,
            "Verlust-Ratio": np.nan, "Gewinn-Konstanz (%)": np.nan,
            "Unterbewertung":"—","Qualität":"—","RSI/Volumen":"—","Buffett-Signal":"—"
        })
        continue
    s = close_df[t].dropna()
    monthly = last_120_monthly(s)
    if monthly.empty:
        rows.append({
            "Ticker": t, "Name": name_map.get(t,t),
            "Sicherheitsscore": np.nan, "GeoPAK10 (%)": np.nan,
            "Verlust-Ratio": np.nan, "Gewinn-Konstanz (%)": np.nan,
            "Unterbewertung":"—","Qualität":"—","RSI/Volumen":"—","Buffett-Signal":"—"
        })
        continue

    geo = geopak10_from_months(monthly)                 # in %
    vr  = verlust_ratio_weighted(monthly)               # Verhältnis (>0)
    gk  = gewinn_konstanz_7260(monthly)                 # in %
    score = sicherheits_score(geo, vr, gk)

    sigs = buffett_signale(t, monthly)

    rows.append({
        "Ticker": t,
        "Name": name_map.get(t, t),
        # Sicherheitsscore direkt neben GeoPAK10:
        "Sicherheitsscore": round(score, 4) if pd.notna(score) else np.nan,
        "GeoPAK10 (%)": round(geo, 2) if pd.notna(geo) else np.nan,
        "Verlust-Ratio": round(vr, 3) if pd.notna(vr) else np.nan,
        "Gewinn-Konstanz (%)": round(gk, 1) if pd.notna(gk) else np.nan,
        **sigs
    })
    if pd.notna(score):
        ok += 1

df = pd.DataFrame(rows)

# Ranking nur für berechnete Scores
ranked = df.dropna(subset=["Sicherheitsscore"]).copy()
ranked = ranked.sort_values(["Sicherheitsscore","GeoPAK10 (%)"], ascending=[False, False]).reset_index(drop=True)
ranked["Rank"] = np.arange(1, len(ranked)+1)

# Nicht berechnete (zu wenig Historie etc.) hintenan
rest = df[df["Sicherheitsscore"].isna()].copy()
final = pd.concat([ranked, rest], axis=0, ignore_index=True)

cols = [
    "Rank", "Ticker", "Name",
    "Sicherheitsscore", "GeoPAK10 (%)", "Verlust-Ratio", "Gewinn-Konstanz (%)",
    "Unterbewertung", "Qualität", "RSI/Volumen", "Buffett-Signal"
]
for c in cols:
    if c not in final.columns:
        final[c] = np.nan

st.success(f"Erfolgreich berechnet: {ok}/{len(tickers)} (benötigt genau 120 Monatskurse).")
st.dataframe(final[cols], use_container_width=True)
st.download_button("📥 CSV herunterladen", final[cols].to_csv(index=False).encode("utf-8"), "champions_scores.csv", "text/csv")

st.caption("Hinweis: Buffett-Signale sind heuristische, informative Checks. Kennzahlen basieren auf 120 Monatsdaten gemäß Aktienbrief-Logik.")
