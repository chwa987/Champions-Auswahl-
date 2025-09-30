# champions_app.py
# Version 2.0 – Champions Analyse: GeoPAK10 + Sicherheits-Score + Buffett-Signale (live mit yfinance)

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

st.set_page_config(page_title="🏆 Champions-Analyse", page_icon="🏆", layout="wide")

# =========================================================
# Helpers: CSV robust einlesen
# =========================================================
def read_champions_csv(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> pd.DataFrame:
    """
    Erwartet mind. 'Ticker'; 'Name' optional. Robust ggü. Komma/Semikolon, Groß/Klein.
    """
    import io
    # 1. Versuch: normal lesen
    try:
        df = pd.read_csv(uploaded_file)
    except Exception:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, sep=";")

    # Spaltennamen normalisieren
    df.columns = [c.strip().lower() for c in df.columns]
    if "ticker" not in df.columns:
        # fallback: erste Spalte als Ticker interpretieren
        df = df.rename(columns={df.columns[0]: "ticker"})
    if "name" not in df.columns:
        df["name"] = df["ticker"]

    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["name"]   = df["name"].astype(str).str.strip()
    df = df[df["ticker"] != ""].drop_duplicates(subset=["ticker"])
    return df[["ticker", "name"]]


# =========================================================
# Helpers: Daten laden (Close & Volume)
# =========================================================
@st.cache_data(show_spinner=False, ttl=60*60)
def fetch_prices_volumes(tickers, start, end):
    """
    Lädt historische Kurse/Volumina für mehrere Ticker in EINEM Download.
    Gibt (close_df, volume_df) mit gemeinsamen Datumsindex zurück.
    """
    if isinstance(tickers, str):
        tickers = [t.strip() for t in tickers.split(",") if t.strip()]

    data = yf.download(
        tickers=" ".join(tickers),
        start=start,
        end=end,
        auto_adjust=True,   # für Preisreihen (Splits/Divs adjustiert)
        progress=False,
        threads=True,
    )
    if data is None or len(data) == 0:
        return pd.DataFrame(), pd.DataFrame()

    close_dict, vol_dict = {}, {}

    if isinstance(data.columns, pd.MultiIndex):
        # Multi-Index (pro Ticker Unterspalten)
        for t in tickers:
            try:
                sub = data[t]
                if "Close" in sub.columns:
                    close_dict[t] = sub["Close"].rename(t)
                if "Volume" in sub.columns:
                    vol_dict[t] = sub["Volume"].rename(t)
            except Exception:
                continue
    else:
        # Single-Index (ein Ticker)
        if "Close" in data.columns:
            close_dict[tickers[0]] = data["Close"].rename(tickers[0])
        if "Volume" in data.columns:
            vol_dict[tickers[0]] = data["Volume"].rename(tickers[0])

    close_df = pd.concat(close_dict.values(), axis=1) if close_dict else pd.DataFrame()
    vol_df   = pd.concat(vol_dict.values(), axis=1) if vol_dict else pd.DataFrame()
    if not close_df.empty:
        close_df = close_df.sort_index().dropna(how="all")
    if not vol_df.empty:
        vol_df = vol_df.reindex(close_df.index)
    return close_df, vol_df


# =========================================================
# Kennzahlen: GeoPAK10, Verlust-Ratio, Gewinnkonstanz
# =========================================================
def annual_returns_from_series(s: pd.Series) -> pd.Series:
    """Jahresrenditen aus täglichen Kursen: Schlusskurs pro Jahr → pct_change()."""
    y = s.resample("Y").last().pct_change().dropna()
    return y

def calc_geopak10_from_yearly(yearly: pd.Series) -> float:
    """Geometrisches Mittel/CAGR der letzten 10 Jahresrenditen (wenn vorhanden)."""
    if yearly is None or len(yearly) < 10:
        return np.nan
    # Geom. Mittel über genau 10 Jahre:
    y10 = yearly.tail(10)
    geo = (np.prod(1 + y10) ** (1/len(y10)) - 1)  # als Dezimal
    return float(geo)

def calc_verlust_ratio_from_yearly(yearly: pd.Series) -> float:
    """
    Verlust-Ratio nach Aktienbrief-Logik-Nähe:
    Verhältnis Ø-Verlustjahre / Ø-Gewinnjahre (Beträge). Niedriger ist besser (> durch ^-1.2 in Score).
    """
    if yearly is None or len(yearly) == 0:
        return np.nan
    gains = yearly[yearly > 0]
    losses = yearly[yearly < 0]
    if gains.empty or losses.empty:
        return np.nan
    return float(abs(losses.mean()) / gains.mean())

def calc_gewinnkonstanz_from_yearly(yearly: pd.Series) -> float:
    """% positive Jahre."""
    if yearly is None or len(yearly) == 0:
        return np.nan
    return float((yearly > 0).mean() * 100.0)

def safety_score(geo: float, verlustratio: float, gk_percent: float) -> float:
    """
    Sicherheits-Score gemäß deiner Formel:
      Score = GEO^0.8 * (Verlust-Ratio)^(-1.2) * (Gewinnkonstanz/100)^(1.5)
    """
    if any(pd.isna([geo, verlustratio, gk_percent])) or verlustratio <= 0 or geo <= -1:
        return np.nan
    return float((geo ** 0.8) * (verlustratio ** -1.2) * ((gk_percent / 100.0) ** 1.5))


# =========================================================
# Technische Hilfskennzahlen für Buffett-Momentum-Info
# =========================================================
def rsi(series: pd.Series, period: int = 14) -> float:
    s = series.dropna()
    if len(s) < period + 1:
        return np.nan
    delta = s.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    rsi_series = 100 - (100 / (1 + rs))
    return float(rsi_series.iloc[-1])

def volume_ratio(vol_series: pd.Series, lookback: int = 60) -> float:
    v = vol_series.dropna()
    if len(v) < lookback:
        return np.nan
    base = v.rolling(lookback).mean().iloc[-1]
    cur  = v.iloc[-1]
    if base is None or base == 0:
        return np.nan
    return float(cur / base)


# =========================================================
# Fundamentals aus yfinance (gecached)
# =========================================================
@st.cache_data(show_spinner=False, ttl=24*60*60)
def get_fundamentals(ticker: str) -> dict:
    try:
        tk = yf.Ticker(ticker)
        try:
            info = tk.get_info()  # neuere yfinance-Versionen
        except Exception:
            info = tk.info       # Fallback
        if not isinstance(info, dict):
            return {}
        return info
    except Exception:
        return {}


def buffett_signals(ticker: str,
                    price_s: pd.Series | None,
                    vol_s: pd.Series | None) -> dict:
    """
    Liefert informativ:
      - Unterbewertung (PE/PB Schwellen)
      - Qualität (ROE, Marge, FCF, Verschuldung)
      - RSI/Volumen (technischer Check)
      - Buffett-Signal (Unterbewertung UND Qualität)
    """
    info = get_fundamentals(ticker)

    # Unterbewertung (sehr einfache Heuristik)
    pe = info.get("trailingPE", None) or info.get("forwardPE", None)
    pb = info.get("priceToBook", None)
    underval = None
    try:
        underval = (pe is not None and pe > 0 and pe < 15) and (pb is not None and pb > 0 and pb < 3)
    except Exception:
        underval = None

    # Qualität (Moat/Management/Profitabilität-Proxy)
    roe = info.get("returnOnEquity", None)  # typ. in Dezimal (0.15 = 15%)
    margin = info.get("operatingMargins", None)
    fcf = info.get("freeCashflow", None)    # absoluter Betrag (Währung)
    dte = info.get("debtToEquity", None)    # Verhältnis
    total_debt = info.get("totalDebt", None)

    qual = None
    try:
        roe_ok = (roe is not None and roe >= 0.15)
        mar_ok = (margin is not None and margin >= 0.12)
        fcf_ok = (fcf is not None and fcf > 0)
        lev_ok = ((dte is not None and dte < 1.5) or
                  (total_debt is not None and fcf is not None and fcf > 0 and total_debt < 3 * fcf))
        qual = roe_ok and mar_ok and fcf_ok and lev_ok
    except Exception:
        qual = None

    # RSI/Volumen
    rsi_ok = None
    vol_ok = None
    try:
        r = rsi(price_s) if price_s is not None else np.nan
        vr = volume_ratio(vol_s) if vol_s is not None else np.nan
        rsi_ok = (not pd.isna(r)) and (45 <= r <= 70)
        vol_ok = (not pd.isna(vr)) and (vr >= 1.0)
    except Exception:
        pass

    buffett_ok = None
    if isinstance(underval, bool) and isinstance(qual, bool):
        buffett_ok = (underval and qual)

    def lab(x_true):
        if x_true is True:  return "Ja"
        if x_true is False: return "Nein"
        return "—"

    return {
        "Unterbewertung": lab(underval),
        "Qualität": lab(qual),
        "RSI/Volumen": "OK" if (rsi_ok and vol_ok) else ("Warnung" if (rsi_ok is not None or vol_ok is not None) else "—"),
        "Buffett-Signal": lab(buffett_ok),
    }


# =========================================================
# Champions-Berechnung (pro Ticker)
# =========================================================
def compute_champions(close_df: pd.DataFrame, vol_df: pd.DataFrame) -> pd.DataFrame:
    """
    Berechnet alle Kennzahlen auf Basis der letzten 10 Jahresrenditen.
    Nur wenn >=10 Jahreszahlen vorhanden: Aufnahme ins Ranking.
    """
    out = []
    for t in close_df.columns:
        s = close_df[t].dropna()
        yearly = annual_returns_from_series(s)
        if len(yearly) < 10:
            # Nicht genug Historie → nicht berechnen (leere Werte)
            out.append({
                "Ticker": t, "Name": t,
                "Sicherheits-Score": np.nan,
                "GeoPAK10 (%)": np.nan,
                "Verlust-Ratio": np.nan,
                "Gewinnkonstanz (%)": np.nan,
                "Unterbewertung": "—",
                "Qualität": "—",
                "RSI/Volumen": "—",
                "Buffett-Signal": "—",
            })
            continue

        geo = calc_geopak10_from_yearly(yearly)             # Dezimal
        vr  = calc_verlust_ratio_from_yearly(yearly)        # Verhältnis
        gk  = calc_gewinnkonstanz_from_yearly(yearly)       # Prozent
        score = safety_score(geo, vr, gk)

        sigs = buffett_signals(t, s, vol_df.get(t) if isinstance(vol_df, pd.DataFrame) else None)

        out.append({
            "Ticker": t,
            "Name": t,  # wird später gemappt
            "Sicherheits-Score": round(score, 4) if pd.notna(score) else np.nan,
            "GeoPAK10 (%)": round(geo * 100.0, 2) if pd.notna(geo) else np.nan,
            "Verlust-Ratio": round(vr, 3) if pd.notna(vr) else np.nan,
            "Gewinnkonstanz (%)": round(gk, 1) if pd.notna(gk) else np.nan,
            **sigs
        })

    df = pd.DataFrame(out)

    # Ranking nur für berechnete Scores
    df_ranked = df.dropna(subset=["Sicherheits-Score"]).copy()
    df_ranked = df_ranked.sort_values("Sicherheits-Score", ascending=False).reset_index(drop=True)
    df_ranked["Rank"] = np.arange(1, len(df_ranked) + 1)

    # nicht berechnete hinterher anhängen (ohne Rank)
    df_na = df[df["Sicherheits-Score"].isna()].copy()
    final = pd.concat([df_ranked, df_na], axis=0, ignore_index=True)

    return final


# =========================================================
# UI
# =========================================================
st.title("🏆 Champions – GeoPAK10, Sicherheits-Score & Buffett-Signale (live)")

st.sidebar.header("⚙️ Einstellungen")
# Unabhängig von der UI laden wir ~12 Jahre, damit i. d. R. 10 volle Jahresrenditen vorhanden sind:
end_date = st.sidebar.date_input("Ende", value=datetime.today())
start_date = st.sidebar.date_input("Start (nur Anzeige, Berechnung nutzt ~12 Jahre)", value=datetime.today() - timedelta(days=365*10))

uploaded = st.file_uploader("CSV mit Champions (mind. 'Ticker'; optional 'Name')", type=["csv"])

if not uploaded:
    st.info("Bitte eine CSV hochladen. Beispiel-Spalten: Ticker, Name")
    st.stop()

# CSV lesen
try:
    df_in = read_champions_csv(uploaded)
except Exception as e:
    st.error(f"CSV konnte nicht gelesen werden: {e}")
    st.stop()

parsed = len(df_in)
st.markdown(f"**{parsed}/{parsed} ausgelesen.**")

tickers = df_in["ticker"].tolist()
name_map = dict(zip(df_in["ticker"], df_in["name"]))

# Historie laden (ca. 12 Jahre, unabhängig von den Sidebar-Daten)
hist_end = pd.to_datetime(datetime.today().date())
hist_start = hist_end - pd.DateOffset(years=12)

with st.spinner("Lade Kurs- und Volumendaten …"):
    close_df, vol_df = fetch_prices_volumes(tickers, hist_start, hist_end)

if close_df.empty:
    st.warning("Keine Kursdaten geladen.")
    st.stop()

with st.spinner("Berechne Kennzahlen …"):
    df = compute_champions(close_df, vol_df)

# Namen mappen
df["Name"] = df["Ticker"].map(name_map).fillna(df["Name"])

# Anzeige: Score vor GeoPAK10
cols = [
    "Rank", "Ticker", "Name", "Sicherheits-Score",
    "GeoPAK10 (%)", "Verlust-Ratio", "Gewinnkonstanz (%)",
    "Unterbewertung", "Qualität", "RSI/Volumen", "Buffett-Signal"
]
# Einige Reihen können kein Rank haben (zu wenig Historie)
if "Rank" not in df.columns:
    df["Rank"] = np.nan

# Sortierte Darstellung: erst Ranked, dann die ohne Historie
df_display = df.copy()
ranked = df_display["Rank"].notna()
df_display_ranked = df_display[ranked]
df_display_na = df_display[~ranked]
df_display = pd.concat([df_display_ranked, df_display_na], ignore_index=True)

st.dataframe(df_display[cols], use_container_width=True)

# Zählung „erfolgreich berechnet“
success = int(df["Sicherheits-Score"].notna().sum())
total = len(tickers)
st.markdown(f"**Erfolgreich berechnet:** {success}/{total} (≥10 Jahresrenditen erforderlich)")

st.download_button(
    "📥 Ergebnisse (CSV)",
    df_display[cols].to_csv(index=False).encode("utf-8"),
    "champions_scores.csv",
    "text/csv"
)

st.caption("Hinweis: Buffett-Signale sind heuristische Annäherungen auf Basis öffentlich verfügbarer Felder aus yfinance-Infos. Nur zu Informations- und Ausbildungszwecken.")
