#!/usr/bin/env python3
"""
200W EMA Pullback — stock scanner
=================================
Python port of the "200W EMA Pullback — Screener" Pine indicator.

Pulls daily OHLC from Yahoo (via yfinance), builds weekly bars, computes the
200W / 200D EMAs, the % band, the regime filter and the entry condition, and
prints / saves / pushes the symbols that are signalling.

Usage
-----
  python ema200w_scanner.py                              # S&P 500, signal in last 4 weeks
  python ema200w_scanner.py --universe nasdaq100
  python ema200w_scanner.py --universe ftse100
  python ema200w_scanner.py --tickers-file my_list.txt   # one symbol per line
  python ema200w_scanner.py --mode reclaim --recent 1    # only this week's reclaims
  python ema200w_scanner.py --in-band                    # also list anything sitting in the band now
  python ema200w_scanner.py --push                       # send hits to Telegram

Telegram: set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in the environment.

Install:  pip install yfinance pandas numpy requests lxml
"""

import argparse
import datetime as dt
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# EMA that seeds exactly like Pine's ta.ema (SMA of the first `length` bars,
# then standard recursion). pandas .ewm() seeds on the first value, which can
# differ slightly for years on a 200-period EMA.
# --------------------------------------------------------------------------- #
def pine_ema(series: pd.Series, length: int) -> pd.Series:
    vals = series.to_numpy(dtype=float)
    out = np.full(len(vals), np.nan)
    if len(vals) < length:
        return pd.Series(out, index=series.index)
    alpha = 2.0 / (length + 1)
    out[length - 1] = vals[:length].mean()
    for i in range(length, len(vals)):
        out[i] = alpha * vals[i] + (1 - alpha) * out[i - 1]
    return pd.Series(out, index=series.index)


def bars_since(cond: pd.Series) -> pd.Series:
    """Pine ta.barssince: bars since cond was last true (NaN if never)."""
    out = np.full(len(cond), np.nan)
    last = None
    for i, c in enumerate(cond.to_numpy()):
        if c:
            last = i
        if last is not None:
            out[i] = i - last
    return pd.Series(out, index=cond.index)


# --------------------------------------------------------------------------- #
# Core logic — one symbol
# --------------------------------------------------------------------------- #
def evaluate(daily: pd.DataFrame, *, len_w=200, len_d=200, band_pct=3.0,
             mode="touch", need_regime=True, regime_min=2) -> pd.DataFrame | None:
    """
    daily: DataFrame with columns Open/High/Low/Close indexed by date.
    Returns a weekly DataFrame with the same outputs as the Pine screener.
    """
    daily = daily.dropna(subset=["Close"]).copy()
    if len(daily) < len_d + 5:
        return None

    # 200D EMA on daily closes; request.security("D") on a weekly chart with
    # lookahead off gives the daily value as of the last daily bar of each week.
    daily["emaD"] = pine_ema(daily["Close"], len_d)

    # Weekly bars, Mon–Fri, labelled by the week's Friday
    wk = daily.resample("W-FRI").agg(
        Open=("Open", "first"), High=("High", "max"),
        Low=("Low", "min"), Close=("Close", "last"), emaD=("emaD", "last"),
    ).dropna(subset=["Close"])

    if len(wk) < len_w + regime_min + 2:
        return None

    wk["emaW"] = pine_ema(wk["Close"], len_w)
    wk["bandHi"] = wk["emaW"] * (1 + band_pct / 100)
    wk["bandLo"] = wk["emaW"] * (1 - band_pct / 100)

    regime = (wk["emaD"] > wk["emaW"]) if need_regime else pd.Series(True, index=wk.index)
    regime_age = bars_since(~regime)
    regime_ok = regime & (regime_age.isna() | (regime_age >= regime_min))

    touched = (wk["Low"] <= wk["bandHi"]) & (wk["Low"] >= wk["bandLo"] * 0.98)
    hold_above = wk["Close"] > wk["emaW"]
    reclaim = (wk["Close"].shift(1) < wk["emaW"].shift(1)) & (wk["Close"] > wk["emaW"])
    entry = (touched & hold_above) if mode == "touch" else reclaim

    signal = regime_ok & entry & wk["emaW"].notna() & wk["emaD"].notna()

    wk["signal"] = signal.astype(int)
    wk["weeks_since_signal"] = bars_since(signal)
    wk["regime_on"] = regime_ok.astype(int)
    wk["regime_age"] = regime_age
    wk["pct_from_200w"] = (wk["Close"] / wk["emaW"] - 1) * 100
    wk["low_pct_from_200w"] = (wk["Low"] / wk["emaW"] - 1) * 100
    wk["touched"] = touched.astype(int)
    wk["in_band"] = ((wk["Close"] >= wk["bandLo"]) & (wk["Close"] <= wk["bandHi"])).astype(int)
    return wk


# --------------------------------------------------------------------------- #
# Universe helpers
# --------------------------------------------------------------------------- #
_TABLE_CACHE: dict[str, list[pd.DataFrame]] = {}


def _tables(url: str) -> list[pd.DataFrame]:
    import requests
    from io import StringIO
    if url in _TABLE_CACHE:
        return _TABLE_CACHE[url]
    r = requests.get(url, headers={"User-Agent": "ema200w-scanner/1.0 (github.com/tobyjfowler/ema200w-scanner)"}, timeout=30)
    r.raise_for_status()
    tables = pd.read_html(StringIO(r.text))
    _TABLE_CACHE[url] = tables
    # flatten multi-level headers to plain strings
    for t in tables:
        t.columns = [" ".join(str(x) for x in c if str(x) != "nan").strip() if isinstance(c, tuple) else str(c)
                     for c in t.columns]
    return tables


def _find_col(t: pd.DataFrame, *names) -> str | None:
    for c in t.columns:
        cl = str(c).lower()
        if any(n in cl for n in names):
            return c
    return None


def _constituents(url: str, ticker_names=("ticker", "symbol", "epic")) -> tuple[pd.DataFrame, str]:
    """Largest table on the page that has a ticker-like column."""
    best = None
    for t in _tables(url):
        col = _find_col(t, *ticker_names)
        if col and len(t) >= 50 and (best is None or len(t) > len(best[0])):
            best = (t, col)
    if best is None:
        raise RuntimeError(f"no constituents table found at {url}")
    return best


_UNIVERSE_URL = {
    "sp500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "nasdaq100": "https://en.wikipedia.org/wiki/Nasdaq-100",
    "ftse100": "https://en.wikipedia.org/wiki/FTSE_100_Index",
}


def _yahoo(sym: str, name: str) -> str:
    sym = str(sym).strip()
    if name == "ftse100":
        return sym.replace(".", "") + ".L"
    return sym.replace(".", "-")     # BRK.B -> BRK-B


def load_universe(name: str) -> list[str]:
    if name not in _UNIVERSE_URL:
        raise ValueError(f"unknown universe {name}")
    tbl, col = _constituents(_UNIVERSE_URL[name])
    return sorted({_yahoo(s, name) for s in tbl[col].dropna().astype(str) if s.strip()})


def load_universe_meta(name: str) -> dict[str, dict]:
    """symbol -> {name, sector} from the same table. Best effort — never fails the scan."""
    try:
        tbl, col = _constituents(_UNIVERSE_URL[name])
        ncol = _find_col(tbl, "security", "company", "name")
        scol = _find_col(tbl, "sector", "industry")
        return {_yahoo(r[col], name): {"name": str(r[ncol]) if ncol else "", "sector": str(r[scol]) if scol else ""}
                for _, r in tbl.iterrows() if str(r[col]).strip()}
    except Exception as e:
        print(f"meta for {name} failed: {e}", file=sys.stderr)
        return {}


def load_tickers_file(path: str) -> list[str]:
    with open(path) as f:
        return [ln.strip().upper() for ln in f if ln.strip() and not ln.startswith("#")]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def fetch_daily(tickers: list[str], years: int) -> dict[str, pd.DataFrame]:
    import yfinance as yf
    out = {}
    chunk = 100
    for i in range(0, len(tickers), chunk):
        batch = tickers[i:i + chunk]
        # auto_adjust=False -> Close is split-adjusted but not dividend-adjusted,
        # which is what TradingView shows by default.
        data = None
        for attempt in range(3):
            try:
                data = yf.download(batch, period=f"{years}y", interval="1d", group_by="ticker",
                                   auto_adjust=False, threads=True, progress=False)
            except Exception as e:
                print(f"  download error for batch {i // chunk + 1}: {e}", file=sys.stderr)
                data = None
            if data is not None and len(data):
                break
            print(f"  empty download for batch {i // chunk + 1}, retrying…", file=sys.stderr)
            time.sleep(10 * (attempt + 1))
        if data is None or not len(data):
            continue
        for t in batch:
            try:
                df = data[t] if len(batch) > 1 else data
                df = df[["Open", "High", "Low", "Close"]].dropna(subset=["Close"])
                if len(df):
                    out[t] = df
            except Exception:
                pass
        print(f"  fetched {min(i + chunk, len(tickers))}/{len(tickers)}", file=sys.stderr)
    return out


# --------------------------------------------------------------------------- #
# Telegram
# --------------------------------------------------------------------------- #
def push(message: str):
    import requests
    token, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not (token and chat):
        print("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set — skipping push", file=sys.stderr)
        return
    r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage", data={
        "chat_id": chat, "text": message[:4000], "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }, timeout=15)
    if not r.ok:
        print(f"Telegram error {r.status_code}: {r.text[:200]}", file=sys.stderr)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="200W EMA pullback scanner")
    ap.add_argument("--universe", choices=["sp500", "nasdaq100", "ftse100"], default="sp500")
    ap.add_argument("--tickers-file", help="text file, one symbol per line (overrides --universe)")
    ap.add_argument("--mode", choices=["touch", "reclaim"], default="touch",
                    help="touch = Touch & hold, reclaim = close back above 200W")
    ap.add_argument("--band", type=float, default=3.0, help="band around 200W EMA, %% (default 3)")
    ap.add_argument("--len-w", type=int, default=200)
    ap.add_argument("--len-d", type=int, default=200)
    ap.add_argument("--no-regime", action="store_true", help="drop the 200D > 200W requirement")
    ap.add_argument("--regime-min", type=int, default=2, help="regime must be on for N weeks")
    ap.add_argument("--recent", type=int, default=4,
                    help="report symbols with a signal in the last N weeks (default 4)")
    ap.add_argument("--in-band", action="store_true",
                    help="also list regime-on symbols whose close is inside the band now")
    ap.add_argument("--completed-only", action="store_true",
                    help="evaluate on the last completed week, ignoring the in-progress bar")
    ap.add_argument("--years", type=int, default=15, help="history to download (default 15)")
    ap.add_argument("--out", default="scan_results.csv")
    ap.add_argument("--push", action="store_true", help="send results to Telegram")
    args = ap.parse_args()

    tickers = load_tickers_file(args.tickers_file) if args.tickers_file else load_universe(args.universe)
    print(f"Universe: {len(tickers)} symbols", file=sys.stderr)

    data = fetch_daily(tickers, args.years)
    print(f"Got data for {len(data)} symbols", file=sys.stderr)

    rows = []
    for sym, daily in data.items():
        if args.completed_only:
            # drop any days from the current Mon–Fri week
            today = pd.Timestamp(dt.date.today())
            week_start = today - pd.Timedelta(days=today.weekday())
            daily = daily[daily.index < week_start]
        wk = evaluate(daily, len_w=args.len_w, len_d=args.len_d, band_pct=args.band,
                      mode=args.mode, need_regime=not args.no_regime, regime_min=args.regime_min)
        if wk is None or wk["emaW"].isna().iloc[-1]:
            continue
        last = wk.iloc[-1]
        rows.append({
            "symbol": sym,
            "week": wk.index[-1].date(),
            "close": round(last["Close"], 2),
            "ema200w": round(last["emaW"], 2),
            "ema200d": round(last["emaD"], 2),
            "signal_now": int(last["signal"]),
            "weeks_since_signal": None if np.isnan(last["weeks_since_signal"]) else int(last["weeks_since_signal"]),
            "regime_on": int(last["regime_on"]),
            "pct_from_200w": round(last["pct_from_200w"], 2),
            "low_pct_from_200w": round(last["low_pct_from_200w"], 2),
            "touched": int(last["touched"]),
            "in_band": int(last["in_band"]),
        })

    res = pd.DataFrame(rows)
    if res.empty:
        print("ERROR: no symbols evaluated — data download returned nothing "
              "(Yahoo rate limit?) or history too short.", file=sys.stderr)
        sys.exit(2)
    res.to_csv(args.out, index=False)

    hits = res[(res["weeks_since_signal"].notna()) & (res["weeks_since_signal"] <= args.recent)]
    hits = hits.sort_values(["weeks_since_signal", "pct_from_200w"])

    cols = ["symbol", "weeks_since_signal", "close", "ema200w", "pct_from_200w", "low_pct_from_200w", "regime_on"]
    print(f"\n=== Signals in the last {args.recent} week(s) — mode: {args.mode}, band: {args.band}% ===")
    print(hits[cols].to_string(index=False) if len(hits) else "  none")

    watch = pd.DataFrame()
    if args.in_band:
        watch = res[(res["regime_on"] == 1) & (res["in_band"] == 1) & ~res["symbol"].isin(hits["symbol"])]
        watch = watch.sort_values("pct_from_200w")
        print(f"\n=== Regime on & close inside the band now (no signal yet) ===")
        print(watch[cols].to_string(index=False) if len(watch) else "  none")

    print(f"\nFull table ({len(res)} symbols) saved to {args.out}")

    if args.push:
        uni = "custom list" if args.tickers_file else args.universe.upper()
        mode = "touch & hold" if args.mode == "touch" else "reclaim"
        head = f"<b>200W EMA scan — {uni}</b>\n{mode}, ±{args.band:g}% band, last {args.recent}w\n"
        if len(hits):
            lines = []
            for r in hits.itertuples():
                when = "this week" if r.weeks_since_signal == 0 else f"{r.weeks_since_signal}w ago"
                lines.append(f"<code>{r.symbol:<8}</code> {when:<10} {r.pct_from_200w:+.1f}% from 200W")
            body = "\n".join(lines)
        else:
            body = "No signals."
        msg = f"{head}\n{body}"
        if args.in_band and len(watch):
            msg += "\n\n<i>In band, no signal yet:</i> " + ", ".join(watch["symbol"].head(20))
        msg += f"\n\n<i>{len(res)} stocks scanned · week ending {res['week'].iloc[0]}</i>"
        push(msg)

if __name__ == "__main__":
    main()
