#!/usr/bin/env python3
"""
Build the dashboard data files.

Scans every universe, writes:
  data/latest.json   — one record per stock with all metrics + a 2-year sparkline
  data/history.json  — every run's breadth numbers + a log of live signals

Sends a Telegram message only when there are NEW signals since the last run
(or always with --always-push, used for the Friday summary).

Usage:  python build_dashboard.py [--universes sp500,nasdaq100,ftse100] [--push]
"""

import argparse
import datetime as dt
import json
import os
import sys

import numpy as np
import pandas as pd

from ema200w_scanner import evaluate, fetch_daily, load_universe, load_universe_meta, push

DATA_DIR = "data"
SPARK_WEEKS = 104          # weeks of close / 200W EMA kept per stock for the mini chart
SIGNAL_LOOKBACK_W = 260    # how far back to list historic signals (5 years)
FWD = (4, 12, 26)          # forward-return horizons in weeks
CLUSTER_GAP = 8            # weeks without a signal before a new one counts as a fresh setup

UNIVERSE_LABEL = {"sp500": "S&P 500", "nasdaq100": "Nasdaq 100", "sp400": "S&P 400", "ftse100": "FTSE 100", "ftse250": "FTSE 250",
                  "watchlist": "Watchlist", "markets": "Markets"}

# Indexes, commodities, rates and crypto: Yahoo symbol -> (name, category, TradingView symbol)
MARKETS = {
    "^GSPC": ("S&P 500", "Index", "SP:SPX"),
    "^NDX": ("Nasdaq 100", "Index", "NASDAQ:NDX"),
    "^DJI": ("Dow Jones Industrial", "Index", "DJ:DJI"),
    "^RUT": ("Russell 2000", "Index", "TVC:RUT"),
    "^FTSE": ("FTSE 100", "Index", "LSE:UKX"),
    "^FTMC": ("FTSE 250", "Index", "LSE:MCX"),
    "^GDAXI": ("DAX", "Index", "XETR:DAX"),
    "^STOXX50E": ("Euro Stoxx 50", "Index", "TVC:SX5E"),
    "^N225": ("Nikkei 225", "Index", "TVC:NI225"),
    "^HSI": ("Hang Seng", "Index", "TVC:HSI"),
    "DX-Y.NYB": ("US Dollar Index", "Index", "TVC:DXY"),
    "GC=F": ("Gold", "Commodity", "COMEX:GC1!"),
    "SI=F": ("Silver", "Commodity", "COMEX:SI1!"),
    "HG=F": ("Copper", "Commodity", "COMEX:HG1!"),
    "PL=F": ("Platinum", "Commodity", "NYMEX:PL1!"),
    "CL=F": ("Crude oil (WTI)", "Commodity", "NYMEX:CL1!"),
    "BZ=F": ("Brent crude", "Commodity", "NYMEX:BB1!"),
    "NG=F": ("Natural gas", "Commodity", "NYMEX:NG1!"),
    "ZC=F": ("Corn", "Commodity", "CBOT:ZC1!"),
    "ZW=F": ("Wheat", "Commodity", "CBOT:ZW1!"),
    "ZS=F": ("Soybeans", "Commodity", "CBOT:ZS1!"),
    "TLT": ("US 20y+ Treasuries (TLT)", "Rates & credit", "NASDAQ:TLT"),
    "HYG": ("US high yield (HYG)", "Rates & credit", "AMEX:HYG"),
    "BTC-USD": ("Bitcoin", "Crypto", "INDEX:BTCUSD"),
    "ETH-USD": ("Ether", "Crypto", "INDEX:ETHUSD"),
}


# Yahoo exchange code -> TradingView exchange prefix
_TV_EXCHANGE = {"NMS": "NASDAQ", "NGM": "NASDAQ", "NCM": "NASDAQ", "NAS": "NASDAQ",
                "NYQ": "NYSE", "ASE": "AMEX", "PCX": "AMEX", "BTS": "CBOE", "LSE": "LSE"}
_EXCH_PATH = os.path.join(DATA_DIR, "exchanges.json")


def load_exchanges(symbols: list[str]) -> dict[str, str]:
    """Yahoo exchange code per symbol, cached in data/exchanges.json; only new symbols are looked up."""
    import yfinance as yf
    from concurrent.futures import ThreadPoolExecutor
    cache = json.load(open(_EXCH_PATH)) if os.path.exists(_EXCH_PATH) else {}
    todo = [s for s in symbols if s not in cache]
    if todo:
        print(f"looking up exchange for {len(todo)} new symbols", file=sys.stderr)

        def one(sym):
            try:
                return sym, str(yf.Ticker(sym).fast_info["exchange"])
            except Exception:
                return sym, ""
        with ThreadPoolExecutor(max_workers=6) as ex:
            for sym, code in ex.map(one, todo):
                if code:
                    cache[sym] = code
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(_EXCH_PATH, "w") as f:
            json.dump(cache, f, indent=0, sort_keys=True)
    return cache


def tv_symbol(sym: str, exch: str = "") -> str:
    """TradingView symbol (EXCHANGE:TICKER) for a Yahoo ticker."""
    if sym.endswith(".L"):
        return "LSE:" + sym[:-2]
    pref = _TV_EXCHANGE.get(exch, "")
    base = sym.replace("-", ".")
    return f"{pref}:{base}" if pref else base


def r2(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 2)


def scan_universe(name: str, args, exchanges: dict[str, str]) -> tuple[list[dict], list[dict]]:
    if name == "markets":
        tickers = list(MARKETS)
        meta = {k: {"name": v[0], "sector": v[1]} for k, v in MARKETS.items()}
        tv_override = {k: v[2] for k, v in MARKETS.items()}
    elif name == "watchlist":
        # watchlist.txt in the repo root: one Yahoo symbol per line, optional "# comment".
        # Lines like "SYM, Company name" set the display name.
        if not os.path.exists("watchlist.txt"):
            return [], []
        tickers, meta, tv_override = [], {}, {}
        for ln in open("watchlist.txt"):
            ln = ln.split("#")[0].strip()
            if not ln:
                continue
            sym, _, nm = ln.partition(",")
            sym = sym.strip().upper()
            tickers.append(sym)
            meta[sym] = {"name": nm.strip(), "sector": "Watchlist"}
        if not tickers:
            return [], []
        exchanges.update(load_exchanges([t for t in tickers if not t.endswith(".L")]))
    else:
        try:
            tickers = load_universe(name)
        except Exception as e:
            print(f"[{name}] could not load constituents: {e}", file=sys.stderr)
            return [], []
        meta = load_universe_meta(name)
        tv_override = {}
        exchanges.update(load_exchanges([t for t in tickers if not t.endswith(".L")]))
    print(f"[{name}] {len(tickers)} symbols", file=sys.stderr)
    data = fetch_daily(tickers, args.years)
    print(f"[{name}] data for {len(data)}", file=sys.stderr)

    stocks, signals = [], []
    for sym, daily in data.items():
        wk = evaluate(daily, band_pct=args.band, mode=args.mode, regime_min=args.regime_min)
        if wk is None or wk["emaW"].isna().iloc[-1]:
            continue
        # also compute the other entry mode so the page can show both
        wk_other = evaluate(daily, band_pct=args.band,
                            mode="reclaim" if args.mode == "touch" else "touch",
                            regime_min=args.regime_min)
        last = wk.iloc[-1]
        m = meta.get(sym, {})

        tail = wk.tail(SPARK_WEEKS)
        spark = {d.strftime("%Y-%m-%d"): (r2(c), r2(e), int(sg))
                 for d, c, e, sg in zip(tail.index, tail["Close"], tail["emaW"], tail["signal"])}
        # 52-week high/low context
        last52 = wk.tail(52)
        hi52, lo52 = last52["High"].max(), last52["Low"].min()

        stocks.append({
            "symbol": sym,
            "tv": tv_override.get(sym) or tv_symbol(sym, exchanges.get(sym, "")),
            "name": m.get("name", ""),
            "sector": m.get("sector", ""),
            "universe": name,
            "week": wk.index[-1].strftime("%Y-%m-%d"),
            "close": r2(last["Close"]),
            "ema200w": r2(last["emaW"]),
            "ema200d": r2(last["emaD"]),
            "pct_from_200w": r2(last["pct_from_200w"]),
            "low_pct_from_200w": r2(last["low_pct_from_200w"]),
            "pct_to_band": r2((last["Close"] / last["bandHi"] - 1) * 100),  # distance above band top
            "regime_on": int(last["regime_on"]),
            "regime_weeks": None if np.isnan(last["regime_age"]) else int(last["regime_age"]),
            "touched": int(last["touched"]),
            "in_band": int(last["in_band"]),
            "signal_now": int(last["signal"]),
            "weeks_since_signal": None if np.isnan(last["weeks_since_signal"]) else int(last["weeks_since_signal"]),
            "weeks_since_other": None if np.isnan(wk_other["weeks_since_signal"].iloc[-1]) else int(wk_other["weeks_since_signal"].iloc[-1]),
            "pct_from_52w_high": r2((last["Close"] / hi52 - 1) * 100),
            "slope_200w": r2((last["emaW"] / wk["emaW"].iloc[-14] - 1) * 100) if len(wk) > 14 and pd.notna(wk["emaW"].iloc[-14]) else None,
            "pct_from_52w_low": r2((last["Close"] / lo52 - 1) * 100),
            "ret_4w": r2((last["Close"] / wk["Close"].iloc[-5] - 1) * 100) if len(wk) > 5 else None,
            "ret_12w": r2((last["Close"] / wk["Close"].iloc[-13] - 1) * 100) if len(wk) > 13 else None,
            "spark": spark,
        })

        # historic signals with forward returns (both modes). Touch & hold re-fires
        # every week price sits in the band, so collapse runs: keep the first signal
        # of a cluster (no signal in the previous CLUSTER_GAP weeks) and count repeats.
        for mode, frame in ((args.mode, wk), ("reclaim" if args.mode == "touch" else "touch", wk_other)):
            sig = frame["signal"].to_numpy()
            idx = np.where(sig == 1)[0]
            closes = frame["Close"].to_numpy()
            for i in idx:
                if i < len(frame) - SIGNAL_LOOKBACK_W:
                    continue
                if sig[max(0, i - CLUSTER_GAP):i].any():
                    continue
                repeats = int(sig[i + 1:i + 1 + CLUSTER_GAP].sum())
                rec = {
                    "symbol": sym, "universe": name, "mode": mode,
                    "date": frame.index[i].strftime("%Y-%m-%d"),
                    "price": r2(closes[i]), "ema200w": r2(frame["emaW"].iloc[i]),
                    "weeks_ago": int(len(frame) - 1 - i), "repeats": repeats,
                }
                for h in FWD:
                    rec[f"fwd_{h}w"] = r2((closes[i + h] / closes[i] - 1) * 100) if i + h < len(closes) else None
                rec["fwd_now"] = r2((closes[-1] / closes[i] - 1) * 100)
                # max drawdown from entry to +26w (or to now)
                end = min(i + 26, len(closes) - 1)
                rec["mdd_26w"] = r2((closes[i:end + 1].min() / closes[i] - 1) * 100)
                signals.append(rec)
    return stocks, signals


def breadth(stocks: list[dict]) -> dict:
    n = len(stocks)
    if not n:
        return {}
    return {
        "n": n,
        "regime_on_pct": round(100 * sum(s["regime_on"] for s in stocks) / n, 1),
        "in_band_pct": round(100 * sum(s["in_band"] for s in stocks) / n, 1),
        "below_200w_pct": round(100 * sum(1 for s in stocks if s["pct_from_200w"] < 0) / n, 1),
        "signals_4w": sum(1 for s in stocks if s["weeks_since_signal"] is not None and s["weeks_since_signal"] <= 4),
        "median_pct_from_200w": round(float(np.median([s["pct_from_200w"] for s in stocks])), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universes", default="sp500,nasdaq100,sp400,ftse100,ftse250,watchlist,markets")
    ap.add_argument("--mode", choices=["touch", "reclaim"], default="touch")
    ap.add_argument("--band", type=float, default=3.0)
    ap.add_argument("--regime-min", type=int, default=2)
    ap.add_argument("--years", type=int, default=15)
    ap.add_argument("--push", action="store_true", help="Telegram on new signals")
    ap.add_argument("--always-push", action="store_true", help="Telegram even with no new signals (weekly summary)")
    args = ap.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    hist_path = os.path.join(DATA_DIR, "history.json")
    history = json.load(open(hist_path)) if os.path.exists(hist_path) else {"runs": [], "live_signals": {}}

    now = dt.datetime.now(dt.timezone.utc)
    all_stocks, all_signals, per_universe, exchanges = [], [], {}, {}
    for u in [x.strip() for x in args.universes.split(",") if x.strip()]:
        stocks, signals = scan_universe(u, args, exchanges)
        if not stocks:
            print(f"[{u}] WARNING: nothing scanned, skipping", file=sys.stderr)
            continue
        all_stocks += stocks
        all_signals += signals
        per_universe[u] = {"label": UNIVERSE_LABEL.get(u, u), "breadth": breadth(stocks)}

    if not all_stocks:
        print("ERROR: no data at all", file=sys.stderr)
        sys.exit(2)

    # --- live signal log: anything signalling now that we haven't logged before
    new_signals = []
    for s in all_stocks:
        if s["signal_now"]:
            key = f"{s['symbol']}|{s['week']}|{args.mode}"
            if key not in history["live_signals"]:
                history["live_signals"][key] = {
                    "symbol": s["symbol"], "universe": s["universe"], "mode": args.mode,
                    "week": s["week"], "price": s["close"], "ema200w": s["ema200w"],
                    "first_seen": now.strftime("%Y-%m-%d"),
                }
                new_signals.append(s)

    history["runs"].append({
        "ts": now.strftime("%Y-%m-%dT%H:%MZ"),
        "universes": {u: v["breadth"] for u, v in per_universe.items()},
    })
    history["runs"] = history["runs"][-400:]

    # align every sparkline on one shared week axis (ending at the latest week seen)
    end = max(pd.Timestamp(s["week"]) for s in all_stocks)
    weeks = [d.strftime("%Y-%m-%d") for d in pd.date_range(end=end, periods=SPARK_WEEKS, freq="W-FRI")]
    for s in all_stocks:
        sp = s.pop("spark")
        s["c"] = [sp.get(w, (None,))[0] for w in weeks]
        s["e"] = [sp.get(w, (None, None))[1] for w in weeks]
        s["sg"] = [i for i, w in enumerate(weeks) if sp.get(w, (0, 0, 0))[2]]

    latest = {
        "generated": now.strftime("%Y-%m-%dT%H:%MZ"),
        "weeks": weeks,
        "params": {"mode": args.mode, "band_pct": args.band, "regime_min": args.regime_min,
                   "len_w": 200, "len_d": 200},
        "universes": per_universe,
        "stocks": all_stocks,
        "signals": sorted(all_signals, key=lambda r: r["date"], reverse=True),
    }
    with open(os.path.join(DATA_DIR, "latest.json"), "w") as f:
        json.dump(latest, f, separators=(",", ":"))
    with open(hist_path, "w") as f:
        json.dump(history, f, separators=(",", ":"))
    print(f"wrote {len(all_stocks)} stocks, {len(all_signals)} historic signals, "
          f"{len(new_signals)} new live signals", file=sys.stderr)

    # --- Telegram
    if args.push and (new_signals or args.always_push):
        page = os.environ.get("DASHBOARD_URL", "")
        lines = [f"<b>200W EMA scan</b> — {now.strftime('%a %d %b')}"]
        if new_signals:
            lines.append(f"\n<b>New signals ({len(new_signals)})</b>")
            for s in sorted(new_signals, key=lambda x: x["pct_from_200w"]):
                lines.append(f"<code>{s['symbol']:<8}</code> {UNIVERSE_LABEL.get(s['universe'], s['universe'])}  "
                             f"{s['pct_from_200w']:+.1f}% from 200W")
        else:
            lines.append("\nNo new signals this week.")
        inband = [s for s in all_stocks if s["regime_on"] and s["in_band"] and not s["signal_now"]]
        if inband:
            lines.append("\n<i>In band, waiting:</i> " + ", ".join(s["symbol"] for s in sorted(inband, key=lambda x: x["pct_from_200w"])[:20]))
        for u, v in per_universe.items():
            b = v["breadth"]
            lines.append(f"\n{v['label']}: {b['regime_on_pct']:.0f}% regime on · {b['in_band_pct']:.0f}% in band · {b['below_200w_pct']:.0f}% below 200W")
        if page:
            lines.append(f"\n{page}")
        push("\n".join(lines))


if __name__ == "__main__":
    main()
