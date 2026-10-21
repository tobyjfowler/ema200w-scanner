#!/usr/bin/env python3
"""
Fund tab data: daily price with 200D / 100W / 200W EMAs, level ladder, touch
history and alerts for a small list of funds. Writes data/funds.json and sends a
Telegram line when a level alert first fires.

Usage:  python build_fund.py [--push]
"""
import argparse
import datetime as dt
import json
import os
import sys

import numpy as np
import pandas as pd

from ema200w_scanner import fetch_daily, pine_ema, push

FUNDS = [
    # yahoo symbol, display name, short label
    ("0P0000TKZO.L", "Vanguard LifeStrategy 100% Equity (Acc)", "LS100"),
]
BAND = 3.0                 # % — "touch" = within this of a level, or below it
LEVELS = (("ema200d", "200D"), ("ema100w", "100W"), ("ema200w", "200W"))
DATA_DIR = "data"
OUT = os.path.join(DATA_DIR, "funds.json")


def r2(x, d=2):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), d)


def fwd(c, i, h):
    return r2((c[i + h] / c[i] - 1) * 100, 1) if i + h < len(c) else None


def episodes(d: pd.DataFrame, col: str, gap_days: int = 5) -> list[dict]:
    """Contiguous runs where close <= level*(1+band), allowing short gaps. Daily index."""
    pct = (d["Close"] / d[col] - 1) * 100
    near = (pct <= BAND).to_numpy()
    below = (pct < 0).to_numpy()
    c = d["Close"].to_numpy()
    out, i, n = [], 0, len(d)
    while i < n:
        if not near[i] or np.isnan(pct.iloc[i]):
            i += 1
            continue
        j = i
        while j + 1 < n and near[j + 1:j + 2 + gap_days].any():
            j += 1
        out.append({
            "start": d.index[i].strftime("%Y-%m-%d"), "end": d.index[j].strftime("%Y-%m-%d") if j < n - 1 else None,
            "days": int(j - i + 1), "deepest": r2(pct.iloc[i:j + 1].min(), 1), "days_below": int(below[i:j + 1].sum()),
            "f1m": fwd(c, i, 21), "f3m": fwd(c, i, 63), "f6m": fwd(c, i, 126), "f12m": fwd(c, i, 252),
        })
        i = j + 1
    return out


def reclaims(d: pd.DataFrame, col: str, min_below: int = 5) -> list[dict]:
    below = (d["Close"] < d[col]).to_numpy()
    c = d["Close"].to_numpy()
    out = []
    for k in range(min_below + 1, len(d)):
        if not below[k] and below[k - min_below:k].all() and not np.isnan(d[col].iloc[k]):
            mdd = r2((c[k:min(k + 127, len(c))].min() / c[k] - 1) * 100, 1)
            out.append({"date": d.index[k].strftime("%Y-%m-%d"), "close": r2(c[k]), "level": r2(d[col].iloc[k]),
                        "f1m": fwd(c, k, 21), "f3m": fwd(c, k, 63), "f6m": fwd(c, k, 126), "f12m": fwd(c, k, 252), "mdd6m": mdd})
    return out


def build_one(sym: str, name: str, label: str, daily: pd.DataFrame) -> dict:
    d = daily[["Close"]].copy()
    d["ema200d"] = pine_ema(d["Close"], 200)
    # weekly closes, labelled by the last trading day actually in each week (so the
    # in-progress week's value includes today's close, as on a live weekly chart)
    grp = d["Close"].groupby(d.index.to_period("W-FRI"))
    wk = pd.Series(grp.last().to_numpy(), index=grp.apply(lambda g: g.index[-1]).to_numpy())
    e100 = pine_ema(wk, 100)
    e200 = pine_ema(wk, 200)
    d["ema100w"] = e100.reindex(d.index, method="ffill")
    d["ema200w"] = e200.reindex(d.index, method="ffill")
    c = d["Close"].to_numpy()
    last = d.iloc[-1]

    def level(col, short):
        v = last[col]
        if pd.isna(v):
            return None
        pct = (last["Close"] / v - 1) * 100
        hist = (d["Close"] / d[col] - 1) * 100
        hist = hist[d[col].notna()]
        eps = episodes(d[d[col].notna()], col)
        last_touch = eps[-1]["start"] if eps else None
        # slope over ~3 months (63 trading days)
        s = d[col].dropna()
        slope = (s.iloc[-1] / s.iloc[-64] - 1) * 100 if len(s) > 64 else None
        return {
            "key": col, "label": short, "value": r2(v), "pct": r2(pct, 1), "fall_to": r2(-pct / (1 + pct / 100), 1),
            "slope_3m": r2(slope, 1), "last_touch": last_touch,
            "weeks_since_touch": None if not eps else int((d.index[-1] - pd.Timestamp(eps[-1]["start"])).days // 7),
            "pct_percentile": r2(100 * (hist < pct).mean(), 0),
            "days_total": int(len(hist)), "days_in_band": int((hist.abs() <= BAND).sum()), "days_below": int((hist < 0).sum()),
            "closest_by_year": {str(y): r2(g.min(), 1) for y, g in hist.groupby(hist.index.year)},
            "episodes": eps[::-1][:40],
            "reclaims": reclaims(d[d[col].notna()], col)[::-1][:30],
        }

    levels = [lv for lv in (level(col, short) for col, short in LEVELS) if lv]
    # alerts: a condition that is true now
    alerts = []
    for lv in levels:
        if lv["pct"] is not None and lv["pct"] <= BAND:
            alerts.append({"key": f"touch_{lv['key']}", "text": f"{label} at its {lv['label']} ({lv['pct']:+.1f}%)"})
    rc = [r for r in (levels[0]["reclaims"] if levels else []) if r["date"] == d.index[-1].strftime("%Y-%m-%d")]
    if rc:
        alerts.append({"key": "reclaim_ema200d", "text": f"{label} reclaimed its 200D after a break"})

    # series for the chart: daily, rounded. ~3000 points × 4 values is fine.
    ser = d.dropna(subset=["ema200d"])
    series = {
        "dates": [t.strftime("%Y-%m-%d") for t in ser.index],
        "close": [r2(v) for v in ser["Close"]],
        "ema200d": [r2(v) for v in ser["ema200d"]],
        "ema100w": [r2(v) for v in ser["ema100w"]],
        "ema200w": [r2(v) for v in ser["ema200w"]],
    }
    state = "above all three" if all(lv["pct"] > 0 for lv in levels) else \
        "below the " + ", ".join(lv["label"] for lv in levels if lv["pct"] <= 0)
    return {
        "symbol": sym, "name": name, "label": label, "tv": "",
        "date": d.index[-1].strftime("%Y-%m-%d"), "close": r2(last["Close"]),
        "ret_1m": fwd(c, len(c) - 22, 21), "ret_3m": fwd(c, len(c) - 64, 63), "ret_12m": fwd(c, len(c) - 253, 252),
        "state": state, "levels": levels, "alerts": alerts, "series": series,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()
    os.makedirs(DATA_DIR, exist_ok=True)
    prev = json.load(open(OUT)) if os.path.exists(OUT) else {"funds": [], "fired": {}}
    fired = prev.get("fired", {})

    data = fetch_daily([f[0] for f in FUNDS], 30)
    funds, new_alerts = [], []
    for sym, name, label in FUNDS:
        if sym not in data or len(data[sym]) < 1100:
            print(f"[fund] {sym}: no/insufficient data", file=sys.stderr)
            continue
        f = build_one(sym, name, label, data[sym])
        funds.append(f)
        for a in f["alerts"]:
            k = f"{sym}|{a['key']}"
            if k not in fired:                      # fire once per episode
                fired[k] = f["date"]
                new_alerts.append(a["text"])
        # clear fired flags for conditions no longer true, so they can fire again next time
        active = {f"{sym}|{a['key']}" for a in f["alerts"]}
        for k in [k for k in fired if k.startswith(sym + "|") and k not in active]:
            del fired[k]
        print(f"[fund] {label}: {f['close']} · " + " · ".join(f"{lv['label']} {lv['value']} ({lv['pct']:+.1f}%)" for lv in f["levels"]), file=sys.stderr)

    out = {"generated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ"), "band_pct": BAND, "funds": funds, "fired": fired}
    with open(OUT, "w") as fh:
        json.dump(out, fh, separators=(",", ":"))
    print(f"wrote {len(funds)} funds, {len(new_alerts)} new alerts", file=sys.stderr)

    if args.push and new_alerts:
        page = os.environ.get("DASHBOARD_URL", "")
        push("<b>Vanguard levels</b>\n" + "\n".join("• " + a for a in new_alerts) + (f"\n{page}" if page else ""))


if __name__ == "__main__":
    main()
