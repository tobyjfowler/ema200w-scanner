#!/usr/bin/env python3
"""
Full 200W EMA history for one instrument.

  python research/instrument_history.py 0P0000TKZO.L "Vanguard LifeStrategy 100 Acc"

Writes research/history_<symbol>.md with:
  - summary stats (time above/below 200W, regime on/off, number of episodes)
  - every "episode" of interaction with the band: when price first entered the ±band
    (or closed below the 200W), the deepest point, when it left, and the return
    4 / 12 / 26 / 52 weeks after the first touch and after the exit
  - every touch & hold and reclaim signal with forward returns
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ema200w_scanner import evaluate, fetch_daily  # noqa: E402

FWD = (4, 12, 26, 52)


def fwd(closes, i, h):
    return (closes[i + h] / closes[i] - 1) * 100 if i + h < len(closes) else None


def f(v, d=1):
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:+.{d}f}%"


def main():
    sym = sys.argv[1]
    name = sys.argv[2] if len(sys.argv) > 2 else sym
    data = fetch_daily([sym], 30)
    if sym not in data:
        print(f"no data for {sym}", file=sys.stderr)
        sys.exit(2)
    daily = data[sym]
    # funds: High/Low may be missing or equal to Close
    for c in ("High", "Low", "Open"):
        if c not in daily or daily[c].isna().all():
            daily[c] = daily["Close"]
    wk_t = evaluate(daily, mode="touch")
    wk_r = evaluate(daily, mode="reclaim")
    if wk_t is None:
        print("not enough history", file=sys.stderr)
        sys.exit(2)
    wk = wk_t
    closes = wk["Close"].to_numpy()
    n = len(wk)
    valid = wk["emaW"].notna()
    first = wk.index[valid.argmax()]

    L = [f"# {name} ({sym}) — 200W EMA history\n",
         f"Daily data from {daily.index[0].date()} to {daily.index[-1].date()}; the 200W EMA is defined from "
         f"{first.date()} ({int(valid.sum())} weeks of history with an EMA). Band ±3%, regime = 200D > 200W for ≥2 weeks.\n"]

    v = wk[valid]
    above = (v["Close"] > v["emaW"]).mean() * 100
    inband = v["in_band"].mean() * 100
    reg = v["regime_on"].mean() * 100
    L.append("## Summary\n")
    L.append(f"- Weeks closing above the 200W: **{above:.0f}%** · inside the band: {inband:.0f}% · regime on: {reg:.0f}%")
    L.append(f"- Now: close {closes[-1]:.2f}, 200W {wk['emaW'].iloc[-1]:.2f} ({f(wk['pct_from_200w'].iloc[-1])}), "
             f"regime {'on' if wk['regime_on'].iloc[-1] else 'off'}, 200W slope (13w) {f((wk['emaW'].iloc[-1] / wk['emaW'].iloc[-14] - 1) * 100)}")
    L.append(f"- Touch & hold signals (clustered): {int(wk_t['signal'].sum())} raw · reclaim signals: {int(wk_r['signal'].sum())} raw\n")

    # ---- episodes: contiguous runs where the weekly low is inside/below the band top
    near = ((wk["Low"] <= wk["bandHi"]) & valid).to_numpy()
    L.append("## Episodes at or below the band\n")
    L.append("Each row is a contiguous run of weeks where the weekly low was at or below the top of the band "
             "(i.e. price was interacting with the 200W). Returns are from the first week's close.\n")
    L.append("| # | first touch | weeks | deepest close vs 200W | closed below 200W? | regime at start | +4w | +12w | +26w | +52w | left band on |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    i, ep = 0, 0
    episodes = []
    while i < n:
        if not near[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and (near[j + 1] or (j + 2 < n and near[j + 2])):   # allow 1-week gaps
            j += 1
        seg = wk.iloc[i:j + 1]
        deepest = seg["pct_from_200w"].min()
        below = (seg["Close"] < seg["emaW"]).any()
        ep += 1
        episodes.append((i, j))
        L.append(f"| {ep} | {wk.index[i].date()} | {j - i + 1} | {f(deepest)} | {'yes' if below else 'no'} | "
                 f"{'on' if wk['regime_on'].iloc[i] else 'off'} | {f(fwd(closes, i, 4))} | {f(fwd(closes, i, 12))} | "
                 f"{f(fwd(closes, i, 26))} | {f(fwd(closes, i, 52))} | {wk.index[j].date() if j < n - 1 else 'ongoing'} |")
        i = j + 1
    if not episodes:
        L.append("| – | no interaction with the band in the available history | | | | | | | | | |")

    # ---- signals
    for label, frame in (("Touch & hold", wk_t), ("Reclaim", wk_r)):
        idx = np.where(frame["signal"].to_numpy() == 1)[0]
        L.append(f"\n## {label} signals\n")
        if not len(idx):
            L.append("_none_")
            continue
        L.append("| date | close | 200W | +4w | +12w | +26w | +52w | max DD next 26w |")
        L.append("|---|---|---|---|---|---|---|---|")
        last = -99
        for k in idx:
            if k - last < 8:       # collapse clusters, keep the first
                continue
            last = k
            mdd = (closes[k:min(k + 27, n)].min() / closes[k] - 1) * 100
            L.append(f"| {frame.index[k].date()} | {closes[k]:.2f} | {frame['emaW'].iloc[k]:.2f} | {f(fwd(closes, k, 4))} | "
                     f"{f(fwd(closes, k, 12))} | {f(fwd(closes, k, 26))} | {f(fwd(closes, k, 52))} | {f(mdd)} |")

    # ---- year by year distance
    L.append("\n## Year-end snapshot\n")
    L.append("| year | close | 200W | % from 200W | regime | weeks in band that year |")
    L.append("|---|---|---|---|---|---|")
    for y, g in v.groupby(v.index.year):
        last = g.iloc[-1]
        L.append(f"| {y} | {last['Close']:.2f} | {last['emaW']:.2f} | {f(last['pct_from_200w'])} | {'on' if last['regime_on'] else 'off'} | {int(g['in_band'].sum())} |")

    out = f"research/history_{sym.replace('=', '').replace('^', '').replace('.', '_')}.md"
    os.makedirs("research", exist_ok=True)
    open(out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
