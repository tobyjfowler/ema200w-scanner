#!/usr/bin/env python3
"""
Daily-timeframe EMA history for one instrument (default 200D).

  python research/ema_daily_history.py 0P0000TKZO.L "Vanguard LifeStrategy 100 Acc" [200] [3.0]

Writes research/history_<symbol>_<len>D.md with:
  - summary (time above / in band / below)
  - every episode where the close entered the ±band or went below the EMA:
    start, length, deepest close, whether it closed below, and returns 1/3/6/12 months
    after the first touch
  - every reclaim (first close back above the EMA after being below) with forward returns
  - closest approach to the EMA in each calendar year
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ema200w_scanner import fetch_daily, pine_ema  # noqa: E402

FWD = ((21, "1m"), (63, "3m"), (126, "6m"), (252, "12m"))


def f(v, d=1):
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:+.{d}f}%"


def fwd(c, i, h):
    return (c[i + h] / c[i] - 1) * 100 if i + h < len(c) else None


def main():
    sym = sys.argv[1]
    name = sys.argv[2] if len(sys.argv) > 2 else sym
    length = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    band = float(sys.argv[4]) if len(sys.argv) > 4 else 3.0

    data = fetch_daily([sym], 30)
    if sym not in data:
        sys.exit(f"no data for {sym}")
    d = data[sym].copy()
    d["ema"] = pine_ema(d["Close"], length)
    d = d[d["ema"].notna()].copy()
    d["pct"] = (d["Close"] / d["ema"] - 1) * 100
    d["in_band"] = d["pct"].abs() <= band
    d["below"] = d["Close"] < d["ema"]
    d["near"] = d["pct"] <= band          # at or below the band top
    c = d["Close"].to_numpy()
    pct = d["pct"].to_numpy()
    near = d["near"].to_numpy()
    below = d["below"].to_numpy()
    n = len(d)

    slope = (d["ema"].iloc[-1] / d["ema"].iloc[-64] - 1) * 100 if n > 64 else None
    L = [f"# {name} ({sym}) — {length}D EMA history\n",
         f"Daily closes {d.index[0].date()} to {d.index[-1].date()} ({n} trading days with an EMA). Band ±{band:g}%.\n",
         "## Summary\n",
         f"- Days closing above the {length}D: **{100 * (~d['below']).mean():.0f}%** · inside the band: {100 * d['in_band'].mean():.0f}% · below: {100 * d['below'].mean():.0f}%",
         f"- Now: close {c[-1]:.2f}, {length}D {d['ema'].iloc[-1]:.2f} ({f(pct[-1])}), EMA slope over the last 3 months {f(slope)}\n"]

    # ---- episodes: contiguous runs (allowing gaps of up to 5 trading days) where close <= band top
    L.append(f"## Episodes at or below the band\n")
    L.append(f"Contiguous runs of days where the close was within {band:g}% of, or below, the {length}D EMA. Returns are from the close on the first day.\n")
    L.append("| # | first touch | trading days | deepest close vs EMA | days closed below | 1m | 3m | 6m | 12m | ended |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    i, ep = 0, 0
    while i < n:
        if not near[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and near[j + 1:j + 7].any():
            j += 1
        seg = slice(i, j + 1)
        ep += 1
        L.append(f"| {ep} | {d.index[i].date()} | {j - i + 1} | {f(pct[seg].min())} | {int(below[seg].sum())} | "
                 f"{f(fwd(c, i, 21))} | {f(fwd(c, i, 63))} | {f(fwd(c, i, 126))} | {f(fwd(c, i, 252))} | "
                 f"{d.index[j].date() if j < n - 1 else 'ongoing'} |")
        i = j + 1
    if ep == 0:
        L.append("| – | never reached the band | | | | | | | | |")

    # ---- reclaims
    L.append(f"\n## Reclaims (first close back above the {length}D after at least 5 days below)\n")
    L.append("| date | close | EMA | 1m | 3m | 6m | 12m | max DD next 6m |")
    L.append("|---|---|---|---|---|---|---|---|")
    cnt = 0
    for k in range(6, n):
        if not below[k] and below[k - 5:k].all():
            cnt += 1
            mdd = (c[k:min(k + 127, n)].min() / c[k] - 1) * 100
            L.append(f"| {d.index[k].date()} | {c[k]:.2f} | {d['ema'].iloc[k]:.2f} | {f(fwd(c, k, 21))} | {f(fwd(c, k, 63))} | {f(fwd(c, k, 126))} | {f(fwd(c, k, 252))} | {f(mdd)} |")
    if cnt == 0:
        L.append("| – | none | | | | | | |")

    # ---- closest approach per year
    L.append("\n## Closest approach each year\n")
    L.append(f"| year | date | close | {length}D | % from EMA | days in band | days below | year-end % from EMA |")
    L.append("|---|---|---|---|---|---|---|---|")
    for y, g in d.groupby(d.index.year):
        k = g["pct"].idxmin()
        L.append(f"| {y} | {k.date()} | {g.loc[k, 'Close']:.2f} | {g.loc[k, 'ema']:.2f} | {f(g.loc[k, 'pct'])} | "
                 f"{int(g['in_band'].sum())} | {int(g['below'].sum())} | {f(g['pct'].iloc[-1])} |")

    out = f"research/history_{sym.replace('=', '').replace('^', '').replace('.', '_')}_{length}D.md"
    os.makedirs("research", exist_ok=True)
    open(out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
