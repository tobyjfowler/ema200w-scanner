#!/usr/bin/env python3
"""
Does EMA slope add anything to the 200W pullback signal?

For every clustered historic signal (touch & hold and reclaim, last ~8 years),
measure at the signal bar:
  slope_w   = % change of the 200W EMA over the previous 13 weeks
  slope_d   = % change of the 200D EMA over the previous 13 weeks (weekly-sampled)
  accel_d   = slope_d now minus slope_d 13 weeks earlier (is the 200D steepening or rolling over?)
  off_high  = % below the 52-week high at the signal (how hard did it fall to get here?)
  spread    = (200D / 200W - 1) % — how far above the 200W the 200D sits

then bucket each measure into quintiles and report forward returns at 4 / 12 / 26 / 52
weeks, win rate and max drawdown per bucket. Writes research/slope_report.md.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ema200w_scanner import evaluate, fetch_daily, load_universe  # noqa: E402

LOOKBACK_W = 420      # ~8 years of signals
CLUSTER_GAP = 8
FWD = (4, 12, 26, 52)
SLOPE_N = 13


def collect(universes=("sp500", "nasdaq100", "ftse100"), years=20) -> pd.DataFrame:
    rows = []
    for u in universes:
        try:
            tickers = load_universe(u)
        except Exception as e:
            print(f"[{u}] skip: {e}", file=sys.stderr)
            continue
        data = fetch_daily(tickers, years)
        print(f"[{u}] {len(data)} symbols", file=sys.stderr)
        for sym, daily in data.items():
            for mode in ("touch", "reclaim"):
                wk = evaluate(daily, mode=mode)
                if wk is None:
                    continue
                emaW, emaD, close = wk["emaW"].to_numpy(), wk["emaD"].to_numpy(), wk["Close"].to_numpy()
                high = wk["High"].to_numpy()
                sig = wk["signal"].to_numpy()
                n = len(wk)
                for i in np.where(sig == 1)[0]:
                    if i < max(n - LOOKBACK_W, 2 * SLOPE_N + 52) or sig[max(0, i - CLUSTER_GAP):i].any():
                        continue
                    if i + FWD[-1] >= n:        # need a full year of outcome
                        continue
                    slope_d_now = (emaD[i] / emaD[i - SLOPE_N] - 1) * 100
                    slope_d_prev = (emaD[i - SLOPE_N] / emaD[i - 2 * SLOPE_N] - 1) * 100
                    rec = dict(
                        symbol=sym, universe=u, mode=mode, date=wk.index[i].strftime("%Y-%m-%d"),
                        slope_w=(emaW[i] / emaW[i - SLOPE_N] - 1) * 100,
                        slope_d=slope_d_now,
                        accel_d=slope_d_now - slope_d_prev,
                        off_high=(close[i] / high[i - 52:i + 1].max() - 1) * 100,
                        spread=(emaD[i] / emaW[i] - 1) * 100,
                        mdd_26=(close[i:i + 27].min() / close[i] - 1) * 100,
                    )
                    for h in FWD:
                        rec[f"fwd_{h}"] = (close[i + h] / close[i] - 1) * 100
                    rows.append(rec)
    return pd.DataFrame(rows)


def table(df: pd.DataFrame, col: str, q: int = 5) -> pd.DataFrame:
    d = df.copy()
    try:
        d["bucket"] = pd.qcut(d[col], q, duplicates="drop")
    except ValueError:
        return pd.DataFrame()
    g = d.groupby("bucket", observed=True)
    out = pd.DataFrame({
        "n": g.size(),
        **{f"avg +{h}w": g[f"fwd_{h}"].mean().round(1) for h in FWD},
        "win% +26w": (g["fwd_26"].apply(lambda s: 100 * (s > 0).mean())).round(0),
        "median +26w": g["fwd_26"].median().round(1),
        "avg maxDD 26w": g["mdd_26"].mean().round(1),
    })
    out.index = [f"{iv.left:.1f} to {iv.right:.1f}" for iv in out.index]
    return out


def md(df: pd.DataFrame) -> str:
    return df.to_markdown() if len(df) else "_not enough data_"


def main():
    df = collect()
    os.makedirs("research", exist_ok=True)
    df.to_csv("research/slope_signals.csv", index=False)

    lines = [f"# EMA slope study\n", f"{len(df)} clustered signals, {df['date'].min()} to {df['date'].max()}, "
             f"universes: {', '.join(sorted(df['universe'].unique()))}. Forward returns are simple price "
             "returns from the signal week's close; survivorship bias applies (today's index members only).\n"]
    base = {h: df[f"fwd_{h}"].mean() for h in FWD}
    lines.append("## Baseline (all signals)\n")
    lines.append(f"| mode | n | avg +4w | avg +12w | avg +26w | avg +52w | win% +26w | avg maxDD |\n|---|---|---|---|---|---|---|---|")
    for mode, g in df.groupby("mode"):
        lines.append(f"| {mode} | {len(g)} | {g.fwd_4.mean():+.1f}% | {g.fwd_12.mean():+.1f}% | {g.fwd_26.mean():+.1f}% | "
                     f"{g.fwd_52.mean():+.1f}% | {100 * (g.fwd_26 > 0).mean():.0f}% | {g.mdd_26.mean():+.1f}% |")
    lines.append(f"| all | {len(df)} | {base[4]:+.1f}% | {base[12]:+.1f}% | {base[26]:+.1f}% | {base[52]:+.1f}% | "
                 f"{100 * (df.fwd_26 > 0).mean():.0f}% | {df.mdd_26.mean():+.1f}% |\n")

    measures = [
        ("slope_w", "200W EMA slope — % change over prior 13 weeks"),
        ("slope_d", "200D EMA slope — % change over prior 13 weeks"),
        ("accel_d", "200D slope acceleration — slope now minus slope 13w earlier (negative = rolling over)"),
        ("off_high", "% below 52-week high at signal (how far it fell to reach the 200W)"),
        ("spread", "200D above 200W, % (how stretched the regime is)"),
    ]
    for col, title in measures:
        lines.append(f"## {title}\n")
        lines.append("Quintiles, lowest to highest. Touch & hold:\n")
        lines.append(md(table(df[df["mode"] == "touch"], col)) + "\n")
        lines.append("Reclaim:\n")
        lines.append(md(table(df[df["mode"] == "reclaim"], col)) + "\n")

    # a couple of concrete filter candidates
    lines.append("## Candidate filters (touch & hold)\n")
    t = df[df["mode"] == "touch"]
    cands = {
        "baseline": t,
        "200D slope > 0": t[t.slope_d > 0],
        "200D slope > 0 and not rolling over (accel ≥ 0)": t[(t.slope_d > 0) & (t.accel_d >= 0)],
        "200W slope > 2%/13w": t[t.slope_w > 2],
        "fell < 20% from 52w high": t[t.off_high > -20],
        "fell > 30% from 52w high": t[t.off_high < -30],
        "200D slope > 0 and fell < 20%": t[(t.slope_d > 0) & (t.off_high > -20)],
        "200D slope < 0 (rolling over)": t[t.slope_d < 0],
    }
    lines.append("| filter | n | kept % | avg +12w | avg +26w | avg +52w | win% +26w | avg maxDD |\n|---|---|---|---|---|---|---|---|")
    for name, g in cands.items():
        if not len(g):
            continue
        lines.append(f"| {name} | {len(g)} | {100 * len(g) / len(t):.0f}% | {g.fwd_12.mean():+.1f}% | {g.fwd_26.mean():+.1f}% | "
                     f"{g.fwd_52.mean():+.1f}% | {100 * (g.fwd_26 > 0).mean():.0f}% | {g.mdd_26.mean():+.1f}% |")
    lines.append("\n## By year (touch & hold, all signals) — sanity check that results aren't one regime\n")
    t2 = t.assign(year=t.date.str[:4]).groupby("year")
    lines.append("| year | n | avg +26w | win% +26w |\n|---|---|---|---|")
    for y, g in t2:
        lines.append(f"| {y} | {len(g)} | {g.fwd_26.mean():+.1f}% | {100 * (g.fwd_26 > 0).mean():.0f}% |")

    open("research/slope_report.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
