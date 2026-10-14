# 200W EMA pullback scanner

Weekly stock scan that finds names pulling back to their 200-week EMA while the
200-day EMA is above it (uptrend regime). A Python port of the TradingView
"200W EMA Pullback — Screener" indicator.

## How it runs

- **Automatically**: every weekday evening (UK) via GitHub Actions
  (`.github/workflows/scan.yml`). `build_dashboard.py` scans the S&P 500,
  Nasdaq 100 and FTSE 100, writes `data/latest.json` + `data/history.json`,
  commits them, and publishes the dashboard (`index.html`) to GitHub Pages.
- **Telegram**: a message on any day with new signals, plus a summary every Friday.
- **On demand**: Actions tab → *200W EMA scan* → *Run workflow*.
- **Dashboard**: https://tobyjfowler.github.io/ema200w-scanner/

`ema200w_scanner.py` is the original single-universe CLI and still works on its own.

## Setup (one-off)

1. Repo → Settings → Pages → *Source: GitHub Actions*.
2. Repo → Settings → Secrets and variables → Actions → add:

| Secret | What |
|---|---|
| `TELEGRAM_BOT_TOKEN` | Bot token from @BotFather |
| `TELEGRAM_CHAT_ID` | Your chat ID with that bot |

## Signal logic

Same as the Pine indicator, defaults in brackets:

- `emaW` = 200-week EMA of weekly closes (`--len-w`)
- `emaD` = 200-day EMA of daily closes, sampled at week end (`--len-d`)
- Band = `emaW` ± 3% (`--band`)
- Regime = `emaD > emaW`, on for ≥ 2 weeks (`--regime-min`, `--no-regime` to disable)
- **Touch & hold** (`--mode touch`): weekly low enters the band and the week closes above `emaW`
- **Reclaim** (`--mode reclaim`): last week closed below `emaW`, this week closes above

Data: Yahoo Finance via `yfinance`, split-adjusted but not dividend-adjusted
closes (what TradingView shows by default). EMA is seeded with an SMA like
Pine's `ta.ema`.

## Running locally

```
pip install -r requirements.txt
python ema200w_scanner.py --universe sp500 --in-band
python ema200w_scanner.py --tickers-file mylist.txt --recent 8
```
