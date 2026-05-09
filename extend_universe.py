"""
extend_universe.py
──────────────────
Always rebuilds from the original 39 VN stock CSVs, then adds 4 simulated
diversifiers (VN govt bond daily price data is not publicly available):

  GLD  — SPDR Gold ETF → simulates world gold / SJC gold in VND (HIGH similarity;
          SJC carries a domestic premium vs. world price)
  AGG  — iShares US Aggregate Bond ETF → simulates VN fixed-income behavior
          (MEDIUM similarity; comparable yield & low equity corr in normal years,
          diverged in 2022 when Fed hiked while SBV cut rates)
  EMB  — iShares USD EM Bond ETF → simulates VN government bond exposure
          (MEDIUM similarity; literally holds VN govt bonds ~4% of fund,
          but diluted across 30+ EM countries)
  VNM  — VanEck Vietnam ETF in USD → Vietnam equity via foreign investor channel
          (HIGH similarity; same VN stocks but adds USD/VND FX dimension)

All USD assets converted to VND via USDVND=X.

Run: python3 extend_universe.py
"""

import json, numpy as np, pandas as pd, yfinance as yf
from pathlib import Path
from datetime import datetime

HALFLIFE   = 252
DATA_DIR   = Path("portfolio_data/processed")
RAW_DIR    = Path("portfolio_data/raw/prices")
FX_TICKER  = "USDVND=X"
DATE_START = "2018-01-01"
DATE_END   = "2026-04-30"

VN_TICKERS = [
    "ANV","BIC","BID","BMI","CII","CMG","CSV","CTD","DBC","DGW",
    "DPM","DXG","ELC","FPT","GAS","GIL","HAH","HAX","HCM","HPG",
    "IMP","KDC","KDH","LCG","LSS","MBB","NKG","PC1","PNJ","PVT",
    "REE","SAB","SSI","STK","TCM","VCB","VCI","VIC","VSC",
]
NEW_TICKERS = {
    "GLD": "SPDR Gold ETF (simulates world gold / SJC gold in VND)",
    "AGG": "iShares US Aggregate Bond ETF (simulates VN fixed-income behavior)",
    "EMB": "iShares USD EM Bond ETF (simulates VN govt bond exposure; holds ~4% VN bonds)",
    "VNM": "VanEck Vietnam ETF (Vietnam equity via foreign investor / FX channel)",
}

print("=" * 60)
print("  Building universe: 39 VN stocks + GLD,AGG,EMB,VNM = 43")
print("=" * 60)

# ── Step 1: Load VN trading dates ─────────────────────────────────────────
print("\n[1] Loading VN trading calendar from dates.npy …")
existing_dates  = np.load(DATA_DIR / "dates.npy", allow_pickle=True)
vn_dates        = pd.to_datetime(existing_dates)
T               = len(vn_dates)
with open(DATA_DIR / "meta.json") as f:
    meta_orig = json.load(f)
print(f"   {T} trading days: {vn_dates[0].date()} → {vn_dates[-1].date()}")

# ── Step 2: Rebuild 39-stock returns from raw CSVs ────────────────────────
print("\n[2] Rebuilding 39-stock returns from raw price CSVs …")
vn_prices = []
for t in VN_TICKERS:
    s = (pd.read_csv(RAW_DIR / f"{t}.csv", parse_dates=["Date"])
           .set_index("Date")["Close"]
           .reindex(vn_dates, method="ffill", limit=5)
           .bfill(limit=5))
    vn_prices.append(s.values)

P_vn = np.column_stack(vn_prices)                      # [T × 39]
R_vn = np.log(P_vn[1:] / P_vn[:-1])                  # [T-1 × 39]
# Pad first row with 0 so we keep T rows total
R_vn = np.vstack([np.zeros((1, 39)), R_vn])           # [T × 39]
print(f"   VN returns matrix: {R_vn.shape}")

# ── Step 3: Download FX rate ──────────────────────────────────────────────
print("\n[3] Downloading USD/VND exchange rate …")
fx = (yf.download(FX_TICKER, start=DATE_START, end=DATE_END,
                  progress=False, auto_adjust=True)["Close"]
        .squeeze())
fx.index = pd.to_datetime(fx.index)
fx = fx.reindex(vn_dates, method="ffill", limit=5).bfill(limit=5)
print(f"   FX: {len(fx)} rows, missing: {int(fx.isna().sum())}")

# ── Step 4: Download new assets and compute returns in VND ───────────────
print("\n[4] Downloading GLD, EMB, VNM and converting to VND …")
new_ret_cols = []
for ticker in NEW_TICKERS:
    raw = (yf.download(ticker, start=DATE_START, end=DATE_END,
                       progress=False, auto_adjust=True)["Close"]
             .squeeze())
    raw.index = pd.to_datetime(raw.index)
    p_vnd = (raw.reindex(vn_dates, method="ffill", limit=5)
                .bfill(limit=5) * fx)
    r = np.log(p_vnd / p_vnd.shift(1)).reindex(vn_dates).fillna(0).values
    new_ret_cols.append(r)
    vol  = np.std(r) * np.sqrt(252) * 100
    corr = float(np.corrcoef(r, R_vn[:, :39].mean(axis=1))[0, 1])
    print(f"   {ticker}: vol={vol:.1f}%, corr_w_VN={corr:.3f}")

R_new = np.column_stack(new_ret_cols)                 # [T × 3]

# ── Step 5: Combined return matrix ────────────────────────────────────────
R = np.hstack([R_vn, R_new])                          # [T × 42]
all_tickers = VN_TICKERS + list(NEW_TICKERS.keys())
N = len(all_tickers)
print(f"\n[5] Combined matrix: {R.shape}  ({T} days × {N} assets)")

# ── Step 6: EWMA μ and Σ ─────────────────────────────────────────────────
print(f"\n[6] EWMA μ and Σ (half-life={HALFLIFE}) …")
decay   = np.exp(-np.log(2) / HALFLIFE)
raw_w   = np.array([decay ** (T - 1 - t) for t in range(T)])
w       = raw_w / raw_w.sum()

mu_daily    = w @ R
R_dm        = R - mu_daily
sigma_daily = (R_dm * w[:, None]).T @ R_dm
mu_annual   = mu_daily * 252
sigma_annual = (sigma_daily * 252)
sigma_annual = (sigma_annual + sigma_annual.T) / 2   # ensure symmetry

print(f"   μ: {mu_annual.min()*100:.1f}% → {mu_annual.max()*100:.1f}%")
print(f"   σ diagonal: {np.sqrt(np.diag(sigma_annual)).min()*100:.1f}%"
      f" → {np.sqrt(np.diag(sigma_annual)).max()*100:.1f}%")

# ── Step 7: Eigendecomposition + Marchenko-Pastur ────────────────────────
print("\n[7] Eigendecomposition …")
ev, evc = np.linalg.eigh(sigma_annual)
idx     = np.argsort(ev)[::-1]
ev, evc = ev[idx], evc[:, idx]

avg_var   = float(np.mean(np.diag(sigma_annual)))
mp_thresh = avg_var * (1 + np.sqrt(N / T)) ** 2
n_signal  = int(np.sum(ev > mp_thresh))
print(f"   M-P threshold: {mp_thresh:.4f}")
print(f"   Signal factors: {n_signal} / {N}")

# ── Step 8: Save processed files ─────────────────────────────────────────
print("\n[8] Saving processed/ artifacts …")
np.save(DATA_DIR / "mu.npy",           mu_annual.astype(np.float64))
np.save(DATA_DIR / "sigma.npy",        sigma_annual.astype(np.float64))
np.save(DATA_DIR / "eigenvalues.npy",  ev.astype(np.float64))
np.save(DATA_DIR / "eigenvectors.npy", evc.astype(np.float64))
np.save(DATA_DIR / "returns.npy",      R.astype(np.float64))
# dates.npy unchanged

meta_new = {
    **meta_orig,
    "tickers":          all_tickers,
    "n_tickers":        N,
    "t_days":           T,
    "t_n_ratio":        round(T / N, 2),
    "n_signal_factors": n_signal,
    "n_noise_factors":  N - n_signal,
    "last_updated":     datetime.now().isoformat(timespec="seconds"),
    "extension_note":   "Added 4 simulated diversifiers (VN govt bond daily data unavailable): "
                        "GLD (world gold proxy, high similarity to SJC), "
                        "AGG (simulates VN fixed-income behavior, medium similarity), "
                        "EMB (simulates VN govt bond exposure via EM bond fund, medium similarity), "
                        "VNM (Vietnam equity via foreign investor/FX channel, high similarity) "
                        "— all USD assets converted to VND via USDVND=X",
}
with open(DATA_DIR / "meta.json", "w") as f:
    json.dump(meta_new, f, indent=2)

# Save raw CSVs for the 3 new assets
RAW_DIR.mkdir(parents=True, exist_ok=True)
for ticker in NEW_TICKERS:
    raw = (yf.download(ticker, start=DATE_START, end=DATE_END,
                       progress=False, auto_adjust=True)["Close"].squeeze())
    raw.index = pd.to_datetime(raw.index)
    p_vnd = (raw.reindex(vn_dates, method="ffill", limit=5)
                .bfill(limit=5) * fx)
    pd.DataFrame({"Date": vn_dates, "Close": p_vnd.values}).to_csv(
        RAW_DIR / f"{ticker}.csv", index=False)
    print(f"   Saved {ticker}.csv")

print(f"\n{'='*60}")
print(f"  Done.  {N} assets  |  {n_signal} signal factors  |  T/N = {T/N:.1f}")
print(f"  Restart Streamlit to load new data.")
print(f"{'='*60}")
