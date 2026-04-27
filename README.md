# Vietnam Stock Portfolio Optimizer

A quantitative portfolio optimization engine built from linear algebra fundamentals.
Given an investor's target annual return, the system finds the **minimum-risk portfolio**
that achieves that target — no black-box ML, every number is traceable to a closed-form formula.

---

## The Problem

Markowitz optimization is theoretically elegant but notoriously fragile in practice.
With 39 stocks and only a finite history of daily returns, the covariance matrix is
poorly conditioned and the expected return estimates are noisy — small errors in inputs
produce wildly unstable portfolio weights. This project addresses that instability directly.

---

## Mathematical Pipeline

### 1. Return Estimation
Daily log-returns are computed as `r = ln(P_t / P_{t-1})` and aggregated using an
**Exponentially Weighted Moving Average** (half-life = 252 trading days). This means
data from 2018 carries less than 0.01% of the weight of today's data — old market
regimes do not contaminate the current estimate.

### 2. Covariance Matrix & Noise Filtering
The EWMA covariance matrix is decomposed into its **eigenvalues and eigenvectors**.
The [Marchenko–Pastur theorem](https://en.wikipedia.org/wiki/Marchenko%E2%80%93Pastur_distribution)
provides a theoretical upper bound for eigenvalues that arise purely from sampling noise.
This separates genuine risk factors from statistical artifacts — a result from random matrix theory.

### 3. Stabilizing the Inputs
Two independent regularization layers prevent the optimizer from over-fitting to noise:

- **James–Stein shrinkage (μ-side):** Expected returns estimated from historical data
  are unreliable — a stock that happened to perform well is not necessarily a better
  investment. James–Stein shrinkage pulls each estimate toward the cross-sectional mean,
  reducing the outsized influence of historically lucky stocks on the final weights.

- **Tikhonov regularization (Σ-side):** Adding a small multiple of the identity matrix
  `Σ_reg = Σ + λI` improves the condition number of the covariance matrix, preventing
  the matrix inverse from amplifying noise into extreme weights.

### 4. Portfolio Construction
The optimizer solves a **constrained quadratic program**: find weights that minimize
portfolio variance subject to hitting the target return, fully investing capital,
holding long-only positions, and capping any single stock at 15%.

```
minimize    wᵀ Σ_reg w
subject to  μᵀw = r*          (target return)
            1ᵀw = 1           (fully invested)
            0 ≤ wᵢ ≤ 0.15    (long-only, position cap)
```

### 5. Concentration Risk & Reliable Zone
Portfolio concentration is measured via the **Herfindahl–Hirschman Index**
`N_eff = 1 / Σwᵢ²`, which gives the effective number of independent positions.
When `N_eff < 3`, the optimizer is being asked to concentrate heavily into a few
stocks — a sign that the target return is too ambitious for the estimation quality
of the input data. The system flags this as a **caution zone** rather than
silently returning an unreliable answer.

---

## Validation

A **walk-forward out-of-sample test** avoids lookahead bias:

| Period | Role |
|---|---|
| Jan 2018 – Dec 2023 | Training (covariance + return estimation) |
| Jan 2024 – Apr 2026 | Out-of-sample test |

The minimum-variance portfolio is benchmarked against **1/N equal-weighting**,
following the methodology of DeMiguel et al. (2009) *"Optimal Versus Naive Diversification"*.

---

## Dataset

| Parameter | Value |
|---|---|
| Universe | 39 HOSE/HNX Vietnamese equities |
| Date range | Jan 2018 – Apr 2026 |
| Observations | ~2,090 daily closes |
| Risk-free rate | 4.5% p.a. (Vietnamese government bond proxy) |
| EWMA half-life | 252 trading days |

**Tickers:** ANV, BIC, BID, BMI, CII, CMG, CSV, CTD, DBC, DGW, DPM, DXG, ELC, FPT,
GAS, GIL, HAH, HAX, HCM, HPG, IMP, KDC, KDH, LCG, LSS, MBB, NKG, PC1, PNJ, PVT,
REE, SAB, SSI, STK, TCM, VCB, VCI, VIC, VSC

---

## Repository Structure

```
LinearProject/
├── LinearCode.ipynb              Full optimization engine (documented step-by-step)
├── HANDOVER.txt                  API contract and UI spec for frontend developers
└── portfolio_data/
    ├── raw/prices/               Daily closing prices (one CSV per ticker)
    └── processed/
        ├── meta.json             Metadata: tickers, date range, parameters
        ├── mu.npy                EWMA expected annual returns  [39]
        ├── sigma.npy             EWMA covariance matrix        [39 × 39]
        ├── eigenvalues.npy       Sorted eigenvalues            [39]
        ├── eigenvectors.npy      Eigenvector matrix Q          [39 × 39]
        ├── returns.npy           Log-return matrix             [T × 39]
        └── dates.npy             Date index for returns
```

---

## Stack

- **Python 3.10+** — NumPy, SciPy, Pandas
- **Solver** — `scipy.optimize.minimize` with SLSQP method
- **Notebook** — Jupyter, with one markdown cell per mathematical step

---

## Limitations

- Expected returns are backward-looking (EWMA of historical log-returns only)
- Arithmetic annualization `μ_annual = μ_daily × 252` — geometric compounding not applied
- Transaction costs and market impact are not modelled
- Universe is fixed; no dynamic inclusion or exclusion of tickers
- This is an academic project — **not financial advice**

## Future improvement
- Integreate sentiment analysis modeling
- Intergrate prediction models utilizing deep learning models for time series data prediction
