"""
Vietnam Stock Portfolio Optimizer — FastAPI Server
===================================================
Generated from HANDOVER.txt spec.
Thảo: chỉ cần review phần load_artifacts() và optimizer() là đủ,
còn lại (CORS, endpoints, validation) đã đúng spec rồi.

Cách chạy:
    pip install fastapi uvicorn numpy scipy
    uvicorn main:app --reload --port 8000

Sau đó test thử:
    http://localhost:8000/meta
    http://localhost:8000/docs   ← Swagger UI tự động
"""

import json
import numpy as np
from pathlib import Path
from scipy.optimize import minimize
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ─────────────────────────────────────────────
#  CONFIG  (từ HANDOVER.txt Section 3)
# ─────────────────────────────────────────────
HALFLIFE    = 252
LAMBDA_REG  = 0.05
ALPHA_MIN   = 0.05
RFR         = 0.045   # risk-free rate: 4.5% p.a.
CAP_WEIGHT  = 0.15    # max weight per stock: 15%
FFILL_LIMIT = 5

DATA_DIR = Path(__file__).parent / "processed"

# ─────────────────────────────────────────────
#  APP + CORS  (BẮT BUỘC phải có để browser gọi được)
# ─────────────────────────────────────────────
app = FastAPI(title="VN Portfolio Optimizer", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # khi deploy thật thì đổi thành domain cụ thể
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
#  LOAD ARTIFACTS  (Thảo kiểm tra phần này)
# ─────────────────────────────────────────────
def load_artifacts():
    """Load pre-computed files từ portfolio_data/processed/"""
    with open(DATA_DIR / "meta.json") as f:
        meta = json.load(f)

    mu    = np.load(DATA_DIR / "mu.npy")        # [N] expected annual returns
    sigma = np.load(DATA_DIR / "sigma.npy")     # [N x N] covariance matrix
    eigenvalues  = np.load(DATA_DIR / "eigenvalues.npy")
    eigenvectors = np.load(DATA_DIR / "eigenvectors.npy")

    return meta, mu, sigma, eigenvalues, eigenvectors

try:
    META, MU_RAW, SIGMA_RAW, EIGENVALUES, EIGENVECTORS = load_artifacts()
    N = len(META["tickers"])
    print(f"✅ Loaded {N} tickers, date range {META['date_start']} → {META['date_end']}")
except Exception as e:
    print(f"⚠️  Không load được artifacts: {e}")
    print("   Chạy LinearCode.ipynb Steps 1–5 trước để tạo file processed/ nhé.")
    META, MU_RAW, SIGMA_RAW, EIGENVALUES, EIGENVECTORS = None, None, None, None, None
    N = 39

# ─────────────────────────────────────────────
#  OPTIMIZER CORE  (Thảo kiểm tra phần này)
# ─────────────────────────────────────────────
def james_stein_shrinkage(mu: np.ndarray, sigma: np.ndarray, T_eff: float) -> np.ndarray:
    """James-Stein shrinkage trên expected returns (Bug Fix #2 từ HANDOVER)"""
    n = len(mu)
    mu_bar = np.mean(mu)
    avg_var = 252 * np.mean(np.diag(sigma)) / T_eff   # đã fix bug factor-252
    total_sq_dev = np.sum((mu - mu_bar) ** 2)
    if total_sq_dev < 1e-10:
        return mu.copy()
    alpha = max(ALPHA_MIN, 1 - ((n - 3) * avg_var) / total_sq_dev)
    alpha = min(alpha, 1.0)
    return mu_bar + alpha * (mu - mu_bar)


def tikhonov_regularize(sigma: np.ndarray) -> np.ndarray:
    """Tikhonov (L2) regularization: Σ_reg = Σ + λI"""
    return sigma + LAMBDA_REG * np.eye(len(sigma))


def solve_qp(mu_js: np.ndarray, sigma_reg: np.ndarray, r_target: float):
    """
    Giải Quadratic Program:
        minimize    wᵀ Σ_reg w
        subject to  μᵀw = r_target
                    1ᵀw = 1
                    0 ≤ wᵢ ≤ 0.15
    """
    n = len(mu_js)
    w0 = np.ones(n) / n  # khởi điểm: equal weight

    def portfolio_variance(w):
        return w @ sigma_reg @ w

    def grad_variance(w):
        return 2 * sigma_reg @ w

    constraints = [
        {"type": "eq", "fun": lambda w: w @ mu_js - r_target},   # target return
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},        # fully invested
    ]
    bounds = [(0.0, CAP_WEIGHT)] * n  # long-only, max 15%

    result = minimize(
        portfolio_variance,
        w0,
        jac=grad_variance,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    return result


def compute_frontier(mu_js, sigma_reg, n_points=60):
    """Tính toàn bộ efficient frontier"""
    ret_min = float(mu_js.min()) + 0.001
    ret_max = float(mu_js.max()) * CAP_WEIGHT * len(mu_js) * 0.9  # realistic upper bound

    points = []
    for r in np.linspace(ret_min, ret_max, n_points):
        res = solve_qp(mu_js, sigma_reg, r)
        if res.success:
            w = res.x
            vol = float(np.sqrt(w @ sigma_reg @ w))
            sharpe = float((r - RFR) / vol) if vol > 0 else 0.0
            points.append({"ret": round(r, 5), "vol": round(vol, 5), "sharpe": round(sharpe, 5)})
    return points


def n_effective(weights: np.ndarray) -> float:
    """Herfindahl-Hirschman Index: N_eff = 1 / Σwᵢ²"""
    return float(1.0 / np.sum(weights ** 2))


# ─────────────────────────────────────────────
#  REQUEST / RESPONSE SCHEMAS
# ─────────────────────────────────────────────
class OptimizeRequest(BaseModel):
    r_target: float   # e.g. 0.15 for 15%


# ─────────────────────────────────────────────
#  ENDPOINTS
# ─────────────────────────────────────────────
@app.get("/meta")
def get_meta():
    """
    Trả về metadata: tickers, date range, zone bounds.
    Frontend gọi cái này đầu tiên khi load trang.
    """
    if META is None:
        raise HTTPException(503, "Artifacts chưa được load. Chạy LinearCode.ipynb trước.")

    # Tính zone bounds bằng cách sweep frontier
    mu_js     = james_stein_shrinkage(MU_RAW, SIGMA_RAW, META["t_days"])
    sigma_reg = tikhonov_regularize(SIGMA_RAW)
    frontier  = compute_frontier(mu_js, sigma_reg, n_points=80)

    ret_gmv = frontier[0]["ret"] if frontier else 0.0

    # Tìm ret_reliable_max: điểm cuối cùng có N_eff >= 3
    ret_reliable_max = ret_gmv
    for pt in frontier:
        res = solve_qp(mu_js, sigma_reg, pt["ret"])
        if res.success and n_effective(res.x) >= 3.0:
            ret_reliable_max = pt["ret"]

    ret_caution_max = frontier[-1]["ret"] if frontier else 0.0

    return {
        "tickers":          META["tickers"],
        "n_tickers":        META["n_tickers"],
        "date_start":       META["date_start"],
        "date_end":         META["date_end"],
        "risk_free_rate":   RFR,
        "ret_gmv":          round(ret_gmv, 5),
        "ret_reliable_max": round(ret_reliable_max, 5),
        "ret_caution_max":  round(ret_caution_max, 5),
    }


@app.get("/frontier")
def get_frontier():
    """
    Trả về toàn bộ efficient frontier (không cần r_target).
    Frontend dùng để vẽ biểu đồ Efficient Frontier tab.
    """
    if MU_RAW is None:
        raise HTTPException(503, "Artifacts chưa được load.")

    mu_js     = james_stein_shrinkage(MU_RAW, SIGMA_RAW, META["t_days"])
    sigma_reg = tikhonov_regularize(SIGMA_RAW)
    points    = compute_frontier(mu_js, sigma_reg, n_points=80)

    return {"frontier": points}


@app.post("/optimize")
def optimize(req: OptimizeRequest):
    """
    Core endpoint. Nhận r_target, trả về portfolio tối ưu.

    Input:  { "r_target": 0.15 }
    Output: weights, port_vol, port_ret, sharpe, n_eff, zone, ...
    """
    if MU_RAW is None:
        raise HTTPException(503, "Artifacts chưa được load. Chạy LinearCode.ipynb trước.")

    r = req.r_target

    # ── Validation ──────────────────────────────────────────
    mu_js     = james_stein_shrinkage(MU_RAW, SIGMA_RAW, META["t_days"])
    sigma_reg = tikhonov_regularize(SIGMA_RAW)
    frontier  = compute_frontier(mu_js, sigma_reg, n_points=80)

    ret_gmv         = frontier[0]["ret"] if frontier else 0.0
    ret_caution_max = frontier[-1]["ret"] if frontier else 1.0

    if r < ret_gmv:
        raise HTTPException(
            400,
            f"r_target ({r:.1%}) thấp hơn minimum-variance return ({ret_gmv:.1%}). "
            f"Hãy tăng target lên."
        )
    if r > ret_caution_max:
        raise HTTPException(
            400,
            f"r_target ({r:.1%}) vượt quá mức tối đa ({ret_caution_max:.1%}). "
            f"Không thể đạt được với universe hiện tại."
        )

    # ── Solve QP ────────────────────────────────────────────
    result = solve_qp(mu_js, sigma_reg, r)
    if not result.success:
        raise HTTPException(422, f"Optimizer không hội tụ: {result.message}")

    w     = result.x
    p_vol = float(np.sqrt(w @ sigma_reg @ w))
    p_ret = float(mu_js @ w)
    neff  = n_effective(w)

    # ── Zone ────────────────────────────────────────────────
    ret_reliable_max = ret_gmv
    for pt in frontier:
        res2 = solve_qp(mu_js, sigma_reg, pt["ret"])
        if res2.success and n_effective(res2.x) >= 3.0:
            ret_reliable_max = pt["ret"]

    zone = "reliable" if neff >= 3.0 else "caution"

    return {
        "weights":          {META["tickers"][i]: round(float(w[i]), 6) for i in range(N)},
        "port_ret":         round(p_ret, 6),
        "port_vol":         round(p_vol, 6),
        "sharpe":           round((p_ret - RFR) / p_vol, 4) if p_vol > 0 else 0.0,
        "n_eff":            round(neff, 3),
        "zone":             zone,
        "ret_gmv":          round(ret_gmv, 5),
        "ret_reliable_max": round(ret_reliable_max, 5),
        "ret_caution_max":  round(ret_caution_max, 5),
        "frontier":         frontier,
    }


# ─────────────────────────────────────────────
#  CHẠY TRỰC TIẾP  (python main.py)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
