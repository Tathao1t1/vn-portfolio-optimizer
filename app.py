"""
ChainX Optima — Vietnam Portfolio Optimizer
pip install streamlit numpy scipy plotly
streamlit run app.py
"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from scipy.optimize import minimize
from datetime import datetime
import json


st.set_page_config(
    page_title="ChainX Optima",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
:root {
  --bg:#0f172a; --panel:#1e293b; --card:rgba(255,255,255,0.04);
  --border:rgba(255,255,255,0.09); --purple:#38bdf8; --purple2:#0ea5e9;
  --purple3:#0369a1; --grad:linear-gradient(135deg,#38bdf8 0%,#0369a1 100%);
  --white:#f1f5f9; --gray:#94a3b8; --green:#34d399; --amber:#f59e0b; --red:#f87171;
}

/* Base */
.stApp{background:var(--bg)!important;color:var(--white);font-family:'Inter',sans-serif;}
[data-testid="stSidebar"]{background:var(--panel)!important;border-right:1px solid var(--border);}
[data-testid="stHeader"]{background:transparent!important;}
[data-testid="stMainBlockContainer"]{padding-top:1.2rem;}

/* Metric cards */
div[data-testid="stMetric"]{
  background:var(--card);border:1px solid var(--border);
  border-radius:14px;padding:20px 22px;
  box-shadow:0 4px 20px rgba(0,0,0,.4);transition:transform .2s,box-shadow .2s;
}
div[data-testid="stMetric"]:hover{transform:translateY(-3px);box-shadow:0 8px 30px rgba(56,189,248,.2);}
div[data-testid="stMetric"] label{color:var(--gray)!important;font-size:11px!important;text-transform:uppercase;letter-spacing:1.2px;}
div[data-testid="stMetricValue"]>div{color:var(--white)!important;font-size:28px!important;font-weight:800!important;}
div[data-testid="stMetricDelta"]{color:var(--gray)!important;font-size:11px!important;}

/* Cards/containers */
[data-testid="stVerticalBlockBorderWrapper"]{
  border:1px solid var(--border)!important;border-radius:14px!important;
  background:var(--card)!important;box-shadow:0 4px 24px rgba(0,0,0,.3)!important;
}

/* Primary button */
[data-testid="stSidebar"] button[kind="primary"]{
  background:var(--grad)!important;color:#fff!important;border:none!important;
  border-radius:12px!important;font-weight:700!important;font-size:15px!important;
  padding:14px!important;min-height:54px!important;
  box-shadow:0 4px 20px rgba(56,189,248,.4);width:100%!important;
  transition:all .25s;margin-top:8px;
}
[data-testid="stSidebar"] button[kind="primary"]:hover{
  transform:translateY(-2px);box-shadow:0 8px 28px rgba(56,189,248,.65);
}

/* Tabs */
div[data-testid="stTabs"] button{
  color:var(--gray)!important;font-size:13px!important;
  font-weight:500!important;padding:10px 20px!important;
}
div[data-testid="stTabs"] button[aria-selected="true"]{
  color:var(--white)!important;border-bottom:2px solid var(--purple)!important;
}

/* Scrollbar */
::-webkit-scrollbar{width:4px;}
::-webkit-scrollbar-track{background:var(--panel);}
::-webkit-scrollbar-thumb{background:var(--purple3);border-radius:2px;}

/* Inputs — let Streamlit handle its own input rendering natively */

/* Ticker strip */
.ticker-wrap{
  overflow:hidden;background:#0c1628;padding:10px 0;
  border-radius:10px;border:1px solid var(--border);margin-bottom:24px;
}
.ticker-inner{display:flex;width:max-content;animation:scroll 60s linear infinite;}
.ticker-inner:hover{animation-play-state:paused;}
@keyframes scroll{0%{transform:translateX(0)}100%{transform:translateX(-50%)}}
.t-item{
  display:flex;align-items:center;padding:0 18px;
  font-size:12px;color:var(--white);
  border-right:1px solid var(--border);gap:8px;white-space:nowrap;
}
.t-badge{
  width:24px;height:24px;border-radius:5px;
  background:rgba(3,105,161,.7);display:inline-flex;
  align-items:center;justify-content:center;
  font-size:9px;font-weight:800;color:#bae6fd;
  font-family:monospace;flex-shrink:0;
}
.t-name{font-weight:700;letter-spacing:.3px;}
.t-mu{color:var(--gray);font-size:10px;}
.pos{color:#10b981;font-weight:600;}
.neg{color:#ef4444;font-weight:600;}

/* Tooltips */
.tt-wrap{display:inline-flex;align-items:center;gap:7px;position:relative;cursor:default;}
.tt-icon{
  width:16px;height:16px;border-radius:50%;
  background:rgba(56,189,248,.2);border:1px solid rgba(56,189,248,.5);
  display:inline-flex;align-items:center;justify-content:center;
  font-size:9px;color:var(--purple);font-weight:800;flex-shrink:0;
  cursor:help;transition:background .15s;
}
.tt-icon:hover{background:rgba(56,189,248,.45);}
.tt-box{
  visibility:hidden;opacity:0;position:absolute;left:50%;
  transform:translateX(-50%);bottom:calc(100% + 10px);
  background:#0f2744;border:1px solid rgba(56,189,248,.4);
  border-radius:10px;padding:12px 14px;width:280px;
  z-index:9999;font-size:12px;color:#d4d4d8;line-height:1.65;
  box-shadow:0 10px 40px rgba(0,0,0,.7);transition:opacity .15s;
  pointer-events:none;text-align:left;font-weight:400;
}
.tt-box::after{
  content:'';position:absolute;top:100%;left:50%;
  transform:translateX(-50%);border:6px solid transparent;border-top-color:#0f2744;
}
.tt-wrap:hover .tt-box{visibility:visible;opacity:1;}

/* Section titles */
.sec-title{
  font-size:15px;font-weight:700;color:var(--white);
  margin:0 0 14px;display:flex;align-items:center;gap:7px;
}

/* Stock rows */
.c-row{display:flex;align-items:center;padding:10px 0;border-bottom:1px solid var(--border);gap:10px;}
.c-row:last-child{border-bottom:none;}
.c-ticker{font-weight:700;font-size:12px;color:var(--white);background:rgba(56,189,248,.18);padding:3px 8px;border-radius:5px;font-family:monospace;min-width:42px;text-align:center;}
.c-name{color:var(--gray);font-size:12px;flex:1;line-height:1.3;}
.c-pct{color:var(--purple);font-weight:700;font-size:13px;min-width:42px;text-align:right;}
.c-amt{color:var(--white);font-weight:700;font-size:12px;min-width:100px;text-align:right;}

/* Zone banners */
.zone-reliable{
  background:rgba(16,185,129,.08);border:1px solid rgba(16,185,129,.35);
  border-radius:12px;padding:16px 20px;font-weight:600;font-size:14px;
  display:flex;align-items:flex-start;gap:14px;margin:20px 0;
}
.zone-caution{
  background:rgba(245,158,11,.08);border:1px solid rgba(245,158,11,.35);
  border-radius:12px;padding:16px 20px;font-weight:600;font-size:14px;
  display:flex;align-items:flex-start;gap:14px;margin:20px 0;
}
.zone-icon{font-size:22px;flex-shrink:0;}
.zone-title{color:var(--white);}
.zone-desc{font-weight:400;font-size:12px;color:#a1a1aa;margin-top:4px;line-height:1.6;}

/* Sidebar labels */
.sb-lbl{color:var(--gray);font-size:11px;text-transform:uppercase;letter-spacing:.8px;margin:16px 0 6px;font-weight:600;}

/* Upgrade block */
.upg{
  background:linear-gradient(135deg,#0ea5e9 0%,#0369a1 100%);
  border-radius:16px;padding:20px 16px;text-align:center;
  margin-top:24px;box-shadow:0 10px 30px -5px rgba(14,165,233,.5);
  border:1px solid rgba(56,189,248,.3);
}
.upg h4{margin:8px 0 4px;font-size:15px;font-weight:800;color:#fff;}
.upg p{margin:0;font-size:11px;color:rgba(255,255,255,.75);line-height:1.5;}

/* Divider */
.hline{height:1px;background:var(--border);margin:20px 0;}

/* Disclaimer banner */
.disclaimer{
  background:rgba(245,158,11,.06);border:1px solid rgba(245,158,11,.22);
  border-radius:10px;padding:12px 16px;font-size:12px;color:#a1a1aa;
  margin:20px 0 8px;line-height:1.65;
}

/* Section header */
.section-header{
  font-size:11px;font-weight:700;color:var(--gray);
  text-transform:uppercase;letter-spacing:1.2px;margin:28px 0 16px;
  display:flex;align-items:center;gap:10px;
}
.section-header::after{content:'';flex:1;height:1px;background:var(--border);}

/* Metric description */
.metric-desc{font-size:11px;color:var(--gray);margin-top:6px;line-height:1.55;padding:0 2px;}

/* Chart entrance animation */
@keyframes fadeSlideUp{
  from{opacity:0;transform:translateY(18px);}
  to{opacity:1;transform:translateY(0);}
}
@keyframes fadeIn{
  from{opacity:0;}
  to{opacity:1;}
}
[data-testid="stPlotlyChart"]{
  animation:fadeSlideUp .55s cubic-bezier(.22,.68,0,1.2) both;
}
div[data-testid="stMetric"]{
  animation:fadeSlideUp .45s cubic-bezier(.22,.68,0,1.2) both;
}
div[data-testid="stMetric"]:nth-child(2){animation-delay:.07s;}
div[data-testid="stMetric"]:nth-child(3){animation-delay:.14s;}
div[data-testid="stMetric"]:nth-child(4){animation-delay:.21s;}
</style>
""", unsafe_allow_html=True)


# ── TOOLTIP HELPER ────────────────────────────────────────────────────────────
def tt(label, tip, size="15px"):
    return (
        f'<div class="sec-title" style="font-size:{size}">'
        f'{label}'
        f'<div class="tt-wrap">'
        f'<div class="tt-icon">i</div>'
        f'<div class="tt-box">{tip}</div>'
        f'</div></div>'
    )


# ── CONSTANTS ─────────────────────────────────────────────────────────────────
RFR      = 0.045
CAP      = 0.15
DATA_DIR = Path(__file__).parent / "portfolio_data" / "processed"

COMPANY = {
    "ANV":"Nam Viet Corp",      "BIC":"BIDV Insurance",      "BID":"BIDV",
    "BMI":"Bao Minh Insurance", "CII":"HCM Infrastructure",  "CMG":"CMC Corp",
    "CSV":"Southern Chemicals", "CTD":"Coteccons",            "DBC":"Dabaco Group",
    "DGW":"Digiworld",          "DPM":"DPM Fertilizer",      "DXG":"Dat Xanh Group",
    "ELC":"Elcom Corp",         "FPT":"FPT Corp",             "GAS":"PV GAS",
    "GIL":"Gilimex",            "HAH":"Hai An Logistics",     "HAX":"Haxaco",
    "HCM":"HSC Securities",     "HPG":"Hoa Phat Group",       "IMP":"Imexpharm",
    "KDC":"Kido Group",         "KDH":"Khang Dien",           "LCG":"Licogi 16",
    "LSS":"Lam Son Sugar",      "MBB":"MB Bank",              "NKG":"Nam Kim Steel",
    "PC1":"PC1 Group",          "PNJ":"PNJ Gold",             "PVT":"PVTrans",
    "REE":"REE Corp",           "SAB":"Sabeco",               "SSI":"SSI Securities",
    "STK":"Century Fiber",      "TCM":"Thanh Cong Textile",   "VCB":"Vietcombank",
    "VCI":"Vietcap Securities", "VIC":"Vingroup",             "VSC":"Viconship",
}


# ── LOAD DATA ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_data():
    try:
        with open(DATA_DIR / "meta.json") as f:
            meta = json.load(f)
        mu  = np.load(DATA_DIR / "mu.npy")
        sig = np.load(DATA_DIR / "sigma.npy")
        ev  = np.load(DATA_DIR / "eigenvalues.npy")
        evc = np.load(DATA_DIR / "eigenvectors.npy")
        # Slice to 39 VN tickers only (VN stocks are always first in the matrix)
        tickers = meta["tickers"][:39]
        mu  = mu[:39]
        sig = sig[:39, :39]
        ev, evc = np.linalg.eigh(sig)
        idx = np.argsort(ev)[::-1]
        ev, evc = ev[idx], evc[:, idx]
        meta = {**meta, "tickers": tickers, "n_tickers": 39}
        return meta, tickers, mu, sig, ev, evc, True
    except Exception:
        n = 39
        np.random.seed(42)
        tickers = sorted(COMPANY.keys())
        meta = {
            "tickers": tickers, "n_tickers": n,
            "date_start": "2018-01-03", "date_end": "2026-04-24",
            "risk_free_rate": RFR, "t_days": 2090,
        }
        mu   = np.linspace(0.05, 0.17, n)
        vols = np.random.uniform(0.18, 0.35, n)
        corr = np.full((n, n), 0.32)
        np.fill_diagonal(corr, 1.0)
        for i in range(0, n, 4):
            for j in range(i, min(i+4, n)):
                for k in range(i, min(i+4, n)):
                    corr[j, k] = corr[k, j] = 0.62
        np.fill_diagonal(corr, 1.0)
        D   = np.diag(vols)
        sig = D @ corr @ D + np.eye(n) * .001
        ev, evc = np.linalg.eigh(sig)
        idx = np.argsort(ev)[::-1]
        return meta, tickers, mu, sig, ev[idx], evc[:, idx], False


META, TICKERS, MU_RAW, SIGMA_RAW, EIGENVALUES, EIGENVECTORS, IS_REAL = load_data()
N = len(TICKERS)


# ── SHARED HELPERS ────────────────────────────────────────────────────────────
def _js_mu():
    """James-Stein shrinkage: pull each stock's μ toward the cross-sectional mean."""
    mu_bar  = np.mean(MU_RAW)
    avg_var = 252 * np.mean(np.diag(SIGMA_RAW)) / META["t_days"]
    sq      = np.sum((MU_RAW - mu_bar) ** 2)
    alpha   = max(0.05, min(1., 1. - ((N - 3) * avg_var) / sq)) if sq > 1e-10 else 0.05
    return mu_bar + alpha * (MU_RAW - mu_bar)


def _sr():
    """Tikhonov-regularised covariance: Σ + 0.05·I (matches HANDOVER LAMBDA_REG)."""
    return SIGMA_RAW + 0.05 * np.eye(N)


# ── OPTIMIZER ─────────────────────────────────────────────────────────────────
@st.cache_data
def optimize(r_target: float):
    mu_js = _js_mu()
    sr    = _sr()
    res   = minimize(
        lambda w: w @ sr @ w,
        np.ones(N) / N,
        method="SLSQP",
        jac=lambda w: 2 * sr @ w,
        bounds=[(0., CAP)] * N,
        constraints=[
            {"type": "eq", "fun": lambda w: w @ mu_js - r_target},
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.},
        ],
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not res.success:
        return None
    w    = res.x
    pv   = float(np.sqrt(w @ sr @ w))
    pr   = float(mu_js @ w)
    neff = float(1. / np.sum(w ** 2))
    sh   = (pr - RFR) / pv if pv > 0 else 0.
    return {
        "weights":  {TICKERS[i]: round(float(w[i]), 6) for i in range(N)},
        "port_ret": round(pr, 5),
        "port_vol": round(pv, 5),
        "sharpe":   round(sh, 4),
        "n_eff":    round(neff, 3),
        "zone":     "reliable" if neff >= 3. else "caution",
    }


# ── GMV: TRUE CONSTRAINED QP (fixed — no longer uses raw eigenvector) ─────────
@st.cache_data
def compute_gmv():
    mu_js = _js_mu()
    sr    = _sr()
    res   = minimize(
        lambda w: w @ sr @ w,
        np.ones(N) / N,
        method="SLSQP",
        jac=lambda w: 2 * sr @ w,
        bounds=[(0., CAP)] * N,
        constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.}],
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not res.success:
        return None
    w    = res.x
    pv   = float(np.sqrt(w @ sr @ w))
    pr   = float(mu_js @ w)
    neff = float(1. / np.sum(w ** 2))
    return {
        "weights":  {TICKERS[i]: round(float(w[i]), 6) for i in range(N)},
        "port_ret": round(pr, 5),
        "port_vol": round(pv, 5),
        "n_eff":    round(neff, 3),
    }


# ── EFFICIENT FRONTIER (fixed: JS mu, positive rmin) ──────────────────────────
@st.cache_data
def frontier(n_pts: int = 60):
    mu_js = _js_mu()
    sr    = _sr()
    rmin  = max(0.01, float(mu_js.min()) + .005)
    rmax  = float(np.sort(mu_js)[-3]) * .88
    if rmin >= rmax:
        return []
    pts = []
    for r in np.linspace(rmin, rmax, n_pts):
        res = minimize(
            lambda w: w @ sr @ w,
            np.ones(N) / N,
            method="SLSQP",
            jac=lambda w: 2 * sr @ w,
            bounds=[(0, CAP)] * N,
            constraints=[
                {"type": "eq", "fun": lambda w, r=r: w @ mu_js - r},
                {"type": "eq", "fun": lambda w: np.sum(w) - 1.},
            ],
            options={"maxiter": 500, "ftol": 1e-10},
        )
        if res.success:
            w  = res.x
            v  = float(np.sqrt(w @ sr @ w))
            sh = (r - RFR) / v if v > 0 else 0.
            pts.append({
                "ret":      round(r * 100, 2),
                "vol":      round(v * 100, 2),
                "sharpe":   round(sh, 3),
                "reliable": (1. / np.sum(w ** 2)) >= 3.,
            })
    return pts


def fmt_vnd(val: float) -> str:
    if val >= 1e9:  return f"{val/1e9:,.2f} tỷ"
    if val >= 1e6:  return f"{val/1e6:,.2f}M"
    if val >= 1e3:  return f"{val/1e3:,.1f}K"
    return f"{val:,.0f}"


CHART_BG  = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                 font=dict(color="#a1a1aa", family="Inter,sans-serif"))
GRID      = dict(gridcolor="rgba(255,255,255,.05)")
PURPLE_CS = [[0, "#0c4a6e"], [.5, "#0ea5e9"], [1, "#bae6fd"]]


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:10px 0 24px">
      <div style="font-size:24px;font-weight:900;color:#fff;letter-spacing:-1px">
        ✨ ChainX <span style="color:#38bdf8">Optima</span>
      </div>
      <div style="font-size:11px;color:#a1a1aa;margin-top:5px;letter-spacing:.6px;font-weight:500">
        Linear Algebra · Vietnam Equities
      </div>
      <div style="font-size:11px;color:#52525b;margin-top:4px">Thao · Ngan · Nhu</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='hline' style='margin:0 0 4px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='sb-lbl'>Initial Capital (triệu VND)</div>", unsafe_allow_html=True)
    inv_m = st.number_input(
        "Capital",
        min_value=1,
        max_value=100_000,
        value=100,
        step=10,
        format="%d",
        label_visibility="collapsed",
        help="Enter your capital in millions of VND. Example: 100 = 100,000,000 VND.",
    )
    inv = inv_m * 1_000_000
    # show human-readable equivalent below the input
    if inv >= 1_000_000_000:
        inv_label = f"{inv/1_000_000_000:,.2f} tỷ VND"
    else:
        inv_label = f"{inv/1_000_000:,.0f} triệu VND"
    st.markdown(
        f"<div style='font-size:11px;color:#6b7280;margin:-6px 0 8px;text-align:right'>"
        f"= {inv_label}</div>",
        unsafe_allow_html=True,
    )

    # Dynamic slider bounds from GMV and frontier
    _gmv = compute_gmv()
    _slider_min = max(4, int((_gmv["port_ret"] if _gmv else 0.05) * 100) + 1)
    _slider_max = min(25, int(float(np.sort(_js_mu())[-3]) * 88))
    _slider_def = max(_slider_min, min(9, _slider_max))

    st.markdown("<div class='sb-lbl'>Target Annual Return</div>", unsafe_allow_html=True)
    r_pct = st.slider("", _slider_min, _slider_max, _slider_def,
                      label_visibility="collapsed",
                      help="Annual return you want the optimizer to achieve. Higher targets may concentrate into fewer stocks.")
    r_tgt = r_pct / 100.
    st.markdown(
        f"<div style='text-align:center;font-size:22px;font-weight:900;color:#38bdf8;margin:-4px 0 12px'>"
        f"{r_pct}% / year</div>",
        unsafe_allow_html=True,
    )

    go_btn = st.button("Optimize Portfolio", use_container_width=True, type="primary")

    st.markdown("<div class='hline' style='margin:16px 0'></div>", unsafe_allow_html=True)

    # Static data info (replaces the broken "Investment Horizon" selectbox)
    dot_color = "#10b981" if IS_REAL else "#f87171"
    dot_label = "LIVE DATA" if IS_REAL else "MOCK DATA"
    st.markdown(f"""
    <div style="background:rgba(255,255,255,.03);border:1px solid var(--border);
                border-radius:10px;padding:12px 14px;font-size:11px;color:#a1a1aa;line-height:1.8">
      <span style="color:{dot_color};font-weight:700">● {dot_label}</span><br>
      {META.get('n_tickers', N)} Vietnamese stocks<br>
      {META.get('date_start','2018-01-03')} → {META.get('date_end','2026-04-24')}<br>
      Risk-free rate: {RFR*100:.1f}% p.a. (SBV reference rate)
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="upg" style="margin-top:20px">
      <div style="font-size:28px">👑</div>
      <h4>Upgrade</h4>
      <p>Unlock all ChainX Optima features</p>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PANEL
# ══════════════════════════════════════════════════════════════════════════════

# ── Header ────────────────────────────────────────────────────────────────────
badge_html = (
    "<span style='background:rgba(16,185,129,.12);color:#10b981;font-size:11px;"
    "padding:3px 10px;border-radius:20px;font-weight:700;border:1px solid rgba(16,185,129,.35)'>● LIVE DATA</span>"
    if IS_REAL else
    "<span style='background:rgba(239,68,68,.1);color:#f87171;font-size:11px;"
    "padding:3px 10px;border-radius:20px;font-weight:700;border:1px solid rgba(239,68,68,.3)'>● MOCK DATA</span>"
)
st.markdown(f"""
<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:6px">
  <div>
    <h2 style="margin:0;font-size:24px;font-weight:900;color:#fff;letter-spacing:-0.5px">
      Vietnam Portfolio Optimizer
    </h2>
    <p style="color:#a1a1aa;font-size:13px;margin:4px 0 0">
      Find the <b style="color:#bae6fd">minimum-risk portfolio</b> that hits your target return
      — powered by linear algebra, not black-box ML.
    </p>
  </div>
  <div style="display:flex;align-items:center;gap:10px;margin-top:4px;flex-shrink:0">
    {badge_html}
    <span style="color:#52525b;font-size:11px">{datetime.now().strftime("%b %d, %Y")}</span>
  </div>
</div>
<div class="hline"></div>
""", unsafe_allow_html=True)

# ── Ticker strip (fixed: CSS badges, correct "Expected Annual Return" labels) ──
td = sorted(zip(TICKERS, MU_RAW), key=lambda x: x[1], reverse=True)
strip_stocks = td[:10] + td[-5:]
items = "".join(
    f'<div class="t-item">'
    f'<span class="t-badge">{t[:2]}</span>'
    f'<span class="t-name">{t}</span>'
    f'<span class="t-mu">μ&nbsp;</span>'
    f'<span class="{"pos" if m >= 0 else "neg"}">'
    f'{"+" if m >= 0 else ""}{m*100:.1f}%</span>'
    f'</div>'
    for t, m in strip_stocks
)
st.markdown(
    f'<div class="ticker-wrap"><div class="ticker-inner">{items}{items}</div></div>',
    unsafe_allow_html=True,
)

# ── Optimizer state (fixed: failure sentinel prevents infinite retry loop) ────
if "res"      not in st.session_state: st.session_state.res      = None
if "failed_r" not in st.session_state: st.session_state.failed_r = None

if go_btn:
    if st.session_state.failed_r != r_tgt:
        with st.spinner("Solving quadratic program…"):
            result = optimize(r_tgt)
        if result is None:
            st.session_state.failed_r = r_tgt
            st.session_state.res      = None
        else:
            st.session_state.res      = result
            st.session_state.failed_r = None
elif st.session_state.res is None and st.session_state.failed_r != r_tgt:
    with st.spinner("Loading initial portfolio…"):
        result = optimize(r_tgt)
    if result is None:
        st.session_state.failed_r = r_tgt
    else:
        st.session_state.res = result

if st.session_state.failed_r == r_tgt:
    st.error(
        "Optimizer could not converge for this target return. "
        "Try lowering the target — very high returns may be infeasible for this stock universe."
    )
    st.stop()

if st.session_state.res is None:
    st.info("Set your target return in the sidebar and click **Optimize Portfolio**.")
    st.stop()

R = st.session_state.res

# ── Section 1: Portfolio at a Glance ─────────────────────────────────────────
st.markdown('<div class="section-header">Your Portfolio at a Glance</div>', unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Expected Return", f"{R['port_ret']*100:.1f}%")
    st.markdown(
        "<div class='metric-desc'>Average annual gain this portfolio is expected to earn, "
        "based on EWMA of historical returns (2018–2026).</div>",
        unsafe_allow_html=True,
    )
with m2:
    st.metric("Portfolio Risk (σ)", f"{R['port_vol']*100:.1f}%")
    st.markdown(
        "<div class='metric-desc'>Annualised volatility — how much the portfolio value "
        "can swing up or down in a typical year. Lower is safer.</div>",
        unsafe_allow_html=True,
    )
with m3:
    st.metric("Sharpe Ratio", f"{R['sharpe']:.2f}")
    st.markdown(
        "<div class='metric-desc'>Return per unit of risk, above the 4.5% risk-free rate. "
        "Above 1.0 is generally considered good.</div>",
        unsafe_allow_html=True,
    )
with m4:
    st.metric("Effective Positions", f"{R['n_eff']:.1f}")
    st.markdown(
        "<div class='metric-desc'>How many independent bets the portfolio is making "
        "(HHI-based). Higher = better diversification.</div>",
        unsafe_allow_html=True,
    )

# Zone banner (dynamic — based on actual n_eff from optimizer, not hardcoded threshold)
if R["zone"] == "reliable":
    st.markdown(f"""
    <div class="zone-reliable">
      <div class="zone-icon">✅</div>
      <div>
        <div class="zone-title">Reliable Zone — Portfolio is well-diversified</div>
        <div class="zone-desc">
          N_eff = {R['n_eff']:.1f} (≥ 3.0) · Capital is spread across enough positions that
          estimation errors in individual stock returns won't dominate the result.
          You can proceed with confidence.
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="zone-caution">
      <div class="zone-icon">⚠️</div>
      <div>
        <div class="zone-title">Caution Zone — Portfolio is concentrated</div>
        <div class="zone-desc">
          N_eff = {R['n_eff']:.1f} (< 3.0) · This target forces the optimizer into very few stocks.
          The math is correct, but high concentration means historical return estimates dominate —
          and those estimates are unreliable. Consider lowering your target return.
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── Section 2: Detailed Analysis ─────────────────────────────────────────────
st.markdown('<div class="section-header">Detailed Analysis</div>', unsafe_allow_html=True)

tw, te = st.tabs(["📊  Holdings", "🔬  Math Engine"])


# ════ TAB 1: Holdings ════════════════════════════════════════════════════════
with tw:
    wts = R["weights"]
    sw  = sorted(wts.items(), key=lambda x: x[1], reverse=True)
    top = [(t, w) for t, w in sw if w > 0.001]
    tl  = [t for t, _ in top]
    ow  = [round(w * 100, 1) for _, w in top]
    H   = 440

    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.markdown(tt(
                "Optimal Allocation",
                "Weights from the <b>Quadratic Program</b>: minimise wᵀΣw subject to "
                "μᵀw = target return, 1ᵀw = 1, 0 ≤ wᵢ ≤ 15%. "
                "Stocks receiving near-zero weight don't reduce risk at your target — "
                "excluding them is mathematically correct, not a bug.",
            ), unsafe_allow_html=True)
            fig = go.Figure(go.Bar(
                y=tl, x=ow, orientation="h",
                marker=dict(color=ow, colorscale=PURPLE_CS, showscale=False, line=dict(width=0)),
                text=[f"{v}%" for v in ow], textposition="outside",
                textfont=dict(color="#fff", size=11),
            ))
            fig.update_layout(
                height=H, margin=dict(l=0, r=55, t=4, b=20),
                xaxis=dict(title="Weight (%)", **GRID, color="#a1a1aa"),
                yaxis=dict(tickfont=dict(family="monospace", color="#fff", size=11), **GRID),
                transition=dict(duration=700, easing="cubic-in-out"),
                **CHART_BG,
            )
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        with st.container(border=True):
            st.markdown(tt(
                "Capital Distribution",
                "<b>Donut chart</b>: how your capital is spread across holdings. "
                "A well-diversified portfolio shows no single dominant slice. "
                "Stocks below 1.5% are grouped into 'Others' for readability.",
            ), unsafe_allow_html=True)
            pl  = [t for t, w in top if w >= .015]
            pv  = [round(w * 100, 1) for _, w in top if w >= .015]
            oth = sum(w for _, w in top if w < .015)
            if oth > .001:
                pl.append("Others")
                pv.append(round(oth * 100, 1))
            nc   = len(pv)
            cpie = px.colors.sample_colorscale("Blues_r", [i / max(1, nc - 1) for i in range(nc)])
            fig2 = go.Figure(go.Pie(
                labels=pl, values=pv, hole=.6,
                textinfo="label+percent", textfont=dict(color="#fff", size=11),
                marker=dict(colors=cpie, line=dict(color="#0f172a", width=2)),
            ))
            fig2.update_layout(
                height=H, margin=dict(l=20, r=80, t=4, b=20),
                showlegend=False,
                transition=dict(duration=700, easing="cubic-in-out"),
                **CHART_BG,
            )
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    r1, r2 = st.columns(2)
    rem     = sum(w for _, w in top[10:])
    has_oth = rem > .001

    with r1:
        with st.container(border=True):
            st.markdown(tt(
                "Top Holdings",
                "Stocks sorted by weight. The optimizer assigns near-zero weight to "
                "stocks that don't help reduce risk at your target — those are excluded. "
                "Weights sum to 100%.",
            ), unsafe_allow_html=True)
            rows = "".join(
                f'<div class="c-row">'
                f'<span class="c-ticker">{t}</span>'
                f'<span class="c-name">{COMPANY.get(t,"")}</span>'
                f'<span class="c-pct">{w*100:.1f}%</span>'
                f'</div>'
                for t, w in top[:10]
            )
            st.markdown(f"<div style='padding:0 6px'>{rows}</div>", unsafe_allow_html=True)

    with r2:
        with st.container(border=True):
            st.markdown(tt(
                "Allocated Amount",
                f"Amount to invest per stock: <b>weight × {fmt_vnd(inv)} VND capital</b>. "
                "This is a suggested allocation based on historical data — "
                "not a buy order. Always verify prices and availability with your broker.",
            ), unsafe_allow_html=True)
            vrows = "".join(
                f'<div class="c-row">'
                f'<span class="c-ticker">{t}</span>'
                f'<span class="c-pct">{w*100:.1f}%</span>'
                f'<span class="c-amt">{fmt_vnd(inv*w)} VND</span>'
                f'</div>'
                for t, w in top[:10]
            )
            if has_oth:
                vrows += (
                    f'<div class="c-row">'
                    f'<span class="c-ticker" style="opacity:.4">···</span>'
                    f'<span class="c-pct" style="color:#52525b">{rem*100:.1f}%</span>'
                    f'<span class="c-amt" style="color:#52525b">{fmt_vnd(inv*rem)} VND</span>'
                    f'</div>'
                )
            st.markdown(f"<div style='padding:0 6px'>{vrows}</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # GMV section (fixed: uses true constrained QP, not raw eigenvector)
    with st.container(border=True):
        st.markdown(tt(
            "Lowest Risk Portfolio (Global Minimum Variance)",
            "The portfolio with the <b>absolute lowest achievable risk</b>, ignoring return target. "
            "Found by solving the same optimizer without a return constraint. "
            "Compare its risk to yours above — the gap is the cost of your return ambition. "
            "This uses the true constrained QP solution, not an eigenvector approximation.",
        ), unsafe_allow_html=True)

        with st.spinner("Computing GMV…"):
            GMV = compute_gmv()

        if GMV:
            gmv_items = sorted(GMV["weights"].items(), key=lambda x: x[1], reverse=True)
            gmv_top   = [(t, w) for t, w in gmv_items if w > 0.001][:12]
            gv = [round(w * 100, 1) for _, w in gmv_top]
            gl = [t for t, _ in gmv_top]

            gc1, gc2, gc3 = st.columns(3)
            with gc1:
                st.metric("GMV Return",     f"{GMV['port_ret']*100:.1f}%", "Minimum-risk baseline")
            with gc2:
                st.metric("GMV Risk (σ)",   f"{GMV['port_vol']*100:.1f}%", "Lowest achievable vol")
            with gc3:
                delta_vol = (R["port_vol"] - GMV["port_vol"]) * 100
                st.metric(
                    "Your Risk Premium",
                    f"+{delta_vol:.1f}%",
                    f"extra vol for {r_pct}% target",
                )

            fig3 = go.Figure(go.Bar(
                y=gl, x=gv, orientation="h",
                marker=dict(color=gv, colorscale=PURPLE_CS, showscale=False, line=dict(width=0)),
                text=[f"{v}%" for v in gv], textposition="outside",
                textfont=dict(color="#fff", size=11),
            ))
            fig3.update_layout(
                height=320, margin=dict(l=0, r=55, t=4, b=20),
                xaxis=dict(title="Weight (%)", **GRID, color="#a1a1aa"),
                yaxis=dict(tickfont=dict(family="monospace", color="#fff", size=11)),
                transition=dict(duration=700, easing="cubic-in-out"),
                **CHART_BG,
            )
            st.plotly_chart(fig3, use_container_width=True)


# ════ TAB 2: Math Engine ══════════════════════════════════════════════════════
with te:
    st.markdown("""
    <div style="background:rgba(56,189,248,.06);border:1px solid rgba(56,189,248,.2);
                border-radius:12px;padding:16px 20px;margin-bottom:20px;
                font-size:13px;color:#bae6fd;line-height:1.75">
      <b>What is happening under the hood?</b><br>
      The covariance matrix Σ captures how all 39 Vietnamese stocks move together.
      We decompose it into <b>eigenvalues</b> (how much variance each direction carries)
      and <b>eigenvectors</b> (the directions themselves).
      Eigenvalues above the <b>Marchenko-Pastur threshold</b> represent genuine market structure —
      those below are statistical noise from limited data.
      Filtering noise before optimizing makes the portfolio more stable and less sensitive to data quirks.
    </div>
    """, unsafe_allow_html=True)

    # Eigenvalue spectrum (M-P formula fixed: avg_variance × (1 + √(N/T))²)
    with st.container(border=True):
        st.markdown(tt(
            "Eigenvalue Spectrum of Σ",
            "Each bar = one eigenvalue of the covariance matrix Σ = QΛQᵀ. "
            "<b>Purple</b> = signal (above Marchenko-Pastur threshold λ+ = avg_var × (1+√(N/T))²). "
            "<b>Gray</b> = noise (sampling artifact, below threshold). "
            "<b>Light purple</b> = smallest eigenvalue = minimum-variance direction. "
            "<b>Red dashed</b> = M-P threshold separating signal from noise.",
        ), unsafe_allow_html=True)

        n        = N
        T        = META["t_days"]
        # FIXED: correct Marchenko-Pastur formula — scales by average diagonal variance
        mp       = float(np.mean(np.diag(SIGMA_RAW))) * (1 + np.sqrt(n / T)) ** 2
        ev_plot  = EIGENVALUES[:n]
        n_signal = int(np.sum(ev_plot > mp))

        cols = [
            "#bae6fd" if i == len(ev_plot) - 1
            else ("#38bdf8" if v > mp else "#3f3f46")
            for i, v in enumerate(ev_plot)
        ]
        labs = [
            "λ₁ market" if i == 0
            else (f"λ{n} min-var" if i == n - 1
                  else (f"λ{i+1}" if i % 5 == 0 else ""))
            for i in range(n)
        ]

        fe = go.Figure()
        fe.add_trace(go.Bar(
            x=labs, y=ev_plot,
            marker_color=cols, marker_line_width=0,
            hovertemplate="λ = %{y:.4f}<extra></extra>",
        ))
        fe.add_hline(
            y=mp,
            line=dict(color="#ef4444", width=2, dash="dash"),
            annotation_text=f"M-P λ+ = {mp:.4f}",
            annotation_position="top right",
            annotation_font=dict(color="#ef4444", size=11),
        )
        fe.update_layout(
            height=400,
            xaxis=dict(title="Eigenvalue index", **GRID, color="#a1a1aa", tickangle=0),
            yaxis=dict(title="Magnitude",         **GRID, color="#a1a1aa"),
            showlegend=False,
            margin=dict(l=0, r=0, t=4, b=40),
            transition=dict(duration=600, easing="cubic-in-out"),
            **CHART_BG,
        )
        st.plotly_chart(fe, use_container_width=True)

        lc1, lc2, lc3, lc4 = st.columns(4)
        def _leg(color, text):
            return (
                f"<span style='color:{color};font-size:16px;line-height:1'>■</span>"
                f" <span style='color:#a1a1aa;font-size:12px'>{text}</span>"
            )
        with lc1: st.markdown(_leg("#38bdf8", "Signal (real market risk)"),    unsafe_allow_html=True)
        with lc2: st.markdown(_leg("#3f3f46", "Noise (below M-P threshold)"),  unsafe_allow_html=True)
        with lc3: st.markdown(_leg("#bae6fd", "Min-variance direction"),        unsafe_allow_html=True)
        with lc4: st.markdown(
            "<span style='color:#ef4444;font-size:13px'>- - -</span>"
            " <span style='color:#a1a1aa;font-size:12px'>Marchenko-Pastur λ+</span>",
            unsafe_allow_html=True,
        )

        st.info(
            f"**{n_signal} signal factor{'s' if n_signal != 1 else ''}** out of {n} eigenvalues "
            f"exceed the Marchenko-Pastur threshold (λ+ = {mp:.4f}). "
            f"The remaining {n - n_signal} are noise — filtered before optimization. "
            f"T/N ratio = {T/n:.1f} (higher = more reliable covariance estimate)."
        )

    # James-Stein shrinkage visualisation
    with st.container(border=True):
        mu_bar    = float(np.mean(MU_RAW))
        avg_var_v = 252 * float(np.mean(np.diag(SIGMA_RAW))) / META["t_days"]
        sq_v      = float(np.sum((MU_RAW - mu_bar) ** 2))
        alpha_v   = max(0.05, min(1., 1. - ((N - 3) * avg_var_v) / sq_v)) if sq_v > 1e-10 else 0.05
        mu_js_v   = mu_bar + alpha_v * (MU_RAW - mu_bar)

        st.markdown(tt(
            "James-Stein Return Shrinkage",
            "Raw historical returns are noisy — a stock that performed well may just have been lucky. "
            "James-Stein shrinkage pulls each stock's expected return toward the cross-sectional mean, "
            "reducing the influence of outliers. "
            f"Shrinkage factor α = {alpha_v:.3f}: closer to 0 means more shrinkage toward the mean. "
            "The optimizer uses the shrunken values (purple bars).",
        ), unsafe_allow_html=True)

        sorted_idx = np.argsort(MU_RAW)
        t_labels   = [TICKERS[i] for i in sorted_idx]
        js_fig     = go.Figure()
        js_fig.add_trace(go.Bar(
            x=t_labels,
            y=[MU_RAW[i] * 100 for i in sorted_idx],
            name="Raw historical μ",
            marker_color="rgba(56,189,248,0.2)",
            hovertemplate="%{x}: %{y:.1f}%<extra>Raw</extra>",
        ))
        js_fig.add_trace(go.Bar(
            x=t_labels,
            y=[mu_js_v[i] * 100 for i in sorted_idx],
            name="JS-shrunken μ (used by optimizer)",
            marker_color="#0ea5e9",
            hovertemplate="%{x}: %{y:.1f}%<extra>Shrunken</extra>",
        ))
        js_fig.add_hline(
            y=mu_bar * 100,
            line=dict(color="#10b981", width=1.5, dash="dot"),
            annotation_text=f"Cross-sectional mean = {mu_bar*100:.1f}%",
            annotation_font=dict(color="#10b981", size=11),
        )
        js_fig.update_layout(
            height=320, barmode="group",
            xaxis=dict(title="Ticker", **GRID, color="#a1a1aa", tickangle=45),
            yaxis=dict(title="Annual Return (%)", **GRID, color="#a1a1aa"),
            legend=dict(orientation="h", y=1.1, font=dict(color="#fff", size=12)),
            margin=dict(l=0, r=0, t=30, b=60),
            transition=dict(duration=600, easing="cubic-in-out"),
            **CHART_BG,
        )
        st.plotly_chart(js_fig, use_container_width=True)
        st.caption(
            f"Shrinkage factor α = {alpha_v:.3f}. "
            "All return estimates are pulled toward the cross-sectional mean. "
            "Outlier stocks (very high or very low raw μ) are trusted less — "
            "reducing the optimizer's tendency to over-concentrate into historically lucky stocks."
        )


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<div class='hline'></div>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;font-size:11px;color:#52525b;padding:8px 0 16px;line-height:1.7">
  ⚠ <b style="color:#6b7280">Not financial advice.</b>
  This tool is for educational purposes only. Past performance does not guarantee future results.
  Expected returns are estimated from EWMA historical data (Jan 2018 – Apr 2026) and do not
  incorporate analyst forecasts, macro signals, or forward-looking information.
  Transaction costs and brokerage fees are not modelled.<br>
  <span style="color:#3f3f46">ChainX Optima · Built with linear algebra · Thao · Ngan · Nhu</span>
</div>
""", unsafe_allow_html=True)
