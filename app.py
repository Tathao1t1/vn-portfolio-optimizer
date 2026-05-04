"""
ChainX Optima — Vietnam Portfolio Optimizer (final)
pip install streamlit numpy scipy plotly
streamlit run app.py
"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import datetime
import json, random


st.set_page_config(page_title="ChainX Optima", page_icon="✨", layout="wide",
                    initial_sidebar_state="expanded")


st.markdown("""
<style>
:root{
  --bg:#0e0e11; --panel:#15151a; --card:rgba(255,255,255,0.03);
  --border:rgba(255,255,255,0.07); --purple:#9D4EDD; --purple2:#7c3aed;
  --purple3:#4C1D95; --grad:linear-gradient(135deg,#9D4EDD 0%,#4C1D95 100%);
  --white:#ffffff; --gray:#a1a1aa; --green:#10b981; --red:#ef4444;
}
.stApp{background:var(--bg)!important;color:var(--white);font-family:'Inter',sans-serif;}
[data-testid="stSidebar"]{background:var(--panel)!important;border-right:1px solid var(--border);}
[data-testid="stHeader"]{background:transparent!important;}
[data-testid="stMainBlockContainer"]{padding-top:.8rem;}
section[data-testid="stSidebar"] > div {padding-top: 1rem; padding-left: 0.5rem; padding-right: 0.5rem;}

/* metrics */
div[data-testid="stMetric"]{
  background:var(--card);border:1px solid var(--border);
  border-radius:14px;padding:18px 22px;
  box-shadow:0 4px 20px rgba(0,0,0,.4);transition:transform .2s,box-shadow .2s;
}
div[data-testid="stMetric"]:hover{transform:translateY(-3px);box-shadow:0 8px 30px rgba(157,78,221,.2);}
div[data-testid="stMetric"] label{color:var(--gray)!important;font-size:11px!important;text-transform:uppercase;letter-spacing:1.2px;}
div[data-testid="stMetricValue"]>div{color:var(--white)!important;font-size:30px!important;font-weight:800!important;}
div[data-testid="stMetricDelta"]{color:var(--purple)!important;font-size:12px!important;}

/* cards */
[data-testid="stVerticalBlockBorderWrapper"]{
  border:1px solid var(--border)!important;border-radius:14px!important;
  background:var(--card)!important;box-shadow:0 4px 24px rgba(0,0,0,.3)!important;
}

/* radio → chainx nav */
div[role="radiogroup"]{gap:6px!important;margin-top:12px; margin-bottom: 24px;}
div[role="radiogroup"]>label>div:first-of-type{display:none!important;}
div[role="radiogroup"]>label>div:last-of-type{margin-left:0!important;padding-left:0!important;}
div[role="radiogroup"]>label{
  padding:12px 18px;border-radius:12px;cursor:pointer;
  transition:all .2s ease;font-weight:600;color:var(--gray);font-size:14px;
}
div[role="radiogroup"]>label:hover{background:rgba(255,255,255,.05);color:var(--white); transform: translateX(4px);}
div[role="radiogroup"]>label[aria-checked="true"]{
  background:rgba(157,78,221,.15)!important;color:var(--white)!important;
  border-left: 4px solid var(--purple);
  border-radius: 8px 12px 12px 8px;
}

/* primary button (Optimize) */
[data-testid="stSidebar"] button[kind="primary"]{
  background:var(--grad)!important;color:#fff!important;border:none!important;
  border-radius:12px!important;font-weight:700!important;font-size:14px!important;
  padding:14px!important;min-height:52px!important;
  box-shadow:0 4px 20px rgba(157,78,221,.4);width:100%!important;transition:all .25s;
  margin-bottom: 12px;
}
[data-testid="stSidebar"] button[kind="primary"]:hover{
  transform:translateY(-2px);box-shadow:0 8px 28px rgba(157,78,221,.65);
}

/* tabs */
div[data-testid="stTabs"] button{
  color:var(--gray)!important;font-size:13px!important;
  font-weight:500!important;padding:8px 18px!important;
}
div[data-testid="stTabs"] button[aria-selected="true"]{
  color:var(--white)!important;border-bottom:2px solid var(--purple)!important;
}

/* tooltip & tickers ... (giữ nguyên của bạn) */
.ticker-wrap{overflow:hidden;background:#121217;padding:10px 0;border-radius:10px;border:1px solid var(--border);margin-bottom:22px;}
.ticker-inner{display:flex;width:max-content;animation:scroll 50s linear infinite;}
.ticker-inner:hover{animation-play-state:paused;}
@keyframes scroll{0%{transform:translateX(0)}100%{transform:translateX(-50%)}}
.t-item{display:flex;align-items:center;padding:0 20px;font-size:12px;color:var(--white);border-right:1px solid var(--border);gap:7px;white-space:nowrap;}
.t-name{font-weight:700;letter-spacing:.3px;}
.pos{color:#10b981;font-weight:600;}
.neg{color:#ef4444;font-weight:600;}

.tt-wrap{display:inline-flex;align-items:center;gap:7px;position:relative;cursor:default;}
.tt-icon{width:16px;height:16px;border-radius:50%;background:rgba(157,78,221,.2);border:1px solid rgba(157,78,221,.5);display:inline-flex;align-items:center;justify-content:center;font-size:9px;color:var(--purple);font-weight:800;flex-shrink:0;cursor:help;transition:background .15s;}
.tt-icon:hover{background:rgba(157,78,221,.45);}
.tt-box{visibility:hidden;opacity:0;position:absolute;left:50%;transform:translateX(-50%);bottom:calc(100% + 10px);background:#1a1625;border:1px solid rgba(157,78,221,.4);border-radius:10px;padding:12px 14px;width:270px;z-index:9999;font-size:12px;color:#d4d4d8;line-height:1.65;box-shadow:0 10px 40px rgba(0,0,0,.7);transition:opacity .15s;pointer-events:none;text-align:left;}
.tt-box::after{content:'';position:absolute;top:100%;left:50%;transform:translateX(-50%);border:6px solid transparent;border-top-color:#1a1625;}
.tt-wrap:hover .tt-box{visibility:visible;opacity:1;}

.sec-title{font-size:15px;font-weight:700;color:var(--white);margin:0 0 14px;display:flex;align-items:center;gap:7px;}

.c-row{display:flex;align-items:center;padding:10px 0;border-bottom:1px solid var(--border);gap:10px;}
.c-row:last-child{border-bottom:none;}
.c-ticker{font-weight:700;font-size:12px;color:var(--white);background:rgba(157,78,221,.18);padding:3px 8px;border-radius:5px;font-family:monospace;min-width:42px;text-align:center;}
.c-name{color:var(--gray);font-size:12px;flex:1;line-height:1.3;}
.c-pct{color:var(--purple);font-weight:700;font-size:13px;min-width:42px;text-align:right;}
.c-amt{color:var(--white);font-weight:700;font-size:12px;min-width:75px;text-align:right;}

/* Zone blocks (Reliable/Caution) */
.zone-r{
  background:rgba(16,185,129,.1);border:1px solid rgba(16,185,129,.4);
  color:#10b981;border-radius:12px;padding:14px;font-weight:600;font-size:14px;
  text-align:center; display:flex; align-items:center; justify-content:center; gap:8px;
  margin: 16px 0 20px 0; box-shadow: 0 4px 12px rgba(16,185,129,.05);
}
.zone-c{
  background:rgba(239,68,68,.1);border:1px solid rgba(239,68,68,.4);
  color:#ef4444;border-radius:12px;padding:14px;font-weight:600;font-size:14px;
  text-align:center; display:flex; align-items:center; justify-content:center; gap:8px;
  margin: 16px 0 20px 0; box-shadow: 0 4px 12px rgba(239,68,68,.05);
}

/* sidebar labels */
.sb-lbl{color:var(--gray);font-size:11px;text-transform:uppercase;letter-spacing:.8px;margin:18px 0 8px;font-weight:600;font-family:'Inter',sans-serif;}

/* upgrade block */
.upg{
  background: linear-gradient(135deg, #8B5CF6 0%, #4C1D95 100%);
  border-radius:16px;padding:24px 16px;text-align:center;
  margin-top:32px; margin-bottom: 20px;
  box-shadow:0 10px 30px -5px rgba(139,92,246,.5);
  border: 1px solid rgba(157,78,221,.3);
  position: relative; overflow: hidden;
}
.upg::before {
  content: ''; position: absolute; top: -50%; left: -50%; width: 200%; height: 200%;
  background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 60%);
  opacity: 0.5; pointer-events: none;
}
.upg .ic{font-size:32px;margin-bottom:8px; display:inline-block; filter: drop-shadow(0 4px 6px rgba(0,0,0,0.3));}
.upg h4{margin:0 0 6px 0;font-size:16px;font-weight:800;color:#fff; letter-spacing: 0.5px;}
.upg p{margin:0;font-size:12px;color:rgba(255,255,255,.75); line-height:1.5;}

/* scrollbar */
::-webkit-scrollbar{width:4px;}
::-webkit-scrollbar-track{background:var(--panel);}
::-webkit-scrollbar-thumb{background:var(--purple3);border-radius:2px;}

/* inputs */
[data-testid="stSelectbox"]>div,[data-testid="stNumberInput"] input{
  background:rgba(255,255,255,.04)!important;
  border:1px solid var(--border)!important;border-radius:10px!important;color:var(--white)!important;
  padding: 4px 8px !important;
}

/* divider */
.hline{height:1px;background:var(--border);margin:24px 0;}
</style>
""", unsafe_allow_html=True)


# ── TOOLTIP HELPER ──────────────────────────────────────────
def tt(label, tip, size="15px"):
    return f"""<div class="sec-title" style="font-size:{size}">
        {label}
        <div class="tt-wrap">
            <div class="tt-icon">i</div>
            <div class="tt-box">{tip}</div>
        </div>
    </div>"""


# ── CONSTANTS ───────────────────────────────────────────────
RFR = 0.045
CAP = 0.15
DATA_DIR = Path(__file__).parent / "portfolio_data" / "processed"


COMPANY = {
    "ANV":"Nam Viet Corp","BIC":"BIDV Insurance","BID":"BIDV",
    "BMI":"Bao Minh Insurance","CII":"Ho Chi Minh Infrastructure","CMG":"CMC Corp",
    "CSV":"Southern Chemicals","CTD":"Coteccons","DBC":"Dabaco Group",
    "DGW":"Digiworld","DPM":"DPM Fertilizer","DXG":"Dat Xanh Group",
    "ELC":"Elcom Corp","FPT":"FPT Corp","GAS":"PV GAS",
    "GIL":"Gilimex","HAH":"Hai An Logistics","HAX":"Haxaco",
    "HCM":"HSC Securities","HPG":"Hoa Phat Group","IMP":"Imexpharm",
    "KDC":"Kido Group","KDH":"Khang Dien","LCG":"Licogi 16",
    "LSS":"Lam Son Sugar","MBB":"MB Bank","NKG":"Nam Kim Steel",
    "PC1":"PC1 Group","PNJ":"PNJ Gold","PVT":"PVTrans",
    "REE":"REE Corp","SAB":"Sabeco","SSI":"SSI Securities",
    "STK":"Century Fiber","TCM":"Thanh Cong Textile","VCB":"Vietcombank",
    "VCI":"Vietcap Securities","VIC":"Vingroup","VSC":"Viconship",
}


# ── LOAD ARTIFACTS ──────────────────────────────────────────
@st.cache_resource
def load_data():
    try:
        with open(DATA_DIR/"meta.json") as f: meta=json.load(f)
        mu  = np.load(DATA_DIR/"mu.npy")
        sig = np.load(DATA_DIR/"sigma.npy")
        ev  = np.load(DATA_DIR/"eigenvalues.npy")
        evc = np.load(DATA_DIR/"eigenvectors.npy")
        tickers = meta["tickers"]          # ← use order from meta.json
        return meta, tickers, mu, sig, ev, evc, True
    except Exception:
        n=39; np.random.seed(42)
        tickers = sorted(COMPANY.keys())   # sorted ABC for mock
        meta={"tickers":tickers,"n_tickers":n,"date_start":"2018-01-03",
              "date_end":"2026-04-24","risk_free_rate":RFR,"t_days":2090}
        mu=np.linspace(0.05,0.17,n)
        vols=np.random.uniform(0.18,0.35,n)
        corr=np.full((n,n),0.32); np.fill_diagonal(corr,1.0)
        for i in range(0,n,4):
            for j in range(i,min(i+4,n)):
                for k in range(i,min(i+4,n)): corr[j,k]=corr[k,j]=0.62
        np.fill_diagonal(corr,1.0)
        D=np.diag(vols); sig=D@corr@D+np.eye(n)*.001
        ev,evc=np.linalg.eigh(sig); idx=np.argsort(ev)[::-1]
        return meta, tickers, mu, sig, ev[idx], evc[:,idx], False


META, TICKERS, MU_RAW, SIGMA_RAW, EIGENVALUES, EIGENVECTORS, IS_REAL = load_data()
N = len(TICKERS)


# ── OPTIMIZER ───────────────────────────────────────────────
def optimize(r_target):
    from scipy.optimize import minimize
    mu=MU_RAW.copy(); mu_bar=np.mean(mu)
    avg_var=252*np.mean(np.diag(SIGMA_RAW))/META["t_days"]
    sq=np.sum((mu-mu_bar)**2)
    alpha=max(0.05,1-((N-3)*avg_var)/sq) if sq>1e-10 else 0.05
    mu_js=mu_bar+min(alpha,1.)*(mu-mu_bar)
    sr=SIGMA_RAW+0.05*np.eye(N)
    res=minimize(lambda w:w@sr@w, np.ones(N)/N, method="SLSQP",
        jac=lambda w:2*sr@w, bounds=[(0.,CAP)]*N,
        constraints=[{"type":"eq","fun":lambda w:w@mu_js-r_target},
                     {"type":"eq","fun":lambda w:np.sum(w)-1.}],
        options={"maxiter":1000,"ftol":1e-12})
    if not res.success: return None
    w=res.x; pv=float(np.sqrt(w@sr@w)); pr=float(mu_js@w)
    neff=float(1./np.sum(w**2)); sh=(pr-RFR)/pv if pv>0 else 0.
    # key fix: use TICKERS list that matches MU_RAW index
    return {"weights":{TICKERS[i]:round(float(w[i]),6) for i in range(N)},
            "port_ret":round(pr,5),"port_vol":round(pv,5),
            "sharpe":round(sh,4),"n_eff":round(neff,3),
            "zone":"reliable" if neff>=3. else "caution"}


@st.cache_data
def frontier(n_pts=60):
    from scipy.optimize import minimize
    sr=SIGMA_RAW+0.05*np.eye(N); mu_js=MU_RAW.copy()
    rmin=float(MU_RAW.min())+.005; rmax=float(np.sort(MU_RAW)[-3])*.88
    pts=[]
    for r in np.linspace(rmin,rmax,n_pts):
        res=minimize(lambda w:w@sr@w,np.ones(N)/N,method="SLSQP",
        jac=lambda w:2*sr@w,bounds=[(0,CAP)]*N,
        constraints=[{"type":"eq","fun":lambda w,r=r:w@mu_js-r},
                     {"type":"eq","fun":lambda w:np.sum(w)-1.}],
        options={"maxiter":500,"ftol":1e-10})
        if res.success:
            w=res.x; v=float(np.sqrt(w@sr@w)); sh=(r-RFR)/v if v>0 else 0.
            pts.append({"ret":round(r*100,2),"vol":round(v*100,2),
                        "sharpe":round(sh,3),"reliable":(1/np.sum(w**2))>=3.})
    return pts


def fmt_money(val,cur):
    if val>=1e9: s=f"{val/1e9:,.2f}B"
    elif val>=1e6: s=f"{val/1e6:,.2f}M"
    elif val>=1e3: s=f"{val/1e3:,.1f}K"
    else: s=f"{val:,.0f}"
    return f"{s} {cur}"


CHART_BG = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#a1a1aa",family="Inter,sans-serif"))
GRID     = dict(gridcolor="rgba(255,255,255,.05)")


# ── SIDEBAR ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:10px 0 28px">
      <div style="font-size:26px;font-weight:900;color:#fff;letter-spacing:-1px">
        ✨ ChainX <span style="color:#9D4EDD">Optima</span>
      </div>
      <div style="font-size:11px;color:#a1a1aa;margin-top:6px;letter-spacing:.6px;font-weight:500">
        Linear Algebra · Vietnam Equities
      </div>
      <div style="font-size:12px;color:#fff;margin-top:8px;font-weight:700">
        Thao · Ngan · Nhu
      </div>
    </div>""", unsafe_allow_html=True)

    menu = st.radio("", [
        "🎛️  Dashboard","📈  Market Summary",
        "💼  Portfolio","📉  Analytics",
        "⚙️  Settings","🎧  Help & Support",
    ], label_visibility="collapsed")

    if menu=="🎛️  Dashboard":
        st.markdown("<div class='sb-lbl'>Investment Horizon</div>", unsafe_allow_html=True)
        st.selectbox("",["Jan 2018 – Apr 2026","Jan 2020 – Apr 2026","Jan 2022 – Apr 2026"],
                     label_visibility="collapsed")

        st.markdown("<div class='sb-lbl'>Initial Capital</div>", unsafe_allow_html=True)
        ca,cu=st.columns([3,1])
        with ca: inv=st.number_input("",min_value=1.,value=100_000_000.,step=1_000_000.,
                                      format="%.0f",label_visibility="collapsed")
        with cu: cur=st.selectbox("",["VND","USD"],label_visibility="collapsed")

        st.markdown("<div class='sb-lbl'>Target Return</div>", unsafe_allow_html=True)
        r_pct=st.slider("",9,28,15,label_visibility="collapsed")
        r_tgt=r_pct/100.

        # Khối Reliable / Caution được đẩy sát xuống dưới input và có thêm padding
        if r_pct<=19:
            st.markdown(f"<div class='zone-r'>✅ <span style='color:#e4e4e7'>Reliable - {r_pct}%</span></div>",unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='zone-c'>⚠️ <span style='color:#e4e4e7'>Caution - {r_pct}%</span></div>",unsafe_allow_html=True)

        go_btn=st.button("Optimize Portfolio",use_container_width=True,type="primary")

    st.markdown("""
    <div class="upg">
      <div class="ic">👑</div>
      <h4>Upgrade</h4>
      <p>Unlock all ChainX Optima features</p>
    </div>""", unsafe_allow_html=True)


# ── DASHBOARD ───────────────────────────────────────────────
if menu=="🎛️  Dashboard":


    # greeting
    h=datetime.now().hour
    greet,icon=(("Good morning","☀️") if 5<=h<12 else
                ("Good afternoon","🌤️") if 12<=h<18 else
                ("Good evening","🌙") if 18<=h<22 else
                ("Good night","🌟"))
    badge=("<span style='background:rgba(16,185,129,.12);color:#10b981;font-size:11px;padding:4px 12px;border-radius:20px;font-weight:700;border:1px solid rgba(16,185,129,.35)'>● LIVE DATA</span>"
           if IS_REAL else
           "<span style='background:rgba(239,68,68,.1);color:#f87171;font-size:11px;padding:4px 12px;border-radius:20px;font-weight:700;border:1px solid rgba(239,68,68,.3)'>● MOCK DATA</span>")


    st.markdown(f"""
    <div style="display:flex;justify-content:space-between;align-items:flex-start">
      <div>
        <h2 style="margin:0;font-size:26px;font-weight:800;color:#fff">
          {greet} {icon}, Welcome back!
        </h2>
        <p style="color:#a1a1aa;font-size:13px;margin:5px 0 0">
          Here is the latest update on your optimized portfolio based on historical data.
        </p>
      </div>
      <div style="display:flex;align-items:center;gap:10px;margin-top:4px;flex-shrink:0">
        {badge}
        <span style="color:#52525b;font-size:11px">{datetime.now().strftime("%b %d, %Y · %H:%M")}</span>
      </div>
    </div>
    <div class="hline"></div>""", unsafe_allow_html=True)


    # ticker
    td=sorted(zip(TICKERS,MU_RAW),key=lambda x:x[1],reverse=True)
    mixed=td[:10]+td[-10:]; random.seed(datetime.now().strftime("%Y-%m-%d")); random.shuffle(mixed)
    items="".join(
        f'<div class="t-item">'
        f'<img src="https://ui-avatars.com/api/?name={t}&background=4C1D95&color=fff&rounded=true&size=26&font-size=0.4" width="22" height="22" style="border-radius:5px">'
        f'<span class="t-name">{t}</span>'
        f'<span class="{"pos" if m*100>=0 else "neg"}">{"▲ +" if m*100>=0 else "▼ "}{m*100:.1f}%</span>'
        f'</div>'
        for t,m in mixed)
    st.markdown(f'<div class="ticker-wrap"><div class="ticker-inner">{items}{items}</div></div>',
                unsafe_allow_html=True)


    # optimizer
    if "res" not in st.session_state: st.session_state.res=None
    if go_btn or st.session_state.res is None:
        with st.spinner("⚙️  Solving quadratic program…"):
            st.session_state.res=optimize(r_tgt)


    R=st.session_state.res
    if R is None:
        st.error("⚠️  Optimizer failed to converge — try lowering Target Return.")
        st.stop()


    # metrics row
    m1,m2,m3,m4=st.columns(4)
    with m1: st.metric("Expected Return",   f"{R['port_ret']*100:.1f}%","Optimal")
    with m2: st.metric("Portfolio Risk (σ)",f"{R['port_vol']*100:.1f}%","Controlled")
    with m3: st.metric("Sharpe Ratio",       f"{R['sharpe']:.2f}",       "AI Assessed")
    with m4:
        lbl="Reliable ✅" if R["zone"]=="reliable" else "Caution ⚠️"
        st.metric("Effective Positions",f"{R['n_eff']:.1f}",lbl)


    st.markdown("<div style='height:18px'></div>",unsafe_allow_html=True)


    # ── TABS
    tw,tf,te=st.tabs(["📊  Portfolio Weights","📈  Efficient Frontier","🔬  Eigenanalysis"])


    # ════ TAB 1 ═══════════════════════════════════
    with tw:
        wts    = R["weights"]
        sw     = sorted(wts.items(),key=lambda x:x[1],reverse=True)
        top    = [(t,w) for t,w in sw if w>0.001]
        tl     = [t for t,_ in top]
        ow     = [round(w*100,1) for _,w in top]
        H      = 440
        purple_cs = [[0,"#3b0764"],[.5,"#7c3aed"],[1,"#d8b4fe"]]


        # row 1: bar + pie
        c1,c2=st.columns(2)
        with c1:
            with st.container(border=True):
                st.markdown(tt("Optimal Allocation",
                    "Optimal weights from <b>Quadratic Program</b>: minimize wᵀΣw s.t. μᵀw = r*, Σw = 1, 0 ≤ wᵢ ≤ 15%. Optimizer allocates capital to achieve target return with minimum risk."),
                    unsafe_allow_html=True)
                fig=go.Figure(go.Bar(
                    y=tl,x=ow,orientation="h",
                    marker=dict(color=ow,colorscale=purple_cs,showscale=False,line=dict(width=0)),
                    text=[f"{v}%" for v in ow],textposition="outside",
                    textfont=dict(color="#fff",size=11)))
                fig.update_layout(height=H,margin=dict(l=0,r=55,t=4,b=20),
                    xaxis=dict(title="Weight (%)",**GRID,color="#a1a1aa"),
                    yaxis=dict(tickfont=dict(family="monospace",color="#fff",size=11),**GRID),
                    **CHART_BG)
                st.plotly_chart(fig,use_container_width=True)


        with c2:
            with st.container(border=True):
                st.markdown(tt("Distribution",
                    "<b>Donut chart</b> shows capital allocation. Good portfolio has no stock > 15% (optimizer hard cap). <b>High N_eff</b> = diversified allocation, lower risk."),
                    unsafe_allow_html=True)
                pl=[t for t,w in top if w>=.015]; pv=[round(w*100,1) for t,w in top if w>=.015]
                oth=sum(w for _,w in top if w<.015)
                if oth>.001: pl.append("Others"); pv.append(round(oth*100,1))
                nc=len(pv)
                cpie=px.colors.sample_colorscale("Purples_r",[i/max(1,nc-1) for i in range(nc)])
                fig2=go.Figure(go.Pie(labels=pl,values=pv,hole=.6,
                    textinfo="label+percent",textfont=dict(color="#fff",size=11),
                    marker=dict(colors=cpie,line=dict(color="#15151a",width=2))))
                fig2.update_layout(height=H,margin=dict(l=20,r=80,t=4,b=20),
                    showlegend=False,**CHART_BG)
                st.plotly_chart(fig2,use_container_width=True)


        st.markdown("<div style='height:10px'></div>",unsafe_allow_html=True)


        # row 2: tables
        r1,r2=st.columns(2)
        rem=sum(w for _,w in top[10:]); has_oth=rem>.001


        with r1:
            with st.container(border=True):
                st.markdown(tt("Top Holdings",
                    "Selected stocks sorted by weight descending. Stocks with weight < 0.1% are excluded as optimizer assigns ≈ 0 weight — they don't help portfolio optimization."),
                    unsafe_allow_html=True)
                rows="".join(f"""<div class="c-row">
                    <span class="c-ticker">{t}</span>
                    <span class="c-name">{COMPANY.get(t,'')}</span>
                    <span class="c-pct">{w*100:.1f}%</span>
                    </div>""" for t,w in top[:10])
                st.markdown(f"<div style='padding:0 6px'>{rows}</div>",unsafe_allow_html=True)


        with r2:
            with st.container(border=True):
                st.markdown(tt("Allocated Amount",
                    "Specific dollar amounts per stock. Formula: <b>Amount = weight × initial capital</b>. Example: 15% weight × 100M VND = buy 15M VND worth. This is a suggestion — not financial advice."),
                    unsafe_allow_html=True)
                vrows="".join(f"""<div class="c-row">
                    <span class="c-ticker">{t}</span>
                    <span class="c-pct">{w*100:.1f}%</span>
                    <span class="c-amt">{fmt_money(inv*w,cur)}</span>
                    </div>""" for t,w in top[:10])
                if has_oth:
                    vrows+=f'<div class="c-row"><span class="c-ticker" style="opacity:.4">···</span><span class="c-pct" style="color:#52525b">{rem*100:.1f}%</span><span class="c-amt" style="color:#52525b">{fmt_money(inv*rem,cur)}</span></div>'
                st.markdown(f"<div style='padding:0 6px'>{vrows}</div>",unsafe_allow_html=True)


        st.markdown("<div style='height:10px'></div>",unsafe_allow_html=True)


        # row 3: GMV
        with st.container(border=True):
            st.markdown(tt("Global Minimum Variance (GMV) Portfolio",
                "Lowest risk portfolio — corresponds to <b>minimum eigenvalue eigenvector</b> of Σ. This is the leftmost point of Efficient Frontier, independent of target return."),
                unsafe_allow_html=True)
            vm=np.abs(EIGENVECTORS[:,-1]); vm/=vm.sum()
            tv=sorted(zip(TICKERS,vm),key=lambda x:x[1],reverse=True)[:12]
            vw=[round(w*100,1) for _,w in tv]
            fig3=go.Figure(go.Bar(
                y=[t for t,_ in tv],x=vw,orientation="h",
                marker=dict(color=vw,colorscale=purple_cs,showscale=False,line=dict(width=0)),
                text=[f"{v}%" for v in vw],textposition="outside",
                textfont=dict(color="#fff",size=11)))
            fig3.update_layout(height=340,margin=dict(l=0,r=55,t=4,b=20),
                xaxis=dict(title="Weight (%)",**GRID,color="#a1a1aa"),
                yaxis=dict(tickfont=dict(family="monospace",color="#fff",size=11)),
                **CHART_BG)
            st.plotly_chart(fig3,use_container_width=True)


    # ════ TAB 2 ═══════════════════════════════════
    with tf:
        with st.container(border=True):
            st.markdown(tt("Efficient Frontier",
                "Each point = optimal portfolio. <b>★</b> = your portfolio. Dark purple = high Sharpe ratio. Red circles = Caution zone (N_eff < 3, too concentrated). All points below curve are suboptimal."),
                unsafe_allow_html=True)
            with st.spinner("Computing…"):
                fr=frontier()
            if fr:
                rel=[p for p in fr if p["reliable"]]; cau=[p for p in fr if not p["reliable"]]
                ff=go.Figure()
                if rel:
                    ff.add_trace(go.Scatter(
                        x=[p["vol"] for p in rel],y=[p["ret"] for p in rel],mode="markers",
                        marker=dict(color=[p["sharpe"] for p in rel],colorscale="Purples",size=10,
                            colorbar=dict(title="Sharpe",thickness=10,len=.55,
                            tickfont=dict(color="#a1a1aa"),title_font=dict(color="#a1a1aa"))),
                        name="Reliable",hovertemplate="Return:%{y:.1f}%<br>Risk:%{x:.1f}%<extra></extra>"))
                if cau:
                    ff.add_trace(go.Scatter(
                        x=[p["vol"] for p in cau],y=[p["ret"] for p in cau],mode="markers",
                        marker=dict(color="#ef4444",size=8,symbol="circle-open",line=dict(width=2)),
                        name="Caution",hovertemplate="Return:%{y:.1f}%<br>Risk:%{x:.1f}%<extra></extra>"))
                ff.add_trace(go.Scatter(
                    x=[R["port_vol"]*100],y=[R["port_ret"]*100],mode="markers",
                    marker=dict(color="#fff",size=20,symbol="star",line=dict(color="#9D4EDD",width=2)),
                    name="Your portfolio",
                    hovertemplate=f"Return:{R['port_ret']*100:.1f}%<br>Risk:{R['port_vol']*100:.1f}%<br>Sharpe:{R['sharpe']:.2f}<extra></extra>"))
                ff.update_layout(height=460,
                    xaxis=dict(title="Volatility (%)",**GRID,color="#a1a1aa"),
                    yaxis=dict(title="Return (%)",**GRID,color="#a1a1aa"),
                    legend=dict(orientation="h",y=-0.14,x=0,font=dict(color="#fff",size=12)),
                    margin=dict(l=0,r=0,t=4,b=40),**CHART_BG)
                st.plotly_chart(ff,use_container_width=True)


    # ════ TAB 3 ═══════════════════════════════════
    with te:
        with st.container(border=True):
            st.markdown(tt("Eigenvalue Spectrum",
                "39 eigenvalues of Σ = QΛQᵀ. <b>Purple = Signal</b> (λ > M-P threshold, real market risk). <b>Gray = Noise</b> (statistical noise). <b>Light purple = λ_min</b> → GMV portfolio. <b>Red dashed</b> = Marchenko-Pastur λ+ threshold separating signal from noise."),
                unsafe_allow_html=True)
            n=N; T=META["t_days"]; mp=((1+np.sqrt(n/T))**2)/T
            ev=EIGENVALUES[:n]
            cols=["#c4b5fd" if i==len(ev)-1 else ("#9D4EDD" if v>mp else "#3f3f46")
                  for i,v in enumerate(ev)]
            labs=["λ₁ market" if i==0 else (f"λ{n} min var" if i==n-1
            else (f"λ{i+1}" if i%5==0 else "")) for i in range(n)]
            fe=go.Figure()
            fe.add_trace(go.Bar(x=labs,y=ev,marker_color=cols,marker_line_width=0,
                hovertemplate="λ = %{y:.6f}<extra></extra>"))
            fe.add_hline(y=mp,line=dict(color="#ef4444",width=2,dash="dash"),
                annotation_text=f"M-P λ+ = {mp:.5f}",
                annotation_position="top right",
                annotation_font=dict(color="#ef4444",size=11))
            fe.update_layout(height=420,
                xaxis=dict(title="Eigenvalue index",**GRID,color="#a1a1aa",tickangle=0),
                yaxis=dict(title="Magnitude",**GRID,color="#a1a1aa"),
                showlegend=False,margin=dict(l=0,r=0,t=4,b=40),**CHART_BG)
            st.plotly_chart(fe,use_container_width=True)


            lc1,lc2,lc3,lc4=st.columns(4)
            def leg(col,txt): return f"<span style='color:{col};font-size:16px;line-height:1'>■</span> <span style='color:#a1a1aa;font-size:12px'>{txt}</span>"
            with lc1: st.markdown(leg("#9D4EDD","Signal (real market risk)"),unsafe_allow_html=True)
            with lc2: st.markdown(leg("#3f3f46","Noise (below M-P)"),unsafe_allow_html=True)
            with lc3: st.markdown(leg("#c4b5fd","Min-var direction"),unsafe_allow_html=True)
            with lc4: st.markdown("<span style='color:#ef4444;font-size:13px'>- - -</span> <span style='color:#a1a1aa;font-size:12px'>Marchenko-Pastur λ+</span>",unsafe_allow_html=True)
            st.info(f"**{int(np.sum(ev>mp))} signal factors** / {n} total · T/N = {T/n:.1f} (higher = more reliable covariance estimate)")


else:
    name=menu.split("  ",1)[-1]
    st.markdown(f"<h2 style='color:#fff;margin-top:1rem'>{name}</h2>",unsafe_allow_html=True)
    st.markdown("<p style='color:#a1a1aa'>Module under development. Select <b>Dashboard</b> to use portfolio optimizer.</p>",unsafe_allow_html=True)