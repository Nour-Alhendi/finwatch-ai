"""
FinWatch AI — Main Entry Point
================================
Run:  streamlit run finwatch/app.py

This file wires all sub-modules together.
dashboard.py is left untouched; this is the modular equivalent.
"""

import sys
from pathlib import Path

# Allow imports like `from data.loader import ...` when running from project root
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

st.set_page_config(page_title="FinWatch AI", page_icon="📡", layout="wide")

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

/* ── Base ── */
html,body,[class*="css"]{
    font-family:'Inter',sans-serif;
    background:#060a0f;
    color:#e2e8f0;
}
/* Dot-grid background for depth */
[data-testid="stAppViewContainer"]{
    background:
        radial-gradient(ellipse 80% 50% at 10% 0%, rgba(29,233,182,0.04) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 90% 100%, rgba(88,166,255,0.03) 0%, transparent 60%),
        radial-gradient(rgba(26,35,50,0.55) 1px, transparent 1px)
        #060a0f;
    background-size:auto, auto, 26px 26px;
}

/* ── Sidebar ── */
[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#080d15 0%,#060a0f 100%)!important;
    border-right:1px solid rgba(30,45,65,0.8)!important;
    width:220px!important;min-width:220px!important;
}
[data-testid="stSidebar"] > div:first-child{width:220px!important;min-width:220px!important}
[data-testid="stSidebar"] .block-container{padding-top:0.6rem!important}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"]{gap:0px!important}
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"]{gap:2px!important;margin-bottom:0!important}
[data-testid="stSidebar"] .stButton{margin:0!important;padding:0!important}

[data-testid="stSidebar"] button{
    background:transparent!important;
    border:none!important;
    color:#637a91!important;
    font-family:'IBM Plex Mono',monospace!important;
    font-size:11px!important;
    text-align:left!important;
    padding:2px 10px!important;
    border-radius:0!important;
    border-left:2px solid transparent!important;
    width:100%!important;
    min-height:24px!important;
    line-height:1.4!important;
    transition:all 0.15s ease!important;
}
[data-testid="stSidebar"] button:hover{
    background:rgba(29,233,182,0.05)!important;
    color:#cdd9e5!important;
    border-left:2px solid rgba(29,233,182,0.3)!important;
}

/* ── Tooltips (JS-driven, see _comp.html below) ── */
.tip,
.an-text[data-tip]{
    border-bottom:1px dotted rgba(88,166,255,0.6);
    cursor:help;
}
#fw-tooltip{
    position:fixed;
    background:#1c2a3a;
    border:1px solid #3a5068;
    border-radius:6px;
    box-shadow:0 4px 20px rgba(0,0,0,0.8);
    padding:6px 12px;
    font-size:11px;
    font-family:'IBM Plex Mono',monospace;
    font-weight:400;
    color:#ffffff;
    line-height:1.5;
    max-width:260px;
    pointer-events:none;
    opacity:0;
    transition:opacity 0.12s ease;
    z-index:999999;
}

/* ── Main layout ── */
.main .block-container{padding:0 1.2rem 2rem!important;padding-top:0!important;max-width:100%!important}
#MainMenu,footer,header{display:none!important}
[data-testid="stToolbar"]{display:none!important}
[data-testid="stHeader"]{display:none!important}
[data-testid="stDecoration"]{display:none!important}
section.main > div{padding-top:0!important}

/* ── Logo ── */
.logo{
    font-family:'IBM Plex Mono',monospace;
    font-size:17px;font-weight:700;
    padding:14px 0 10px;
    letter-spacing:0.5px;
    background:linear-gradient(135deg,#e2e8f0 0%,#1de9b6 40%,#58a6ff 75%,#a371f7 100%);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    background-clip:text;
    display:inline-block;
}
.logo span{
    /* gradient covers both words */
}
.sb-label{
    font-size:8px;letter-spacing:2.5px;color:#3d5266;
    text-transform:uppercase;padding:8px 0 4px;
    font-family:'IBM Plex Mono',monospace;
}

/* ── Stock Header ── */
.stock-header{
    padding:12px 0 10px;
    display:flex;align-items:center;gap:12px;
    border-bottom:1px solid rgba(30,45,65,0.8);
    margin-bottom:2px;
}
[data-testid="stVerticalBlock"]>[data-testid="element-container"]{margin-bottom:0!important}
div[data-testid="element-container"]>.stMarkdown{margin-bottom:0!important}
.sh-name{font-size:34px;font-weight:700;color:#e2e8f0;letter-spacing:-0.8px;line-height:1.1}
.sh-sub{font-size:11px;color:#4e5f72;font-family:'IBM Plex Mono',monospace;letter-spacing:1px;margin-top:3px}
.sh-price{font-size:22px;font-weight:500;color:#e2e8f0;margin-left:auto;font-family:'IBM Plex Mono',monospace}
.sh-up{font-size:12px;color:#1de9b6;font-family:'IBM Plex Mono',monospace;text-shadow:0 0 12px rgba(29,233,182,0.4)}
.sh-dn{font-size:12px;color:#f85149;font-family:'IBM Plex Mono',monospace;text-shadow:0 0 12px rgba(248,81,73,0.4)}

/* ── Watchlist changes ── */
.wl-chg-up{color:#1de9b6;font-family:'IBM Plex Mono',monospace;font-size:10px;text-align:right;line-height:1.8;padding-top:3px}
.wl-chg-dn{color:#f85149;font-family:'IBM Plex Mono',monospace;font-size:10px;text-align:right;line-height:1.8;padding-top:3px}
.wl-chg-fl{color:#3d5266;font-family:'IBM Plex Mono',monospace;font-size:10px;text-align:right;line-height:1.8;padding-top:3px}

/* ── Chart label ── */
.chart-label{
    font-size:9px;letter-spacing:2px;color:#3d5266;
    text-transform:uppercase;margin-bottom:6px;
    font-family:'IBM Plex Mono',monospace;
}

/* ── Cards / Panels ── */
.rp-section{
    padding:12px 0;
    border-bottom:1px solid rgba(30,45,65,0.6);
}
.rp-title{
    font-size:8px;letter-spacing:2.5px;color:#3d5266;
    text-transform:uppercase;margin-bottom:10px;
    font-family:'IBM Plex Mono',monospace;
}
.sev-big{font-size:17px;font-weight:600;margin-bottom:2px;font-family:'IBM Plex Mono',monospace;letter-spacing:-0.5px}

/* Severity colors + glow */
.s-critical{color:#f85149;text-shadow:0 0 16px rgba(248,81,73,0.35)}
.s-warning{color:#e3b341;text-shadow:0 0 14px rgba(227,179,65,0.3)}
.s-watch{color:#58a6ff;text-shadow:0 0 12px rgba(88,166,255,0.25)}
.s-normal{color:#2ea043}.s-positive{color:#2ea043}.s-review{color:#a371f7}

.rp-row{display:flex;justify-content:space-between;align-items:center;padding:3px 0}
.rp-key{font-size:10px;color:#3d5266;font-family:'IBM Plex Mono',monospace}
.rp-val{font-size:10px;color:#c9d1d9;font-family:'IBM Plex Mono',monospace}
.rp-up{color:#1de9b6!important}.rp-dn{color:#f85149!important}
.dot-row{display:flex;gap:4px;align-items:center}
.d{width:7px;height:7px;border-radius:50%;display:inline-block}
.d-on{background:#2ea043;box-shadow:0 0 6px rgba(46,160,67,0.6)}
.d-off{background:#1a2332}

/* ── News ── */
.news-item{display:block;width:100%;padding:7px 0;border-bottom:1px solid rgba(30,45,65,0.5)}
.news-item:last-child{border-bottom:none}
.news-sent{font-size:8px;padding:1px 6px;border-radius:3px;display:inline-block;margin-bottom:4px;font-family:'IBM Plex Mono',monospace;letter-spacing:0.5px}
.s-neg{background:rgba(248,81,73,0.12);color:#f85149;border:1px solid rgba(248,81,73,0.2)}
.s-pos{background:rgba(46,160,67,0.12);color:#2ea043;border:1px solid rgba(46,160,67,0.2)}
.s-neu{background:rgba(30,45,65,0.6);color:#637a91;border:1px solid rgba(30,45,65,0.8)}
.news-text{font-size:10px;color:#637a91;line-height:1.5}
.summary-text{font-size:10px;color:#637a91;line-height:1.7}

/* ── Analysis Panel ── */
.analysis-panel{
    background:rgba(11,17,26,0.7);
    border:1px solid rgba(30,45,65,0.6);
    border-radius:10px;
    padding:14px 18px;
    margin-top:10px;
    box-shadow:0 4px 20px rgba(0,0,0,0.35),inset 0 1px 0 rgba(255,255,255,0.03);
    backdrop-filter:blur(6px);
}
.an-header{
    font-size:8px;letter-spacing:2.5px;text-transform:uppercase;
    color:#3d5266;border-bottom:1px solid rgba(30,45,65,0.6);
    padding-bottom:8px;margin-bottom:12px;font-family:'IBM Plex Mono',monospace;
}
.an-date{color:#3d5266;margin-left:10px;font-size:8px}
.an-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}
.an-title{font-size:7px;letter-spacing:2px;text-transform:uppercase;color:#3d5266;margin-bottom:5px;border-bottom:1px solid rgba(30,45,65,0.4);padding-bottom:3px;font-family:'IBM Plex Mono',monospace}
.an-row{display:flex;align-items:flex-start;gap:5px;padding:2px 0}
.an-pos{color:#1de9b6;font-size:11px;flex-shrink:0;line-height:18px;font-family:'IBM Plex Mono',monospace}
.an-risk{color:#f85149;font-size:11px;flex-shrink:0;line-height:18px;font-family:'IBM Plex Mono',monospace}
.an-warn{color:#e3b341;font-size:11px;flex-shrink:0;line-height:18px;font-family:'IBM Plex Mono',monospace}
.an-neu{color:#3d5266;font-size:11px;flex-shrink:0;line-height:18px;font-family:'IBM Plex Mono',monospace}
.an-text{font-size:12px;color:#b8c4ce;line-height:1.8;font-family:'IBM Plex Mono',monospace}

/* ── Tooltip ── */
[data-tip]{position:relative;cursor:help;border-bottom:1px dotted rgba(99,122,145,0.4)}
[data-tip]::after{
    content:attr(data-tip);position:absolute;left:0;top:120%;
    background:#0f1923;color:#c9d1d9;font-size:10px;line-height:1.6;
    padding:7px 12px;border-radius:6px;white-space:normal;
    max-width:240px;min-width:140px;
    border:1px solid rgba(30,45,65,0.8);
    box-shadow:0 8px 24px rgba(0,0,0,0.5);
    z-index:9999;visibility:hidden;opacity:0;
    transition:opacity 0.15s ease;pointer-events:none;
    font-family:'Inter',sans-serif;font-weight:400;
}
[data-tip]:hover::after{visibility:visible;opacity:1}

/* ── Segmented control ── */
[data-testid="stSegmentedControl"] > div{
    background:rgba(11,17,26,0.8)!important;
    border:1px solid rgba(30,45,65,0.7)!important;
    border-radius:6px!important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar{width:4px;height:4px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:rgba(30,45,65,0.8);border-radius:2px}
::-webkit-scrollbar-thumb:hover{background:rgba(48,73,104,0.9)}

/* ── Escalate alert buttons ── */
.escalate-btn button{
    color:#f85149!important;
    border-left:2px solid rgba(248,81,73,0.5)!important;
    background:rgba(248,81,73,0.05)!important;
}
.escalate-btn button:hover{
    background:rgba(248,81,73,0.12)!important;
    border-left:2px solid rgba(248,81,73,0.8)!important;
    color:#ff6b6b!important;
}
</style>
""", unsafe_allow_html=True)

# ── Imports (after sys.path is set) ───────────────────────────────────────────
from data.loader import (
    COMPANY_NAMES, SECTORS, ETF_STOCKS, SEV_CLS,
    load_decisions, load_detection, load_news, load_spx, load_price_summary,
)
from ui.components import (
    render_sidebar, render_stock_header, render_spx_header,
    render_risk_news_row, render_anomaly_selector,
    render_analysis_panel, render_investor_summary, render_llm_report,
    render_strategy_box, show_analysis_modal, render_candle_panel,
)
from ui.charts import render_price_chart, render_rsi_chart, render_spx_chart
from ui.portfolio_page import render_portfolio_page
from data.portfolio import load_portfolios

# ── Session State ─────────────────────────────────────────────────────────────
decisions   = load_decisions()
news_df     = load_news()
price_data  = load_price_summary()
all_tickers = decisions["ticker"].tolist()


def _find_sector(t):
    for s, ts in SECTORS.items():
        if t in ts: return s
    return list(SECTORS.keys())[0]


if "selected"        not in st.session_state: st.session_state.selected        = all_tickers[0]
if "period"          not in st.session_state: st.session_state.period          = "1M"
if "spx_period"      not in st.session_state: st.session_state.spx_period      = "1M"
if "category"        not in st.session_state: st.session_state.category        = "Stocks"
if "anomaly_date"    not in st.session_state: st.session_state.anomaly_date    = None
if "language"        not in st.session_state: st.session_state.language        = "english"
if "llm_cache"       not in st.session_state: st.session_state.llm_cache       = {}
if "lang_modal"      not in st.session_state: st.session_state.lang_modal      = None
if "clicked_date"    not in st.session_state: st.session_state.clicked_date    = None
if "selected_sector" not in st.session_state: st.session_state.selected_sector = _find_sector(st.session_state.selected)
if "page"            not in st.session_state: st.session_state.page            = "landing"
if "portfolios"      not in st.session_state: st.session_state.portfolios      = load_portfolios()
if "active_portfolio" not in st.session_state: st.session_state.active_portfolio = None

# ── JS Tooltip ────────────────────────────────────────────────────────────────
import streamlit.components.v1 as _comp
_comp.html("""
<script>
(function() {
    var par = window.parent;
    if (par._fwTooltipReady) return;
    par._fwTooltipReady = true;

    var doc = par.document;

    // Remove any stale tooltip divs
    doc.querySelectorAll('#fw-tooltip').forEach(function(el) { el.remove(); });

    var tip = doc.createElement('div');
    tip.id = 'fw-tooltip';
    doc.body.appendChild(tip);

    function show(e, text) { tip.textContent = text; tip.style.opacity = '1'; move(e); }
    function move(e) {
        var x = e.clientX + 14, y = e.clientY - 44;
        if (x + 280 > par.innerWidth) x = e.clientX - 294;
        if (y < 8) y = e.clientY + 18;
        tip.style.left = x + 'px';
        tip.style.top  = y + 'px';
    }
    function hide() { tip.style.opacity = '0'; }

    function attach() {
        doc.querySelectorAll('.tip[data-tip], .an-text[data-tip]').forEach(function(el) {
            if (el._fw) return;
            el._fw = true;
            el.addEventListener('mouseover', function(e) {
                e.stopPropagation();
                show(e, el.getAttribute('data-tip'));
            });
            el.addEventListener('mousemove', move);
            el.addEventListener('mouseout',  hide);
        });
    }
    attach();
    new MutationObserver(attach).observe(doc.body, { childList: true, subtree: true });
})();
</script>
""", height=0)

# ── Landing Page ───────────────────────────────────────────────────────────────
def render_landing():
    decisions_l  = load_decisions()
    stocks_l     = decisions_l[~decisions_l["ticker"].str.startswith("^")]
    total        = len(stocks_l) - 1   # ^SPX tracked separately as market reference
    critical     = (stocks_l["severity"] == "CRITICAL").sum()
    warning      = (stocks_l["severity"] == "WARNING").sum()

    st.markdown("""
    <style>
    /* Hide sidebar on landing */
    [data-testid="stSidebar"]{display:none!important}
    .main .block-container{padding:0 2rem 2rem!important;padding-top:0!important}

    /* Landing background */
    [data-testid="stAppViewContainer"]{
        background:
            radial-gradient(ellipse 60% 50% at 10% 15%,  rgba(29,233,182,0.07) 0%, transparent 60%),
            radial-gradient(ellipse 45% 40% at 90% 75%,  rgba(88,166,255,0.06) 0%, transparent 60%),
            radial-gradient(ellipse 35% 30% at 75% 5%,   rgba(163,113,247,0.04) 0%, transparent 55%)
            #060a0f;
    }

    @keyframes float {
        0%,100%{transform:translateY(0px)} 50%{transform:translateY(-6px)}
    }
    @keyframes glow-pulse {
        0%,100%{opacity:0.4} 50%{opacity:1}
    }
    @keyframes draw-line {
        from{stroke-dashoffset:2000} to{stroke-dashoffset:0}
    }
    @keyframes pulse-dot {
        0%,100%{opacity:0.3;r:3} 50%{opacity:1;r:5}
    }
    @keyframes flicker {
        0%,100%{opacity:0.06} 40%{opacity:0.13} 70%{opacity:0.08}
    }

    .hero-wrap{
        display:flex;flex-direction:column;align-items:center;
        justify-content:center;text-align:center;
        padding:3vh 0 2vh;
        position:relative;z-index:1;
    }
    .hero-badge{
        display:inline-block;
        font-family:'IBM Plex Mono',monospace;font-size:10px;
        letter-spacing:3px;text-transform:uppercase;color:#1de9b6;
        border:1px solid rgba(29,233,182,0.3);border-radius:20px;
        padding:4px 16px;margin-bottom:28px;
        background:rgba(29,233,182,0.05);
    }
    .hero-title{
        font-family:'Inter',sans-serif;
        font-size:68px;font-weight:700;letter-spacing:-3px;line-height:1;
        background:linear-gradient(135deg,#e2e8f0 0%,#1de9b6 40%,#58a6ff 75%,#a371f7 100%);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
        background-clip:text;margin-bottom:20px;
    }
    .hero-sub{
        font-family:'Inter',sans-serif;font-size:15px;font-weight:300;
        color:#637a91;max-width:560px;line-height:1.8;margin-bottom:14px;
    }
    .hero-stats{
        display:flex;gap:32px;justify-content:center;margin-top:16px;
        font-family:'IBM Plex Mono',monospace;
    }
    .hs-item{text-align:center}
    .hs-val{font-size:22px;font-weight:500;color:#e2e8f0;display:block}
    .hs-val.red{color:#f85149;text-shadow:0 0 16px rgba(248,81,73,0.4)}
    .hs-val.yellow{color:#e3b341;text-shadow:0 0 14px rgba(227,179,65,0.3)}
    .hs-val.teal{color:#1de9b6;text-shadow:0 0 14px rgba(29,233,182,0.4)}
    .hs-label{font-size:9px;letter-spacing:2px;text-transform:uppercase;color:#3d5266;margin-top:2px}

    .divider{
        width:120px;height:1px;
        background:linear-gradient(90deg,transparent,rgba(29,233,182,0.4),transparent);
        margin:40px auto;
    }

    .cards-title{
        text-align:center;font-family:'IBM Plex Mono',monospace;
        font-size:9px;letter-spacing:3px;text-transform:uppercase;
        color:#3d5266;margin-bottom:24px;
    }

    .lcard-outer{
        padding:1px;
        border-radius:20px;
        background:linear-gradient(135deg,rgba(29,233,182,0.35) 0%,rgba(88,166,255,0.15) 50%,rgba(163,113,247,0.25) 100%);
        box-shadow:0 0 40px rgba(29,233,182,0.06),0 20px 60px rgba(0,0,0,0.6);
        transition:all 0.35s cubic-bezier(0.23,1,0.32,1);
        transform:perspective(900px) rotateX(0deg) rotateY(0deg);
    }
    .lcard-outer:hover{
        transform:perspective(900px) rotateX(-4deg) rotateY(4deg) translateY(-6px) scale(1.02);
        box-shadow:0 0 60px rgba(29,233,182,0.12),0 30px 80px rgba(0,0,0,0.7),
                   -8px 8px 30px rgba(29,233,182,0.08);
        background:linear-gradient(135deg,rgba(29,233,182,0.55) 0%,rgba(88,166,255,0.25) 50%,rgba(163,113,247,0.40) 100%);
    }
    .lcard{
        background:rgba(6,10,15,0.04);
        border-radius:19px;
        padding:36px 28px 30px;
        text-align:center;
        backdrop-filter:blur(6px);
        -webkit-backdrop-filter:blur(6px);
        position:relative;
        overflow:hidden;
    }
    .lcard::before{
        content:'';
        position:absolute;top:0;left:0;right:0;height:1px;
        background:linear-gradient(90deg,transparent,rgba(255,255,255,0.15),transparent);
    }
    .lcard-icon{
        font-size:36px;margin-bottom:16px;display:block;
        animation:float 4s ease-in-out infinite;
        filter:drop-shadow(0 0 12px rgba(29,233,182,0.3));
    }
    .lcard-title{
        font-family:'Inter',sans-serif;font-size:18px;font-weight:700;
        color:#e2e8f0;margin-bottom:10px;letter-spacing:-0.5px;
    }
    .lcard-desc{font-family:'Inter',sans-serif;font-size:12px;color:#6b8099;line-height:1.8;margin-bottom:0}
    .lcard-tag{
        display:inline-block;margin-bottom:18px;
        font-family:'IBM Plex Mono',monospace;font-size:8px;
        letter-spacing:2px;text-transform:uppercase;
        padding:3px 12px;border-radius:20px;
    }
    .tag-live{
        background:rgba(29,233,182,0.08);color:#1de9b6;
        border:1px solid rgba(29,233,182,0.25);
    }

    /* Hide the trigger buttons — cards are clickable via JS */
    [data-testid="column"]:has(.lcard-outer) [data-testid="stButton"]{
        display:none!important;
    }

    .footer-note{
        text-align:center;margin-top:40px;
        font-family:'IBM Plex Mono',monospace;font-size:9px;
        letter-spacing:1.5px;color:#3d5266;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Background SVG — price chart lines ──
    st.markdown("""
    <div style="position:fixed;top:0;left:0;width:100vw;height:100vh;z-index:0;pointer-events:none;overflow:hidden">
      <svg width="100%" height="100%" viewBox="0 0 1440 900" preserveAspectRatio="xMidYMid slice" xmlns="http://www.w3.org/2000/svg">
        <!-- Candlestick-style price chart line 1 — teal, bottom third -->
        <polyline points="0,620 60,600 100,630 160,570 220,610 280,540 340,580 400,510 460,550 520,490 580,530 640,460 700,500 760,440 820,480 880,415 940,455 1000,390 1060,430 1120,370 1180,410 1240,350 1300,395 1360,330 1440,375"
          fill="none" stroke="#1de9b6" stroke-width="1.2" opacity="0.10"
          stroke-dasharray="2000" stroke-dashoffset="2000"
          style="animation:draw-line 4s ease-out 0.2s forwards"/>
        <!-- Candlestick-style price chart line 2 — blue, middle -->
        <polyline points="0,480 80,460 140,500 200,430 260,470 330,400 390,445 450,375 510,420 580,355 640,395 710,330 770,370 840,300 900,345 970,275 1030,320 1100,255 1160,295 1230,230 1290,275 1360,210 1440,255"
          fill="none" stroke="#58a6ff" stroke-width="1" opacity="0.08"
          stroke-dasharray="2000" stroke-dashoffset="2000"
          style="animation:draw-line 4.5s ease-out 0.8s forwards"/>
        <!-- Faint area fill under line 1 -->
        <polygon points="0,620 60,600 100,630 160,570 220,610 280,540 340,580 400,510 460,550 520,490 580,530 640,460 700,500 760,440 820,480 880,415 940,455 1000,390 1060,430 1120,370 1180,410 1240,350 1300,395 1360,330 1440,375 1440,900 0,900"
          fill="url(#chart-grad)" opacity="0.04"/>
        <!-- Vertical grid lines (subtle) -->
        <line x1="240" y1="0" x2="240" y2="900" stroke="#1de9b6" stroke-width="0.4" opacity="0.06" style="animation:flicker 6s ease-in-out infinite"/>
        <line x1="480" y1="0" x2="480" y2="900" stroke="#1de9b6" stroke-width="0.4" opacity="0.05" style="animation:flicker 7s ease-in-out 1s infinite"/>
        <line x1="720" y1="0" x2="720" y2="900" stroke="#58a6ff" stroke-width="0.4" opacity="0.05" style="animation:flicker 8s ease-in-out 2s infinite"/>
        <line x1="960" y1="0" x2="960" y2="900" stroke="#1de9b6" stroke-width="0.4" opacity="0.04" style="animation:flicker 6s ease-in-out 3s infinite"/>
        <line x1="1200" y1="0" x2="1200" y2="900" stroke="#58a6ff" stroke-width="0.4" opacity="0.04" style="animation:flicker 9s ease-in-out 0.5s infinite"/>
        <!-- Pulsing data points on line 1 -->
        <circle cx="280" cy="540" r="3" fill="#1de9b6" opacity="0" style="animation:pulse-dot 3s ease-in-out 2s infinite"/>
        <circle cx="640" cy="460" r="3" fill="#1de9b6" opacity="0" style="animation:pulse-dot 3s ease-in-out 2.8s infinite"/>
        <circle cx="1000" cy="390" r="3" fill="#1de9b6" opacity="0" style="animation:pulse-dot 3s ease-in-out 3.5s infinite"/>
        <circle cx="1360" cy="330" r="3" fill="#1de9b6" opacity="0" style="animation:pulse-dot 3s ease-in-out 4.2s infinite"/>
        <!-- Alert markers — red dot for anomaly -->
        <circle cx="400" cy="510" r="4" fill="#f85149" opacity="0" style="animation:pulse-dot 4s ease-in-out 3s infinite"/>
        <circle cx="880" cy="415" r="4" fill="#f85149" opacity="0" style="animation:pulse-dot 4s ease-in-out 4.5s infinite"/>
        <defs>
          <linearGradient id="chart-grad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stop-color="#1de9b6"/>
            <stop offset="100%" stop-color="#1de9b6" stop-opacity="0"/>
          </linearGradient>
        </defs>
      </svg>
    </div>
    """, unsafe_allow_html=True)

    # ── Hero ──
    entry_count = 0
    if "trading_signal" in stocks_l.columns:
        entry_count = int((stocks_l["trading_signal"] == "ENTRY").sum())

    st.markdown(f"""
    <div class="hero-wrap">
        <div class="hero-badge">AI · FINANCE · RISK ASSESSMENT</div>
        <div class="hero-title">FinWatch AI</div>
        <div class="hero-sub">
            AI-powered risk assessment and decision support for equity markets.<br>
            Detect anomalies. Assess risk. Know what to do next.
        </div>
        <div class="hero-stats">
            <div class="hs-item">
                <span class="hs-val teal">{total}</span>
                <div class="hs-label">Stocks Monitored</div>
            </div>
            <div class="hs-item">
                <span class="hs-val red">{int(critical)}</span>
                <div class="hs-label">Critical Today</div>
            </div>
            <div class="hs-item">
                <span class="hs-val yellow">{int(warning)}</span>
                <div class="hs-label">Warnings Today</div>
            </div>
            <div class="hs-item">
                <span class="hs-val teal">{entry_count}</span>
                <div class="hs-label">Entry Signals</div>
            </div>
        </div>
    </div>
    <div class="divider"></div>
    """, unsafe_allow_html=True)

    # ── Cards ──
    _, c1, c2, _ = st.columns([1, 2, 2, 1], gap="large")

    with c1:
        st.markdown("""<div class="lcard-outer">
        <div class="lcard">
            <span class="lcard-icon">📈</span>
            <div class="lcard-tag tag-live">Live</div>
            <div class="lcard-title">Stocks</div>
            <div class="lcard-desc">
                64 stocks across 12 sectors.<br>
                Real-time anomaly alerts,<br>
                risk scores &amp; AI analysis.
            </div>
        </div></div>""", unsafe_allow_html=True)
        if st.button(" ", key="go_stocks", use_container_width=True):
            st.session_state.page = "stocks"
            st.rerun()

    with c2:
        st.markdown("""<div class="lcard-outer">
        <div class="lcard">
            <span class="lcard-icon">💼</span>
            <div class="lcard-tag tag-live">Live</div>
            <div class="lcard-title">My Portfolio</div>
            <div class="lcard-desc">
                Track positions with live P&amp;L.<br>
                AI signal per holding: hold,<br>
                reduce, or exit — now.
            </div>
        </div></div>""", unsafe_allow_html=True)
        if st.button(" ", key="go_portfolio", use_container_width=True):
            st.session_state.page = "portfolio"
            st.rerun()

    st.markdown('<div class="footer-note">FINWATCH AI · NOUR AL HENDI · 2026</div>', unsafe_allow_html=True)

    # ── Make cards clickable via JS ──
    import streamlit.components.v1 as _components
    _components.html("""
    <script>
    (function attach() {
        var doc = window.parent.document;
        var cards = doc.querySelectorAll('.lcard-outer');
        if (!cards.length) { setTimeout(attach, 300); return; }
        cards.forEach(function(card, i) {
            card.style.cursor = 'pointer';
            card.addEventListener('click', function() {
                var btns = doc.querySelectorAll('[data-testid="stButton"] button');
                if (btns[i]) btns[i].click();
            });
        });
    })();
    </script>
    """, height=0)


if st.session_state.page == "landing":
    render_landing()
    st.stop()


# ── ETF Page ──────────────────────────────────────────────────────────────────
def render_etf_page():
    SEV_ORDER  = {"CRITICAL":0,"WARNING":1,"WATCH":2,"REVIEW":3,"POSITIVE_MOMENTUM":4,"NORMAL":5}
    SEV_COLOR  = {"CRITICAL":"#f85149","WARNING":"#e3b341","WATCH":"#58a6ff","NORMAL":"#3fb950","POSITIVE_MOMENTUM":"#3fb950","REVIEW":"#a371f7"}

    with st.sidebar:
        st.markdown('<div class="logo">Fin<span>Watch</span> AI</div>', unsafe_allow_html=True)
        if st.button("← Home", key="etf_home", use_container_width=True):
            st.session_state.page = "landing"
            st.rerun()

    st.markdown("""
    <div style="padding:14px 0 10px;border-bottom:1px solid rgba(30,45,65,0.8);margin-bottom:16px;display:flex;align-items:center;gap:14px">
        <div>
            <div style="font-size:20px;font-weight:700;color:#e2e8f0;letter-spacing:-0.5px">Sector ETF Overview</div>
            <div style="font-size:10px;color:#3d5266;font-family:'IBM Plex Mono',monospace;margin-top:2px">
                9 SECTORS · RISK AGGREGATION · AI MONITORING
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    for etf_name, tickers in ETF_STOCKS.items():
        sec_dec = decisions[decisions["ticker"].isin(tickers)]
        if sec_dec.empty:
            continue

        # Aggregate stats
        worst_sev = (sec_dec.assign(_o=sec_dec["severity"].map(SEV_ORDER).fillna(99))
                     .sort_values("_o")["severity"].iloc[0])
        sev_counts = sec_dec["severity"].value_counts()
        critical   = int(sev_counts.get("CRITICAL", 0))
        warning    = int(sev_counts.get("WARNING", 0))
        watch      = int(sev_counts.get("WATCH", 0))
        normal     = int(sev_counts.get("NORMAL", 0) + sev_counts.get("POSITIVE_MOMENTUM", 0))

        # Price changes from price_data
        changes = []
        for t in tickers:
            if t in price_data:
                _, pct = price_data[t]
                changes.append((t, pct))
        changes.sort(key=lambda x: x[1])
        worst_t = changes[0]  if changes else None
        best_t  = changes[-1] if changes else None

        sev_color = SEV_COLOR.get(worst_sev, "#637a91")
        sev_css   = SEV_CLS.get(worst_sev, "")

        # Severity bar (proportional)
        n = len(tickers)
        bar_html = ""
        for sev, cnt, col in [
            ("CRITICAL", critical, "#f85149"),
            ("WARNING",  warning,  "#e3b341"),
            ("WATCH",    watch,    "#58a6ff"),
            ("NORMAL",   normal,   "#2ea043"),
        ]:
            if cnt > 0:
                w = round(cnt / n * 100)
                bar_html += f'<div style="width:{w}%;background:{col};height:4px;border-radius:2px;display:inline-block" title="{cnt} {sev}"></div>'

        worst_html = ""
        best_html  = ""
        if worst_t:
            arrow = "▼" if worst_t[1] < 0 else "▲"
            worst_html = f'<span style="color:#f85149;font-family:IBM Plex Mono;font-size:10px">{worst_t[0]} {arrow}{abs(worst_t[1]):.1f}%</span>'
        if best_t and best_t != worst_t:
            arrow = "▲" if best_t[1] >= 0 else "▼"
            best_html = f'<span style="color:#1de9b6;font-family:IBM Plex Mono;font-size:10px">{best_t[0]} {arrow}{abs(best_t[1]):.1f}%</span>'

        st.markdown(f"""
        <div style="background:rgba(11,17,26,0.6);border:1px solid rgba(30,45,65,0.5);
                    border-radius:10px;padding:14px 18px;margin-bottom:10px;
                    box-shadow:0 4px 16px rgba(0,0,0,0.3)">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
                <div>
                    <span style="font-size:13px;font-weight:600;color:#e2e8f0;letter-spacing:-0.2px">{etf_name}</span>
                </div>
                <div style="display:flex;align-items:center;gap:20px">
                    <div style="text-align:right">
                        <div style="font-size:8px;letter-spacing:2px;text-transform:uppercase;color:#3d5266;font-family:IBM Plex Mono;margin-bottom:2px">Best 1D</div>
                        {best_html}
                    </div>
                    <div style="text-align:right">
                        <div style="font-size:8px;letter-spacing:2px;text-transform:uppercase;color:#3d5266;font-family:IBM Plex Mono;margin-bottom:2px">Worst 1D</div>
                        {worst_html}
                    </div>
                    <div style="text-align:right;min-width:80px">
                        <div style="font-size:8px;letter-spacing:2px;text-transform:uppercase;color:#3d5266;font-family:IBM Plex Mono;margin-bottom:2px">Sector Risk</div>
                        <span style="font-size:12px;font-weight:600;color:{sev_color};font-family:IBM Plex Mono">{worst_sev}</span>
                    </div>
                </div>
            </div>
            <div style="display:flex;gap:3px;width:100%;margin-bottom:8px">{bar_html}</div>
            <div style="display:flex;gap:16px">
                {'<span style="font-size:9px;color:#f85149;font-family:IBM Plex Mono">'+str(critical)+'C</span>' if critical else ''}
                {'<span style="font-size:9px;color:#e3b341;font-family:IBM Plex Mono">'+str(warning)+'W</span>' if warning else ''}
                {'<span style="font-size:9px;color:#58a6ff;font-family:IBM Plex Mono">'+str(watch)+'Wt</span>' if watch else ''}
                {'<span style="font-size:9px;color:#2ea043;font-family:IBM Plex Mono">'+str(normal)+'N</span>' if normal else ''}
                <span style="font-size:9px;color:#3d5266;font-family:IBM Plex Mono;margin-left:auto">{" · ".join(tickers)}</span>
            </div>
        </div>""", unsafe_allow_html=True)


if st.session_state.page == "etfs":
    render_etf_page()
    st.stop()

if st.session_state.page == "portfolio":
    with st.sidebar:
        st.markdown('<div class="logo">Fin<span>Watch</span> AI</div>', unsafe_allow_html=True)
        if st.button("← Home", key="pf_home", use_container_width=True):
            st.session_state.page = "landing"
            st.rerun()
        if st.button("📡 Stocks", key="pf_to_stocks", use_container_width=True):
            st.session_state.page = "stocks"
            st.rerun()
        st.markdown("<hr style='border-color:#1a2332;margin:6px 0'>", unsafe_allow_html=True)

        # ── Portfolio News Feed ────────────────────────────────────────────────
        portfolios   = st.session_state.get("portfolios", {})
        active_pf    = st.session_state.get("active_portfolio")
        pf_positions = portfolios.get(active_pf, []) if active_pf else []
        pf_tickers   = [p["ticker"] for p in pf_positions]

        # Fallback: if no portfolio, show all tickers sorted by severity
        if not pf_tickers:
            sev_order = {"CRITICAL":0,"WARNING":1,"WATCH":2,"REVIEW":3,"POSITIVE_MOMENTUM":4,"NORMAL":5}
            sorted_dec = decisions.copy()
            sorted_dec["_sev_rank"] = sorted_dec["severity"].map(lambda s: sev_order.get(s, 9))
            pf_tickers = sorted_dec.sort_values("_sev_rank")["ticker"].tolist()
            sidebar_label = "ALL STOCKS · BY RISK"
        else:
            sidebar_label = f"PORTFOLIO · {active_pf.upper()}"

        st.markdown(
            f'<div style="font-size:7px;letter-spacing:2.5px;color:#3d5266;'
            f'font-family:\'IBM Plex Mono\',monospace;text-transform:uppercase;'
            f'margin-bottom:8px">{sidebar_label}</div>',
            unsafe_allow_html=True,
        )

        _sent_color = {"positive":"#1de9b6","negative":"#f85149","neutral":"#637a91","mixed":"#e3b341"}
        _sev_col    = {"CRITICAL":"#f85149","WARNING":"#e3b341","WATCH":"#58a6ff",
                       "POSITIVE_MOMENTUM":"#1de9b6","NORMAL":"#2ea043","REVIEW":"#a371f7"}

        for t in pf_tickers:
            dec_row  = decisions[decisions["ticker"] == t]
            sev      = dec_row.iloc[0]["severity"] if not dec_row.empty else "NORMAL"
            sev_c    = _sev_col.get(sev, "#637a91")
            price, chg = price_data.get(t, (0.0, 0.0))
            chg_c    = "#1de9b6" if chg >= 0 else "#f85149"
            name     = COMPANY_NAMES.get(t, t)

            # Headlines for this ticker
            headlines, sentiments = [], []
            if news_df is not None and t in news_df["ticker"].values:
                nr         = news_df[news_df["ticker"] == t].iloc[0]
                raw_news   = nr.get("top_news", [])
                raw_sent   = nr.get("news_sentiment", [])
                headlines  = list(raw_news)[:3] if hasattr(raw_news, "__iter__") else []
                sentiments = list(raw_sent)[:3] if hasattr(raw_sent, "__iter__") else []

            # Build news HTML
            news_html = ""
            for i, h in enumerate(headlines):
                s     = sentiments[i] if i < len(sentiments) else "neutral"
                sc    = _sent_color.get(str(s).lower(), "#637a91")
                short = str(h)[:72] + ("…" if len(str(h)) > 72 else "")
                news_html += (
                    f'<div style="display:flex;gap:5px;align-items:flex-start;'
                    f'padding:3px 0;border-bottom:1px solid rgba(30,45,65,0.3)">'
                    f'<span style="color:{sc};font-size:8px;flex-shrink:0;margin-top:1px">●</span>'
                    f'<span style="color:#8b9aab;font-size:9px;line-height:1.4">{short}</span>'
                    f'</div>'
                )
            if not headlines:
                news_html = '<div style="color:#3d5266;font-size:9px;padding:3px 0">No news available</div>'

            st.markdown(f"""
<div style="background:rgba(11,17,26,0.7);border:1px solid rgba(30,45,65,0.5);
     border-left:2px solid {sev_c};border-radius:6px;padding:8px 10px;margin-bottom:6px">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:5px">
    <span style="color:#e2e8f0;font-weight:700;font-family:'IBM Plex Mono',monospace;font-size:11px">{t}</span>
    <span style="color:{chg_c};font-family:'IBM Plex Mono',monospace;font-size:9px">
      ${price:.2f} <span style="opacity:0.7">({chg:+.1f}%)</span>
    </span>
  </div>
  <div style="font-size:8px;color:{sev_c};font-family:'IBM Plex Mono',monospace;
       letter-spacing:1px;margin-bottom:5px">{sev}</div>
  {news_html}
</div>""", unsafe_allow_html=True)

    render_portfolio_page()
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────
render_sidebar(decisions, price_data)

# ── Main ──────────────────────────────────────────────────────────────────────
ticker = st.session_state.selected

# ── Market Overview (^SPX) ────────────────────────────────────────────────────
if ticker == "^SPX":
    spx_df = load_spx()
    render_spx_header(spx_df)

    _sl, _sr = st.columns([6, 4], gap="small")
    with _sl:
        st.markdown('<div class="chart-label" style="padding-top:6px">S&P 500 Price History</div>', unsafe_allow_html=True)
    with _sr:
        _new_spx_period = st.segmented_control(
            "SPX Period", ["1M", "3M", "6M", "1Y", "All"],
            default=st.session_state.spx_period,
            label_visibility="collapsed",
            key="spx_period_ctrl",
        )
        if _new_spx_period and _new_spx_period != st.session_state.spx_period:
            st.session_state.spx_period = _new_spx_period
            st.rerun()
    render_spx_chart(spx_df, st.session_state.spx_period)

    # Sector Alert Overview
    st.markdown('<div class="chart-label">Sector Alert Overview</div>', unsafe_allow_html=True)
    sev_order = {"CRITICAL": 0, "WARNING": 1, "WATCH": 2, "REVIEW": 3, "POSITIVE_MOMENTUM": 4, "NORMAL": 5}
    sector_rows_html = ""
    for sector, etf_tickers in ETF_STOCKS.items():
        sector_dec = decisions[decisions["ticker"].isin(etf_tickers)]
        if sector_dec.empty:
            continue
        critical = (sector_dec["severity"] == "CRITICAL").sum()
        warning  = (sector_dec["severity"] == "WARNING").sum()
        watch    = (sector_dec["severity"] == "WATCH").sum()
        worst    = (sector_dec.assign(_o=sector_dec["severity"].map(sev_order).fillna(99))
                               .sort_values("_o").iloc[0]["severity"])
        sev_c    = SEV_CLS.get(worst, "s-normal")
        sector_rows_html += f"""
        <div class="rp-row">
          <span class="rp-key" style="width:180px">{sector}</span>
          <span class="rp-val {sev_c}" style="min-width:60px">{worst.replace("_"," ")}</span>
          <span style="font-size:9px;color:#f85149;font-family:IBM Plex Mono;min-width:30px">{f'{critical}C' if critical else ''}</span>
          <span style="font-size:9px;color:#e3b341;font-family:IBM Plex Mono;min-width:30px">{f'{warning}W' if warning else ''}</span>
          <span style="font-size:9px;color:#58a6ff;font-family:IBM Plex Mono">{f'{watch}Wt' if watch else ''}</span>
        </div>"""
    st.markdown(f'<div class="rp-section">{sector_rows_html}</div>', unsafe_allow_html=True)

    # Market Pulse + Top Critical + Trading Signals
    _spx_c1, _spx_c2, _spx_c3 = st.columns(3, gap="small")
    with _spx_c1:
        total    = len(decisions)
        by_sev   = decisions["severity"].value_counts()
        critical = by_sev.get("CRITICAL", 0)
        warning  = by_sev.get("WARNING", 0)
        watch    = by_sev.get("WATCH", 0)
        normal   = by_sev.get("NORMAL", 0) + by_sev.get("POSITIVE_MOMENTUM", 0)
        st.markdown(f"""
        <div class="rp-section">
          <div class="rp-title">Market Pulse</div>
          <div class="rp-row"><span class="rp-key">Total Stocks</span><span class="rp-val">{total}</span></div>
          <div class="rp-row"><span class="rp-key s-critical">Critical</span><span class="rp-val s-critical">{critical}</span></div>
          <div class="rp-row"><span class="rp-key s-warning">Warning</span><span class="rp-val s-warning">{warning}</span></div>
          <div class="rp-row"><span class="rp-key s-watch">Watch</span><span class="rp-val s-watch">{watch}</span></div>
          <div class="rp-row"><span class="rp-key s-normal">Normal</span><span class="rp-val s-normal">{normal}</span></div>
        </div>""", unsafe_allow_html=True)
    with _spx_c2:
        top_critical = decisions[decisions["severity"] == "CRITICAL"].head(8)
        if not top_critical.empty:
            crit_html = ""
            for _, r in top_critical.iterrows():
                t = r["ticker"]
                n = COMPANY_NAMES.get(t, t)
                crit_html += f'<div class="rp-row"><span class="rp-key">{n}</span><span class="rp-val s-critical">{t}</span></div>'
            st.markdown(f'<div class="rp-section"><div class="rp-title">Top Critical</div>{crit_html}</div>', unsafe_allow_html=True)
    with _spx_c3:
        if "trading_signal" in decisions.columns:
            sig_counts = decisions["trading_signal"].value_counts()
            n_entry    = int(sig_counts.get("ENTRY", 0))
            n_exit     = int(sig_counts.get("EXIT", 0))
            n_hold     = int(sig_counts.get("HOLD", 0))
            n_neutral  = int(sig_counts.get("NEUTRAL", 0))
            entry_tickers = decisions[decisions["trading_signal"] == "ENTRY"]["ticker"].tolist()
            entry_names   = ", ".join([COMPANY_NAMES.get(t, t) for t in entry_tickers[:6]])
            entry_row_html = (
                f'<div style="margin-top:6px;font-size:9px;color:#1de9b6;font-family:IBM Plex Mono;line-height:1.6">'
                f'{entry_names}{"…" if len(entry_tickers) > 6 else ""}</div>'
            ) if entry_tickers else ""
            st.markdown(f"""
            <div class="rp-section">
              <div class="rp-title">AI Trading Signals</div>
              <div class="rp-row">
                <span class="rp-key" style="color:#1de9b6">▲ ENTRY</span>
                <span class="rp-val" style="color:#1de9b6;font-weight:600">{n_entry}</span>
              </div>
              <div class="rp-row">
                <span class="rp-key" style="color:#f85149">▼ EXIT</span>
                <span class="rp-val" style="color:#f85149;font-weight:600">{n_exit}</span>
              </div>
              <div class="rp-row">
                <span class="rp-key" style="color:#58a6ff">◆ HOLD</span>
                <span class="rp-val" style="color:#58a6ff">{n_hold}</span>
              </div>
              <div class="rp-row">
                <span class="rp-key">— NEUTRAL</span>
                <span class="rp-val">{n_neutral}</span>
              </div>
              {entry_row_html}
            </div>""", unsafe_allow_html=True)

    st.stop()

# ── Normal Stock View ─────────────────────────────────────────────────────────
dec_row = decisions[decisions["ticker"] == ticker]
if dec_row.empty:
    st.warning(f"No decision data for {ticker}")
    st.stop()
row = dec_row.iloc[0]

det_df = load_detection(ticker)
name   = COMPANY_NAMES.get(ticker, ticker)
lang   = st.session_state.language

render_stock_header(ticker, name, det_df, lang)

if det_df is not None and "Close" in det_df.columns and len(det_df) > 1:
    render_risk_news_row(ticker, row, det_df, decisions, news_df)

    _pl, _pr = st.columns([6, 4], gap="small")
    with _pl:
        st.markdown('<div class="chart-label" style="padding-top:6px">Price  ·  EMA 20</div>', unsafe_allow_html=True)
    with _pr:
        _new_period = st.segmented_control(
            "Period", ["1M", "3M", "6M", "1Y", "All"],
            default=st.session_state.period,
            label_visibility="collapsed",
            key="period_ctrl",
        )
        if _new_period and _new_period != st.session_state.period:
            st.session_state.period = _new_period
            st.rerun()
    _ev = render_price_chart(
        det_df, ticker, st.session_state.period, st.session_state.clicked_date
    )
    try:
        _pts = _ev.selection.points if _ev else []
        if _pts:
            _cd = str(_pts[0].get("x", ""))[:10]
            if _cd:
                st.session_state.clicked_date = _cd
    except (AttributeError, TypeError, KeyError):
        pass

    render_candle_panel(det_df, st.session_state.clicked_date)
    render_anomaly_selector(det_df, name, st.session_state.period, dec_row)

    st.markdown('<div class="chart-label">RSI  ·  RSI MA 14</div>', unsafe_allow_html=True)
    render_rsi_chart(det_df, st.session_state.period)

    render_analysis_panel(det_df, dec_row, ticker, name, lang)
    render_investor_summary(det_df, dec_row, row, news_df, ticker)
    render_strategy_box(det_df, dec_row)

    if st.session_state.lang_modal:
        show_analysis_modal(ticker, name, det_df, news_df, st.session_state.lang_modal)
