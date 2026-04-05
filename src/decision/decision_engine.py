"""
FinWatch AI — Layer 6: Decision Engine (v2)
============================================
Clean, anomaly-first design. No direction forecasting.

Primary signals (in order of reliability):
  1. anomaly_score_weighted  — 4-model ensemble, weighted (most reliable)
  2. p_drawdown              — ML probability of >5% drawdown in 20 days (AUC 0.64)
  3. drawdown (30d actual)   — already happened, hard fact
  4. RSI + momentum          — technical confirmation
  5. news_sentiment_score    — soft signal (Groq contextual)
  6. excess_return           — vs market (filters false positives)

Severity Levels:
  CRITICAL         — Multiple strong signals align: high anomaly + high p_drawdown
  WARNING          — Elevated risk, 1-2 strong signals
  WATCH            — Single weak signal, monitor
  NORMAL           — No significant signals
  POSITIVE_SIGNAL  — Low drawdown risk + positive technicals
  REVIEW           — Conflicting signals

Confidence:
  Based on signal agreement, not on broken ML model probabilities.
  Will be replaced by historical_precision after backtesting.
"""

from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
import pandas as pd

_ROOT         = Path(__file__).resolve().parents[2]
_PRECISION_PATH = _ROOT / "data/backtesting/signal_precision.parquet"

# Historical precision from walk-forward backtest (fallback values if file missing)
# Loaded once at import time; keys = severity label
_HISTORICAL_PRECISION: dict = {}

def _load_precision():
    global _HISTORICAL_PRECISION
    if not _PRECISION_PATH.exists():
        return
    df = pd.read_parquet(_PRECISION_PATH)
    mp = dict(zip(df["signal"], df["precision"]))
    _HISTORICAL_PRECISION = {
        "CRITICAL":        mp.get("severity_critical", 0.48),
        "WARNING":         max(mp.get("severity_warning",  0.36) - mp.get("severity_critical", 0.48) * 0.35, 0.30),
        "WATCH":           0.28,
        "REVIEW":          0.28,
        "NORMAL":          1.0 - mp.get("severity_warning", 0.36),   # precision of "no event"
        "POSITIVE_SIGNAL": 1.0 - mp.get("severity_warning", 0.36),
    }

_load_precision()


# ── Thresholds ─────────────────────────────────────────────────────────────

# Drawdown probability thresholds
P_DRAWDOWN_CRITICAL = 0.60   # >60% chance of 5%+ drawdown → CRITICAL signal
P_DRAWDOWN_WARNING  = 0.45   # >45% → WARNING signal
P_DRAWDOWN_LOW      = 0.30   # <30% + good technicals → POSITIVE

# Anomaly weighted score thresholds
ANOMALY_STRONG  = 0.50   # AE + IF both flagged → strong
ANOMALY_MEDIUM  = 0.30   # at least one ML model flagged
ANOMALY_WEAK    = 0.20   # only z-score flagged

# Drawdown (actual 30d) thresholds — dynamic per volatility
DRAWDOWN_WARNING_CAP  = -0.08
DRAWDOWN_CRITICAL_CAP = -0.15

# RSI
RSI_OVERBOUGHT = 70
RSI_OVERSOLD   = 30

# Tail risk
ES_RATIO_HIGH = 2.0

# News sentiment
SENTIMENT_NEGATIVE = -0.10
SENTIMENT_POSITIVE = +0.10


# ── Data Classes ────────────────────────────────────────────────────────────

@dataclass
class AnomalyInput:
    ticker:                 str
    date:                   str

    # Drawdown Probability Model
    p_drawdown:             float = 0.35   # P(drawdown > 5% in 20 days)
    drawdown_risk:          str   = "low"  # "high" / "low"

    # Anomaly Detection
    anomaly_score:          int   = 0      # 0–4 (integer, backward compat)
    anomaly_score_weighted: float = 0.0    # 0–1 weighted score
    market_anomaly:         bool  = False
    sector_anomaly:         bool  = False

    # Technical signals
    rsi:            float = 50.0
    momentum_5:     float = 0.0
    momentum_10:    float = 0.0
    drawdown:       float = 0.0    # actual 30d max drawdown (negative)
    obv_signal:     float = 0.0
    volatility:     float = 0.02
    excess_return:  float = 0.0
    es_ratio:       float = 1.0
    vix_level:      float = 20.0   # current VIX — used for regime-aware thresholds

    # News sentiment (Groq + VADER)
    vader_score:          float = 0.0
    finbert_score:        float = 0.0
    news_sentiment_score: float = 0.0

    # Fundamental signals (optional — loaded from data/fundamental/)
    days_to_next_earnings: Optional[int]   = None   # None = unknown
    insider_sentiment:     float           = 0.0    # -1 heavy selling … +1 heavy buying
    put_call_ratio:        float           = 0.85   # market avg ~0.85; >1.0 = fear
    options_fear:          int             = 0      # 1 if put_call_ratio > 1.0

    # Valuation signals (optional — loaded from data/fundamental/valuation.parquet)
    pe_ratio:       Optional[float] = None   # trailing P/E; None = unknown / not profitable
    pe_forward:     Optional[float] = None   # forward P/E; None = unknown
    pb_ratio:       Optional[float] = None   # price-to-book; <1.0 = below book value
    revenue_growth: Optional[float] = None   # YoY revenue growth; negative = declining

    # Trend & regime context (loaded from detection layer)
    price_vs_ma200: float = 0.0    # (close/ma200 - 1): positive = above MA200, negative = below
    price_vs_ma50:  float = 0.0    # (close/ma50  - 1): positive = above MA50
    regime:         str   = "unknown"  # bull / bear / sideways / transition_down / transition_up
    volume_trend:   float = 1.0    # volume_ma5/volume_ma20: >1 = rising volume
    trend_strength: float = 0.0    # trend strength score


@dataclass
class DecisionOutput:
    ticker:           str
    date:             str
    severity:         str
    action:           str
    confidence:       float
    context:          str

    # Signal summary (for narrator + portfolio page)
    p_drawdown:             float = 0.0
    anomaly_score:          int   = 0
    anomaly_score_weighted: float = 0.0
    drawdown_risk:          str   = "low"
    momentum_signal:        str   = "neutral"
    caution_flag:           bool  = False
    override_reason:        str   = ""
    summary:                str   = ""
    sentiment_note:         str   = ""
    trading_signal:         str   = "NEUTRAL"   # ENTRY / HOLD / EXIT / AVOID / NEUTRAL


# ── Core Logic ──────────────────────────────────────────────────────────────

def _dynamic_drawdown_threshold(volatility: float) -> tuple[float, float]:
    """Tighter thresholds for calm stocks, capped for volatile ones."""
    warning  = max(DRAWDOWN_WARNING_CAP,  -1.0 * volatility * 20)
    critical = max(DRAWDOWN_CRITICAL_CAP, -2.0 * volatility * 20)
    return warning, critical


def _momentum_label(mom5: float, mom10: float) -> str:
    if mom5 > 0.02 and mom10 > 0.01:   return "positive"
    if mom5 < -0.02 and mom10 < -0.01: return "negative"
    if mom5 < -0.02 and mom10 > 0.01:  return "pullback"   # short dip in uptrend
    if mom5 > 0.02 and mom10 < -0.01:  return "bounce"     # short pop in downtrend
    return "neutral"


def _confidence(inp: AnomalyInput, severity: str) -> float:
    """
    Confidence = historical precision (from walk-forward backtest) anchored per
    severity level, then nudged ±10 pp by how strongly individual signals agree.

    If the backtest file hasn't been generated yet, falls back to a pure
    signal-agreement formula so the system still produces valid output.
    """
    is_risk = severity in ("CRITICAL", "WARNING", "WATCH")

    # ── Base: historical precision ────────────────────────────────────────────
    if _HISTORICAL_PRECISION:
        base = _HISTORICAL_PRECISION.get(severity, 0.35)

        # ── Adjustment: how strongly do current signals agree? (−0.10 … +0.10)
        agree = 0.0
        if is_risk:
            # p_drawdown above/below the warning threshold shifts confidence
            agree += 0.05 * (inp.p_drawdown - P_DRAWDOWN_WARNING) / (1 - P_DRAWDOWN_WARNING)
            agree += 0.03 * inp.anomaly_score_weighted
            if inp.news_sentiment_score <= SENTIMENT_NEGATIVE:
                agree += 0.02
        else:
            agree += 0.05 * (P_DRAWDOWN_LOW - inp.p_drawdown) / P_DRAWDOWN_LOW
            agree += 0.03 * (1 - inp.anomaly_score_weighted)
            if inp.news_sentiment_score >= SENTIMENT_POSITIVE:
                agree += 0.02

        return round(min(max(base + agree, 0.10), 0.95), 2)

    # ── Fallback: signal-agreement (used before backtest is run) ─────────────
    signals = 0.0
    total   = 9.0

    # 1. Drawdown probability (weight 3)
    if is_risk:
        signals += 3 * inp.p_drawdown
    else:
        signals += 3 * (1 - inp.p_drawdown)

    # 2. Weighted anomaly score (weight 3)
    if is_risk:
        signals += 3 * inp.anomaly_score_weighted
    else:
        signals += 3 * (1 - inp.anomaly_score_weighted)

    # 3. Actual drawdown (weight 2)
    dd_warn, _ = _dynamic_drawdown_threshold(inp.volatility)
    if is_risk and inp.drawdown <= dd_warn:
        signals += 2
    elif not is_risk and inp.drawdown > dd_warn * 0.5:
        signals += 2

    # 4. News sentiment alignment (weight 1)
    if is_risk and inp.news_sentiment_score <= SENTIMENT_NEGATIVE:
        signals += 1
    elif not is_risk and inp.news_sentiment_score >= SENTIMENT_POSITIVE:
        signals += 1
    else:
        signals += 0.5

    return round(signals / total, 2)


def _vix_thresholds(vix: float) -> tuple[float, float, float]:
    """
    VIX-regime-aware thresholds.
    FP analysis showed: calm markets (VIX<20) generate the most false positives.
    At low VIX, raise the bar — demand stronger evidence for WARNING/CRITICAL.

    Returns: (p_warning, p_critical, p_watch)
    """
    if vix < 15:                              # very calm market
        return 0.55, 0.68, 0.42
    elif vix < 20:                            # normal market
        return 0.50, 0.63, 0.40
    elif vix < 25:                            # moderately elevated
        return P_DRAWDOWN_WARNING,  P_DRAWDOWN_CRITICAL, 0.38
    else:                                     # high fear — use base thresholds
        return P_DRAWDOWN_WARNING,  P_DRAWDOWN_CRITICAL, 0.35


def decide(inp: AnomalyInput) -> DecisionOutput:
    """
    Core decision logic — anomaly-first, no direction forecasting.
    VIX-regime-aware thresholds + excess_return false-positive filter.
    """
    dd_warn, dd_crit         = _dynamic_drawdown_threshold(inp.volatility)
    mom_label                = _momentum_label(inp.momentum_5, inp.momentum_10)
    anomaly_w                = inp.anomaly_score_weighted
    p_dd                     = inp.p_drawdown
    caution                  = False
    override_reason          = ""

    # VIX-regime thresholds (key improvement from FP analysis)
    p_warn, p_crit, p_watch  = _vix_thresholds(inp.vix_level)

    # ── Determine base severity ────────────────────────────────────────────

    # CRITICAL: strong ML signal + strong anomaly
    if p_dd >= p_crit and anomaly_w >= ANOMALY_MEDIUM:
        severity = "CRITICAL"
        action   = "ESCALATE"
        context  = "high drawdown probability + anomaly confirmed"

    # CRITICAL: extreme actual drawdown already happening
    elif inp.drawdown <= dd_crit:
        severity = "CRITICAL"
        action   = "ESCALATE"
        context  = "severe drawdown in progress"
        override_reason = f"drawdown={inp.drawdown:.1%} ≤ {dd_crit:.1%}"

    # WARNING: elevated ML signal OR medium anomaly
    elif p_dd >= p_warn or anomaly_w >= ANOMALY_MEDIUM:
        severity = "WARNING"
        action   = "MONITOR"
        context  = (
            f"p_drawdown={p_dd:.0%}" if p_dd >= p_warn
            else f"anomaly_weighted={anomaly_w:.2f}"
        )

    # WARNING: meaningful actual drawdown
    elif inp.drawdown <= dd_warn:
        severity = "WARNING"
        action   = "MONITOR"
        context  = f"drawdown={inp.drawdown:.1%}"
        override_reason = "actual drawdown threshold"

    # WATCH: weak single signal
    elif anomaly_w >= ANOMALY_WEAK or p_dd >= p_watch:
        severity = "WATCH"
        action   = "OBSERVE"
        context  = "weak signal — monitor"

    # POSITIVE: low drawdown risk + good technicals
    elif (p_dd < P_DRAWDOWN_LOW
          and inp.rsi < RSI_OVERBOUGHT
          and mom_label in ("positive", "pullback")
          and anomaly_w < ANOMALY_WEAK):
        severity = "POSITIVE_SIGNAL"
        action   = "NONE"
        context  = f"low drawdown risk ({p_dd:.0%}) + {mom_label} momentum"

    # NORMAL
    else:
        severity = "NORMAL"
        action   = "NONE"
        context  = "no significant signals"

    # ── Caution flags (do not override severity, just flag) ────────────────

    # RSI overbought + elevated drawdown risk
    if inp.rsi >= RSI_OVERBOUGHT and p_dd >= 0.40:
        caution = True
        context += " | overbought RSI"

    # Tail risk
    if inp.es_ratio >= ES_RATIO_HIGH and severity not in ("CRITICAL",):
        if severity == "NORMAL":
            severity = "WATCH"
            action   = "OBSERVE"
        context += f" | tail risk (ES={inp.es_ratio:.1f})"

    # Idiosyncratic vs market — if CRITICAL but market-wide, downgrade to WARNING
    if severity == "CRITICAL" and inp.market_anomaly and inp.excess_return > -0.03:
        severity = "WARNING"
        action   = "MONITOR"
        override_reason = "market-wide event, stock not underperforming"
        context  = "broad market stress (not stock-specific)"

    # Outperforming-stock false-positive filter (from backtest FP analysis):
    # 61% of false positives came from stocks with positive excess_return.
    # If the stock is outperforming the market AND drawdown signal isn't very strong,
    # the risk signal is likely caused by the stock going UP unusually fast, not down.
    if (severity == "WARNING"
            and inp.excess_return > 0.02
            and p_dd < p_crit
            and anomaly_w < ANOMALY_STRONG):
        severity = "WATCH"
        action   = "OBSERVE"
        override_reason = "stock outperforming market — reduced false positive risk"
        context  += " | outperforming market (likely upward anomaly)"

    # Negative news confirms risk signals
    sentiment_note = ""
    if inp.news_sentiment_score <= SENTIMENT_NEGATIVE:
        if severity in ("WARNING", "WATCH"):
            severity = "WARNING"
            context += " | negative news confirms"
        sentiment_note = f"bearish news ({inp.news_sentiment_score:+.2f})"
    elif inp.news_sentiment_score >= SENTIMENT_POSITIVE:
        if severity == "POSITIVE_SIGNAL":
            context += " | positive news confirms"
        sentiment_note = f"bullish news ({inp.news_sentiment_score:+.2f})"

    # ── Fundamental signals ────────────────────────────────────────────────

    # Earnings imminent (≤3 days) → always flag caution regardless of severity
    if inp.days_to_next_earnings is not None and inp.days_to_next_earnings <= 3:
        caution = True
        context += f" | earnings in {inp.days_to_next_earnings}d"

    # Heavy insider selling confirms risk signals
    if inp.insider_sentiment <= -0.3:
        if severity in ("WATCH", "WARNING"):
            severity = "WARNING"
            action   = "MONITOR"
        context += f" | insider selling ({inp.insider_sentiment:+.2f})"

    # Options market showing fear → confirm risk
    if inp.options_fear and severity in ("WATCH", "WARNING", "CRITICAL"):
        context += f" | options fear (P/C={inp.put_call_ratio:.2f})"

    # REVIEW: conflicting signals (high anomaly but good fundamentals)
    if (anomaly_w >= ANOMALY_MEDIUM
            and p_dd < P_DRAWDOWN_WARNING
            and inp.excess_return > 0.02
            and severity not in ("CRITICAL",)):
        severity = "REVIEW"
        action   = "FLAG"
        context  = "anomaly detected but drawdown model disagrees — manual check"

    # ── Regime-aware adjustments ───────────────────────────────────────────────
    # Bear market: elevate risk signals (more dangerous to hold in a downtrend)
    if inp.regime == "bear":
        if severity == "WATCH":
            severity = "WARNING"
            action   = "MONITOR"
            context += " | bear market (elevated risk)"
        elif severity in ("NORMAL", "POSITIVE_SIGNAL"):
            caution = True
            context += " | bear market regime"

    # Transitioning down: add caution without changing severity
    if inp.regime == "transition_down" and severity == "POSITIVE_SIGNAL":
        caution = True
        context += " | market transitioning down"

    # Price well below MA200: structural downtrend — add context
    if inp.price_vs_ma200 < -0.10 and severity in ("NORMAL", "POSITIVE_SIGNAL"):
        caution = True
        context += f" | below MA200 ({inp.price_vs_ma200:+.1%})"


    # ── Valuation signals ─────────────────────────────────────────────────────
    valuation_note = ""

    # Negative P/E = company losing money → block ENTRY, add caution
    if inp.pe_ratio is not None and inp.pe_ratio < 0:
        valuation_note = "negative earnings"
        if severity == "POSITIVE_SIGNAL":
            severity = "NORMAL"
            action   = "NONE"
            context += " | negative earnings (no entry)"

    # Highly overvalued (P/E > 50) → block ENTRY signal
    elif inp.pe_ratio is not None and inp.pe_ratio > 50:
        valuation_note = f"overvalued P/E={inp.pe_ratio:.0f}"
        if severity == "POSITIVE_SIGNAL":
            severity = "NORMAL"
            action   = "NONE"
            context += f" | overvalued (P/E={inp.pe_ratio:.0f}, no entry)"

    # Cheap stock (P/E < 15 or P/B < 1.5) → strengthen POSITIVE_SIGNAL
    elif ((inp.pe_ratio is not None and inp.pe_ratio < 15)
          or (inp.pb_ratio is not None and inp.pb_ratio < 1.5)):
        valuation_note = (
            f"cheap P/E={inp.pe_ratio:.0f}" if inp.pe_ratio is not None and inp.pe_ratio < 15
            else f"cheap P/B={inp.pb_ratio:.2f}"
        )

    # Declining revenue confirms risk signals
    if inp.revenue_growth is not None and inp.revenue_growth < -0.05:
        valuation_note += f" rev{inp.revenue_growth:+.0%}"
        if severity in ("WATCH", "WARNING", "CRITICAL"):
            context += f" | revenue declining ({inp.revenue_growth:+.0%})"

    # Forward P/E > trailing P/E means earnings expected to decline → caution on POSITIVE
    if (inp.pe_forward is not None and inp.pe_ratio is not None
            and inp.pe_forward > inp.pe_ratio * 1.2
            and severity == "POSITIVE_SIGNAL"):
        context += f" | earnings contraction expected (fwdPE={inp.pe_forward:.0f})"

    # ── Trading Signal ────────────────────────────────────────────────────────

    # ── Overbought EXIT (profit-taking) — checked before severity gates
    # RSI > 75: stock is strongly overextended, smart to take profits
    # regardless of whether there's an active anomaly signal
    overbought_exit = (
        inp.rsi > 75
        and inp.price_vs_ma200 > 0.15   # well above MA200 = extended run
        and mom_label in ("positive", "bounce")  # still moving up = classic exit point
    )

    if severity == "CRITICAL":
        # CRITICAL = strong ML signal + anomaly → always EXIT
        trading_signal = "EXIT"
        exit_reason    = "risk"

    elif severity == "WARNING":
        # WARNING → EXIT only with strong ML conviction AND no active recovery.
        # Recovery check: if momentum_5 > 0.03 the stock is bouncing back — don't exit.
        recovering = inp.momentum_5 > 0.03

        strong_signal = p_dd >= 0.50 or anomaly_w >= 0.35

        if strong_signal and not recovering:
            trading_signal = "EXIT"
            exit_reason    = "risk"
        elif strong_signal and recovering:
            # Risk is elevated but stock is actively recovering → HOLD and watch
            trading_signal = "HOLD"
            exit_reason    = ""
            context += " | recovering momentum — hold, monitor closely"
        else:
            # Past drawdown happened but ML disagrees it continues → HOLD
            trading_signal = "HOLD"
            exit_reason    = ""

    elif overbought_exit and severity not in ("CRITICAL", "WARNING"):
        # Profit-taking EXIT: stock ran too far, RSI overextended
        trading_signal = "EXIT"
        exit_reason    = "overbought"
        context += f" | overbought RSI={inp.rsi:.0f}, +{inp.price_vs_ma200:.0%} above MA200 → take profits"

    elif severity == "WATCH":
        trading_signal = "HOLD"
        exit_reason    = ""

    elif severity == "POSITIVE_SIGNAL":
        exit_reason = ""
        # POSITIVE_SIGNAL already guarantees: p_dd < 30%, no anomaly, RSI < 70,
        # positive or pullback momentum. We add 4 structural gates:

        # Gate 1: Trend intact — not more than 10% below MA200
        above_ma200  = inp.price_vs_ma200 > -0.10

        # Gate 2: Market regime — bear or falling market = don't enter
        regime_ok    = inp.regime not in ("bear", "transition_down")

        # Gate 3: Valuation — no negative earnings, no extreme P/E
        valuation_ok = inp.pe_ratio is None or (inp.pe_ratio > 0 and inp.pe_ratio < 60)

        # Gate 4: Revenue — not collapsing
        revenue_ok   = inp.revenue_growth is None or inp.revenue_growth >= -0.10

        if above_ma200 and regime_ok and valuation_ok and revenue_ok:
            trading_signal = "ENTRY"
        else:
            trading_signal = "HOLD"

    elif severity == "NORMAL":
        exit_reason    = ""
        trading_signal = "NEUTRAL"

    else:  # REVIEW
        exit_reason    = ""
        trading_signal = "NEUTRAL"

    confidence = _confidence(inp, severity)

    # Build human-readable summary
    summary_parts = []
    summary_parts.append(f"P(drawdown>5% / 20d) = {p_dd:.0%}")
    summary_parts.append(f"Anomaly = {anomaly_w:.2f}")
    summary_parts.append(f"RSI = {inp.rsi:.0f}")
    if inp.drawdown < -0.03:
        summary_parts.append(f"30d drawdown = {inp.drawdown:.1%}")
    if sentiment_note:
        summary_parts.append(sentiment_note)
    if inp.insider_sentiment <= -0.3:
        summary_parts.append(f"insiders selling ({inp.insider_sentiment:+.2f})")
    if inp.days_to_next_earnings is not None and inp.days_to_next_earnings <= 7:
        summary_parts.append(f"earnings in {inp.days_to_next_earnings}d")
    if inp.options_fear:
        summary_parts.append(f"options fear (P/C={inp.put_call_ratio:.2f})")
    if valuation_note:
        summary_parts.append(valuation_note)
    summary = " | ".join(summary_parts)

    return DecisionOutput(
        ticker=inp.ticker, date=inp.date,
        severity=severity, action=action,
        confidence=confidence, context=context,
        p_drawdown=p_dd,
        anomaly_score=inp.anomaly_score,
        anomaly_score_weighted=anomaly_w,
        drawdown_risk=inp.drawdown_risk,
        momentum_signal=mom_label,
        caution_flag=caution,
        override_reason=override_reason,
        summary=summary,
        sentiment_note=sentiment_note,
        trading_signal=trading_signal,
    )


def run_decision_engine(records: list) -> list:
    """Process list of dicts → list of DecisionOutput."""
    results = []
    for r in records:
        inp = AnomalyInput(**{
            k: v for k, v in r.items()
            if k in AnomalyInput.__dataclass_fields__
        })
        results.append(decide(inp))
    return results
