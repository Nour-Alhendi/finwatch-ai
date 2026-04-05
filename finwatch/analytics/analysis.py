"""
analytics/analysis.py
HTML analysis panels, anomaly explanations, and investor summaries.
"""

from analytics.signals import calculate_risk_drivers


def _tip(term: str, defn: str) -> str:
    """Wrap a financial term in a CSS tooltip span."""
    return f'<span class="tip" data-tip="{defn}">{term}</span>'


def build_analysis(det_df, dec_row, ticker, name, lang="english"):
    if det_df is None or det_df.empty:
        return None
    r   = det_df.iloc[-1]
    row = dec_row.iloc[0]

    rsi      = r.get("rsi", 50)
    mom5     = r.get("momentum_5", 0)
    mom10    = r.get("momentum_10", 0)
    vol      = r.get("volatility", 0)
    ret_1d   = r.get("returns", 0)
    drawdown = r.get("max_drawdown_30d", 0)
    regime   = str(r.get("regime", "unknown"))
    excess   = r.get("excess_return", 0)
    obv      = r.get("obv_signal", 0)
    sev      = row.get("severity", "NORMAL")
    direction= row.get("direction", "stable")
    p_down   = float(row.get("p_down", 0.33))
    p_up     = max(0.0, 1 - p_down - 0.15)
    mom_sig  = row.get("momentum_signal", "neutral")
    vol_ma20 = r.get("volume_ma20", 1)
    volume   = r.get("Volume", 0)
    date_str = str(r.get("Date", ""))[:10]
    action   = row.get("action", "—")
    caution  = row.get("caution_flag", None)

    def pos(icon="▲"): return f'<span class="an-pos">{icon}</span>'
    def risk(icon="●"): return f'<span class="an-risk">{icon}</span>'
    def warn(icon="▲"): return f'<span class="an-warn">{icon}</span>'
    def neu(icon="→"):  return f'<span class="an-neu">{icon}</span>'

    _TIPS_ALL = {
        "english": {
            "Trend: bullish":   "Bullish: Price is rising — positive market direction",
            "Trend: bearish":   "Bearish: Price is falling — negative market direction",
            "Trend: sideways":  "Sideways: No clear trend — price moves without direction",
            "Momentum: rising": "Rising: Price momentum is accelerating upward",
            "Momentum: falling":"Falling: Price momentum is accelerating downward",
            "Momentum: neutral":"Neutral: No clear directional force in the market",
            "overbought":       "Overbought (RSI >70): Too many buyers — correction possible",
            "oversold":         "Oversold (RSI <30): Too many sellers — recovery possible",
            "RSI:":             "RSI (0–100): Measures if a stock is overbought or oversold. >70 = overbought, <30 = oversold",
            "Volume:":          "Trading volume compared to the 20-day average",
            "elevated":         "Elevated volume: Unusually high trading — often on news or breakouts",
            "thin":             "Thin volume: Low trading activity — price direction less reliable",
            "Drawdown":         "Maximum loss from the last peak over 30 days",
            "Regime:":          "Market regime: Statistical environment of the stock (e.g. calm, volatile, trending)",
            "outperforming":    "Outperformance: Stock rising more than the overall market",
            "underperforming":  "Underperformance: Stock falling more than the overall market",
            "in line":          "In line: Stock moving similarly to the overall market",
            "OBV:":             "OBV (On-Balance Volume): Measures buying vs. selling pressure via volume",
            "buying pressure":  "Buying pressure: More buyers than sellers — bullish signal",
            "selling pressure": "Selling pressure: More sellers than buyers — bearish signal",
            "Return:":          "Daily return: Price change on the last trading day in %",
            "P(down)":          "AI probability: Price will fall in the next 5 days",
            "P(up)":            "AI probability: Price will rise in the next 5 days",
            "Ann. vol":         "Annualised volatility: Estimated yearly price swing in %",
            "Action:":          "Recommended action from the AI system based on all signals",
            "Risk level:":      "Overall risk level of the stock according to AI analysis",
            "CRITICAL":         "CRITICAL: Multiple risk models triggered simultaneously — immediate attention needed",
            "WARNING":          "WARNING: Elevated risk detected — close monitoring recommended",
            "WATCH":            "WATCH: Slightly elevated risk — keep situation under observation",
            "NORMAL":           "NORMAL: No elevated risk — stock behaving within normal range",
        },
        "german": {
            "Trend: bullish":   "Bullish: Der Kurs steigt – positive Marktrichtung",
            "Trend: bearish":   "Bearish: Der Kurs fällt – negative Marktrichtung",
            "Trend: sideways":  "Sideways: Kein klarer Trend – Kurs bewegt sich seitwärts",
            "Momentum: rising": "Rising: Kursdynamik beschleunigt sich nach oben",
            "Momentum: falling":"Falling: Kursdynamik beschleunigt sich nach unten",
            "Momentum: neutral":"Neutral: Keine klare Kraft in eine Richtung",
            "overbought":       "Überkauft (RSI >70): Zu viele Käufe – Korrektur möglich",
            "oversold":         "Überverkauft (RSI <30): Zu viele Verkäufe – Erholung möglich",
            "RSI:":             "RSI (0–100): Misst ob die Aktie über- oder unterkauft ist. >70 = überkauft, <30 = überverkauft",
            "Volume:":          "Handelsvolumen im Vergleich zum 20-Tage-Durchschnitt",
            "elevated":         "Erhöhtes Volumen: Ungewöhnlich viel Handel – oft bei News oder Ausbrüchen",
            "thin":             "Dünnes Volumen: Wenig Handel – Kursrichtung weniger verlässlich",
            "Drawdown":         "Maximaler Verlust vom letzten Höchststand in 30 Tagen",
            "Regime:":          "Marktregime: Statistisches Umfeld der Aktie (z.B. ruhig, volatil, trendend)",
            "outperforming":    "Outperformance: Aktie steigt stärker als der Gesamtmarkt",
            "underperforming":  "Underperformance: Aktie fällt stärker als der Gesamtmarkt",
            "in line":          "In line: Aktie bewegt sich ähnlich wie der Gesamtmarkt",
            "OBV:":             "OBV (On-Balance-Volume): Misst Kauf- vs. Verkaufsdruck anhand des Volumens",
            "buying pressure":  "Kaufdruck: Mehr Käufer als Verkäufer – bullishes Signal",
            "selling pressure": "Verkaufsdruck: Mehr Verkäufer als Käufer – bearishes Signal",
            "Return:":          "Tagesrendite: Kursveränderung am letzten Handelstag in %",
            "P(down)":          "KI-Wahrscheinlichkeit: Kurs fällt in den nächsten 5 Tagen",
            "P(up)":            "KI-Wahrscheinlichkeit: Kurs steigt in den nächsten 5 Tagen",
            "Ann. vol":         "Annualisierte Volatilität: Geschätzte jährliche Kursschwankung in %",
            "Action:":          "Empfohlene Maßnahme des KI-Systems basierend auf allen Signalen",
            "Risk level:":      "Gesamtes Risikoniveau der Aktie laut KI-Analyse",
            "CRITICAL":         "CRITICAL: Mehrere Risikomodelle schlagen an – sofortige Beobachtung nötig",
            "WARNING":          "WARNING: Erhöhtes Risiko erkannt – aufmerksame Beobachtung empfohlen",
            "WATCH":            "WATCH: Leicht erhöhtes Risiko – Situation im Auge behalten",
            "NORMAL":           "NORMAL: Kein erhöhtes Risiko – Aktie verhält sich im Rahmen",
        },
        "arabic": {
            "Trend: bullish":   "صاعد: السعر يرتفع — اتجاه إيجابي في السوق",
            "Trend: bearish":   "هابط: السعر ينخفض — اتجاه سلبي في السوق",
            "Trend: sideways":  "جانبي: لا اتجاه واضح — السعر يتحرك أفقياً",
            "Momentum: rising": "صاعد: زخم السعر يتسارع للأعلى",
            "Momentum: falling":"هابط: زخم السعر يتسارع للأسفل",
            "Momentum: neutral":"محايد: لا قوة اتجاهية واضحة في السوق",
            "overbought":       "مشتراة بإفراط (RSI >70): عمليات شراء مفرطة — تصحيح محتمل",
            "oversold":         "مباعة بإفراط (RSI <30): عمليات بيع مفرطة — ارتداد محتمل",
            "RSI:":             "RSI (0–100): يقيس ما إذا كانت الأسهم مشتراة أو مباعة بإفراط",
            "Volume:":          "حجم التداول مقارنةً بالمتوسط لـ20 يوماً",
            "elevated":         "حجم مرتفع: تداول غير معتاد — غالباً عند الأخبار أو الاختراقات",
            "thin":             "حجم ضعيف: نشاط تداول منخفض — اتجاه السعر أقل موثوقية",
            "Drawdown":         "أقصى خسارة من آخر ذروة خلال 30 يوماً",
            "Regime:":          "نظام السوق: البيئة الإحصائية للسهم (مثل: هادئ، متقلب، في اتجاه)",
            "outperforming":    "أداء متفوق: السهم يرتفع أكثر من السوق العام",
            "underperforming":  "أداء متأخر: السهم ينخفض أكثر من السوق العام",
            "in line":          "متوافق: السهم يتحرك بشكل مشابه للسوق العام",
            "OBV:":             "OBV: يقيس ضغط الشراء مقابل البيع من خلال حجم التداول",
            "buying pressure":  "ضغط شراء: المشترون أكثر من البائعين — إشارة صاعدة",
            "selling pressure": "ضغط بيع: البائعون أكثر من المشترين — إشارة هابطة",
            "Return:":          "العائد اليومي: تغير السعر في آخر يوم تداول بالنسبة المئوية",
            "P(down)":          "احتمال الذكاء الاصطناعي: انخفاض السعر خلال 5 أيام القادمة",
            "P(up)":            "احتمال الذكاء الاصطناعي: ارتفاع السعر خلال 5 أيام القادمة",
            "Ann. vol":         "التذبذب السنوي: تقدير التأرجح السعري السنوي بالنسبة المئوية",
            "Action:":          "الإجراء الموصى به من نظام الذكاء الاصطناعي بناءً على جميع الإشارات",
            "Risk level:":      "مستوى المخاطرة الإجمالي للسهم وفق تحليل الذكاء الاصطناعي",
            "CRITICAL":         "حرج: نماذج مخاطر متعددة تنبّه في آن واحد — يحتاج انتباهاً فورياً",
            "WARNING":          "تحذير: مخاطرة مرتفعة — مراقبة دقيقة موصى بها",
            "WATCH":            "مراقبة: مخاطرة مرتفعة قليلاً — ابقِ الوضع تحت المراقبة",
            "NORMAL":           "طبيعي: لا مخاطرة مرتفعة — السهم يتصرف ضمن النطاق الطبيعي",
        },
    }
    _TIPS = _TIPS_ALL.get(lang, _TIPS_ALL["english"])

    def r_row(ic, txt):
        tip = next((v for k, v in _TIPS.items() if k.lower() in txt.lower()), "")
        if tip:
            tip = tip.replace('&', '&amp;').replace('"', '&quot;').replace('<', '&lt;').replace('>', '&gt;')
        tip_attr = f' data-tip="{tip}"' if tip else ""
        return f'<div class="an-row">{ic}<span class="an-text"{tip_attr}>{txt}</span></div>'

    if "up" in regime or "bull" in regime or (mom5 > 0 and mom10 > 0):
        trend_ic, trend_str = pos(), "bullish"
    elif "down" in regime or "bear" in regime or (mom5 < 0 and mom10 < 0):
        trend_ic, trend_str = risk(), "bearish"
    else:
        trend_ic, trend_str = neu(), "sideways"

    mom_ic = pos() if mom_sig == "rising" else risk() if mom_sig == "falling" else neu()
    sev_ic = pos() if sev in ("NORMAL","POSITIVE_MOMENTUM") else risk() if sev in ("CRITICAL","WARNING") else warn()

    if rsi > 70:    rsi_ic, rsi_str = risk(), f"overbought ({rsi:.1f})"
    elif rsi < 30:  rsi_ic, rsi_str = pos(),  f"oversold ({rsi:.1f}) — recovery signal"
    elif rsi > 55:  rsi_ic, rsi_str = pos(),  f"strong ({rsi:.1f})"
    elif rsi < 45:  rsi_ic, rsi_str = risk(), f"weak ({rsi:.1f})"
    else:           rsi_ic, rsi_str = neu(),  f"neutral ({rsi:.1f})"

    vol_ratio = (volume / vol_ma20) if vol_ma20 > 0 else 1
    if vol_ratio > 1.5:    vol_ic, vol_str = warn("▲"), f"{vol_ratio:.1f}x avg (elevated)"
    elif vol_ratio < 0.6:  vol_ic, vol_str = risk(),    f"{vol_ratio:.1f}x avg (thin)"
    elif vol_ratio >= 0.9: vol_ic, vol_str = pos(),     f"{vol_ratio:.1f}x avg (normal)"
    else:                  vol_ic, vol_str = neu(),     f"{vol_ratio:.1f}x avg (below avg)"

    dd_str = f"{drawdown*100:.1f}%" if drawdown else "N/A"
    if drawdown and abs(drawdown) > 0.15:   dd_ic = risk("▼")
    elif drawdown and abs(drawdown) > 0.08: dd_ic = warn("▼")
    else:                                   dd_ic = neu()

    if excess > 0.01:    ctx_ic, ctx_str = pos(),  f"outperforming +{excess*100:.1f}%"
    elif excess < -0.01: ctx_ic, ctx_str = risk(), f"underperforming {excess*100:.1f}%"
    else:                ctx_ic, ctx_str = neu(),  "in line w/ market"

    obv_ic  = pos() if obv > 0 else risk()
    obv_str = "buying pressure" if obv > 0 else "selling pressure"
    ret_ic  = pos() if ret_1d > 0.005 else risk() if ret_1d < -0.005 else neu()

    d_arrow = "▲" if direction == "up" else "▼" if direction == "down" else "→"
    d_conf  = int((1 - p_down) * 100) if direction == "up" else int(p_down * 100)
    dir_ic  = pos(d_arrow) if direction == "up" else risk(d_arrow) if direction == "down" else neu(d_arrow)
    pd_ic   = risk() if p_down > 0.5 else pos()

    vol_ann = vol * 100 * 16
    va_ic   = risk() if vol_ann > 40 else warn() if vol_ann > 25 else pos()
    act_ic  = risk() if sev in ("CRITICAL","WARNING") else pos()

    exec_html = (
        r_row(trend_ic, f"Trend: {trend_str}") +
        r_row(mom_ic,   f"Momentum: {mom_sig}") +
        r_row(sev_ic,   f"Risk level: {sev.replace('_',' ')}")
    )
    tech_html = (
        r_row(rsi_ic, f"RSI: {rsi_str}") +
        r_row(vol_ic, f"Volume: {vol_str}") +
        r_row(dd_ic,  f"Drawdown 30D: {dd_str}") +
        r_row(neu(),  f"Regime: {regime}")
    )
    ctx_html = (
        r_row(ctx_ic, ctx_str) +
        r_row(obv_ic, f"OBV: {obv_str}") +
        r_row(ret_ic, f"Return: {ret_1d*100:+.2f}%")
    )
    fcast_html = (
        r_row(dir_ic, f"{direction.upper()} — {d_conf}% confidence") +
        r_row(pd_ic,  f"P(down) {p_down*100:.0f}% · P(up) {p_up*100:.0f}%") +
        r_row(va_ic,  f"Ann. vol: {vol_ann:.1f}%") +
        r_row(act_ic, f"Action: {action}")
    )
    if caution:
        fcast_html += r_row(warn("⚠"), caution)

    main_html = (
        '<div class="analysis-panel">'
        f'<div class="an-header">Market Analysis \u2014 {name} ({ticker})'
        f'<span class="an-date">{date_str}</span></div>'
        '<div class="an-grid">'
        f'<div class="an-section"><div class="an-title">Executive Summary</div>{exec_html}</div>'
        f'<div class="an-section"><div class="an-title">Technical Picture</div>{tech_html}</div>'
        f'<div class="an-section"><div class="an-title">Market Context</div>{ctx_html}</div>'
        f'<div class="an-section"><div class="an-title">AI Forecast (5D)</div>{fcast_html}</div>'
        "</div></div>"
    )
    return main_html, None


def explain_anomaly(r, ticker_name, date_str, dec_row=None):
    detectors = []
    if r.get("z_anomaly",  False): detectors.append(f"Z-Score (30d): Return deviated strongly from 30-day norm (z = {r.get('z_score', 0):.2f})")
    if r.get("z_anomaly_60",False): detectors.append(f"Z-Score (60d): Deviation confirmed over 60-day window (z = {r.get('z_score_60', 0):.2f})")
    if r.get("if_anomaly", False): detectors.append("Isolation Forest: Multivariate pattern inconsistent with historical calm periods")
    if r.get("ae_anomaly", False): detectors.append(f"LSTM Autoencoder: High reconstruction error — sequence breaks learned pattern (error = {r.get('ae_error', 0):.4f})")
    if not detectors:
        return None
    ctx = []
    ret = r.get("returns", None)
    if ret is not None: ctx.append(f"Daily return: {ret*100:.2f}%")
    vol = r.get("volatility", None)
    if vol is not None: ctx.append(f"Volatility: {vol*100:.2f}%")
    rsi = r.get("rsi", None)
    if rsi is not None: ctx.append(f'{_tip("RSI", "Relative Strength Index — momentum indicator. Above 70: overbought. Below 30: oversold.")}: {rsi:.1f}')
    vz = r.get("volume_zscore", None)
    if vz is not None and abs(vz) > 1.5: ctx.append(f'Volume spike ({_tip("z", "Z-Score — how many standard deviations from the historical average.")} = {vz:.1f})')
    if r.get("market_anomaly", False): ctx.append("Market-wide event")
    if r.get("sector_anomaly", False): ctx.append("Sector-wide event")
    if r.get("is_high_volume", False): ctx.append("Unusually high volume")
    score    = int(r.get("anomaly_score", len(detectors)))
    det_html = "<br>".join(detectors)

    def _pill(label: str, color: str) -> str:
        return (
            f'<span style="display:inline-block;background:rgba(30,45,65,0.6);'
            f'border:1px solid {color}33;border-radius:4px;padding:2px 8px;'
            f'margin:3px 4px 3px 0;font-size:12px;color:{color};'
            f'font-family:\'IBM Plex Mono\',monospace;white-space:nowrap">{label}</span>'
        )

    pills = []
    ret = r.get("returns", None)
    if ret is not None:
        c = "#1de9b6" if ret >= 0 else "#f85149"
        pills.append(_pill(f"Return: {ret*100:+.2f}%", c))
    vol = r.get("volatility", None)
    if vol is not None:
        c = "#f85149" if vol > 0.03 else "#e3b341" if vol > 0.015 else "#b8c4ce"
        vol_tip = _tip("Volatility", "Average daily price swing. Above 3%: high risk. Above 1.5%: elevated.")
        pills.append(_pill(f"{vol_tip}: {vol*100:.2f}%", c))
    rsi = r.get("rsi", None)
    if rsi is not None:
        c = "#f85149" if rsi > 70 else "#1de9b6" if rsi < 30 else "#b8c4ce"
        pills.append(_pill(f'{_tip("RSI", "Relative Strength Index. Above 70: overbought. Below 30: oversold.")}: {rsi:.1f}', c))
    vz = r.get("volume_zscore", None)
    if vz is not None and abs(vz) > 1.5:
        pills.append(_pill(f'Volume spike (z={vz:.1f})', "#e3b341"))
    if r.get("market_anomaly", False):
        pills.append(_pill("Market-wide event", "#a371f7"))
    if r.get("sector_anomaly", False):
        pills.append(_pill("Sector-wide event", "#58a6ff"))
    if r.get("is_high_volume", False):
        pills.append(_pill("High volume", "#e3b341"))

    ctx_html = (
        f'<div style="margin-top:10px;padding-top:8px;'
        f'border-top:1px solid #1a2332">{"".join(pills)}</div>'
    ) if pills else ""

    # ── Why is Risk Elevated? ─────────────────────────────────────────────────
    drivers_html = ""
    if dec_row is not None:
        dec_dict = dec_row.to_dict() if hasattr(dec_row, "to_dict") else dict(dec_row)
        r_dict   = r.to_dict()       if hasattr(r,       "to_dict") else dict(r)
        sev      = str(dec_dict.get("severity", "NORMAL"))
        if sev in ("CRITICAL", "WARNING", "WATCH"):
            drivers = calculate_risk_drivers(r_dict, dec_dict)
            if drivers:
                _lvl_color = {"high": "#f85149", "medium": "#e3b341", "low": "#8b949e"}
                _lvl_icon  = {"high": "●", "medium": "▲", "low": "→"}
                driver_rows = "".join(
                    f'<div style="display:flex;align-items:flex-start;gap:6px;padding:3px 0">'
                    f'<span style="color:{_lvl_color.get(lvl,"#8b949e")};font-size:11px;flex-shrink:0">{_lvl_icon.get(lvl,"→")}</span>'
                    f'<span style="font-size:12px;color:#b8c4ce;font-family:\'IBM Plex Mono\',monospace;line-height:1.6">{txt}</span>'
                    f'</div>'
                    for lvl, txt in drivers[:6]
                )
                drivers_html = (
                    f'<div style="margin-top:12px;padding-top:10px;border-top:1px solid #1a2332">'
                    f'<div style="font-size:10px;letter-spacing:1.5px;text-transform:uppercase;'
                    f'color:#3d5266;font-family:\'IBM Plex Mono\',monospace;margin-bottom:8px">'
                    f'Why is Risk Elevated?</div>'
                    f'{driver_rows}</div>'
                )

    return (
        "<div style=\"background:#0d1117;border:1px solid #1a2332;border-radius:3px;"
        "padding:12px 16px;margin-top:4px\">"
        "<div style=\"font-size:12px;color:#cdd9e5;font-family:'IBM Plex Mono',monospace;"
        f"margin-bottom:4px\">{ticker_name} — {date_str}</div>"
        "<div style=\"font-size:11px;color:#5c7080;font-family:'IBM Plex Mono',monospace;"
        f"margin-bottom:10px\">{score} of 4 detectors triggered</div>"
        "<div style=\"font-size:12px;color:#8b949e;line-height:1.9;"
        f"font-family:'IBM Plex Mono',monospace\">{det_html}</div>"
        f"{ctx_html}"
        f"{drivers_html}"
        "</div>"
    )


def build_investor_summary(det_df, dec_row, row, news_df, ticker) -> str:
    if det_df is None or det_df.empty:
        return ""
    last = det_df.iloc[-1]
    signals = []

    sev    = str(row.get("severity", "NORMAL"))
    action = str(row.get("action", "MONITOR"))
    conf   = float(row.get("confidence", 0))
    sev_color = {"CRITICAL":"#e05252","WARNING":"#e0a030","NORMAL":"#52b788"}.get(sev, "#b8c4ce")
    signals.append(
        f'<span style="color:{sev_color};font-weight:600">{sev}</span>'
        f'<span style="color:#7a8fa8"> — Action: </span>'
        f'<span style="color:#b8c4ce">{action}</span>'
        f'<span style="color:#7a8fa8"> · {_tip("Confidence", "Model precision on similar signals in historical backtests.")}: </span>'
        f'<span style="color:#b8c4ce">{int(conf*100)}%</span>'
    )

    # ── Risk drivers — "why is risk elevated?" ─────────────────────────────
    if sev in ("CRITICAL", "WARNING", "WATCH"):
        drivers = calculate_risk_drivers(
            last.to_dict() if hasattr(last, "to_dict") else dict(last),
            row.to_dict()  if hasattr(row,  "to_dict") else dict(row),
        )
        if drivers:
            _lvl_color = {"high": "#e05252", "medium": "#e0a030", "low": "#8b949e"}
            _lvl_icon  = {"high": "●", "medium": "▲", "low": "→"}
            driver_parts = [
                f'<span style="color:{_lvl_color.get(lvl,"#8b949e")}">{_lvl_icon.get(lvl,"→")}</span> {txt}'
                for lvl, txt in drivers[:5]
            ]
            signals.append(
                f'<span style="color:#7a8fa8;font-size:12px">Risk driven by:</span><br>'
                + "<br>".join(
                    f'<span style="color:#b8c4ce;padding-left:10px">{d}</span>'
                    for d in driver_parts
                )
            )

    detectors = []
    if last.get("z_anomaly"):    detectors.append("Z-Score (30D)")
    if last.get("z_anomaly_60"): detectors.append("Z-Score (60D)")
    if last.get("if_anomaly"):   detectors.append("Isolation Forest")
    if last.get("ae_anomaly"):   detectors.append("LSTM Autoencoder")
    _det_tips = {
        "Z-Score (30D)":      _tip("Z-Score (30D)",      "How unusual the return is vs. its 30-day history."),
        "Z-Score (60D)":      _tip("Z-Score (60D)",      "How unusual the return is vs. its 60-day history."),
        "Isolation Forest":   _tip("Isolation Forest",   "ML model detecting multivariate outliers in price & volume."),
        "LSTM Autoencoder":   _tip("LSTM Autoencoder",   "Deep learning model that flags unusual price sequences."),
    }
    det_labels = [_det_tips.get(d, d) for d in detectors]
    if detectors:
        signals.append(
            f'<span style="color:#e05252">⬤</span>'
            f'<span style="color:#b8c4ce"> {_tip("Anomaly", "AI flagged unusual price or volume behavior.")} detected [{", ".join(det_labels)}]</span>'
        )
    else:
        signals.append(
            f'<span style="color:#52b788">▲</span>'
            f'<span style="color:#b8c4ce"> No {_tip("anomalies", "AI found no unusual price or volume patterns — behavior is within normal range.")} detected</span>'
        )

    direction, p_down = "stable", 0.33
    if dec_row is not None and not dec_row.empty:
        _r = dec_row.iloc[0]
        direction = str(_r.get("direction", "stable"))
        p_down    = float(_r.get("p_down", 0.33))
    p_up = max(0.0, 1 - p_down - 0.15)
    dir_icon  = {"up":"▲","down":"▼","stable":"→"}.get(direction, "→")
    dir_color = {"up":"#1de9b6","down":"#e05252","stable":"#adb5bd"}.get(direction, "#adb5bd")
    signals.append(
        f'<span style="color:{dir_color}">{dir_icon}</span>'
        f'<span style="color:#b8c4ce"> AI 5-day forecast: <strong style="color:{dir_color}">{direction.upper()}</strong>'
        f' — {_tip("P(up)", "AI probability estimate for a price rise in the next 5 days.")} {int(p_up*100)}%'
        f' · {_tip("P(down)", "AI probability estimate for a price drop in the next 5 days.")} {int(p_down*100)}%</span>'
    )

    mom = "neutral"
    if dec_row is not None and not dec_row.empty:
        mom = str(dec_row.iloc[0].get("momentum_signal", "neutral"))
    elif "momentum_signal" in last.index:
        mom = str(last.get("momentum_signal", "neutral"))
    mom_icon  = {"rising":"▲","falling":"▼","neutral":"→"}.get(mom, "→")
    mom_color = {"rising":"#1de9b6","falling":"#e05252","neutral":"#adb5bd"}.get(mom, "#adb5bd")
    signals.append(
        f'<span style="color:{mom_color}">{mom_icon}</span>'
        f'<span style="color:#b8c4ce"> {_tip("Momentum", "Speed and direction of recent price change. Rising = accelerating upward.")}: <strong style="color:{mom_color}">{mom.capitalize()}</strong></span>'
    )

    if news_df is not None and ticker in news_df["ticker"].values:
        nr = news_df[news_df["ticker"] == ticker].iloc[0]
        sentiments = nr.get("news_sentiment", [])
        if isinstance(sentiments, list) and sentiments:
            counts  = {"positive":0,"negative":0,"neutral":0}
            for s in sentiments:
                counts[s] = counts.get(s, 0) + 1
            dominant   = max(counts, key=counts.get)
            sent_icon  = {"positive":"▲","negative":"⬤","neutral":"→"}.get(dominant, "→")
            sent_color = {"positive":"#52b788","negative":"#e05252","neutral":"#adb5bd"}.get(dominant, "#adb5bd")
            signals.append(
                f'<span style="color:{sent_color}">{sent_icon}</span>'
                f'<span style="color:#b8c4ce"> News sentiment: <strong style="color:{sent_color}">{dominant.capitalize()}</strong></span>'
            )

    rows_html = "".join(
        f'<div style="padding:4px 0;border-bottom:1px solid #1a2332;font-size:11px;'
        f'font-family:\'IBM Plex Sans\',sans-serif;line-height:1.6">{s}</div>'
        for s in signals
    )
    return f"""
    <div style="background:#0d1117;border:1px solid #1a2332;border-radius:3px;
                padding:14px 20px;margin-top:8px;margin-bottom:4px">
      <div style="font-size:10px;letter-spacing:2px;text-transform:uppercase;
                  color:#5c7080;font-family:'IBM Plex Mono',monospace;
                  border-bottom:1px solid #1a2332;padding-bottom:8px;margin-bottom:10px">
        Investor Summary
      </div>
      {rows_html}
    </div>"""
