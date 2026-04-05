"""
FinWatch AI — Layer 7E: News + Sentiment Analysis
==================================================
Dual sentiment engine:
  - VADER : social media tone, general headlines (fast, rule-based)
  - Groq  : Llama 3.3 70B — contextual financial analysis of all headlines together
  - Combined: recency-weighted VADER + Groq contextual score

Why Groq instead of FinBERT:
  FinBERT scores each headline in isolation without understanding implications.
  Groq sees all headlines at once and weighs their financial impact in context.
  Example: "CEO sells 50% of shares" → FinBERT: neutral, Groq: bearish ✓

Output columns: vader_score, finbert_score (= groq score), news_sentiment_score (combined)
"""

import os
import json
import math
import time
import logging
import pandas as pd
from pathlib import Path
from datetime import date, timedelta, datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests

ROOT = Path(__file__).resolve().parents[2]

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

FINNHUB_API_KEY  = os.environ.get("FINNHUB_API_KEY")
GROQ_API_KEY     = os.environ.get("GROQ_API_KEY")
NEWS_ENDPOINT    = "https://finnhub.io/api/v1/company-news"
NEWS_LOOKBACK_DAYS = 7

_ANALYZER = SentimentIntensityAnalyzer()


# ── News Fetching ──────────────────────────────────────────────────────────

def fetch_news(ticker: str, limit: int = 5, retries: int = 3):
    """
    Fetch latest news from Finnhub (last 7 days).
    Returns (headlines, sources, timestamps).
    """
    if not FINNHUB_API_KEY:
        return [], [], []

    today  = date.today()
    params = {
        "symbol": ticker,
        "from":   (today - timedelta(days=NEWS_LOOKBACK_DAYS)).isoformat(),
        "to":     today.isoformat(),
        "token":  FINNHUB_API_KEY,
    }

    for attempt in range(retries):
        try:
            r = requests.get(NEWS_ENDPOINT, params=params, timeout=10)
            r.raise_for_status()
            articles = r.json()
            if not isinstance(articles, list):
                return [], [], []
            valid      = [a for a in articles[:limit] if "headline" in a]
            headlines  = [a["headline"] for a in valid]
            sources    = [a.get("source", "") for a in valid]
            timestamps = [a.get("datetime", 0) for a in valid]
            return headlines, sources, timestamps
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response else "?"
            if status == 429:
                wait = 2 ** attempt
                logging.warning(f"[sentiment] Rate-limited for {ticker}, retrying in {wait}s...")
                time.sleep(wait)
            else:
                logging.error(f"[sentiment] HTTP {status} for {ticker}: {e}")
                return [], [], []
        except Exception as e:
            logging.error(f"[sentiment] Request failed for {ticker}: {e}")
            return [], [], []

    return [], [], []


# ── VADER ──────────────────────────────────────────────────────────────────

def _vader_score(headline: str) -> float:
    """VADER compound score (-1 to +1)."""
    return round(_ANALYZER.polarity_scores(headline)["compound"], 4)


# ── Groq Contextual Sentiment ──────────────────────────────────────────────

def _groq_sentiment(headlines: list, ticker: str) -> tuple[float, str]:
    """
    Send all headlines to Groq (Llama 3.3 70B) for contextual financial sentiment.
    Returns (score: float -1 to +1, reasoning: str).
    Falls back to 0.0 if Groq unavailable.
    """
    if not GROQ_API_KEY or not headlines:
        return 0.0, ""

    numbered = "\n".join(f"{i+1}. {h}" for i, h in enumerate(headlines))
    prompt = f"""You are a senior financial analyst. Analyze these {len(headlines)} recent news headlines for {ticker}.

Headlines:
{numbered}

Evaluate the OVERALL financial sentiment considering:
- Direct financial impact (earnings, revenue, costs)
- Management signals (insider selling/buying, guidance)
- Regulatory/legal risks
- Competitive position
- Market sentiment implications

Respond with ONLY valid JSON in this exact format:
{{"score": <float from -1.0 to 1.0>, "reasoning": "<one sentence explanation>"}}

Where: -1.0 = strongly bearish, 0.0 = neutral, +1.0 = strongly bullish"""

    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=120,
            temperature=0.1,   # low temperature for consistent scoring
        )
        raw = response.choices[0].message.content.strip()
        # Extract JSON — handle cases where model adds extra text
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start == -1 or end == 0:
            logging.warning(f"[sentiment] Groq returned no JSON for {ticker}: {raw[:100]}")
            return 0.0, ""
        data      = json.loads(raw[start:end])
        score     = float(data.get("score", 0.0))
        score     = max(-1.0, min(1.0, score))   # clamp to [-1, 1]
        reasoning = str(data.get("reasoning", ""))
        return round(score, 4), reasoning
    except Exception as e:
        logging.warning(f"[sentiment] Groq sentiment failed for {ticker}: {e}")
        return 0.0, ""


# ── Recency Weighting ──────────────────────────────────────────────────────

def _recency_weight(unix_ts: int) -> float:
    """Exponential decay: half-life = 7 days."""
    if unix_ts <= 0:
        return 1.0
    try:
        article_date = datetime.utcfromtimestamp(unix_ts).date()
        days_old     = max(0, (date.today() - article_date).days)
        return math.exp(-days_old / 7)
    except Exception:
        return 1.0


# ── Aggregate Scores ────────────────────────────────────────────────────────

def aggregate_scores(headlines: list, timestamps: list,
                     ticker: str = "") -> dict:
    """
    Compute sentiment scores for a list of headlines.

    VADER: per-headline, recency-weighted average
    Groq:  single contextual call for all headlines together (much better)
    Combined: 0.4 * vader + 0.6 * groq  (groq weighted higher — more intelligent)

    Returns: {vader_score, finbert_score (=groq), news_sentiment_score, n_articles}
    """
    if not headlines:
        return {
            "vader_score": 0.0, "finbert_score": 0.0,
            "news_sentiment_score": 0.0, "n_articles": 0,
        }

    # VADER — per headline with recency weighting
    total_weight = 0.0
    vader_sum    = 0.0
    for headline, ts in zip(headlines, timestamps):
        w          = _recency_weight(ts)
        vader_sum += _vader_score(headline) * w
        total_weight += w
    vader_avg = round(vader_sum / total_weight, 4) if total_weight > 0 else 0.0

    # Groq — one contextual call for all headlines
    groq_score, groq_reasoning = _groq_sentiment(headlines, ticker)

    # Combined: Groq gets higher weight (more context-aware)
    if GROQ_API_KEY and headlines:
        combined = round(0.4 * vader_avg + 0.6 * groq_score, 4)
    else:
        combined = vader_avg   # fallback to VADER only

    return {
        "vader_score":          vader_avg,
        "finbert_score":        groq_score,    # column name kept for compatibility
        "news_sentiment_score": combined,
        "n_articles":           len(headlines),
        "groq_reasoning":       groq_reasoning,
    }


# ── Per-headline label ─────────────────────────────────────────────────────

def label(score: float) -> str:
    if score >= 0.05:  return "positive"
    if score <= -0.05: return "negative"
    return "neutral"


def score_headline(headline: str) -> dict:
    """Score a single headline with VADER only (fast path)."""
    vader = _vader_score(headline)
    return {"vader": vader, "finbert": vader, "combined": vader}


def analyze_sentiment(headline: str) -> str:
    return label(_vader_score(headline))


# ── Batch Enrichment (Layer 7E pipeline) ──────────────────────────────────

def enrich_with_news(df: pd.DataFrame) -> pd.DataFrame:
    """Adds top news + dual sentiment (VADER + Groq) to LLM output."""
    enriched = []
    total    = len(df)

    groq_available = bool(GROQ_API_KEY)
    if groq_available:
        print("  Sentiment engine: VADER + Groq (Llama 3.3 70B)")
    else:
        print("  Sentiment engine: VADER only (set GROQ_API_KEY for Groq)")

    for i, (_, row) in enumerate(df.iterrows(), 1):
        ticker = row["ticker"]
        print(f"  [{i}/{total}] {ticker}...", end=" ", flush=True)

        headlines, sources, timestamps = fetch_news(ticker, limit=5, retries=1)
        time.sleep(1.0)

        if not headlines:
            print("no news")
            enriched.append({
                "ticker":               ticker,
                "llm_summary":          row["llm_summary"],
                "top_news":             [],
                "news_sources":         [],
                "news_sentiment":       [],
                "vader_score":          0.0,
                "finbert_score":        0.0,
                "news_sentiment_score": 0.0,
                "n_articles":           0,
                "groq_reasoning":       "",
            })
            continue

        scores     = aggregate_scores(headlines, timestamps, ticker=ticker)
        sentiments = [label(_vader_score(h)) for h in headlines]

        print(
            f"vader={scores['vader_score']:+.3f}  "
            f"groq={scores['finbert_score']:+.3f}  "
            f"combined={scores['news_sentiment_score']:+.3f}"
            + (f"  → {scores['groq_reasoning'][:60]}" if scores.get("groq_reasoning") else "")
        )

        enriched.append({
            "ticker":               ticker,
            "llm_summary":          row["llm_summary"],
            "top_news":             headlines,
            "news_sources":         sources,
            "news_sentiment":       sentiments,
            "vader_score":          scores["vader_score"],
            "finbert_score":        scores["finbert_score"],
            "news_sentiment_score": scores["news_sentiment_score"],
            "n_articles":           scores["n_articles"],
            "groq_reasoning":       scores.get("groq_reasoning", ""),
        })

    return pd.DataFrame(enriched)


if __name__ == "__main__":
    llm_path    = ROOT / "data/explanations/llm_summaries.parquet"
    df_llm      = pd.read_parquet(llm_path)
    enriched_df = enrich_with_news(df_llm)
    out_path    = ROOT / "data/explanations/llm_news_enriched.parquet"
    enriched_df.to_parquet(out_path, index=False)
    print(f"\nSaved: {out_path}")
