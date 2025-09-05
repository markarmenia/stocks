#!/usr/bin/env python3
"""
Telegram Day-Trader Stock Advisor (Informational Only)

DISCLAIMER: This bot provides *informational* analysis only and is NOT financial advice.
It should not be used as the sole basis for investment decisions. Markets are risky.
"""

from __future__ import annotations

import asyncio
import os
import re
import json
import html
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Callable, Tuple

import httpx
import pyfiglet
from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart, Command
from aiogram.types import (
    Message,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    ReplyKeyboardMarkup,
    KeyboardButton,
    CallbackQuery,
    BufferedInputFile,
)
from openai import OpenAI
from dotenv import load_dotenv
from urllib.parse import urlparse
from tabulate import tabulate
import pandas as pd

# -------------------------------------------------------------------
# Environment / Config
# -------------------------------------------------------------------
load_dotenv()

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
USE_SERPAPI = os.getenv("USE_SERPAPI", "false").lower() == "true"  # enable with USE_SERPAPI=true
SUBSCRIPTIONS_FILE = "subscriptions.json"
MAX_SUBSCRIPTIONS = 5

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not TELEGRAM_BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN environment variable is required. Please set it in your .env file.")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required. Please set it in your .env file.")

# defeatbeta_api import path (fixes ImportError on some installs)
from defeatbeta_api.data.ticker import Ticker

# -------------------------------------------------------------------
# Globals (simple persistence/state)
# -------------------------------------------------------------------
user_subscriptions: Dict[str, List[str]] = {}  # user_id -> [TICKER,...]
user_state: Dict[str, str] = {}  # user_id -> "awaiting_ticker_subscription"

# Data windowing
PRICE_WINDOW_DAYS = 365               # CSV scope for price
EVENTS_WINDOW_DAYS = 365              # CSV scope for earnings, calendar, splits, dividends
PREVIEW_WINDOW_DAYS = 31              # UI preview scope (~1 month)

# Telegram caps
TELEGRAM_HTML_LIMIT = 4096            # Telegram cap per message

# -------------------------------------------------------------------
# Formatting & Safe-Send Helpers
# -------------------------------------------------------------------
def format_df(df: pd.DataFrame) -> str:
    """Return PLAIN TEXT (no HTML)."""
    return tabulate(df, headers="keys", tablefmt="psql")

def format_pretty_table(data: Any) -> str:
    """Return PLAIN TEXT (no HTML)."""
    if isinstance(data, pd.DataFrame):
        return format_df(data)
    return str(data)

async def send_pre_block(target: Message | CallbackQuery, text: str):
    """Send a single <pre> block safely (avoid broken HTML)."""
    block = f"<pre>{html.escape(text)}</pre>"
    if len(block) > TELEGRAM_HTML_LIMIT:
        inner = html.escape(text)[: (TELEGRAM_HTML_LIMIT - len("<pre></pre>"))]
        block = f"<pre>{inner}</pre>"
    if isinstance(target, Message):
        await target.answer(block, parse_mode="HTML")
    else:
        await target.message.answer(block, parse_mode="HTML")

def clamp_len(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"

def is_ticker(text: str) -> Optional[str]:
    m = re.fullmatch(r"\$?[A-Z]{1,5}(?:\.[A-Z]{1,3})?", text.strip())
    return m.group(0).lstrip("$") if m else None

# -------------------------------------------------------------------
# Data Windowing Helpers (filter by last N days)
# -------------------------------------------------------------------
def filter_last_days_df(
    df: pd.DataFrame,
    days: int,
    date_cols: Tuple[str, ...] = ("report_date", "date", "Date", "datetime", "Datetime", "time", "Time"),
) -> pd.DataFrame:
    """
    Filter DataFrame to last <days> using either a DatetimeIndex or a date-like column.
    If no suitable field exists, return a small tail as a fallback.
    """
    if df is None or len(df) == 0:
        return df

    cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=days)

    if isinstance(df.index, pd.DatetimeIndex):
        return df[df.index >= cutoff]

    for col in date_cols:
        if col in df.columns:
            dt = pd.to_datetime(df[col], utc=True, errors="coerce")
            mask = dt >= cutoff
            return df.loc[mask]

    # Fallback: roughly last N rows (heuristic)
    return df.tail(min(len(df), days))

def sort_newest_first(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by datetime index or by a likely date column, newest first."""
    if isinstance(df.index, pd.DatetimeIndex):
        return df.sort_index(ascending=False)
    for c in ("report_date", "date", "Date", "datetime", "Datetime", "time", "Time"):
        if c in df.columns:
            try:
                return df.sort_values(by=c, ascending=False)
            except Exception:
                pass
    return df

# -------------------------------------------------------------------
# Persistence
# -------------------------------------------------------------------
def load_subscriptions():
    global user_subscriptions
    if os.path.exists(SUBSCRIPTIONS_FILE):
        try:
            with open(SUBSCRIPTIONS_FILE, "r") as f:
                user_subscriptions = json.load(f)
        except json.JSONDecodeError:
            user_subscriptions = {}
    else:
        user_subscriptions = {}

def save_subscriptions():
    with open(SUBSCRIPTIONS_FILE, "w") as f:
        json.dump(user_subscriptions, f, indent=4)

# -------------------------------------------------------------------
# News / Social Sources
# -------------------------------------------------------------------
NEWS_ALLOWLIST = [
    "reuters.com",
    "bloomberg.com",
    "wsj.com",
    "ft.com",
    "cnbc.com",
    "apnews.com",
    "marketwatch.com",
    "seekingalpha.com",
    "investopedia.com",
    "finance.yahoo.com",
    "forbes.com",
]

ISO8601 = "%Y-%m-%dT%H:%M:%SZ"

def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def twenty_four_hours_ago() -> datetime:
    return now_utc() - timedelta(hours=24)

def normalize_domain(u: str) -> str:
    try:
        host = urlparse(u).netloc.lower().rstrip(".")
        return host[4:] if host.startswith("www.") else host
    except Exception:
        return ""

# --- News date parsing helpers ---
_REL_UNITS = {
    "min": "minutes", "mins": "minutes", "minute": "minutes", "minutes": "minutes",
    "hour": "hours", "hours": "hours", "hr": "hours", "hrs": "hours",
    "day": "days", "days": "days", "d": "days",
}
def parse_pubdate(raw: Optional[str]) -> Optional[pd.Timestamp]:
    """Parse ISO8601 ('2025-08-24T12:34:56Z') OR relative ('3 hours ago'). Return UTC timestamp or None."""
    if not raw:
        return None
    try:
        return pd.to_datetime(raw, utc=True)  # absolute first
    except Exception:
        pass
    m = re.search(r"(\d+)\s*(min|mins|minute|minutes|hour|hours|hr|hrs|day|days|d)\s+ago", str(raw), re.IGNORECASE)
    if m:
        n = int(m.group(1))
        unit = _REL_UNITS[m.group(2).lower()]
        return pd.Timestamp.utcnow() - pd.to_timedelta(n, unit=unit)
    return None

async def fetch_news_newsapi(client: httpx.AsyncClient, ticker: str) -> List[Dict[str, Any]]:
    key = os.getenv("NEWS_API_KEY")
    if not key or key == "your_news_api_key_here":
        print(f"[NewsAPI] Missing NEWS_API_KEY. Skipping for {ticker}.")
        return []
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": ticker,
        "apiKey": key,
        # Constrain strictly to last 24h and sort by publish time
        "from": twenty_four_hours_ago().strftime(ISO8601),
        "to": now_utc().strftime(ISO8601),
        "sortBy": "publishedAt",
        "language": "en",
        "pageSize": 50,
    }
    try:
        r = await client.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        results = []
        for n in data.get("articles", []):
            link = n.get("url") or ""
            domain = normalize_domain(link)
            if domain and NEWS_ALLOWLIST and domain not in NEWS_ALLOWLIST:
                continue
            results.append({
                "title": n.get("title"),
                "url": link,
                "source": (n.get("source") or {}).get("name") or domain,
                "domain": domain,
                "published_at": n.get("publishedAt"),
                "summary": clamp_len(n.get("description") or "", 600),
            })
        print(f"[NewsAPI] {ticker}: fetched {len(results)} articles after allowlist.")
        return results
    except Exception as e:
        print(f"[NewsAPI] {ticker}: error {e}")
        return []

async def fetch_news_serpapi(client: httpx.AsyncClient, ticker: str) -> List[Dict[str, Any]]:
    if not USE_SERPAPI:
        print(f"[SerpAPI] Disabled via USE_SERPAPI=false. Skipping for {ticker}.")
        return []
    key = os.getenv("SERPAPI_KEY")
    if not key or key == "your_serpapi_key_here":
        print(f"[SerpAPI] Missing SERPAPI_KEY. Skipping for {ticker}.")
        return []
    url = "https://serpapi.com/search.json"
    params = {
        "engine": "google_news",
        "q": f'"{ticker}" stock news',
        "gl": "us",
        "hl": "en",
        "api_key": key,
        "when": "1 day ago",
        "num": 50,
        "tbm": "nws",
    }
    try:
        r = await client.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        results = []
        for n in data.get("news_results", []):
            link = n.get("link") or ""
            domain = normalize_domain(link)
            if domain and NEWS_ALLOWLIST and domain not in NEWS_ALLOWLIST:
                continue
            results.append({
                "title": n.get("title"),
                "url": link,
                "source": (n.get("source") or {}).get("name") or domain,
                "domain": domain,
                "published_at": n.get("date"),
                "summary": clamp_len(n.get("snippet") or "", 600),
            })
        print(f"[SerpAPI] {ticker}: fetched {len(results)} articles after allowlist.")
        return results
    except Exception as e:
        print(f"[SerpAPI] {ticker}: error {e}")
        return []

async def fetch_reddit_mentions(client: httpx.AsyncClient, ticker: str) -> Dict[str, Any]:
    cid = os.getenv("REDDIT_CLIENT_ID")
    secret = os.getenv("REDDIT_CLIENT_SECRET")
    ua = os.getenv("REDDIT_USER_AGENT", f"telegram-bot/0.1 by example")
    if not (cid and secret):
        print(f"[Reddit] Missing credentials. Skipping for {ticker}.")
        return {"mention_count": 0, "top_posts": [], "note": "REDDIT_* env not set"}
    token_resp = await client.post(
        "https://www.reddit.com/api/v1/access_token",
        data={"grant_type": "client_credentials"},
        auth=(cid, secret),
        headers={"User-Agent": ua},
        timeout=20,
    )
    token_resp.raise_for_status()
    token = token_resp.json().get("access_token")
    if not token:
        print(f"[Reddit] Token missing for {ticker}.")
        return {"mention_count": 0, "top_posts": [], "note": "reddit token missing"}
    headers = {"Authorization": f"Bearer {token}", "User-Agent": ua}
    q = f"title:{ticker} OR selftext:{ticker}"
    url = "https://oauth.reddit.com/search"
    params = {"q": q, "sort": "new", "limit": 25, "type": "link,self", "restrict_sr": False}
    r = await client.get(url, headers=headers, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    posts = []
    for p in data.get("data", {}).get("children", []):
        post = p.get("data", {})
        posts.append({
            "title": post.get("title"),
            "url": f"https://reddit.com{post.get('permalink')}",
            "score": post.get("score", 0),
            "subreddit": post.get("subreddit"),
            "created_utc": post.get("created_utc"),
        })
    print(f"[Reddit] {ticker}: found {len(posts)} mentions.")
    return {"mention_count": len(posts), "top_posts": posts[:5]}

async def fetch_twitter_mentions(client: httpx.AsyncClient, ticker: str) -> Dict[str, Any]:
    print(f"[Twitter] Not configured. Skipping for {ticker}.")
    return {"mention_count": 0, "top_posts": [], "note": "TWITTER_* env not set"}

async def gather_sources(ticker: str) -> Dict[str, Any]:
    async with httpx.AsyncClient() as client:
        news_serpapi = await fetch_news_serpapi(client, ticker)
        news_newsapi = await fetch_news_newsapi(client, ticker)

        # Log raw counts from each provider (after allowlist)
        print(f"[Summary] {ticker}: SerpAPI count = {len(news_serpapi)}, NewsAPI count = {len(news_newsapi)}")

        all_news = news_serpapi + news_newsapi
        seen_urls = set()
        unique_news = []
        for article in all_news:
            u = article.get("url")
            if u and u not in seen_urls:
                seen_urls.add(u)
                unique_news.append(article)

        print(f"[Summary] {ticker}: combined={len(all_news)}; unique={len(unique_news)} before 24h filter.")

        # STRICT last-24h filter
        cutoff = pd.Timestamp.utcnow() - pd.Timedelta(hours=24)
        filtered_24h = []
        for a in unique_news:
            dt = parse_pubdate(a.get("published_at"))
            if dt is not None and dt >= cutoff:
                filtered_24h.append(a)

        print(f"[Summary] {ticker}: unique 24h={len(filtered_24h)} after time filter.")

        reddit = await fetch_reddit_mentions(client, ticker)
        twitter = await fetch_twitter_mentions(client, ticker)

    return {
        "news": filtered_24h,           # only last 24h go forward
        "reddit": reddit,
        "twitter": twitter,
        "meta": {
            "counts": {
                "serpapi": len(news_serpapi),
                "newsapi": len(news_newsapi),
                "combined": len(all_news),
                "unique_before_24h": len(unique_news),
                "unique_24h": len(filtered_24h),
            }
        }
    }

# -------------------------------------------------------------------
# OpenAI analysis (structured output expectation)
# -------------------------------------------------------------------
def build_schema() -> Dict[str, Any]:
    return {
        "name": "TradeDecision",
        "schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "verdict": {"type": "string", "enum": ["buy", "sell", "hold"]},
                "overall_sentiment": {"type": "string", "enum": ["positive", "neutral", "negative"]},
                "impact_score": {"type": "integer", "minimum": 1, "maximum": 5},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "rationale": {"type": "string"},
                "news": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "url": {"type": "string"},
                            "source": {"type": "string"},
                            "published_at": {"type": "string"},
                            "sentiment": {"type": "string", "enum": ["positive", "neutral", "negative"]},
                            "impact": {"type": "integer", "minimum": 1, "maximum": 5},
                        },
                        "required": ["title", "url", "source", "published_at", "sentiment", "impact"],
                        "additionalProperties": False,
                    },
                },
                "social": {
                    "type": "object",
                    "properties": {
                        "reddit": {
                            "type": "object",
                            "properties": {
                                "mention_count": {"type": "integer"},
                                "top_posts": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "title": {"type": "string"},
                                            "url": {"type": "string"},
                                            "score": {"type": "integer"},
                                            "sentiment": {"type": "string", "enum": ["positive", "neutral", "negative"]},
                                        },
                                        "required": ["title", "url"],
                                        "additionalProperties": True,
                                    },
                                },
                            },
                            "required": ["mention_count"],
                            "additionalProperties": True,
                        },
                        "twitter": {
                            "type": "object",
                            "properties": {
                                "mention_count": {"type": "integer"},
                                "top_posts": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "text": {"type": "string"},
                                            "url": {"type": "string"},
                                            "like_count": {"type": "integer"},
                                            "sentiment": {"type": "string", "enum": ["positive", "neutral", "negative"]},
                                        },
                                        "required": ["text", "url"],
                                        "additionalProperties": True,
                                    },
                                },
                            },
                            "required": ["mention_count"],
                            "additionalProperties": True,
                        },
                    },
                    "required": [],
                    "additionalProperties": True,
                },
            },
            "required": ["ticker", "verdict", "overall_sentiment", "impact_score", "rationale"],
            "additionalProperties": False,
        },
        "strict": True,
    }

def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def twenty_four_hours_ago() -> datetime:
    return now_utc() - timedelta(hours=24)

def openai_analyze(ticker: str, gathered: Dict[str, Any]) -> Dict[str, Any]:
    client = OpenAI()

    system = (
        "You are a cautious equity news analyst for day trading education. "
        "Analyze only the last-24h items provided. "
        "Rate each article's impact (1=minor, 5=major) and sentiment. "
        "Then decide a BUY/SELL/HOLD *signal* (not advice), include confidence 0..1, and summarize rationale in 1-2 concise bullets. "
        "If evidence is mixed or thin, choose HOLD. "
        "Never guarantee outcomes. Avoid personalized advice. "
        "Return your analysis as a valid JSON object with the following structure: "
        '{"ticker": "TICKER", "verdict": "buy/sell/hold", "overall_sentiment": "positive/neutral/negative", '
        '"impact_score": 1-5, "confidence": 0.0-1.0, "rationale": "explanation", "news": [{"title": "...", "url": "...", '
        '"source": "...", "published_at": "...", "sentiment": "...", "impact": 1-5}]}'
    )

    user_payload = {
        "ticker": ticker,
        "window": {
            "start": twenty_four_hours_ago().strftime(ISO8601),
            "end": now_utc().strftime(ISO8601),
        },
        "sources": gathered,
    }

    combined_input = (
        system
        + "\n\nUSER:\nAnalyze the following JSON payload and return JSON matching the provided schema.\n\n"
        + json.dumps(user_payload)
    )

    try:
        resp = client.responses.create(
            model=OPENAI_MODEL,
            input=combined_input,
            temperature=0.2,
        )

        text_content = None
        if hasattr(resp, "output") and resp.output:
            if isinstance(resp.output, list) and len(resp.output) > 0:
                first_output = resp.output[0]
                if hasattr(first_output, "content") and first_output.content:
                    content = first_output.content
                    if isinstance(content, list) and len(content) > 0:
                        text_content = content[0]
                        if hasattr(text_content, "text"):
                            text_content = text_content.text

        if not text_content:
            raise RuntimeError("Could not extract text content from response")

        import re as _re
        json_match = _re.search(r'```json\s*(.*?)\s*```', text_content, _re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            json_str = text_content.strip()

        return json.loads(json_str)

    except Exception as e:
        print(f"[OpenAI] error: {e}")
        if 'text_content' in locals():
            print(f"[OpenAI] raw text content: {text_content}")
        raise RuntimeError(f"OpenAI API error: {e}")

def format_telegram(decision: Dict[str, Any]) -> str:
    verdict = decision.get("verdict", "hold").upper()
    impact = decision.get("impact_score", 3)
    sentiment = decision.get("overall_sentiment", "neutral").title()
    conf = decision.get("confidence", 0.5)

    news_items = decision.get("news", []) or []
    n = len(news_items)

    parts = [
        "*Not financial advice*. Educational analysis only.",
        f"*Signal*: {verdict}  |  *Impact*: {impact}/5  |  *Sentiment*: {sentiment}  |  *Confidence*: {conf:.0%}",
        f"\n*News (last 24h):* {n} article{'s' if n != 1 else ''}",
    ]

    if n == 0:
        parts.append("_No allowlisted articles in the last 24 hours._")
    else:
        header = "Showing all articles:" if n < 5 else "Top 5 articles:"
        parts.append(f"\n*{header}*")
        for item in news_items[:5]:
            parts.append(
                f"• [{clamp_len(item.get('title',''), 80)}]({item.get('url')}) – {item.get('source','?')} "
                f"| impact {item.get('impact')}/5 | {item.get('sentiment')}"
            )

    # Social mentions footer
    social = decision.get("social", {})
    reddit = (social or {}).get("reddit", {})
    twitter = (social or {}).get("twitter", {})
    parts.append(f"\n*Social (24h)*: Reddit {reddit.get('mention_count', 0)} | X {twitter.get('mention_count', 0)}")

    return "\n".join(parts)

# -------------------------------------------------------------------
# Keyboards / UI
# -------------------------------------------------------------------
def home_kb() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="📊 My Watchlist"), KeyboardButton(text="➕ Add Ticker")],
            [KeyboardButton(text="🗑️ Remove Ticker"), KeyboardButton(text="❓ Help")],
        ],
        resize_keyboard=True
    )

def watchlist_inline_kb(tickers: List[str]) -> InlineKeyboardMarkup:
    rows = []
    for t in tickers:
        rows.append([
            InlineKeyboardButton(text=f"📈 {t}", callback_data=f"dash:{t}"),
            InlineKeyboardButton(text="🔍 Analyze", callback_data=f"analyze:{t}")
        ])
    if not tickers:
        rows.append([InlineKeyboardButton(text="➕ Add first ticker", callback_data="add:start")])
    return InlineKeyboardMarkup(inline_keyboard=rows)

def dashboard_kb(ticker: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="💵 Price (1m preview, CSV 1y)", callback_data=f"info:{ticker}:price")],
        [
            InlineKeyboardButton(text="📑 Financials", callback_data=f"submenu:{ticker}:fin"),
            InlineKeyboardButton(text="🏢 Corp Info", callback_data=f"submenu:{ticker}:corp"),
        ],
        [
            InlineKeyboardButton(text="🗓️ Events (1m preview, CSV 1y)", callback_data=f"submenu:{ticker}:events"),
            InlineKeyboardButton(text="📈 Forecasts", callback_data=f"submenu:{ticker}:fcst"),
        ],
        [InlineKeyboardButton(text="📰 News Analysis (24h)", callback_data=f"news:{ticker}")],
        [InlineKeyboardButton(text="⬅️ Back to Watchlist", callback_data="nav:watchlist"),
         InlineKeyboardButton(text="🏠 Home", callback_data="nav:home")],
    ])

def submenu_kb_fin(ticker: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="Quarterly Income", callback_data=f"info:{ticker}:quarterly_income_statement"),
            InlineKeyboardButton(text="Annual Income", callback_data=f"info:{ticker}:annual_income_statement"),
        ],
        [
            InlineKeyboardButton(text="Quarterly Balance", callback_data=f"info:{ticker}:quarterly_balance_sheet"),
            InlineKeyboardButton(text="Annual Balance", callback_data=f"info:{ticker}:annual_balance_sheet"),
        ],
        [
            InlineKeyboardButton(text="Quarterly Cash Flow", callback_data=f"info:{ticker}:quarterly_cash_flow"),
            InlineKeyboardButton(text="Annual Cash Flow", callback_data=f"info:{ticker}:annual_cash_flow"),
        ],
        [InlineKeyboardButton(text="⬅️ Back", callback_data=f"dash:{ticker}")]
    ])

def submenu_kb_corp(ticker: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="Info", callback_data=f"info:{ticker}:info"),
            InlineKeyboardButton(text="Officers", callback_data=f"info:{ticker}:officers"),
        ],
        [InlineKeyboardButton(text="Summary", callback_data=f"info:{ticker}:summary")],
        [InlineKeyboardButton(text="⬅️ Back", callback_data=f"dash:{ticker}")]
    ])

def submenu_kb_events(ticker: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="Earnings (CSV 1y)", callback_data=f"info:{ticker}:earnings"),
            InlineKeyboardButton(text="Calendar (CSV 1y)", callback_data=f"info:{ticker}:calendar"),
        ],
        [
            InlineKeyboardButton(text="Splits (CSV 1y)", callback_data=f"info:{ticker}:splits"),
            InlineKeyboardButton(text="Dividends (CSV 1y)", callback_data=f"info:{ticker}:dividends"),
        ],
        [InlineKeyboardButton(text="⬅️ Back", callback_data=f"dash:{ticker}")]
    ])

def submenu_kb_fcst(ticker: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="Revenue Forecast", callback_data=f"info:{ticker}:revenue_forecast"),
            InlineKeyboardButton(text="Earnings Forecast", callback_data=f"info:{ticker}:earnings_forecast"),
        ],
        [InlineKeyboardButton(text="⬅️ Back", callback_data=f"dash:{ticker}")]
    ])

# -------------------------------------------------------------------
# Bot + Dispatcher
# -------------------------------------------------------------------
bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher()

# -------------------------------------------------------------------
# Handlers
# -------------------------------------------------------------------
@dp.message(CommandStart())
async def start(msg: Message):
    user_id = str(msg.from_user.id)
    if user_id not in user_subscriptions:
        user_subscriptions[user_id] = []
        save_subscriptions()

    intro_text = "Stock Advisor"
    fancy_intro = pyfiglet.figlet_format(intro_text)
    escaped_intro = html.escape(fancy_intro)

    header = f"<pre>{escaped_intro}</pre>"
    description = (
        "Welcome! Track up to 5 tickers. Open a dashboard per ticker to see "
        "<b>Price (1m preview, CSV 1y), Financials, Events (1m preview, CSV 1y), Forecasts, and News Analysis (24h)</b>.\n\n"
        "Use the buttons below to get started."
    )
    await msg.answer(header, parse_mode="HTML")
    await msg.answer(description, reply_markup=home_kb(), parse_mode="HTML")

@dp.message(Command("help"))
async def help_cmd(msg: Message):
    await msg.answer(
        "• Add a ticker: '➕ Add Ticker'\n"
        "• Watchlist & dashboards: '📊 My Watchlist'\n"
        "• Remove a ticker: '🗑️ Remove Ticker'\n"
        "Tip: You can also send a symbol like AAPL or $TSLA anytime."
    )

@dp.message(Command("subscribe"))
async def add_ticker_cmd(msg: Message):
    user_id = str(msg.from_user.id)
    if len(user_subscriptions.get(user_id, [])) >= MAX_SUBSCRIPTIONS:
        await msg.answer(f"Limit reached ({MAX_SUBSCRIPTIONS}). Remove one first.")
        return
    user_state[user_id] = "awaiting_ticker_subscription"
    await msg.answer("Send a ticker (e.g., AAPL or $TSLA).")

# Home buttons
@dp.message(F.text == "📊 My Watchlist")
async def my_watchlist(msg: Message):
    user_id = str(msg.from_user.id)
    tickers = user_subscriptions.get(user_id, [])
    await msg.answer("Your watchlist:", reply_markup=home_kb())
    await msg.answer(
        "Tap a ticker to open its dashboard or run quick analysis:",
        reply_markup=watchlist_inline_kb(tickers)
    )

@dp.message(F.text == "➕ Add Ticker")
async def add_ticker_prompt(msg: Message):
    user_id = str(msg.from_user.id)
    if len(user_subscriptions.get(user_id, [])) >= MAX_SUBSCRIPTIONS:
        await msg.answer(f"Limit reached ({MAX_SUBSCRIPTIONS}). Remove one first.")
        return
    user_state[user_id] = "awaiting_ticker_subscription"
    await msg.answer("Send a ticker (e.g., AAPL or $TSLA).")

@dp.message(F.text == "🗑️ Remove Ticker")
async def remove_ticker_menu(msg: Message):
    user_id = str(msg.from_user.id)
    tickers = user_subscriptions.get(user_id, [])
    if not tickers:
        await msg.answer("Your watchlist is empty.")
        return
    rows = [[InlineKeyboardButton(text=t, callback_data=f"rm:{t}")] for t in tickers]
    await msg.answer("Select a ticker to remove:", reply_markup=InlineKeyboardMarkup(inline_keyboard=rows))

@dp.message(F.text == "❓ Help")
async def help_btn(msg: Message):
    await help_cmd(msg)

# Text entry (stateful add or direct dashboard/analyze)
@dp.message(F.text)
async def handle_text(msg: Message):
    user_id = str(msg.from_user.id)
    text = msg.text.strip().upper()

    # subscription flow
    if user_state.get(user_id) == "awaiting_ticker_subscription":
        ticker = is_ticker(text)
        if not ticker:
            await msg.reply("Invalid ticker format. Try again (e.g., AAPL or $MSFT).")
            return
        subs = user_subscriptions.setdefault(user_id, [])
        if ticker in subs:
            await msg.answer(f"{ticker} is already in your watchlist.", reply_markup=home_kb())
        elif len(subs) >= MAX_SUBSCRIPTIONS:
            await msg.answer(f"Limit reached ({MAX_SUBSCRIPTIONS}). Remove one first.", reply_markup=home_kb())
        else:
            subs.append(ticker)
            save_subscriptions()
            await msg.answer(f"Added {ticker} ✅", reply_markup=home_kb())
        user_state.pop(user_id, None)
        return

    # free-form analyze/dashboard jump
    ticker = is_ticker(text)
    if ticker:
        await send_dashboard(msg, ticker)
    else:
        await msg.reply("I didn’t recognize that. Use the buttons or send a ticker like AAPL.")

# Inline callbacks: navigation
@dp.callback_query(F.data == "nav:home")
async def nav_home(cb: CallbackQuery):
    await cb.message.answer("Back to Home.", reply_markup=home_kb())
    await cb.answer()

@dp.callback_query(F.data == "nav:watchlist")
async def nav_watchlist(cb: CallbackQuery):
    user_id = str(cb.from_user.id)
    tickers = user_subscriptions.get(user_id, [])
    await cb.message.edit_text("Your watchlist:")
    await cb.message.edit_reply_markup(reply_markup=watchlist_inline_kb(tickers))
    await cb.answer()

# Inline callbacks: add/remove
@dp.callback_query(F.data == "add:start")
async def add_start(cb: CallbackQuery):
    user_id = str(cb.from_user.id)
    if len(user_subscriptions.get(user_id, [])) >= MAX_SUBSCRIPTIONS:
        await cb.answer(f"Limit reached ({MAX_SUBSCRIPTIONS}).", show_alert=True)
        return
    user_state[user_id] = "awaiting_ticker_subscription"
    await cb.message.answer("Send a ticker (e.g., AAPL).")
    await cb.answer()

@dp.callback_query(F.data.startswith("rm:"))
async def rm_ticker(cb: CallbackQuery):
    user_id = str(cb.from_user.id)
    t = cb.data.split(":", 1)[1]
    subs = user_subscriptions.get(user_id, [])
    if t in subs:
        subs.remove(t)
        save_subscriptions()
        await cb.answer(f"Removed {t}")
    else:
        await cb.answer("Not found.")
    tickers = user_subscriptions.get(user_id, [])
    await cb.message.edit_text("Your watchlist (updated):")
    await cb.message.edit_reply_markup(reply_markup=watchlist_inline_kb(tickers))

# Inline callbacks: dashboards & submenus
@dp.callback_query(F.data.startswith("dash:"))
async def open_dash(cb: CallbackQuery):
    ticker = cb.data.split(":", 1)[1]
    await cb.message.edit_text(f"📊 {ticker} dashboard")
    await cb.message.edit_reply_markup(reply_markup=dashboard_kb(ticker))
    await cb.answer()

@dp.callback_query(F.data.startswith("submenu:"))
async def open_submenu(cb: CallbackQuery):
    _, ticker, which = cb.data.split(":")
    if which == "fin":
        kb = submenu_kb_fin(ticker); label = "Financials"
    elif which == "corp":
        kb = submenu_kb_corp(ticker); label = "Corporate Info"
    elif which == "events":
        kb = submenu_kb_events(ticker); label = "Events (1m preview, CSV 1y)"
    elif which == "fcst":
        kb = submenu_kb_fcst(ticker); label = "Forecasts"
    else:
        await cb.answer("Unknown submenu.")
        return
    await cb.message.edit_text(f"{ticker} • {label}")
    await cb.message.edit_reply_markup(reply_markup=kb)
    await cb.answer()

# Inline callbacks: quick analyze/news
@dp.callback_query(F.data.startswith("analyze:"))
async def analyze_now(cb: CallbackQuery):
    ticker = cb.data.split(":", 1)[1]
    await cb.answer("Analyzing…")
    await cb.message.answer(f"Analyzing `{ticker}`… (24h window)", parse_mode="Markdown")
    try:
        gathered = await gather_sources(ticker)
        decision = openai_analyze(ticker, gathered)
        reply = format_telegram(decision)

        # Surface provider counts to chat as well
        meta = (gathered or {}).get("meta", {})
        counts = (meta or {}).get("counts", {})
        c_msg = (
            f"_Sources (allowlisted): SerpAPI {counts.get('serpapi',0)}, "
            f"NewsAPI {counts.get('newsapi',0)}; Unique 24h {counts.get('unique_24h',0)}._"
        )

        await cb.message.answer(reply + "\n\n" + c_msg, parse_mode="Markdown")
    except Exception as e:
        await cb.message.answer(f"Something went wrong: {html.escape(str(e))}\nCheck API keys and logs.", parse_mode="HTML")

@dp.callback_query(F.data.startswith("news:"))
async def news_analysis(cb: CallbackQuery):
    ticker = cb.data.split(":", 1)[1]
    await analyze_now(cb)  # reuse the same flow

# Info dispatcher (mapping instead of long elif)
INFO_DISPATCH: Dict[str, str] = {
    "price": "price",
    "quarterly_income_statement": "quarterly_income_statement",
    "annual_income_statement": "annual_income_statement",
    "quarterly_balance_sheet": "quarterly_balance_sheet",
    "annual_balance_sheet": "annual_balance_sheet",
    "quarterly_cash_flow": "quarterly_cash_flow",
    "annual_cash_flow": "annual_cash_flow",
    "info": "info",
    "officers": "officers",
    "calendar": "calendar",
    "earnings": "earnings",
    "splits": "splits",
    "dividends": "dividends",
    "revenue_forecast": "revenue_forecast",
    "earnings_forecast": "earnings_forecast",
    "summary": "summary",
}

def maybe_filter_last_year(info_type: str, data: Any) -> Any:
    """
    Apply 1-year filtering for relevant datasets (CSV scope).
    """
    if isinstance(data, pd.DataFrame):
        if info_type in {"price", "dividends", "splits", "calendar", "earnings"}:
            days = PRICE_WINDOW_DAYS if info_type == "price" else EVENTS_WINDOW_DAYS
            return filter_last_days_df(data, days)
    return data

async def send_preview_and_csv(
    target: Message | CallbackQuery,
    df: pd.DataFrame,
    filename: str,
    preview_rows: int = 30,
    note: Optional[str] = None,
):
    """
    Send a preview showing ~last 1 month in chat, then attach a CSV containing last 1 year.
    """
    # PREVIEW: last 1 month
    try:
        preview_df = filter_last_days_df(df, PREVIEW_WINDOW_DAYS)
        preview_df = sort_newest_first(preview_df)
        if preview_rows > 0:
            head_df = preview_df.head(preview_rows)
            preview_text = format_df(head_df)
            await send_pre_block(target, preview_text)
    except Exception as e:
        if isinstance(target, Message):
            await target.answer(f"Preview failed: {html.escape(str(e))}", parse_mode="HTML")
        else:
            await target.message.answer(f"Preview failed: {html.escape(str(e))}", parse_mode="HTML")

    # Note for users
    if note is None:
        note = "Preview shows ~last 1 month. Full 1-year data is attached as CSV."
    if isinstance(target, Message):
        await target.answer(note)
    else:
        await target.message.answer(note)

    # CSV: last 1 year (already filtered)
    csv_bytes = df.to_csv(index=True).encode("utf-8")
    file = BufferedInputFile(csv_bytes, filename=filename)
    chat_id = target.chat.id if isinstance(target, Message) else target.message.chat.id
    await bot.send_document(chat_id=chat_id, document=file, caption=f"{filename}")

@dp.callback_query(F.data.startswith("info:"))
async def show_info(cb: CallbackQuery):
    _, ticker, info_type = cb.data.split(":")
    await cb.answer()
    await cb.message.answer(f"Fetching {info_type.replace('_',' ').title()} for {ticker}…")
    try:
        t = Ticker(ticker)
        attr = INFO_DISPATCH.get(info_type)
        if not attr:
            await cb.message.answer("Unknown info type.")
            return
        method: Callable = getattr(t, attr)
        data = method()

        # Apply last-1y filtering where relevant (CSV scope)
        data = maybe_filter_last_year(info_type, data)

        if isinstance(data, pd.DataFrame):
            filename = f"{ticker}_{info_type}_last1y.csv" if info_type in {"price", "dividends", "splits", "calendar", "earnings"} else f"{ticker}_{info_type}.csv"
            await send_preview_and_csv(cb, data, filename=filename, preview_rows=30, note=None)
        else:
            text = format_pretty_table(data)
            await send_pre_block(cb, text[:3000])

        await cb.message.answer(f"Back to {ticker} dashboard:", reply_markup=dashboard_kb(ticker))

    except Exception as e:
        await cb.message.answer(f"Error fetching data for {ticker}: {html.escape(str(e))}", parse_mode="HTML")

# Jump to dashboard from free-form ticker
async def send_dashboard(msg_or_cb: Message | CallbackQuery, ticker: str):
    text = f"📊 {ticker} dashboard"
    kb = dashboard_kb(ticker)
    if isinstance(msg_or_cb, Message):
        await msg_or_cb.answer(text, reply_markup=kb)
    else:
        await msg_or_cb.message.answer(text, reply_markup=kb)

# -------------------------------------------------------------------
# Entrypoint
# -------------------------------------------------------------------
async def main():
    load_subscriptions()
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
