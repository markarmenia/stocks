#!/usr/bin/env python3
"""
Telegram Day-Trader Stock Advisor (Informational Only)

DISCLAIMER: This bot provides *informational* analysis only and is NOT financial advice. 
It should not be used as the sole basis for investment decisions. Markets are risky.

What it does
------------
• Listens for a stock ticker in Telegram and runs a same‑chat analysis.
• Uses the **OpenAI Responses API with the hosted `web_search` tool** to fetch last‑24h news
  (and public web mentions including Reddit & Twitter/X) from reputable sources — **no third‑party APIs** required.
• Produces strict JSON: {verdict, impact_score 1‑5, overall_sentiment, confidence, rationale, per‑article ratings}.
• Sends a concise summary back to Telegram with links and reasoning.

Tech
----
• Python 3.10+
• aiogram==3.x (Telegram)
• openai==1.x (Responses API)

Environment
-----------
Required:
  OPENAI_API_KEY
  TELEGRAM_BOT_TOKEN
Optional flags:
  OPENAI_MODEL=gpt-4o-mini   # override model

Run
---
> pip install aiogram openai python-dotenv
> export OPENAI_API_KEY=... TELEGRAM_BOT_TOKEN=...
> python telegram_day_trader_agent.py

Notes
-----
• No NewsAPI/SerpAPI/Reddit/Twitter keys needed; all retrieval is via OpenAI `web_search` tool.
• All timestamps are handled in UTC and constrained to the last 24 hours by the prompt.

"""
from __future__ import annotations

import asyncio
import os
import re
import json
import argparse
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart, Command
from aiogram.types import Message
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# -----------------------------
# Configuration
# -----------------------------
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
ISO8601 = "%Y-%m-%dT%H:%M:%SZ"

# -----------------------------
# Utilities
# -----------------------------

def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def twenty_four_hours_ago() -> datetime:
    return now_utc() - timedelta(hours=24)


def clamp_len(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def is_ticker(text: str) -> Optional[str]:
    m = re.fullmatch(r"\$?[A-Z]{1,5}(?:\.[A-Z]{1,3})?", text.strip())
    return m.group(0).lstrip("$") if m else None


# -----------------------------
# JSON helpers
# -----------------------------

def _extract_first_json_obj(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                candidate = text[start:i+1]
                try:
                    return json.loads(candidate)
                except Exception:
                    return None
    return None


def _coerce_decision(d: Dict[str, Any], ticker: str) -> Dict[str, Any]:
    if not isinstance(d, dict):
        d = {}
    out = {
        "ticker": d.get("ticker") or ticker,
        "verdict": (d.get("verdict") or "hold").lower(),
        "overall_sentiment": (d.get("overall_sentiment") or "neutral").lower(),
        "impact_score": int(d.get("impact_score") or 2),
        "confidence": float(d.get("confidence") or 0.4),
        "rationale": d.get("rationale") or "Limited or mixed evidence in last-24h sources.",
        "news": d.get("news") or [],
        "social": d.get("social") or {"reddit": {"mention_count": 0}, "twitter": {"mention_count": 0}},
    }
    out["impact_score"] = max(1, min(5, out["impact_score"]))
    if out["overall_sentiment"] not in {"positive", "neutral", "negative"}:
        out["overall_sentiment"] = "neutral"
    if out["verdict"] not in {"buy", "sell", "hold"}:
        out["verdict"] = "hold"
    out["confidence"] = max(0.0, min(1.0, out["confidence"]))
    return out


# -----------------------------
# OpenAI Responses API with hosted web_search tool
# -----------------------------

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


def openai_analyze(ticker: str) -> Dict[str, Any]:
    """
    Use Responses API + hosted web_search tool.
    Compatible with SDKs that don't support `messages` or `response_format`.
    """
    client = OpenAI()

    # Instruct the model to run web searches limited to last 24h and reputable domains.
    system = (
        "You are a cautious equity news analyst for day trading education. "
        "Search the web (use the web_search tool) for last-24h news and public mentions of the given ticker, "
        "prefer reputable outlets (Reuters, Bloomberg, WSJ, FT, CNBC, AP, MarketWatch, Investopedia, Yahoo/Fin, Forbes). "
        "Include Reddit and Twitter(X) *web* mentions if surfaced by search. "
        "For each article, estimate sentiment and impact (1=minor, 5=major). "
        "Then produce a single JSON object with fields: ticker, verdict (buy/sell/hold), overall_sentiment, impact_score (1-5), confidence (0..1), rationale, news[], social{}. "
        "If evidence is thin or mixed, use HOLD. Never guarantee outcomes."
    )

    # Build a single input string (broadest compatibility across SDK variants)
    combined_input = (
        system
        + f"\n\nTICKER: {ticker}\nWINDOW_UTC: {twenty_four_hours_ago().strftime(ISO8601)} to {now_utc().strftime(ISO8601)}\n"
        + "Return ONLY minified JSON, no extra prose."
    )

    # Ask model to use the hosted web_search tool; let it decide how many calls to make.
    # NOTE: Not all SDK builds expose tool result objects uniformly; we only parse final text.
    try:
        resp = client.responses.create(
            model=OPENAI_MODEL,
            input=combined_input,
            tools=[{"type": "web_search"}],
            tool_choice="auto",
            temperature=0,
        )
    except TypeError:
        # Older SDKs may not accept tools/tool_choice; retry without tools (model may still answer)
        resp = client.responses.create(
            model=OPENAI_MODEL,
            input=combined_input,
            temperature=0.2,
        )

    # Parse across SDK variants
    # 1) output_text / text
    text = getattr(resp, "output_text", None) or getattr(resp, "text", None)
    if isinstance(text, str):
        j = _extract_first_json_obj(text)
        if j:
            return _coerce_decision(j, ticker)

    # 2) output -> content blocks
    out = getattr(resp, "output", None)
    if isinstance(out, list) and out:
        try:
            for piece in out:
                content = getattr(piece, "content", None)
                if isinstance(content, list):
                    for block in content:
                        for attr in ("parsed", "output_json", "json", "text"):
                            val = getattr(block, attr, None)
                            if isinstance(val, dict):
                                return _coerce_decision(val, ticker)
                            if isinstance(val, str):
                                j = _extract_first_json_obj(val)
                                if j:
                                    return _coerce_decision(j, ticker)
        except Exception:
            pass

    # Fallback: HOLD
    return _coerce_decision({}, ticker)


def format_telegram(decision: Dict[str, Any]) -> str:
    verdict = decision.get("verdict", "hold").upper()
    impact = decision.get("impact_score", 3)
    sentiment = decision.get("overall_sentiment", "neutral").title()
    conf = decision.get("confidence", 0.5)
    parts = [
        "*Not financial advice*. Educational analysis only.",
        f"*Signal*: {verdict}  |  *Impact*: {impact}/5  |  *Sentiment*: {sentiment}  |  *Confidence*: {conf:.0%}",
        "\n*Why:*",
        decision.get("rationale", "No rationale provided."),
    ]

    news = decision.get("news", [])
    if news:
        parts.append("\n*Top articles (24h):*")
        for n in news[:5]:
            parts.append(f"• [{clamp_len(n.get('title',''), 80)}]({n.get('url')}) – {n.get('source','?')} | impact {n.get('impact')}/5 | {n.get('sentiment')}")

    social = decision.get("social", {})
    reddit = (social or {}).get("reddit", {})
    twitter = (social or {}).get("twitter", {})
    parts.append(
        f"\n*Social (24h)*: Reddit {reddit.get('mention_count', 0)} | X {twitter.get('mention_count', 0)}"
    )

    return "\n".join(parts)


# -----------------------------
# Telegram bot
# -----------------------------

# Required environment variables
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not TELEGRAM_BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN environment variable is required. Please set it in your .env file.")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required. Please set it in your .env file.")

bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher()


@dp.message(CommandStart())
async def start(msg: Message):
    await msg.answer(
        "Send me a stock ticker like `AAPL` or `$TSLA` and I'll analyze last‑24h news + social (via web search), "
        "\nthen reply with an *informational* BUY/SELL/HOLD signal (impact 1‑5, sentiment).",
    )


@dp.message(Command("help"))
async def help_cmd(msg: Message):
    await msg.answer(
        "Usage: send a ticker symbol (e.g., `MSFT`).\n"
        "No extra API keys needed — retrieval uses OpenAI's hosted web_search tool."
    )


@dp.message(F.text.regexp(r"^\$?[A-Za-z]{1,5}(?:\.[A-Za-z]{1,3})?$"))
async def handle_ticker(msg: Message):
    raw = msg.text.strip().upper()
    ticker = is_ticker(raw)
    if not ticker:
        await msg.reply("Could not parse ticker.")
        return

    await msg.answer(f"Analyzing `{ticker}`... (24h window)")

    try:
        decision = openai_analyze(ticker)
        reply = format_telegram(decision)
        await msg.answer(reply, parse_mode="Markdown")
    except Exception as e:
        await msg.answer(
            f"Something went wrong: {e}\n"
            "Check OPENAI_API_KEY / model permissions."
        )


# -----------------------------
# Tests (local, no external calls)
# -----------------------------

def _run_tests() -> None:
    assert is_ticker("AAPL") == "AAPL"
    assert is_ticker("$TSLA") == "TSLA"
    assert is_ticker("msft") is None
    assert is_ticker("AAPL.B") == "AAPL.B"
    assert is_ticker("TOO_LONG") is None

    sample = {
        "verdict": "hold",
        "impact_score": 3,
        "overall_sentiment": "neutral",
        "confidence": 0.42,
        "rationale": "Mixed catalysts; limited incremental news.",
        "news": [
            {"title": "Company beats EPS", "url": "https://example.com/1", "source": "Example", "impact": 3, "sentiment": "positive"},
            {"title": "Sector headwinds", "url": "https://example.com/2", "source": "Example", "impact": 2, "sentiment": "negative"},
        ],
        "social": {"reddit": {"mention_count": 7}, "twitter": {"mention_count": 13}},
    }
    txt = format_telegram(sample)
    assert "*Signal*: HOLD" in txt
    assert "impact 3/5" in txt
    assert "Reddit 7 | X 13" in txt

    # JSON extraction
    jtxt = 'noise {"ticker":"AAPL","verdict":"buy","overall_sentiment":"positive","impact_score":4,"confidence":0.7,"rationale":"x"} noise'
    assert _extract_first_json_obj(jtxt)["ticker"] == "AAPL"

    print("All local tests passed.")


async def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--test", action="store_true", help="run local tests and exit")
    args, _ = parser.parse_known_args()

    if args.test:
        _run_tests()
        return

    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
