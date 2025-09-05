# Telegram Stock Info Bot

This app connects to a Telegram bot, allowing users to:
- Provide a stock ticker (e.g., TSLA)
- Choose the type of info needed (e.g., quarterly balance sheet, news, etc.)
- Fetch all available data for the ticker using the defeatbeta API
- If 'news' is selected, fetch news articles, display them in markdown, and use OpenAI API to analyze sentiment and impact for each ticker mentioned

## Features
- Telegram bot interface
- Stock data and financial statements
- News fetching and AI-powered news analysis

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Set your Telegram bot token and OpenAI API key as environment variables:
   - `TELEGRAM_BOT_TOKEN`
   - `OPENAI_API_KEY`
3. Run the app:
   ```bash
   python main.py
   ```
