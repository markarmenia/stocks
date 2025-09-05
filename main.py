import os
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, CallbackQueryHandler, ContextTypes
import defeatbeta_api
from defeatbeta_api.data.ticker import Ticker
import openai
import requests
import traceback
from datetime import datetime, timedelta

# Load environment variables from .env file
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY

INFO_OPTIONS = [
    ('Price History', 'price'),
    ('Quarterly Income Statement', 'quarterly_income_statement'),
    ('Annual Income Statement', 'annual_income_statement'),
    ('Quarterly Balance Sheet', 'quarterly_balance_sheet'),
    ('Annual Balance Sheet', 'annual_balance_sheet'),
    ('Quarterly Cash Flow', 'quarterly_cash_flow'),
    ('Annual Cash Flow', 'annual_cash_flow'),
    ('Info', 'info'),
    ('Officers', 'officers'),
    ('Calendar', 'calendar'),
    ('Earnings', 'earnings'),
    ('Splits', 'splits'),
    ('Dividends', 'dividends'),
    ('Revenue Forecast', 'revenue_forecast'),
    ('Earnings Forecast', 'earnings_forecast'),
    ('Summary', 'summary'),
    ('News', 'news'),
]

user_state = {}

def format_df(df):
    try:
        import pandas as pd
        if isinstance(df, pd.DataFrame):
            return df.head(20).to_markdown(index=False)
        return str(df)
    except Exception:
        return str(df)

def format_pretty_table(statement):
    try:
        from io import StringIO
        buf = StringIO()
        statement.print_pretty_table(file=buf)
        return buf.getvalue()
    except Exception:
        return str(statement)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('Welcome! Send me a stock ticker (e.g., TSLA) to get started.')

async def handle_ticker(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ticker = update.message.text.strip().upper()
    user_state[update.effective_user.id] = {'ticker': ticker}
    keyboard = [[InlineKeyboardButton(text, callback_data=key)] for text, key in INFO_OPTIONS]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(f'Select info for {ticker}:', reply_markup=reply_markup)

async def handle_info_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id
    ticker_symbol = user_state.get(user_id, {}).get('ticker')
    info_type = query.data
    await query.edit_message_text(f'Fetching {info_type.replace("_", " ").title()} for {ticker_symbol}...')
    try:
        ticker = Ticker(ticker_symbol)
        if info_type == 'price':
            df = ticker.price()
            await query.message.reply_text(format_df(df))
        elif info_type == 'quarterly_income_statement':
            statement = ticker.quarterly_income_statement()
            await query.message.reply_text(format_pretty_table(statement))
        elif info_type == 'annual_income_statement':
            statement = ticker.annual_income_statement()
            await query.message.reply_text(format_pretty_table(statement))
        elif info_type == 'quarterly_balance_sheet':
            statement = ticker.quarterly_balance_sheet()
            await query.message.reply_text(format_pretty_table(statement))
        elif info_type == 'annual_balance_sheet':
            statement = ticker.annual_balance_sheet()
            await query.message.reply_text(format_pretty_table(statement))
        elif info_type == 'quarterly_cash_flow':
            statement = ticker.quarterly_cash_flow()
            await query.message.reply_text(format_pretty_table(statement))
        elif info_type == 'annual_cash_flow':
            statement = ticker.annual_cash_flow()
            await query.message.reply_text(format_pretty_table(statement))
        elif info_type == 'info':
            info = ticker.info()
            await query.message.reply_text(str(info))
        elif info_type == 'officers':
            officers = ticker.officers()
            await query.message.reply_text(str(officers))
        elif info_type == 'calendar':
            calendar = ticker.calendar()
            await query.message.reply_text(str(calendar))
        elif info_type == 'earnings':
            earnings = ticker.earnings()
            await query.message.reply_text(str(earnings))
        elif info_type == 'splits':
            splits = ticker.splits()
            await query.message.reply_text(str(splits))
        elif info_type == 'dividends':
            dividends = ticker.dividends()
            await query.message.reply_text(str(dividends))
        elif info_type == 'revenue_forecast':
            forecast = ticker.revenue_forecast()
            await query.message.reply_text(str(forecast))
        elif info_type == 'earnings_forecast':
            forecast = ticker.earnings_forecast()
            await query.message.reply_text(str(forecast))
        elif info_type == 'summary':
            summary = ticker.summary()
            await query.message.reply_text(str(summary))
        elif info_type == 'news':
            await handle_news_summary(query, ticker, ticker_symbol)
        else:
            await query.message.reply_text('Unknown info type.')
    except Exception as e:
        await query.message.reply_text(f"Error: {str(e)}\n{traceback.format_exc(0)}")

def get_openai_prompt(news_markdown):
    return f"""Your role is to extract stock news info from the following articles. For each ticker, rank the news impact on a scale of 1 to 5 and provide the sentiment (positive/negative/neutral). Summarize the overall news impact for the ticker(s) in a concise paragraph.\n\nArticles:\n{news_markdown}"""

async def analyze_news_with_openai(news_markdown):
    prompt = get_openai_prompt(news_markdown)
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a financial news analyst."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=400
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        return f"OpenAI error: {str(e)}"

async def handle_news_summary(query, ticker, ticker_symbol):
    news_obj = ticker.news()
    news_list = news_obj.get_news_list()
    if news_list.empty:
        await query.message.reply_text('No news found for this ticker.')
        return
    # Filter for past 7 days' news
    today = datetime.utcnow().date()
    week_ago = today - timedelta(days=7)
    news_list['report_date'] = news_list['report_date'].astype(str)
    filtered_news = news_list[news_list['report_date'].apply(lambda d: week_ago.strftime('%Y-%m-%d') <= d[:10] <= today.strftime('%Y-%m-%d'))]
    if filtered_news.empty:
        await query.message.reply_text('No news found for the past 7 days.')
        return
    news_markdowns = []
    news_links = []
    for idx, row in filtered_news.iterrows():
        uuid = row['uuid']
        title = row.get('title', 'Untitled')
        link = row.get('link', '')
        news_md = news_obj.print_pretty_table(uuid)
        news_markdowns.append(f"### {title}\n{news_md}")
        if link:
            news_links.append(f"- [{title}]({link})")
    all_news_md = '\n\n'.join(news_markdowns)
    ai_summary = await analyze_news_with_openai(all_news_md)
    links_md = '\n'.join(news_links)
    reply = f"*AI News Summary for {ticker_symbol}:*\n{ai_summary}\n\n*News Links (past 7 days):*\n{links_md}"
    await query.message.reply_text(reply, parse_mode='Markdown', disable_web_page_preview=True)

if __name__ == '__main__':
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler('start', start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_ticker))
    app.add_handler(CallbackQueryHandler(handle_info_selection))
    app.run_polling()
