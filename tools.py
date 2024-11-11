import google.generativeai as genai
import yfinance as yf

def analyze_financials(company: str):
  ticker = yf.Ticker(company)
  info = ticker.info
  calendar = ticker.calendar
  sec_filings = ticker.sec_filings
  income_stmt = ticker.income_stmt
  quarterly_income_stmt = ticker.quarterly_income_stmt
  balance_sheet = ticker.balance_sheet
  quarterly_balance_sheet = ticker.quarterly_balance_sheet
  cashflow = ticker.cashflow
  quarterly_cashflow = ticker.quarterly_cashflow

  analyst_generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
  }

  analyst_model = genai.GenerativeModel(
  model_name="gemini-1.5-flash",
  generation_config=analyst_generation_config,
  system_instruction="You are an expert financial analyst. Given a set of financial statements of a company, I ask you to analyze the company. You will have access to ticker name, calendar, sec filings, income statement, quarterly income statement, balance sheet, quarterly balance sheet, cashflow, quarterly cashflow."
  )

  analyst_chat_session = analyst_model.start_chat(
  history=[
  ]
  )

  analyst_response = analyst_chat_session.send_message(f"{ticker}, {info}, {calendar}, {sec_filings}, {income_stmt}, {quarterly_income_stmt}, {balance_sheet}, {quarterly_balance_sheet}, {cashflow}, {quarterly_cashflow}")

  return analyst_response.text

def analyze_stock(company: str, period: str):
  ticker = yf.Ticker(company)
  info = ticker.info
  hist = ticker.history(period=period)
  hist_metadata = ticker.history_metadata

  stock_generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
  }

  stock_model = genai.GenerativeModel(
  model_name="gemini-1.5-flash",
  generation_config=stock_generation_config,
  system_instruction="You are an expert financial analyst. Given a set of financial statements of a company, I ask you to analyze the company. You will have access to ticker name, company info, historical data."
  )

  stock_chat_session = stock_model.start_chat(
  history=[
  ]
  )

  stock_response = stock_chat_session.send_message(f"{ticker}, {info}, {hist}, {hist_metadata}")

  return stock_response.text

def analyze_analysts_estimatations(company: str):
  ticker = yf.Ticker(company)
  info = ticker.info
  analyst_price_targets = ticker.analyst_price_targets
  earnings_estimate = ticker.earnings_estimate
  revenue_estimate = ticker.revenue_estimate
  earnings_history = ticker.earnings_history
  eps_trend = ticker.eps_trend
  eps_revisions = ticker.eps_revisions
  growth_estimates = ticker.growth_estimates

  analysts_generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
  }

  analysts_model = genai.GenerativeModel(
  model_name="gemini-1.5-flash",
  generation_config=analysts_generation_config,
  system_instruction="You are an expert financial analyst. Given a set of financial statements of a company, I ask you to analyze the company. You will have access to ticker name, company info, analyst price targets, earnings estimate, revenue estimate, earnings history, eps trend, eps revisions, growth estimates."
  )

  analysts_chat_session = analysts_model.start_chat(
  history=[
  ]
  )

  analysts_response = analysts_chat_session.send_message(f"{ticker}, {info}, {analyst_price_targets}, {earnings_estimate}, {revenue_estimate}, {earnings_history}, {eps_trend}, {eps_revisions}, {growth_estimates}")

  return analysts_response.text

def analyze_news(company: str):
  ticker = yf.Ticker(company)
  news = ticker.news

  news_generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
  }

  news_model = genai.GenerativeModel(
  model_name="gemini-1.5-flash",
  generation_config=news_generation_config,
  system_instruction="You are an expert financial analyst. Given a set of financial news on company, I ask you to analyze the company. You will have access to ticker name, news on yahoo finance."
  )

  news_chat_session = news_model.start_chat(
  history=[
  ]
  )

  news_response = news_chat_session.send_message(f"{ticker}, {news}")

  return news_response.text