import google.generativeai as genai
import os

genai.configure(api_key=os.environ['GEMINI_API_KEY'])

from tools import analyze_financials, analyze_stock, analyze_analysts_estimatations, analyze_news

functions = {
    "analyze_financials": analyze_financials,
    "analyze_stock": analyze_stock,
    "analyze_analysts_estimatations": analyze_analysts_estimatations,
    "analyze_news": analyze_news,
}

model = genai.GenerativeModel(
  model_name="gemini-1.5-flash",
  tools = functions.values(),
  #tool_config={'function_calling_config':'ANY'},
  system_instruction="You are an expert in financial analysis. Given a set of financial statements of a company, I ask you to analyze the company. If analyze_stock function is called determine the period from one of the items in the list depending on the request ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']."
)

chat_session = model.start_chat(
  history=[
  ]
)

response = chat_session.send_message("Summarize analysts views on GOOG stock")

def call_function(function_call, functions):
    function_name = function_call.name
    function_args = function_call.args
    return functions[function_name](**function_args)

part = response.candidates[0].content.parts[0]

if part.function_call:
    result = call_function(part.function_call, functions)

print(result)