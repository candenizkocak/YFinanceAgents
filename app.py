import gradio as gr
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
    model_name="gemini-1.5-pro",
    tools=functions.values(),
    system_instruction="You are an expert in financial analysis. Given a set of financial statements of a company, I ask you to analyze the company. If analyze_stock function is called determine the period from one of the items in the list depending on the request ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']."
)

chat_session = model.start_chat(history=[])

def call_function(function_call, functions):
    function_name = function_call.name
    function_args = function_call.args
    return functions[function_name](**function_args)

def chat_with_bot(user_input, history=[]):
    # Send user message to model
    response = chat_session.send_message(user_input)
    part = response.candidates[0].content.parts[0]

    function_result = ""
    if part.function_call:
        function_result = call_function(part.function_call, functions)
        function_name = part.function_call.name
        bot_response = f"Function `{function_name}` was called. Result:\n\n{function_result}"
    else:
        bot_response = part.text

    history.append((user_input, bot_response))
    return history, history

examples = [
    "Summarize analysts views on GOOG.",
    "Analyze last month's stock data of NVDA.",
    "Provide a brief overview of MSFT news.",
    "Analyze financials of META in detail."
]

with gr.Blocks() as demo:
    gr.Markdown("### Financial Analysis Chatbot")
    
    chatbot = gr.Chatbot()
    state = gr.State([])

    with gr.Row():
        with gr.Column(scale=4):
            msg_input = gr.Textbox(label="Your message", placeholder="Ask something...", lines=1)
        with gr.Column(scale=1):
            send_button = gr.Button("Send")

    gr.Examples(examples=examples, inputs=msg_input, fn=chat_with_bot, outputs=[chatbot, state], cache_examples=False)

    send_button.click(chat_with_bot, inputs=[msg_input, state], outputs=[chatbot, state])
    msg_input.submit(chat_with_bot, inputs=[msg_input, state], outputs=[chatbot, state])

demo.launch()