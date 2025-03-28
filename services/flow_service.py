from langgraph.graph import StateGraph
from prophet import Prophet
from typing_extensions import TypedDict
import pandas as pd
import yfinance as yf
from datetime import date
import matplotlib.pyplot as plt
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import json
import re

today = date.today()
# Define the shared state structure
class FinancialAnalysisState(TypedDict):
    ticker: str
    period: str
    interval: str
    prompt: str
    stock_data: dict
    predictions: list
    report: str


def extract_data(state: dict, prompt: str) -> dict:
    state['prompt'] = prompt
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key="your-api-key", max_tokens=50)
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "Extract the stock ticker, period, and interval from the following prompt. "
                   "Return your answer as a JSON object with keys 'ticker', 'period', and 'interval'."),
        ("user", "{user_input}")
    ])
    formatted_prompt = prompt_template.format_prompt(user_input=prompt)
    response = llm.invoke(formatted_prompt.to_messages()).content

    # Remove any text before the first '{' and after the last '}'
    json_text_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_text_match:
        json_text = json_text_match.group()
        try:
            details = json.loads(json_text)
        except json.JSONDecodeError:
            details = {}
    else:
        details = {}

    state['ticker'] = details.get("ticker")
    state['period'] = details.get("period", "1y")
    state['interval'] = details.get("interval", "1d")
    return state



def process_stock_data(state: FinancialAnalysisState) -> FinancialAnalysisState:
    ticker = state['ticker']
    period = state['period']
    interval = state['interval']

    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval)
    df.reset_index(inplace=True)

    if 'Date' in df.columns:
        df.rename(columns={'Date': 'ds'}, inplace=True)
    else:
        df.rename(columns={'index': 'ds'}, inplace=True)

    df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)

    if 'Close' in df.columns:
        df.rename(columns={'Close': 'y'}, inplace=True)
    elif 'Adj Close' in df.columns:
        df.rename(columns={'Adj Close': 'y'}, inplace=True)
    else:
        raise ValueError("Dataframe does not have a 'Close' or 'Adj Close' column.")

    state['stock_data'] = df.to_dict(orient='list')
    return state


def predict_stock_movement(state: FinancialAnalysisState) -> FinancialAnalysisState:
    df = pd.DataFrame(state['stock_data'])

    if 'ds' not in df.columns or 'y' not in df.columns:
        raise ValueError("Dataframe must have columns 'ds' and 'y'.")

    model = Prophet(daily_seasonality=True)
    model.fit(df)

    # Create a DataFrame for future dates (forecasting 7 days ahead)
    future = model.make_future_dataframe(periods=7)
    forecast = model.predict(future)

    # Extract the last 7 days of forecasts
    forecast_subset = forecast[['ds', 'yhat']].tail(7)
    state['predictions'] = forecast_subset.to_dict(orient='list')
    return state


def generate_report(state: FinancialAnalysisState) -> FinancialAnalysisState:
    prompt = state['prompt']
    predictions = state['predictions']
    df = pd.DataFrame(state['stock_data'])
    last_price = df['y'].iloc[-1]
    avg_price = df['y'].mean()
    min_price = df['y'].min()
    max_price = df['y'].max()

    # Convert predictions to a string representation that is more readable in the report
    predictions_str = "\n".join([f"  {pd.to_datetime(pred['ds']).date()}: {pred['yhat']:.2f}" for pred in [dict(zip(predictions.keys(), values)) for values in zip(*predictions.values())]]) # Convert predictions to string


    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system",
            "You are a financial analyst. Based solely on the following historical stock data and forecast predictions, "
            f"please generate a detailed report that summarizes recent market trends and provides actionable insights for investors. Future predictions start from tomorrow. Note that today's date is {today}\n\n"
            "Historical Data Summary:\n"
            f"- Last Closing Price: {last_price:.2f}\n"
            f"- Average Closing Price: {avg_price:.2f}\n"
            f"- Minimum Closing Price: {min_price:.2f}\n"
            f"- Maximum Closing Price: {max_price:.2f}\n\n"
            f"Predicted Returns for the Next 7 Days:\n{predictions_str}\n\n" # Use predictions_str here
            "Provide your analysis on potential market movements and any recommendations for investors."
            ),
            ("user", "{user_input}"),
        ]
    )

    # Format the prompt template with user input
    formatted_prompt = prompt_template.format_prompt(user_input=prompt)

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key='AIzaSyDk3iIRV2N4mA7wpyHnbe1pjcjXtvfcizE', max_tokens=500)

    # Pass the formatted prompt string to invoke
    report = llm.invoke(formatted_prompt.to_messages())
    state['report'] = report.content
    return state

def plot_stock_trend(state: FinancialAnalysisState) -> FinancialAnalysisState:
    print('about plotting graph')

    # Convert historical stock data to DataFrame
    df_historical = pd.DataFrame(state['stock_data'])

    # Convert predictions dictionary to DataFrame
    df_predictions = pd.DataFrame(state['predictions'])

    # Convert 'ds' columns to datetime format
    df_historical['ds'] = pd.to_datetime(df_historical['ds'])
    df_predictions['ds'] = pd.to_datetime(df_predictions['ds'])

    df_combined_x = pd.concat([df_historical, df_predictions], ignore_index=True)
    df_combined_x['ds'] = pd.to_datetime(df_combined['ds'])

    # Select only the last 7 days of historical data
    df_historical_last_7 = df_historical.tail(7)

    # Rename columns to match for concatenation
    df_historical_last_7 = df_historical_last_7[['ds', 'y']].rename(columns={'y': 'Price'})
    df_predictions = df_predictions.rename(columns={'yhat': 'Price'})

    df_combined_y = pd.concat([df_historical_last_7, df_predictions], ignore_index=True)

    state["predictions"] = {"x": df_combined_x, "y": df_combined_y}
    # Merge historical and prediction data
    df_combined = pd.concat([df_historical_last_7, df_predictions], ignore_index=True)

    plt.figure(figsize=(12, 6))

    # Plot combined data as a single continuous line
    plt.plot(df_combined['ds'], df_combined['Price'], marker='o', linestyle='-', color='blue', label="Stock Price (Actual & Predicted)")

    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.title(f"Stock Price Movement for {state['ticker']} (Last 7 Days & Next 7 Days)")
    plt.xticks(rotation=45)  
    plt.legend()
    plt.grid(True)
    plt.show()

    print('done')
    return state


# Build the state graph
def build_graph():
    workflow = StateGraph(FinancialAnalysisState)

    workflow.add_node("data_ingestion", process_stock_data)
    workflow.add_node("predict_stock", predict_stock_movement)
    workflow.add_node("generate_report", generate_report)
    workflow.add_node("visualize_data", plot_stock_trend)
    workflow.add_node('extract_data', extract_data)

    workflow.add_edge("data_ingestion", "predict_stock")
    workflow.add_edge("predict_stock", "generate_report")
    workflow.add_edge("generate_report", "visualize_data")
    workflow.add_edge("extract_data", "data_ingestion")

    workflow.set_entry_point("extract_data")
    workflow.set_finish_point("visualize_data")

    return workflow


graph = build_graph()
app = graph.compile()

