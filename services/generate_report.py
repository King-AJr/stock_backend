from langchain.llms import OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

def generate_report(predictions, df):
    """
    Generate a financial report using only yfinance data and model predictions.
    
    Args:
        predictions (list or array-like): Forecasted values (e.g., predicted returns) for the next 7 days.
        df (pandas.DataFrame): Historical stock data fetched from yfinance.
        
    Returns:
        str: A generated report summarizing historical trends and forecast insights.
    """
    # Extract summary statistics from the historical data
    last_price = df['Close'].iloc[-1]
    avg_price = df['Close'].mean()
    min_price = df['Close'].min()
    max_price = df['Close'].max()
    
    # Construct the prompt for the language model
    # prompt = (
    #     "You are a financial analyst. Based solely on the following historical stock data and forecast predictions, "
    #     "please generate a detailed report that summarizes recent market trends and provides actionable insights for investors.\n\n"
    #     "Historical Data Summary:\n"
    #     f"- Last Closing Price: {last_price:.2f}\n"
    #     f"- Average Closing Price: {avg_price:.2f}\n"
    #     f"- Minimum Closing Price: {min_price:.2f}\n"
    #     f"- Maximum Closing Price: {max_price:.2f}\n\n"
    #     f"Predicted Returns for the Next 7 Days: {predictions}\n\n"
    #     "Provide your analysis on potential market movements and any recommendations for investors."
    # )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", 
            "You are a financial analyst. Based solely on the following historical stock data and forecast predictions, "
            "please generate a detailed report that summarizes recent market trends and provides actionable insights for investors.\n\n"
            "Historical Data Summary:\n"
            f"- Last Closing Price: {last_price:.2f}\n"
            f"- Average Closing Price: {avg_price:.2f}\n"
            f"- Minimum Closing Price: {min_price:.2f}\n"
            f"- Maximum Closing Price: {max_price:.2f}\n\n"
            f"Predicted Returns for the Next 7 Days: {predictions}\n\n"
            "Provide your analysis on potential market movements and any recommendations for investors."
            ),
            ("user", "{user_input}"), 
        ]
    )
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key='AIzaSyDk3iIRV2N4mA7wpyHnbe1pjcjXtvfcizE', max_tokens=500)
    
    
    report = llm.invoke(prompt)
    return report

# Example usage:
if __name__ == "__main__":
    import yfinance as yf
    import pandas as pd
    from pmdarima import auto_arima
    
    # Fetch historical data using yfinance
    def fetch_stock_data(ticker="AAPL", period="1y", interval="1d"):
        stock = yf.Ticker(ticker)
        return stock.history(period=period, interval=interval)
    
    # Clean the data by dropping missing values
    def clean_data(df):
        return df.dropna()
    
    # Dummy prediction using Auto ARIMA on the 'Close' column returns
    def predict_stock_movement(df):
        df = df.copy()
        df['returns'] = df['Close'].pct_change()
        df = df.dropna()
        model = auto_arima(df['returns'], seasonal=False, stepwise=True)
        forecast = model.predict(n_periods=7)
        return forecast.tolist()
    
    # Process the data
    data = fetch_stock_data("AAPL", "1y", "1d")
    cleaned_data = clean_data(data)
    predictions = predict_stock_movement(cleaned_data)
    
    # Generate the report using the historical data and predictions
    report = generate_report(predictions, cleaned_data)
    print("Generated Report:\n", report)
