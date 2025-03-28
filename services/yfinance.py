import yfinance as yf
import pandas as pd
import numpy as np

def fetch_stock_data(ticker="AAPL", period="1y", interval="1d"):
    """
    Fetch stock data from Yahoo Finance for the given ticker, period, and interval.
    """
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval)
    return df

def clean_data(df):
    """
    Clean the DataFrame by removing rows with missing values.
    Alternatively, consider filling missing data if appropriate.
    """
    # Drop rows with any NaN values
    cleaned_df = df.dropna()
    # If you prefer to fill missing values, you might use:
    # cleaned_df = df.fillna(method='ffill')
    return cleaned_df

def normalize_data(df):
    """
    Normalize numeric columns using Min-Max scaling.
    """
    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_normalized = df.copy()
    # Apply Min-Max scaling to each numeric column
    for col in numeric_cols:
        min_val = df[col].min()
        max_val = df[col].max()
        if max_val - min_val != 0:
            df_normalized[col] = (df[col] - min_val) / (max_val - min_val)
        else:
            df_normalized[col] = 0  # Avoid division by zero if constant column
    return df_normalized

def process_stock_data(ticker="AAPL", period="1y", interval="1d"):
    """
    Complete service: Fetch, clean, and normalize stock data.
    """
    # Fetch data
    df = fetch_stock_data(ticker, period, interval)
    print("Original Data:")
    print(df.head())

    # Clean data
    df_clean = clean_data(df)
    print("\nCleaned Data:")
    print(df_clean.head())

    # Normalize data
    df_normalized = normalize_data(df_clean)
    print("\nNormalized Data:")
    print(df_normalized.head())

    return df_normalized

if __name__ == "__main__":
    # Example usage for Apple Inc.
    normalized_data = process_stock_data("AAPL", "1y", "1d")
