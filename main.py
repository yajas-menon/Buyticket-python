from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import requests
from statsmodels.tsa.arima.model import ARIMA
import os
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from datetime import datetime
import numpy as np
import traceback # Import traceback for detailed error logging

class ForecastRequest(BaseModel):
    months: int

class DimensionForecastRequest(BaseModel):
    months: int
    dimension: str  # e.g., "Sector", "Airline", "Ag Company"
    filter_value: str # e.g., "Tvl Ins", "Indigo", "Your Company Name"

app = FastAPI()

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "null"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Supabase configuration
SUPABASE_API_URL = os.getenv("NEXT_PUBLIC_SUPABASE_URL_REST", "https://splvfnmdkjijhfkdttuf.supabase.co/rest/v1/sales_raw_data")
SUPABASE_API_KEY = os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InNwbHZmbm1ka2ppamhma2R0dHVmIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDg4NTUyMTgsImV4cCI6MjA2NDQzMTIxOH0.9_-PUcjB_j48tnqsDBr7hsC0bmpT2OBH55I5VJip0kE")

def fetch_supabase_data() -> List[Dict[str, Any]]:
    """Fetch sales data from Supabase REST API"""
    headers = {
        "apikey": SUPABASE_API_KEY,
        "Authorization": f"Bearer {SUPABASE_API_KEY}",
        "Prefer": "return=representation"
    }
    response = requests.get(f"{SUPABASE_API_URL}?select=*", headers=headers)
    if response.status_code != 200:
        print(f"Supabase Error Status: {response.status_code}, Response: {response.text}")
        raise Exception(f"Failed to fetch data from Supabase: {response.status_code} - {response.text}")
    
    data = response.json()
    if not isinstance(data, list):
        print(f"Unexpected data format from Supabase: {data}")
        raise Exception("Data fetched from Supabase is not a list.")
    return data

def prepare_data(
    data: List[Dict[str, Any]], 
    dimension: Optional[str] = None, 
    filter_value: Optional[str] = None
) -> pd.Series:
    """Clean and prepare data for forecasting, optionally filtering by a dimension."""
    df = pd.DataFrame(data)
    print(f"DEBUG: Initial DataFrame shape: {df.shape}, columns: {list(df.columns)}")

    if df.empty:
        print("DEBUG: Initial DataFrame is empty.")
        return pd.Series(dtype='float64', index=pd.DatetimeIndex([], freq='MS', name='Month'))

    # Standardize column names (handle spaces and case)
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()
    print(f"DEBUG: DataFrame columns after standardization: {list(df.columns)}")
    
    if 'month' not in df.columns:
        raise ValueError(f"Required column 'month' not found in data. Available columns: {list(df.columns)}")
    if 'revenue' not in df.columns:
        raise ValueError(f"Required column 'revenue' not found in data. Available columns: {list(df.columns)}")

    # Safely process dimension if provided
    processed_dimension = None
    if dimension:
        processed_dimension = dimension.strip().replace(' ', '_').lower()
        print(f"DEBUG: Processed dimension: '{processed_dimension}' for raw dimension: '{dimension}'")

    # Filter by dimension if provided
    if processed_dimension and filter_value:
        print(f"DEBUG: Attempting to filter by dimension '{processed_dimension}' with value '{filter_value}'")
        if processed_dimension not in df.columns:
            available_cols = ', '.join(df.columns)
            raise ValueError(f"Dimension '{dimension}' not found in data columns: {available_cols}")
        
        # Before filtering, convert target column to string type to avoid type mismatch issues
        # and ensure case-insensitive comparison
        df_before_filter = df.copy() # Keep a copy for debugging if needed
        df = df[df[processed_dimension].astype(str).str.lower() == str(filter_value).lower()]
        
        print(f"DEBUG: DataFrame shape after filtering: {df.shape}")
        if df.empty:
            print(f"DEBUG: DataFrame is empty after filtering by '{processed_dimension}' = '{filter_value}'.")
            return pd.Series(dtype='float64', index=pd.DatetimeIndex([], freq='MS', name='Month'))

    # Convert "May'25" or "May'2025" format to datetime
    def parse_month_year(my_str):
        if not my_str or not isinstance(my_str, str):
            return None
        try:
            return datetime.strptime(my_str, "%b'%y")
        except ValueError:
            try:
                return datetime.strptime(my_str, "%b'%Y") # For 4-digit year
            except ValueError:
                return None

    df['MonthDT'] = df['month'].apply(parse_month_year)
    df['Revenue'] = pd.to_numeric(df['revenue'], errors='coerce')
    
    df = df.dropna(subset=['MonthDT', 'Revenue'])
    
    if df.empty:
        print("DEBUG: DataFrame is empty after coercing and dropping NaNs.")
        return pd.Series(dtype='float64', index=pd.DatetimeIndex([], freq='MS', name='MonthDT'))
    
    monthly_revenue = df.groupby('MonthDT')['Revenue'].sum().sort_index()
    
    if monthly_revenue.empty:
        print("DEBUG: No data available after grouping by month.")
        return pd.Series(dtype='float64', index=pd.DatetimeIndex([], freq='MS', name='MonthDT'))

    monthly_revenue = monthly_revenue.asfreq('MS')
    monthly_revenue = monthly_revenue.ffill() 
    
    print(f"DEBUG: Final monthly_revenue series head:\n{monthly_revenue.head()}")
    return monthly_revenue


def generate_forecast_data(
    df_series: pd.Series, 
    months_to_forecast: int, 
    category_name: str = "Revenue"
) -> List[Dict[str, Any]]:
    """Generate forecast data using a simple growth model."""
    if df_series.empty:
        print("DEBUG: Cannot generate forecast: input data series is empty.")
        return []
    
    last_value = df_series.iloc[-1] if not df_series.empty else 0
    forecast_values = [last_value * (1 + 0.02 * (i + 1)) for i in range(months_to_forecast)]

    forecast_dates = pd.date_range(
        start=df_series.index[-1] + pd.DateOffset(months=1) if not df_series.empty else pd.Timestamp.now() + pd.DateOffset(months=1),
        periods=months_to_forecast,
        freq='MS'
    )
    
    return [
        {
            "date": date.strftime("%Y-%m-%d"),
            "month": date.strftime("%b'%y"),
            "category": category_name,
            "actual": None,
            "forecast": round(float(value), 2),
            "is_future": True
        }
        for date, value in zip(forecast_dates, forecast_values)
    ]

@app.post("/api/forecast")
async def get_overall_forecast(request: ForecastRequest):
    try:
        raw_data = fetch_supabase_data()
        df_series = prepare_data(raw_data)
        
        category_name = "Revenue"

        if df_series.empty or len(df_series) < 2:
            print("DEBUG: Not enough historical data for overall forecast.")
            historical_data = [
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "month": date.strftime("%b'%y"),
                    "category": category_name,
                    "actual": round(float(revenue), 2),
                    "forecast": None,
                    "is_future": False
                }
                for date, revenue in zip(df_series.index, df_series)
            ] if not df_series.empty else []
            return {"status": "success", "data": historical_data, "message": "Insufficient data for overall forecast."}


        forecast_data_list = generate_forecast_data(df_series, request.months, category_name)
        
        historical_data_list = [
            {
                "date": date.strftime("%Y-%m-%d"),
                "month": date.strftime("%b'%y"),
                "category": category_name,
                "actual": round(float(revenue), 2),
                "forecast": None,
                "is_future": False
            }
            for date, revenue in zip(df_series.index, df_series)
        ]
        
        return {"status": "success", "data": historical_data_list + forecast_data_list}
        
    except Exception as e:
        print(f"Error in /api/forecast endpoint: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/forecast-by-dimension")
async def get_forecast_by_dimension(request: DimensionForecastRequest):
    try:
        print(f"DEBUG: Received request for dimension '{request.dimension}' with filter value '{request.filter_value}'")
        raw_data = fetch_supabase_data()
        
        df_series_filtered = prepare_data(raw_data, dimension=request.dimension, filter_value=request.filter_value)
        
        category_name = f"Revenue - {request.dimension}: {request.filter_value}"

        if df_series_filtered.empty or len(df_series_filtered) < 2:
            print(f"DEBUG: Not enough historical data for dimension: {request.dimension}, value: {request.filter_value}")
            historical_data_list = [
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "month": date.strftime("%b'%y"),
                    "category": category_name,
                    "actual": round(float(revenue), 2),
                    "forecast": None,
                    "is_future": False
                }
                for date, revenue in zip(df_series_filtered.index, df_series_filtered)
            ] if not df_series_filtered.empty else []
            return {"status": "success", "data": historical_data_list, "message": f"Insufficient data for dimension '{request.dimension}' with value '{request.filter_value}' to generate forecast."}

        forecast_data_list = generate_forecast_data(df_series_filtered, request.months, category_name)
        
        historical_data_list = [
            {
                "date": date.strftime("%Y-%m-%d"),
                "month": date.strftime("%b'%y"),
                "category": category_name,
                "actual": round(float(revenue), 2),
                "forecast": None,
                "is_future": False
            }
            for date, revenue in zip(df_series_filtered.index, df_series_filtered)
        ]
        
        return {"status": "success", "data": historical_data_list + forecast_data_list}
        
    except Exception as e:
        print(f"Error in /api/forecast-by-dimension endpoint: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/forecast-by-dimensions") # This endpoint is for multi-dimensional filtering, not just one.
async def get_forecast_by_dimensions(request: DimensionForecastRequest): # Consider renaming this to handle multiple dimensions if that's the intent.
    try:
        print(f"DEBUG: Received request for multi-dimension '{request.dimension}' with filter value '{request.filter_value}'")
        raw_data = fetch_supabase_data()
        
        df_series_filtered = prepare_data(
            raw_data, 
            dimension=request.dimension, 
            filter_value=request.filter_value
        )
        
        category_name = f"Revenue - {request.dimension}: {request.filter_value}"

        if df_series_filtered.empty or len(df_series_filtered) < 2:
            print(f"DEBUG: Not enough historical data for dimension: {request.dimension}, value: {request.filter_value}")
            historical_data_list = [
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "month": date.strftime("%b'%y"),
                    "category": category_name,
                    "actual": round(float(revenue), 2),
                    "forecast": None,
                    "is_future": False
                }
                for date, revenue in zip(df_series_filtered.index, df_series_filtered)
            ] if not df_series_filtered.empty else []
            return {"status": "success", "data": historical_data_list, "message": f"Insufficient data for dimension '{request.dimension}' with value '{request.filter_value}' to generate forecast."}

        forecast_data_list = generate_forecast_data(df_series_filtered, request.months, category_name)
        
        historical_data_list = [
            {
                "date": date.strftime("%Y-%m-%d"),
                "month": date.strftime("%b'%y"),
                "category": category_name,
                "actual": round(float(revenue), 2),
                "forecast": None,
                "is_future": False
            }
            for date, revenue in zip(df_series_filtered.index, df_series_filtered)
        ]
        
        return {"status": "success", "data": historical_data_list + forecast_data_list}
        
    except Exception as e:
        print(f"Error in /api/forecast-by-dimensions endpoint: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
# In main.py, after your existing imports and app = FastAPI() setup:

@app.get("/api/dimensions/unique-values")
async def get_unique_dimension_values(dimension: str):
    try:
        raw_data = fetch_supabase_data()
        df = pd.DataFrame(raw_data)

        if df.empty:
            return {"status": "success", "data": []}

        # Standardize column names (handle spaces and case)
        df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()
        
        processed_dimension = dimension.strip().replace(' ', '_').lower()

        if processed_dimension not in df.columns:
            raise HTTPException(status_code=404, detail=f"Dimension '{dimension}' not found in data columns.")
        
        # Get unique values, drop NaNs, convert to list, and sort
        unique_values = df[processed_dimension].dropna().astype(str).str.strip().str.lower().unique().tolist()
        unique_values.sort() # Sort alphabetically for display

        return {"status": "success", "data": unique_values}

    except HTTPException as e: # Re-raise HTTPExceptions
        raise e
    except Exception as e:
        print(f"Error in /api/dimensions/unique-values endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to fetch unique dimension values.")
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

app = FastAPI()