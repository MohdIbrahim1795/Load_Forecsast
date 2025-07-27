"""
Historical Weather Data Fetcher - Timezone and Date Range Corrected
"""

import pandas as pd
from datetime import datetime, timedelta
import requests
import pytz
import os
import time
import logging
from typing import Dict, List

# --------------------------
# 1. CONFIGURATION
# --------------------------
BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
LOCATION = {
    'latitude': 33.6868,
    'longitude': -117.8794,
    'timezone': 'America/Los_Angeles'
}
OUTPUT_DIR = r'D:\Data Science Projects\Load Forecast\Source Data'
OUTPUT_FILE = 'Weather_Data.csv'
MAX_RETRIES = 3
RETRY_DELAY = 5
CHUNK_SIZE_DAYS = 90

HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
    "pressure_msl",
    "cloud_cover",
    "uv_index",
    "weather_code"
]

WEATHER_CODES = {
    0: "Clear", 1: "Mainly Clear", 2: "Partly Cloudy",
    3: "Overcast", 45: "Fog", 51: "Light Drizzle",
    61: "Light Rain", 80: "Rain Showers"
}

# --------------------------
# 2. LOGGING SETUP
# --------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('weather_data.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --------------------------
# 3. CORE FUNCTIONS (FIXED TIMEZONE HANDLING)
# --------------------------
def process_weather_data(raw_data: Dict, request_start: datetime) -> pd.DataFrame:
    """Process API response with proper date filtering and timezone handling."""
    try:
        if not raw_data or 'hourly' not in raw_data:
            raise ValueError("Missing 'hourly' data in API response")

        hourly = raw_data['hourly']
        
        # Convert to naive datetime (no timezone)
        time_data = pd.to_datetime(hourly['time'])
        
        # Create DataFrame with naive timestamps
        df = pd.DataFrame({
            'time': time_data,
            'temp_f': pd.to_numeric(hourly['temperature_2m']),
            'humidity': pd.to_numeric(hourly['relative_humidity_2m']),
            'precip_in': pd.to_numeric(hourly['precipitation']),
            'wind_mph': pd.to_numeric(hourly['wind_speed_10m']),
            'wind_degree': pd.to_numeric(hourly['wind_direction_10m']),
            'pressure_mb': pd.to_numeric(hourly['pressure_msl']),
            'cloud_cover': pd.to_numeric(hourly['cloud_cover']),
            'uv_index': pd.to_numeric(hourly['uv_index']),
            'weather_code': pd.to_numeric(hourly['weather_code'])
        })

        df['condition'] = df['weather_code'].map(WEATHER_CODES).fillna('Unknown')
        
        # Filter to only include dates >= our requested start date
        df = df[df['time'] >= request_start]
        
        return df

    except Exception as e:
        logger.error(f"Data processing failed: {str(e)}")
        raise

def fetch_data_chunk(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Fetch data for a date range with proper parameter handling."""
    try:
        params = {
            **LOCATION,
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'hourly': HOURLY_VARIABLES,
            'temperature_unit': 'fahrenheit',
            'wind_speed_unit': 'mph',
            'precipitation_unit': 'inch'
        }
        
        logger.debug(f"Fetching with params: {params}")
        response = requests.get(BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        return process_weather_data(response.json(), start_date)
        
    except Exception as e:
        logger.error(f"Failed chunk {start_date.date()} to {end_date.date()}: {str(e)}")
        return pd.DataFrame()

# --------------------------
# 4. MAIN EXECUTION
# --------------------------
def main():
    try:
        logger.info("Starting weather data collection")
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Set date range (will filter to ensure no earlier dates)
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2024, 12, 31)
        current_date = start_date
        all_data = []

        while current_date <= end_date:
            chunk_end = min(current_date + timedelta(days=CHUNK_SIZE_DAYS), end_date)
            logger.info(f"Processing {current_date.date()} to {chunk_end.date()}")
            
            df_chunk = fetch_data_chunk(current_date, chunk_end)
            if not df_chunk.empty:
                all_data.append(df_chunk)
            
            current_date = chunk_end + timedelta(days=1)
            time.sleep(1)  # Rate limiting

        if all_data:
            final_df = pd.concat(all_data).sort_values('time')
            
            # Ensure no timezone info remains
            final_df['time'] = final_df['time'].dt.tz_localize(None)
            
            # Final filter just in case
            final_df = final_df[final_df['time'] >= start_date]
            
            output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
            final_df.to_csv(output_path, index=False)
            logger.info(f"Success! Saved {len(final_df)} records to {output_path}")
            print("Sample timestamps:")
            print(final_df['time'].head(16).to_string(index=False))
        else:
            logger.error("No data was collected")

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()