from turtle import st
import pandas as pd
from datetime import timedelta, datetime, date
import os
import numpy as np
import requests
import io
from io import BytesIO 
import time
from src.logger_config import setup_logger 
from pathlib import Path
import pandas as pd
from typing import Union, List
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))  # Add project root to path

# Set up the logger
logger = setup_logger(__name__, 'Preprocessed_Analysis.log')


def weather_features(temp_merged: pd.DataFrame) -> pd.DataFrame:
    """
    Generate engineered weather-related features. Missing features will be filled with 0 to maintain column consistency.
    """
    # Derive missing datetime parts if not present
    if 'datetime' in temp_merged.columns:
        temp_merged['datetime'] = pd.to_datetime(temp_merged['datetime'], errors='coerce')
        if 'hour' not in temp_merged.columns:
            temp_merged['hour'] = temp_merged['datetime'].dt.hour
        if 'month' not in temp_merged.columns:
            temp_merged['month'] = temp_merged['datetime'].dt.month
        if 'day_of_week' not in temp_merged.columns:
            temp_merged['day_of_week'] = temp_merged['datetime'].dt.dayofweek + 1  # Monday=1
    else:
        raise ValueError("Missing 'datetime' column for extracting time features")

    # Initialize all expected dynamic columns with 0
    for i in range(24):
        temp_merged[f'CD90_{i}'] = 0.0
        temp_merged[f'HD30Cap5_{i}'] = 0.0
        temp_merged[f'HD30_{i}'] = 0.0
        temp_merged[f'CD80cap10_{i}'] = 0.0
        temp_merged[f'HD35Cap5_{i}'] = 0.0
        temp_merged[f'HD40Cap5_{i}'] = 0.0

    for i in range(24):
        hour_mask = temp_merged['hour'] == i
        temp_merged.loc[hour_mask & (temp_merged['temp'] >= 90), f'CD90_{i}'] = temp_merged['temp'] - 90
        temp_merged.loc[hour_mask & (temp_merged['temp'] >= 30) & (temp_merged['temp'] < 35), f'HD30Cap5_{i}'] = 35 - temp_merged['temp']
        temp_merged.loc[hour_mask & (temp_merged['temp'] < 30), f'HD30_{i}'] = 30 - temp_merged['temp']
        temp_merged.loc[hour_mask & (temp_merged['temp'] >= 80) & (temp_merged['temp'] < 90), f'CD80cap10_{i}'] = temp_merged['temp'] - 80
        temp_merged.loc[hour_mask & (temp_merged['temp'] >= 35) & (temp_merged['temp'] < 40), f'HD35Cap5_{i}'] = 40 - temp_merged['temp']
        temp_merged.loc[hour_mask & (temp_merged['temp'] >= 40) & (temp_merged['temp'] < 45), f'HD40Cap5_{i}'] = 45 - temp_merged['temp']

    # Static features
    temp_merged['SatCD'] = np.where((temp_merged['day_of_week'] == 6) & (temp_merged['temp'] >= 65), temp_merged['temp'] - 65, 0)
    temp_merged['SunCD'] = np.where((temp_merged['day_of_week'] == 7) & (temp_merged['temp'] >= 65), temp_merged['temp'] - 65, 0)
    temp_merged['SatHD'] = np.where((temp_merged['day_of_week'] == 6) & (temp_merged['temp'] < 65), 65 - temp_merged['temp'], 0)
    temp_merged['SunHD'] = np.where((temp_merged['day_of_week'] == 7) & (temp_merged['temp'] < 65), 65 - temp_merged['temp'], 0)

    # Cloud groupings
    temp_merged['Cloud80'] = np.where(temp_merged['cloud'] >= 80, temp_merged['cloud'] - 80, 0)
    temp_merged['Cloud60'] = np.where((temp_merged['cloud'] < 80) & (temp_merged['cloud'] >= 60), temp_merged['cloud'] - 60, 0)
    temp_merged['Cloud40'] = np.where((temp_merged['cloud'] < 60) & (temp_merged['cloud'] >= 40), temp_merged['cloud'] - 40, 0)
    temp_merged['Cloud20'] = np.where((temp_merged['cloud'] < 40) & (temp_merged['cloud'] >= 20), temp_merged['cloud'] - 20, 0)
    temp_merged['Cloud0'] = np.where(temp_merged['cloud'] < 20, temp_merged['cloud'], 0)

    temp_merged['HumidityHot56'] = np.where((temp_merged['temp'] > 80) & (temp_merged['humidity'] <= 30) & (temp_merged['month'].between(5, 6)), (temp_merged['temp'] - 80) * (30 - temp_merged['humidity']), 0)
    temp_merged['HumidityHot78'] = np.where((temp_merged['temp'] > 80) & (temp_merged['humidity'] <= 30) & (temp_merged['month'].between(7, 8)), (temp_merged['temp'] - 80) * (30 - temp_merged['humidity']), 0)

    temp_merged['Low_Peak'] = np.where((temp_merged['month'].between(1, 3)) & (temp_merged['hour'].between(2, 6)) & (temp_merged['temp'] < 50), 50 - temp_merged['temp'], 0)
    temp_merged['High_Peak456'] = np.where((temp_merged['month'].between(4, 6)) & (temp_merged['hour'].between(16, 20)) & (temp_merged['temp'] > 80), temp_merged['temp'] - 80, 0)
    temp_merged['High_Peak789'] = np.where((temp_merged['month'].between(7, 9)) & (temp_merged['hour'].between(16, 20)) & (temp_merged['temp'] > 80), temp_merged['temp'] - 80, 0)

    # Seasonal features
    def add_season_feature(month_range, hour_range, name_prefix):
        morning_mask = month_range & hour_range
        evening_mask = month_range & ~hour_range

        temp_merged[f'{name_prefix}MorningHD'] = np.where(morning_mask & (temp_merged['temp'] < 65), 65 - temp_merged['temp'], 0)
        temp_merged[f'{name_prefix}MorningCD'] = np.where(morning_mask & (temp_merged['temp'] >= 65), temp_merged['temp'] - 65, 0)
        temp_merged[f'{name_prefix}EveningHD'] = np.where(evening_mask & (temp_merged['temp'] < 65), 65 - temp_merged['temp'], 0)
        temp_merged[f'{name_prefix}EveningCD'] = np.where(evening_mask & (temp_merged['temp'] >= 65), temp_merged['temp'] - 65, 0)

    m = temp_merged['month']
    h = temp_merged['hour']
    add_season_feature(m.between(4, 6), h.between(8, 19), 'Spring')
    add_season_feature(m.between(7, 9), h.between(8, 19), 'Summer')
    add_season_feature(m.between(10, 12), h.between(8, 19), 'Autumn')
    add_season_feature(m.between(1, 3), h.between(8, 19), 'Winter')

    temp_merged['CloudsHot'] = np.where(temp_merged['temp'] > 70, temp_merged['cloud'] / 100 * (temp_merged['temp'] - 70), 0)
    temp_merged['CloudsCold'] = np.where(temp_merged['temp'] < 55, temp_merged['cloud'] / 100 * (55 - temp_merged['temp']), 0)

    temp_merged['WindHot70Cap10'] = np.where(temp_merged['temp'] > 70, temp_merged['wind_speed'] * (temp_merged['temp'] - 70), 0)
    temp_merged['WindHot80'] = np.where(temp_merged['temp'] > 80, temp_merged['wind_speed'] * (temp_merged['temp'] - 80), 0)
    temp_merged['WindCold'] = np.where(temp_merged['temp'] < 55, temp_merged['wind_speed'] * (55 - temp_merged['temp']), 0)

    for i in range(1, 13):
        temp_merged[f'Month_is_{i}'] = (temp_merged['month'] == i).astype(int)

    temp_merged.fillna(0, inplace=True)

    # Dummies
    if 'day_of_week' in temp_merged.columns:
        temp_merged = pd.get_dummies(temp_merged, columns=['day_of_week'], prefix='day_of_week')
    if 'hour' in temp_merged.columns:
        temp_merged = pd.get_dummies(temp_merged, columns=['hour'], prefix='hour')

    return temp_merged

def column_selection(df: pd.DataFrame) -> pd.DataFrame:
    """Select features for model training from the available columns."""

    expected_columns = ['yhat',
        # All expected dynamic columns
        *[f'CD90_{i}' for i in range(24)],
        *[f'HD30Cap5_{i}' for i in range(24)],
        *[f'HD30_{i}' for i in range(24)],
        *[f'CD80cap10_{i}' for i in range(24)],
        *[f'HD35Cap5_{i}' for i in range(24)],
        *[f'HD40Cap5_{i}' for i in range(24)],
        'SatCD', 'SunCD', 'SatHD', 'SunHD',
        'Cloud80', 'Cloud60', 'Cloud40', 'Cloud20', 'Cloud0',
        'HumidityHot56', 'HumidityHot78',
        'Low_Peak', 'High_Peak456', 'High_Peak789',
        'SpringMorningHD', 'SpringMorningCD', 'SpringEveningHD', 'SpringEveningCD',
        'SummerMorningHD', 'SummerMorningCD', 'SummerEveningHD', 'SummerEveningCD',
        'AutumnMorningHD', 'AutumnMorningCD', 'AutumnEveningHD', 'AutumnEveningCD',
        'WinterMorningHD', 'WinterMorningCD', 'WinterEveningHD', 'WinterEveningCD',
        'CloudsHot', 'CloudsCold',
        'WindHot70Cap10', 'WindHot80', 'WindCold',
        'Month_is_1', 'Month_is_2', 'Month_is_3', 'Month_is_4', 'Month_is_5',
        'Month_is_6', 'Month_is_7', 'Month_is_8', 'Month_is_9', 'Month_is_10',
        'Month_is_11', 'Month_is_12','day_of_week_1', 'day_of_week_2', 'day_of_week_3',
        'day_of_week_4', 'day_of_week_5', 'day_of_week_6', 'day_of_week_7',
        'hour_0', 'hour_1', 'hour_2', 'hour_3', 'hour_4', 'hour_5', 'hour_6',
        'hour_7', 'hour_8', 'hour_9', 'hour_10', 'hour_11', 'hour_12',
        'hour_13', 'hour_14', 'hour_15', 'hour_16', 'hour_17', 'hour_18',
        'hour_19', 'hour_20', 'hour_21', 'hour_22', 'hour_23']

    # Select only existing columns
    available_columns = [col for col in expected_columns if col in df.columns]

    missing_columns = list(set(expected_columns) - set(available_columns))
    if missing_columns:
        logger.warning(f"Missing {len(missing_columns)} expected columns in selection")

    return df[available_columns]



#Defining the function to create the dataframe for forecasted data

def create_forecasted_df(usage: Union[str, Path]) -> pd.DataFrame:
    """ Args:
        usage: Path to the usage data CSV file (as string or Path object)
        
    Returns:
        Processed DataFrame with forecast data
        
    Raises:
        FileNotFoundError: If usage file doesn't exist
        ValueError: If required columns are missing or data processing fails
    """
    logger.info("Creating forecasted DataFrame")
    try:
        # Use the provided usage path parameter
        usage_data = pd.read_csv(usage)
        logger.info(f"Successfully loaded usage data with {len(usage_data)} rows")
        
        # Parse datetime with format
        usage_data['Datetime'] = pd.to_datetime(usage_data['Datetime'].astype(str).str.strip(), errors='coerce')
        usage_data = usage_data.rename(columns={
            "Feeder ID": "Feeder_ID",
            "Datetime": "datetime",
            "Usage (kW)": "final_usage"
        })
        usage_data['month'] = usage_data['datetime'].dt.month
        usage_data['day_of_week'] = usage_data['datetime'].dt.dayofweek + 1
        usage_data['hour'] = usage_data['datetime'].dt.hour
        
        # Sort by datetime
        print(f"Null_count datetime usage_data: {usage_data['datetime'].isnull().sum()}")
        forecast_data = usage_data.sort_values('datetime')
        return forecast_data
        
    except FileNotFoundError as e:
        logger.error(f"Usage data file not found: {usage}")
        raise
    except Exception as e:
        logger.error(f"Error processing usage data: {str(e)}")
        raise ValueError("Usage data processing failed") from e
    
# Creating function to return weather _data Dataframe
def weather_data(weather_path: Union[str, Path]) -> pd.DataFrame:
    logger.info("Loading and processing weather data")
    try:
        weather_df = pd.read_csv(weather_path)
        logger.info(f"Successfully loaded weather data with {len(weather_df)} rows")

        weather_columns = {"time", "temp_f", "wind_mph", "cloud_cover", "humidity"}
        weather_df['datetime'] = pd.to_datetime(weather_df['time'].astype(str).str.strip(), errors='coerce') 
        weather_df= weather_df.rename(columns={
            "temp_f": "temp",
             "wind_mph": "wind_speed",
            "cloud_cover": "cloud"
        })
        weather_df= weather_df[['datetime', 'temp', 'humidity', 'cloud', 'wind_speed']]
        print(f"Null_count for weather_data: {weather_df['datetime'].isnull().sum()}")
        return weather_df

    except FileNotFoundError as e:
        logger.error(f"Weather data file not found: {weather_path}")
        raise
    except Exception as e:
        logger.error(f"Error processing weather data: {str(e)}")
        raise ValueError("Weather data processing failed") from e



def get_feeder_list(feeder_path: Union[str, Path]) -> List[str]:
    """Get list of feeder IDs from feeder data file (as strings).
    
    Args:
        feeder_path: Path to the feeder data CSV file
        
    Returns:
        List of feeder ID strings
        
    Raises:
        FileNotFoundError: If feeder file doesn't exist
        ValueError: If data processing fails
    """
    try:
        # Read and process data
        df = pd.read_csv(feeder_path)
        df = df.rename(columns={"feeders": "Feeder_ID"})
        
        logger.info("Processing feeder IDs")
        
        # Convert to string and clean whitespace
        df['Feeder_ID'] = df['Feeder_ID'].astype(str).str.strip()
        
        # Get unique IDs
        sp_id_list = df['Feeder_ID'].unique().tolist()
        
        logger.debug(f"Processed {len(sp_id_list)} feeder IDs: {sp_id_list}")
        return sp_id_list
            
    except FileNotFoundError as e:
        logger.error(f"Feeder data file not found: {feeder_path}")
        raise
    except pd.errors.EmptyDataError as e:
        logger.error("Feeder data file is empty")
        raise ValueError("Empty feeder data file") from e
    except Exception as e:
        logger.error(f"Error processing feeder data: {str(e)}")
        raise ValueError("Feeder data processing failed") from e


def Merged_Data(forecast: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    logger.info("Merging datasets")
    try:
        # Change from strict validation to allow duplicates in weather data
        merged_df = forecast.merge(weather, on='datetime', how='inner', validate='many_to_one')
        logger.info(f"Merged dataset contains {len(merged_df)} rows")
        return merged_df

    except Exception as e:
        logger.error(f"Error merging datasets: {str(e)}")
        raise ValueError("Data merging failed") from e


if __name__ == "__main__":
    """Test the data processing pipeline"""
    try:
        # Example file paths - replace with your actual paths
        base_path = "D:/Data Science Projects/Load Forecast/Source Data/"
        feeder_path = f"{base_path}Feeder.csv"
        usage_path = f"{base_path}Historical_Usage.csv" 
        weather_path = f"{base_path}Weather_Data.csv"
        
        print("\nStarting data preprocessing...")
        
        # 1. Get feeder list
        feeders = get_feeder_list(feeder_path)
        print(f"Found {len(feeders)} feeders to process")
        
        # 2. Process usage data
        forecast_df = create_forecasted_df(usage_path)
        print(f"Processed forecast data: {forecast_df.shape[0]} rows")
        
        # 3. Process weather data
        weather_df = weather_data(weather_path)
        print(f"Processed weather data: {weather_df.shape[0]} rows")
        
        # 4. Merge data
        merged_df = Merged_Data(forecast_df, weather_df)
        print(f"Merged dataset: {merged_df.shape[0]} rows")
        
        # 5. Apply weather features
        processed_df = weather_features(merged_df)
        print("Completed weather feature engineering")
        
        
        
        print("\nPreprocessing completed successfully!")
        print("Check Preprocessed_Analysis.log for detailed processing logs")
        
    except Exception as e:
        print(f"\nError in preprocessing: {str(e)}")
        logger.error(f"Preprocessing failed: {str(e)}")