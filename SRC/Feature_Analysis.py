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

    expected_columns = [
       'Feeder_ID',
        'yhat',
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
        *[f'Month_is_{i}' for i in range(1, 13)],
    ]

    # Include dummies for day_of_week and hour
    expected_columns += [col for col in df.columns if col.startswith("day_of_week_")]
    expected_columns += [col for col in df.columns if col.startswith("hour_")]

    # Select only existing columns
    available_columns = [col for col in expected_columns if col in df.columns]

    missing_columns = list(set(expected_columns) - set(available_columns))
    if missing_columns:
        logger.warning(f"Missing {len(missing_columns)} expected columns in selection")

    return df[available_columns]


def get_feeder_list(feeder_path: Union[str, Path]) -> List[str]:

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


def Merged_Data(forecast_data: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Merge forecast data with weather data.
    
    Args:
        forecast_data: DataFrame containing forecast features
        weather: DataFrame containing weather features

    Returns:
        DataFrame containing merged weather and load features ready for forecasting
        
    Raises:
        ValueError: If data merging fails
        Exception: For other unexpected errors
    """
    logger.info("Merging datasets")
    try:
        # Change from strict validation to allow duplicates in weather data
        merged_df = forecast_data.merge(weather, on='datetime', how='inner', validate='many_to_one')
        logger.info(f"Merged dataset contains {len(merged_df)} rows")
        return merged_df

    except Exception as e:
        logger.error(f"Error merging datasets: {str(e)}")
        raise ValueError("Data merging failed") from e
    

#Defining the function to create the dataframe for forecasted data
def read_data(feeder_path: Path, usage_path: Path, weather_path: Path) -> pd.DataFrame:
    try:
        feeder_data = pd.read_csv(feeder_path)
        usage_data = pd.read_csv(usage_path)
        weather_data = pd.read_csv(weather_path)
        logger.info("Data read successfully from all sources")
        print("Feeder Data:\n", feeder_data.head())
        print("Usage Data:\n", usage_data.head())
        print("Weather Data:\n", weather_data.head())
        print(f"Feeder data type: {feeder_data.dtypes}")
        print(f"Usage data type: {usage_data.dtypes}")
        print(f"Weather data type: {weather_data.dtypes}")
    except Exception as e:
        logger.error(f"Error reading data: {e}")
        
        #Renaming columns for weather and usage data
        logger.info("Renaming columns for weather and usage data and merging it")
    
    try:
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
    except Exception as e:
        logger.error(f"Error in handling Usage Data: {e}")
        raise
    #Handling Weather Data
    try:
        weather_columns = {"time", "temp_f", "wind_mph", "cloud_cover", "humidity"}
        weather_data['datetime'] = pd.to_datetime(weather_data['time'].astype(str).str.strip(), errors='coerce') 
        weather_data = weather_data.rename(columns={
            "temp_f": "temp",
             "wind_mph": "wind_speed",
            "cloud_cover": "cloud"
        })
        weather_data = weather_data[['datetime', 'temp', 'humidity', 'cloud', 'wind_speed']]
        print(f"Null_count for weather_data: {weather_data['datetime'].isnull().sum()}")
    except Exception as e:
        logger.error(f"Error in handling weather data: {e}")
        raise   
    try:    
        Data_df= Merged_Data(forecast_data, weather_data)
        USM=weather_features(Data_df)
        # Selecting column for which dype = bool
        Incorrect_dtype = [col for col in USM.columns if USM[col].dtype == bool]
        print(f"Incorrect_column_dtype: {Incorrect_dtype}")

        for col in Incorrect_dtype:
            USM[col] = USM[col].map({True: 1, False: 0}).astype('int64')

        # Check again if any bool columns remain
        U_Incorrect_dtype = [col for col in USM.columns if USM[col].dtype == bool]
        print(f"Remaining bool columns: {U_Incorrect_dtype}")
        #Transforming the boolean values into float

        logger.info("Data reading process completed successfully")
        logger.info("Exposing to column_selection function")
        
    except Exception as e:
        logger.critical(f"Program failed: {str(e)}", exc_info=True)
    
    return USM

if __name__ == "__main__":
    """Test the data processing pipeline"""
    try:
        print("\nStarting data preprocessing...")
        feeder_path = Path("D:/Data Science Projects/Load Forecast/Source Data/Feeder.csv")
        usage_path = Path("D:/Data Science Projects/Load Forecast/Source Data/Historical_Usage.csv")
        weather_path = Path("D:/Data Science Projects/Load Forecast/Source Data/Weather_Data.csv")

        
        # 1. Get feeder list
        feeders = get_feeder_list(feeder_path)
        print(f"Found {len(feeders)} feeders to process")
        
        
        # 2. Reading data
        final_df = read_data(feeder_path, usage_path, weather_path)
        print(f"Final dataset: {final_df.shape[0]} rows")
         
        
        print("\nFeature Engineering completed successfully exposed to funxtion weather_features!")
        print("Check Preprocessed_Analysis.log for detailed processing logs")
        
    except Exception as e:
        print(f"\nError in preprocessing: {str(e)}")
        logger.error(f"Preprocessing failed: {str(e)}")