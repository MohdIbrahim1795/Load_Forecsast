import pandas as pd
import os
from typing import List, Dict
from src.logger_config import setup_logger
from src.preprocessed_analysis import (weather_features, Merged_Data, column_selection)  # Import our custom logger  
from pathlib import Path

#Reading all three csv files using pandas 
# Set up the logger
logger = setup_logger(__name__, 'sample_data.log')

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
    return Merged_Data(forecast_data, weather_data)

def main():
    """Main function to read and display data from CSV files."""    
    try:
        logger.info("Starting data reading process")
        feeder_path = Path("D:/Data Science Projects/Load Forecast/Source Data/Feeder.csv")
        usage_path = Path("D:/Data Science Projects/Load Forecast/Source Data/Historical_Usage.csv")
        weather_path = Path("D:/Data Science Projects/Load Forecast/Source Data/Weather_Data.csv")

        Data_df = read_data(feeder_path, usage_path, weather_path)
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

        
        #saving data in csv for column adn their corresponding dtype
        col_dtype_df = pd.DataFrame({
        'Column': USM.columns,
        'DataType': USM.dtypes.astype(str)  # convert dtype objects to strings
        })

        # Save to CSV
        
        Path_dir=Path("D:/Data Science Projects/Load Forecast/Processed Data")
        Filename_1 = "USM_Column_Types.csv"
        Path_dir.mkdir(parents=True, exist_ok=True)
        dtype_csv_path = Path_dir / Filename_1
        col_dtype_df.to_csv(dtype_csv_path, index=False)

       #Creating five rows of USM and saving it in datarame
        sample_data = USM
        filename_2 = "USM_Sample_Data.csv"
        sample_data_path = Path_dir / filename_2
        sample_data.to_csv(sample_data_path, index=False)

        logger.info("Data reading process completed successfully")
        logger.info("Exposing to column_selection function")
        SMC = column_selection(USM)
        
    
    except Exception as e:
        logger.critical(f"Program failed: {str(e)}", exc_info=True) 

if __name__ == "__main__":
    main()
