# Standard library imports (alphabetical)
import copy
from datetime import date, datetime, timedelta
import os
import pickle

# Third-party imports (alphabetical by package)
import lightgbm as lgb
import numpy as np
import pandas as pd
import psutil
import requests
import scipy as sp
from lightgbm.sklearn import LGBMRegressor
from prophet import Prophet
from sklearn import metrics
from sklearn.datasets import dump_svmlight_file
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import *
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from statsmodels.api import add_constant
from statsmodels.iolib.smpickle import load_pickle
from statsmodels.regression import linear_model
import sys
from pathlib import Path
from src.Feature_Analysis import read_data
sys.path.append(str(Path(__file__).parent.parent))  # Add project root to path


from src.preprocessed_analysis import (
    weather_features,
    column_selection,
    get_feeder_list,
    create_forecasted_df,
    weather_data,
    Merged_Data
)

# wb.weather_features
# wb.column_selection
# wb.create_forecasted_df
# wb.get_feeder_list
# wb.weather_data
# wb.Merged_Data

# Local application/library specific imports
from src.logger_config import setup_logger
# Set up the logger
logger = setup_logger(__name__, 'Model_Training.log')

#"""Defining two separate functions for saving the model in defined path and loading the model from defined path"""

def save_model(model, path: str) -> None:
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Model saved successfully at {path}")
    except Exception as e:
        logger.error(f"Error saving model at {path}: {e}")

def load_model(path):
    try:
        with open(path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded successfully from {path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model from {path}: {e}")
        return None
    

# Defining the function to train the model
def train_model():
    
    try:
        result_summary = pd.DataFrame([])
        logger.info("Data received for model training")
        feeder_path = Path("D:/Data Science Projects/Load Forecast/Source Data/Feeder.csv")
        usage_path = Path("D:/Data Science Projects/Load Forecast/Source Data/Historical_Usage.csv")
        weather_path = Path("D:/Data Science Projects/Load Forecast/Source Data/Weather_Data.csv")
        Feeders_List = get_feeder_list(feeder_path)
        forecast_data = create_forecasted_df(usage_path)
        Weather_data= weather_data(weather_path)
    
    except Exception as e:
        logger.error(f"Error in data preparation: {str(e)}")
        raise ValueError("Data preparation failed") from e  
    
    try:
        logger.info("Starting model training process")
        for feeder in Feeders_List:
            logger.info(f"Processing feeder: {feeder}")
            forecast_data = forecast_data
            feeder_data = forecast_data.loc[forecast_data['Feeder_ID'] == feeder]
            temp_merged = Merged_Data(feeder_data, Weather_data)
            # Inside your training loop, add progress reporting
        
            # Filter the forecasted data for the current feeder
            
            if temp_merged.empty:
                logger.warning(f"No forecast data found for feeder {feeder}. Skipping.")
                continue
                
            # Get weather data and merge with forecast data
        
            if temp_merged is None or temp_merged.empty:
                logger.error(f"No merged data found for feeder {feeder}. Skipping.")
                continue
                
            # Process weather features
            T8_data = temp_merged
            fb_df = T8_data[['datetime', 'final_usage']]
            fb_df = fb_df.rename(columns={"datetime": "ds", "final_usage": "y"})

            # Train Prophet model
            logger.info("Training Prophet model")           
            m = Prophet(yearly_seasonality=True, growth='flat')
            m.add_country_holidays(country_name='US')
            m.fit(fb_df)
    
            # Save model
            model_path = os.path.join("D:/Data Science Projects/Load Forecast/src/Model_P", 
                                    f"{feeder}_flat_prophet.pickle")
            save_model(m, model_path)

            #Training the Linear Regression model
            logger.info("Training Linear Regression model")
            m_model= load_model(f"D:/Data Science Projects/Load Forecast/src/Model_P/{feeder}_flat_prophet.pickle")
            if m_model is None:
                logger.error(f"Failed to load Prophet model for feeder {feeder}. Skipping Linear Regression training.")
                continue
            future = pd.DataFrame(fb_df['ds'])
            forecast = m_model.predict(future)
            print(forecast.columns)
            Fb_result = forecast[['ds', 'yhat']]
            D_Prophet = pd.merge(T8_data, Fb_result, left_on='datetime', right_on='ds')
            D_Prophet['yhat'] = D_Prophet['yhat'].astype(float)
            temp_merged = D_Prophet.copy()
            temp_merged = weather_features(temp_merged)


            logger.info("Predictions from fb prophet is cascaded to Merged DataFrame")

            logger.info("Started Preparing data for Linear Regression")
            f1_mask = temp_merged['Feeder_ID'] == feeder
            
            X = column_selection(temp_merged)
            Y = temp_merged['final_usage']
            
            X_Y_DF=pd.DataFrame({
            'Column': X.columns,
            'DataType': X.dtypes.astype(str)  # convert dtype objects to strings
            })

            Path_dir=Path("D:/Data Science Projects/Load Forecast/Processed Data")
            Filename_1 = "X_dtype.csv"
            Path_dir.mkdir(parents=True, exist_ok=True)
            dtype_csv_path = Path_dir / Filename_1
            X_Y_DF.to_csv(dtype_csv_path, index=False)
            
            # Ensure all data is float
            # Drop rows with any NaNs after conversion
            X = X.dropna()
            Y = Y.loc[X.index]
            print("\n Data type check before training:")
            print(X.dtypes.value_counts())

            non_numeric_cols = X.columns[X.dtypes == 'object']
            if not non_numeric_cols.empty:
                print(f"\n These columns are still non-numeric: {list(non_numeric_cols)}")
                print(X[non_numeric_cols].head())
            X = X.apply(pd.to_numeric, errors='coerce').astype(np.float32)

            # Add constant for intercept
            X_const = sm.add_constant(X, has_constant='add')
        

            # Train OLS model
            ols_model = sm.OLS(Y, X_const)
            results = ols_model.fit()
            
            model_path = os.path.join("D:/Data Science Projects/Load Forecast/src/Model_P", 
                                    f"{feeder}_LR.pickle")
            save_model(results, model_path)
            logger.info(f"Linear Regression model saved for feeder {feeder}")
            
            #Training the LightGBM model
    
            logger.info("Training LightGBM model")
            X = column_selection(temp_merged)
            Y = temp_merged['final_usage']
             # Ensure all data is float
            # Drop rows with any NaNs after conversion
            X = X.dropna()
            Y = Y.loc[X.index]
            # --- Prepare train/test split (as before) ---
            X_train = X.iloc[:int(X.shape[0]*0.9)].reset_index(drop=True)
            X_test  = X.iloc[int(X.shape[0]*0.9):].reset_index(drop=True)
            y_train = Y.iloc[:int(Y.shape[0]*0.9)].squeeze().reset_index(drop=True)  # squeeze -> Series
            y_test  = Y.iloc[int(Y.shape[0]*0.9):].squeeze().reset_index(drop=True)

            # quick checks
            if X_train.empty or X_test.empty or y_train.empty or y_test.empty:
                logger.warning(f"Empty split for feeder {feeder}. Skipping LGBM.")
            else:
                lgb_train = lgb.Dataset(X_train.values, label=y_train.values)
                lgb_eval  = lgb.Dataset(X_test.values,  label=y_test.values, reference=lgb_train)

            params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': 'mape',
            'num_leaves': 200,
            'learning_rate': 0.05,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.7,
            'bagging_freq': 50,
            'verbose': 1
             }

            print('Start training...')
           
            lgb_model = lgb.train(
            params,
            lgb_train,
            num_boost_round=2000,
            valid_sets=[lgb_eval],
            callbacks=[
            lgb.early_stopping(stopping_rounds=300),
            lgb.log_evaluation(period=50)   # optional: logs eval every 50 rounds
            ]
            )

            # Save trained model (use distinct var name)
            model_path = os.path.join("D:/Data Science Projects/Load Forecast/src/Model_P",
                              f"{feeder}_flat_LGBM.pickle")
            save_model(lgb_model, model_path)
            logger.info(f"LightGBM model saved for feeder: {feeder}")
        return temp_merged
                 
    except Exception as e:
        logger.error(f"Error in model training process: {str(e)}")
        raise ValueError("Model training failed") from e
            

def main():
    """
    Main function to execute the model training pipeline.
    Handles data preparation, model training, and error logging.
    """
    try:
        logger.info("Starting model training pipeline")
        
        # Define paths
        base_dir = Path("D:/Data Science Projects/Load Forecast")
        data_dir = base_dir / "Source Data"
        model_dir = base_dir / "src" / "Model_P"
        
        # Ensure directories exist
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Load and prepare data
        feeder_path = data_dir / "Feeder.csv"
        usage_path = data_dir / "Historical_Usage.csv"
        weather_path = data_dir / "Weather_Data.csv"
        
        logger.info("Loading and preparing data")
        Feeders_List = get_feeder_list(feeder_path)
        forecast_data = create_forecasted_df(usage_path)
        Weather_data = weather_data(weather_path)
        merged_data = Merged_Data(forecast_data, Weather_data)
        
        # Train models
        logger.info("Starting model training process")
        results = train_model()
        
        logger.info("Model training completed successfully")
        print("Training completed successfully. Check log file for details.")
        return 0
        
    except Exception as e:
        logger.error(f"Fatal error in main execution: {str(e)}", exc_info=True)
        print(f"Error occurred: {str(e)}")
        return 1

if __name__ == "__main__":
    # No logging setup needed here since it's already imported
    exit_code = main()
    sys.exit(exit_code)