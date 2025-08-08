import os
import sys 
import pickle
import requests
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
from src.logger_config import setup_logger
from pathlib import Path
import statsmodels.api as sm
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
logger = setup_logger(__name__, 'Model_Testing.log')
from src.preprocessed_analysis import (
    weather_features,
    column_selection,
    get_feeder_list,
    create_forecasted_df,
    weather_data,
    Merged_Data
)

logger.info("Model Testing started")

def load_model(path):
    try:
        with open(path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded successfully from {path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model from {path}: {e}")
        return None
    

#Creating function to perform the testing on test data
def test_data():
    
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
        logger.info("Starting model testing process")
        for feeder in Feeders_List:
            logger.info(f"Processing feeder: {feeder}")
            forecast_data = forecast_data
            feeder_data = forecast_data.loc[forecast_data['Feeder_ID'] == feeder]
            temp_merged = Merged_Data(feeder_data, Weather_data)

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
            logger.info("Predicting from Prophet model")           
        
            # Load Model Prophet
            model_path = os.path.join("D:/Data Science Projects/Load Forecast/src/Model_P", 
                                    f"{feeder}_flat_prophet.pickle")
            fbp_model = load_model(model_path)
            if fbp_model is None:
                logger.error(f"Failed to load Prophet model for feeder {feeder}. Skipping Linear Regression training.")
                continue
            future = pd.DataFrame(fb_df['ds'])
            forecast = fbp_model.predict(future)
            Fb_result = forecast[['ds', 'yhat']]
            D_Prophet = pd.merge(T8_data, Fb_result, left_on='datetime', right_on='ds')
            D_Prophet['yhat'] = D_Prophet['yhat'].astype(float)
            temp_merged = D_Prophet.copy()
            temp_merged = weather_features(temp_merged)
            logger.info(f"Processing feeder: {feeder}")
           
            if temp_merged.empty:
                logger.warning(f"No forecast data found for feeder {feeder}. Skipping.")
                continue
        
            if temp_merged is None or temp_merged.empty:
                logger.error(f"No merged data found for feeder {feeder}. Skipping.")
                continue
                
            # Loading LR Model and LGB model
            model_path_1= os.path.join("D:/Data Science Projects/Load Forecast/src/Model_P", 
                                    f"{feeder}_LR.pickle")
            model_path_2= os.path.join("D:/Data Science Projects/Load Forecast/src/Model_P",
                              f"{feeder}_flat_LGBM.pickle")
            LR_model = load_model(model_path_1)
            LGB_model = load_model(model_path_2)
            X=column_selection(temp_merged)
            Y=temp_merged['final_usage']

            #Metrics for FB Prophet model on complete Training Data
            logger.info(f"PE, APE and RSE calculated f0r FBP Model for {feeder}")
    
            temp_merged['FBP_percentage_err'] = (D_Prophet['yhat']-temp_merged['final_usage'])/temp_merged['final_usage']
            temp_merged['FBP_abs_percentage_err'] = abs((D_Prophet['yhat']-temp_merged['final_usage'])/temp_merged['final_usage'])
            temp_merged['FBP_RMSE'] = np.sqrt((D_Prophet['yhat']-temp_merged['final_usage'])**2)

            #Prediction using Linear model complete Training Data
            # Match training format by adding constant
            X_const = sm.add_constant(X, has_constant='add')
            preds = LR_model.predict(X_const)
            temp_merged['LR_Result'] = pd.Series(preds, index=X_const.index)
            logger.info(f"Prediction completed on complete Training data on Linear Model for {feeder}")
            logger.info(f"PE, APE and RSE calculated f0r Linear Model for {feeder}")

            #Metrics for LR model on complete Training Data
            temp_merged['LR_percentage_err'] = (pd.to_numeric(temp_merged['LR_Result'], errors='coerce') -
            pd.to_numeric(temp_merged['final_usage'], errors='coerce'))/(pd.to_numeric(temp_merged['final_usage'], errors='coerce'))

            temp_merged['LR_abs_percentage_err'] = abs((pd.to_numeric(temp_merged['LR_Result'], errors='coerce') -
            pd.to_numeric(temp_merged['final_usage'], errors='coerce'))/(pd.to_numeric(temp_merged['final_usage'], errors='coerce')))

            temp_merged['LR_RMSE'] = np.sqrt((pd.to_numeric(temp_merged['LR_Result'], errors='coerce') -
            pd.to_numeric(temp_merged['final_usage'], errors='coerce')) ** 2)

            #Train Test splitting

            X_train = X.iloc[:int(X.shape[0]*0.9)].reset_index(drop=True)
            X_test  = X.iloc[int(X.shape[0]*0.9):].reset_index(drop=True)
            y_train = Y.iloc[:int(Y.shape[0]*0.9)].reset_index(drop=True).to_frame(name='final_usage')
            y_test  = Y.iloc[int(Y.shape[0]*0.9):].reset_index(drop=True).to_frame(name='final_usage')
            
            
            print('Start predicting...')
            y_test['y_hat'] = LGB_model.predict(X_test, num_iteration=LGB_model.best_iteration)
            
            #Prediction using LGB complete Training Data
            temp_merged['GBDT_Result'] = LGB_model.predict(X)
            logger.info(f"Prediction completed on Test data on LGB Model for {feeder}")
            
            #Metrics for LGB model on Complete Training Data and Test data        
            #Prediction yhat on test data using LGB
            logger.info(f"PE, APE and RSE calculated f0r LGB for {feeder}")
            y_test['GBDT_abs_percentage_err_Y'] = abs((y_test['y_hat']-y_test['final_usage'])/y_test['final_usage'])
            y_test['GBDT_RMSE_Y'] = np.sqrt((y_test['y_hat']-y_test['final_usage'])**2)

            temp_merged['GBDT_percentage_err'] = (temp_merged['GBDT_Result']-temp_merged['final_usage'])/temp_merged['final_usage']
            temp_merged['GBDT_abs_percentage_err'] = abs((temp_merged['GBDT_Result']-temp_merged['final_usage'])/temp_merged['final_usage'])
            temp_merged['GBDT_RMSE'] = np.sqrt((temp_merged['GBDT_Result']-temp_merged['final_usage'])**2)


    
             # Collect metric names and values only once per feeder
            metric_names = [
                "FBP_percentage_err",
                "FBP_abs_percentage_err",
                "FBP_RMSE",
                "LR_abs_percentage_err",
                "LR_percentage_err",
                "LR_RMSE",
                "GBDT_RMSE",
                "GBDT_abs_percentage_err",
                "GBDT_percentage_err",
                "GBDT_abs_percentage_err_Y",
                "GBDT_RMSE_Y"
                ]   
    
            metric_values = [
            float(temp_merged.describe()['FBP_percentage_err'][1]),
            float(temp_merged.describe()['FBP_abs_percentage_err'][1]),
            float(temp_merged.describe()['FBP_RMSE'][1]),
            float(temp_merged.describe()['LR_abs_percentage_err'][1]),
            float(temp_merged.describe()['LR_percentage_err'][1]),
            float(temp_merged.describe()['LR_RMSE'][1]),
            float(temp_merged.describe()['GBDT_RMSE'][1]),
            float(temp_merged.describe()['GBDT_abs_percentage_err'][1]),
            float(temp_merged.describe()['GBDT_percentage_err'][1]),
            float(y_test.describe()['GBDT_abs_percentage_err_Y'][1]),
            float(y_test.describe()['GBDT_RMSE_Y'][1])
            ]
    
            # Create a single column DataFrame for this feeder
            df = pd.DataFrame({feeder: metric_values}, index=metric_names)
    
            # Join into main table
            if result_summary.empty:
                result_summary = df
            else:
                result_summary = result_summary.join(df, how="outer")

            # After loop: save with metric_name as index
            result_summary.index.name = "metric_name"
            # Saving output to csv
            Path_dir=Path("D:/Data Science Projects/Load Forecast/src/Metrics_Result_Summary")
            Filename_1 = "Metrics_Result_Summary.csv"
            Path_dir.mkdir(parents=True, exist_ok=True)
            dtype_csv_path = Path_dir / Filename_1
            result_summary.to_csv(dtype_csv_path)

            # === Read CSV ===
            csv_path = Path("D:/Data Science Projects/Load Forecast/src/Metrics_Result_Summary/Metrics_Result_Summary.csv")
            df = pd.read_csv(csv_path)

            # --- Identify Best Models for MAPE and RMSE ---
            # Filter for only absolute percentage error metrics (MAPE equivalents)
            mape_metrics = df[df["metric_name"].str.contains("abs_percentage_err$", case=False)]

            # Filter for only RMSE metrics
            rmse_metrics = df[df["metric_name"].str.contains("RMSE$", case=False)]

            # Get the metric_name with the lowest value for each feeder
            best_mape = mape_metrics.set_index("metric_name").idxmin()
            best_rmse = rmse_metrics.set_index("metric_name").idxmin()

            summary = pd.DataFrame({
                "Best MAPE Model": best_mape,
                "Best RMSE Model": best_rmse
            })

            # Save summary to CSV
            summary_path = csv_path.parent / "Best_Model_Summary.csv"
            summary.to_csv(summary_path)
            print(f"Best model summary saved to: {summary_path}")

            # Print summary
            print("\n=== Best Model per Feeder ===")
            print(summary)

            # --- Heatmap for all values ---
            plt.figure(figsize=(10, 6))
            sns.heatmap(df.set_index("metric_name"), annot=True, fmt=".3f", cmap="YlGnBu")
            plt.title("Model Metric Comparison per Feeder")
            plt.tight_layout()
            plt.show()

    except Exception as e:
        logger.error(f"Error in model Testing process: {str(e)}")
        raise ValueError("Model Testing failed") from e
    
def main():
    try:
        logger.info("Starting model testing pipeline")
        # Test model
        logger.info("Starting model testing process")
        results = test_data()
        
        logger.info('Model Testing Successfully')
        print("Testing completed successfully. Check log file for details.")
        return 0
        
    except Exception as e:
        logger.error(f"Fatal error in main execution: {str(e)}", exc_info=True)
        print(f"Error occurred: {str(e)}")
        return 1

if __name__ == "__main__":
    # No logging setup needed here since it's already imported
    exit_code = main()
    sys.exit(exit_code)