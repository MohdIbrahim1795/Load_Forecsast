import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os
from typing import List, Dict

#Setting up the logging configurtaion
def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('feeder_data_generation.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logger()

#Configuring the output directoey name and filename
CONFIG = {
    'output': {
        'directory': './Source Data',
        'filename': 'Historical_Usage.csv'},
        'simulation': {
            'feeders': ['FEEDER_01', 'FEEDER_02', 'FEEDER_03'],
            'min_usage': 10,  # kW
            'max_usage': 500  # kW
        },
        'date_range': {'start': '2022-01-01', 
                       'end': '2024-12-31'}  
        }


def generate_hourly_dates(start_date: str, end_date: str) -> List[datetime]:
    """Generate hourly datetime objects between two dates."""
    try:
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        delta = timedelta(hours=1)

        dates = []
        while start <= end:
            dates.append(start)
            start += delta
            logger.info(f"Generated ({len(dates)}) hourly timestamps")
        return dates

    except Exception as e:
            logger.error(f"Data generation failed: {str(e)}")
            raise


def simulate_feeder_usage(dates: List[datetime], feeders: List[str]) -> pd.DataFrame:
    """Create simulated feeder usage data"""
    try:
        data = []
        for dt in dates:
            for feeder in feeders:
                # Simulate usage with some randomness
                usage = np.random.uniform(
                    CONFIG['simulation']['min_usage'],
                    CONFIG['simulation']['max_usage']
                )
                data.append({
                    'Datetime': dt.strftime('%Y-%m-%d %H:%M:%S'),
                    'Feeder ID': feeder,
                    'Usage (kW)': round(usage, 2)
                })
        
        df = pd.DataFrame(data)
        df.reset_index(drop=True, inplace=True)
        df = df.sort_values(['Feeder ID', 'Datetime'])

        logger.info(f"Generated {len(df)} records of feeder data")
        logger.info(f"Data sorted by Feeder ID and Datetime")
    
        return df
    except Exception as e:
        logger.error(f"Data simulation failed: {str(e)}")
        raise

# --------------------------
# 4. FILE OPERATIONS
# --------------------------
def save_to_csv(df: pd.DataFrame, directory: str, filename: str) -> None:
    """Save DataFrame to CSV with error handling"""
    try:
        os.makedirs(directory, exist_ok=True)
        filepath = os.path.join(directory, filename)
        df.to_csv(filepath, index=False)
        logger.info(f"Successfully saved data to {filepath}")
    except PermissionError:
        logger.error(f"Permission denied when writing to {directory}")
        raise
    except Exception as e:
        logger.error(f"Failed to save CSV: {str(e)}")
        raise

# --------------------------
# 5. MAIN EXECUTION
# --------------------------
def main():
    try:
        logger.info("Starting feeder data generation")
        
        # Generate dates
        dates = generate_hourly_dates(
            CONFIG['date_range']['start'],
            CONFIG['date_range']['end']
        )
        
        # Simulate data
        df = simulate_feeder_usage(dates, CONFIG['simulation']['feeders'])
        
        # Save results
        save_to_csv(
            df,
            CONFIG['output']['directory'],
            CONFIG['output']['filename']
        )
        
        logger.info("Data generation completed successfully")
        
    except Exception as e:
        logger.critical(f"Program failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()