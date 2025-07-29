import pandas as pd
import os
from typing import List, Dict
from src.logger_config import setup_logger  # Import our custom logger

# Set up the logger
logger = setup_logger(__name__, 'feeder.log')

Feeder_ID={
'feeders': ['FEEDER_01', 'FEEDER_02', 'FEEDER_03']
}   
def read_feeder_data(Feeder_ID: Dict[str, List[str]]) -> pd.DataFrame:
    """Read feeder data from a dict in dataframe saving it in file Feeder.csv"""
    try:
        Feeders = pd.DataFrame(Feeder_ID)
        Feeders.reset_index(drop=True, inplace=True)
        logger.info("Feeder data read successfully")
        return Feeders
    except FileNotFoundError:
        logger.error("Feeder data file not found.")
        raise
    except Exception as e:
        logger.error(f"Error reading feeder data: {str(e)}")
        raise
    return pd.DataFrame()   

def save_feeder_data(df: pd.DataFrame,  directory: str, filename: str) -> None:
    """Save the feeder data to a CSV file."""
    try:
        # Create directory first (returns None)
        os.makedirs(directory, exist_ok=True)
        
        # Then create full path
        output_path = os.path.join(directory, filename)

        df.to_csv(output_path, index=False)
        logger.info(f"Feeder data saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save feeder data: {str(e)}")
        raise
    
def main():
    """Main function to read, process, and save feeder data."""
    try:
        logger.info("Starting feeder data processing")
        
        # Read feeder data
        df = read_feeder_data(Feeder_ID)
        
        # Save the processed data
        save_feeder_data(
            df,
            './Source Data',  # Replace with your desired output directory
            'Feeder.csv'         # Replace with your desired output filename
        )
        
        logger.info("Feeder data processing completed successfully")
        
    except Exception as e:
        logger.critical(f"Program failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()