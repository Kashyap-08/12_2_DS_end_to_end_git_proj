import logging
from src.DimondPricePrediction.logger import logging
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from src.DimondPricePrediction.exception import customException
import sys

class DataIngestionConfig:
    raw_data_path:str = os.path.join("artifact", "raw.csv")
    train_data_path:str = os.path.join("artifact", "train.csv")
    test_data_path:str = os.path.join("artifact", "test.csv")


class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config = DataIngestionConfig()


    def initiate_data_ingestion(self):
        logging.info("Starting Data Ingestion")

        try:
            # Read raw data
            path = os.path.join("notebooks","data","gemstone.csv")
            df = pd.read_csv(path)
            logging.info("Read file as csv into a DataFrame")

            # Save Raw data
            logging.info("Save the raw data to artifcat folder")
            raw_path = self.ingestion_config.raw_data_path

            # Create folder where all the data will be stored
            dir_path = os.path.dirname(raw_path)
            os.makedirs(dir_path, exist_ok=True)
            logging.info(f"make {dir} directoy where all data will be stored")

            df.to_csv(raw_path, index=False)
            logging.info(f"Raw data saved to path: {raw_path}")

            # Split data into Train Test Split
            logging.info("Perform Train Test Split")
            train, test = train_test_split(df, test_size=0.25)
            logging.info("Train Test split completed")

            # save train test dataset
            train_path = self.ingestion_config.train_data_path
            test_path = self.ingestion_config.test_data_path

            train.to_csv(train_path, index=False)
            logging.info(f"Saved Train data to path: {train_path}")

            test.to_csv(test_path, index=False)
            logging.info(f"Saved Test data to path: {test_path}")

            logging.info("Return Data path for train and test datsets")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info("Exception during Data Ingestion Step")
            raise customException(e, sys)