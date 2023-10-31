from src.DimondPricePrediction.components.data_ingestion import DataIngestion
from src.DimondPricePrediction.logger import logging
from src.DimondPricePrediction.components.data_transformation import DataTransformation
from src.DimondPricePrediction.components.model_trainer import ModelTrainer

logging.info("Creating Object of DataIngestion class")
data_ingestion = DataIngestion()

train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

data_transformation  = DataTransformation()

train_arr, test_arr = data_transformation.initialize_data_transformation(train_data_path, test_data_path)

model_training_obj = ModelTrainer()
model_training_obj.initiate_model_trainer(train_arr, test_arr)

