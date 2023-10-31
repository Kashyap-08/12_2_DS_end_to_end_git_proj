from dataclasses import dataclass
import os
from src.DimondPricePrediction.logger import logging
from src.DimondPricePrediction.exception import customException
import sys
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from src.DimondPricePrediction.utils.utils import evaluate_model, save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifact', 'model.pkl')

class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config =  ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Spliting Dependent and Independent Variables")
            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )

            models = {
            'LinearRegression':LinearRegression(),
            'Lasso':Lasso(),
            'Ridge':Ridge(),
            'Elasticnet':ElasticNet()
            }

            model_report:dict = evaluate_model(X_train, y_train, X_test, y_test, models)
            print(model_report)
            print('\n=============================================================\n')
            logging.info(f"Model Report: {model_report}")

            # To get best model score from dictionary
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            print(f"Best model found, Model Name: {best_model_name}, R2_Score: {best_model_score}")
            print("\n =========================================================== \n")
            logging.info(f"Best model found, Model Name: {best_model_name}, R2_Score: {best_model_score}")

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

        except Exception as e:
            logging.info('Exception in initiate_model_trainer() method')
            raise customException(e, sys)