import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.DimondPricePrediction.logger import logging
from src.DimondPricePrediction.exception import customException
import sys
import os
from src.DimondPricePrediction.utils.utils import load_object
import mlflow
from urllib.parse import urlparse

class ModelEvaluation:
    def __init__(self) -> None:
        pass

    def eval_metrics(self, actual, pred):
        mse = mean_squared_error(actual, pred)
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)

        return mse, rmse, mae, r2
    
    def initiate_model_evaluation(self, train_arr, test_arr):
        try:
            x_test, y_test = test_arr[:,:-1], test_arr[:,-1]

            # extract model object stored in artifact folder
            model_path = os.path.join("artifact", "model.pkl")
            model = load_object(model_path)

            mlflow.set_registry_uri("https://dagshub.com/kpkolhe1998/12_2_DS_end_to_end_git_proj.mlflow")

            tracking_url = mlflow.get_tracking_uri()
            track_url_type_store = urlparse(tracking_url).scheme

            print(track_url_type_store)

            with mlflow.start_run():
                pred_val = model.predict(x_test)

                mse, rmse, mae, r2 = self.eval_metrics(y_test, pred_val)

                mlflow.log_metric("mean_squared_error", mse)
                mlflow.log_metric("root_mean_squared_error", rmse)
                mlflow.log_metric("mean_absolute_error", mae)
                mlflow.log_metric("r2_score", r2)

                # Model registory does not work with file storage
                if track_url_type_store != "file":
                    # Register the model
                    # There are other ways for model registory, depnds upon the use case,
                    # please reger to the doc for more details:
                    # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                    mlflow.sklearn.log_model(model, "model", registered_model_name = "Ml_model")
                else:
                    mlflow.sklearn.log_model(model, "model")



        except Exception as e:
            logging.info("Error while model evaluation")
            raise customException(e, sys)

