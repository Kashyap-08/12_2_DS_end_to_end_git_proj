from src.DimondPricePrediction.logger import logging
from src.DimondPricePrediction.exception import customException
import sys
import os
import pickle
from sklearn.metrics import r2_score

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        logging.info(f"Making Directory named: {dir_path}")
        os.makedirs(dir_path, exist_ok=True)

        logging.info("Directory Created")

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

        logging.info(f"Dumped Object into {file_path}")

    except Exception as e:
        logging.info("Exception in save_object() method")
        raise customException(e, sys)
    

def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report = {}

        for i in range(len(models)):
            model = list(models.values())[i]

            #Train model
            model.fit(X_train, y_train)

            #Predict Test data
            y_test_predict = model.predict(X_test)

            # Get r2_score form train and test dataset
            test_model_score = r2_score(y_test, y_test_predict)

            report[list(models.keys())[i]] = test_model_score

        return report
    
    except Exception as e:
        logging.info('Exception Occured Due to model Evaluation')
        raise customException(e, sys)