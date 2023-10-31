from src.DimondPricePrediction.logger import logging
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from src.DimondPricePrediction.exception import customException
import sys
import pandas as pd
import numpy as np
from src.DimondPricePrediction.utils.utils import save_object
import os
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocess_obj_file_path = os.path.join('artifact', 'preprocess.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation(self):
        '''
        Method create and initialize pipeline for Numerical and categorical values.
        param: 
        return: preprocess object 
        '''

        logging.info("Initiating Data Transformation")

        try:
            # Columns that should be ordinal encoded
            categorical_col = ['cut', 'color', 'clarity']

            # columns that should be scaled
            numerical_col = ['carat', 'depth', 'table', 'x', 'y', 'z']

            # Define custom ranking for each ordinal variable
            cut_category = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
            clarity_category = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]
            color_category = ["D", "E", "F", "G", "H", "I", "J"]

            logging.info("Initiate Pipeline")

            logging.info("creating Numerical Pipeline")
            # Numerical Pipeline
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),  # handling Missing values
                    ('scaler', StandardScaler())  # Handling Feature Scaling
                ]
            )

            logging.info("Creating Categoricla Pipeline")
            # Categorical Pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),  # Handling Missing values
                    ('ordinal_encoding', OrdinalEncoder(categories=[cut_category, color_category, clarity_category])),  # Encode ordinal data
                    ('Scaler', StandardScaler()) # Handle Feature Scaling
                ]
            )


            logging.info("Assembling the Numerical and Categorical Pipeline into preprocess variable")
            preprocess = ColumnTransformer([
                ('Numericla_Pipeline', num_pipeline, numerical_col),
                ('Categorical_Pipeline', cat_pipeline, categorical_col)
            ])


            logging.info("Return Preprocess Object")
            return preprocess

        except Exception as e:
            logging.info("Exception in get_data_transformation() method")

            raise customException(e, sys)
        

        
    def initialize_data_transformation(self, train_path, test_path):
        try:
            logging.info(f'Reading train and test data stored in train_path: {train_path}, test_path: {test_path}')
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info(f"Train DatFrame head :\n{train_df.head().to_string()}")
            logging.info(f"Test DataFrame head :\n{test_df.head().to_string()}")


            logging.info('Get Object of preporcess pipeline')
            preprocess_obj = self.get_data_transformation()

            target_col_name = 'price'
            drop_col = [target_col_name, 'id']

            logging.info("Seprate Feature and Target columns for Train and Test DF")
            feature_train_data = train_df.drop(drop_col, axis=1)
            target_train_data = train_df[target_col_name]

            feature_test_data = test_df.drop(drop_col, axis=1)
            target_test_data = test_df[target_col_name]

            logging.info("Fitting and transforming data in preprocess object")
            feature_train_arr = preprocess_obj.fit_transform(feature_train_data)
            feature_test_arr = preprocess_obj.transform(feature_test_data)

            logging.info(f"Save preprocess object to path: {self.data_transformation_config.preprocess_obj_file_path}")
            save_object(
                file_path= self.data_transformation_config.preprocess_obj_file_path,
                obj = preprocess_obj
            )   

            # Use numpy.c_ to concatenate them horizontally
            logging.info("Concate the transform data and target data")
            target_arr = np.c_[feature_train_arr, np.array(target_train_data)]
            test_arr = np.c_[feature_test_arr, np.array(target_test_data)]

            logging.info("Retunr created train test arrays")
            return (
                target_arr,
                test_arr
            )

        except Exception as e:
            logging.info("Exception in initialize_data_taransformation() method")

            raise customException(e, sys)