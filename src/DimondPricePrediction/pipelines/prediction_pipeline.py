from src.DimondPricePrediction.exception import customException
import sys
import os
from src.DimondPricePrediction.utils.utils import load_object
from src.DimondPricePrediction.logger import logging
import pandas as pd

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, feature):
        try:
            preporcess_path = os.path.join("artifact", "preprocess.pkl")
            model_path = os.path.join("artifact", "model.pkl")

            preprocess = load_object(preporcess_path)
            model = load_object(model_path)

            scaled_data = preprocess.transform(feature)

            pred = model.predict(scaled_data)

            return pred
        except Exception as e:
            logging.info("Exception in predict() method")
            raise customException(e, sys)
        
class CustomData:
    def __init__(self,
                 carat,
                 cut,
                 color,
                 clarity,
                 depth,
                 table,
                 x,
                 y,
                 z):
        
        self.carat = carat
        self.cut = cut
        self.color = color
        self.clarity = clarity
        self.depth = depth
        self.table = table
        self.x = x
        self.y = y
        self.z = z

        logging.info("Colleced Data for creating DF")

    def get_data_as_dataframe(self):
        try:
            custom_data_dict = {
                'carat' : [self.carat],
                'cut' : [self.cut],
                'color' : [self.color],
                'clarity' : [self.clarity],
                'depth' : [self.depth],
                'table' : [self.table],
                'x' : [self.x],
                'y' : [self.y],
                'z' : [self.z]
            }

            df = pd.DataFrame(custom_data_dict)
            logging.info("DataFrame Gathered")

            return df
        
        except Exception as e:
            logging.info("Exception occured while creating DataFrame")
            customException(e, sys)

