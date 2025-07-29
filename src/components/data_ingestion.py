# What is Data Ingestion?
#Data ingestion is the first step in any machine learning or data pipeline. It refers to collecting and importing data from various sources so that it can be used for processing, analysis, and model building.
#Key Goals of Data Ingestion:
#Read raw data (e.g., from CSV, database, API, or other formats)
#Validate or clean basic structure (e.g., remove nulls, fix formats)
#Split the data into:
#Training data
#Testing data
#Save the datasets to specific locations for further use

import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from src.components.data_transformation import data_transformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts',"train.csv")
    test_data_path:str = os.path.join('artifacts',"test.csv")
    raw_data_path:str = os.path.join('artifacts',"raw.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv('notebook\data\stud.csv')
            logging.info('Read the dataset as DataFrame')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index = False, header = True)

            logging.info("Train test split initiated")

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
            test_set.to_csv(self.ingestion_config.test_data_path, index = False, header = True)

            logging.info("Ingestion of the Data is Completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
        
            )
        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transform = data_transformation()
    train_arr, test_arr,_ = data_transform.initiate_data_transformation(train_data, test_data)

    model_train = ModelTrainer()
    print(model_train.initiate_model_trainer(train_arr, test_arr))
