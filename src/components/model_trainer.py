import os
import sys
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.utils import save_object
from src.utils import evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.best_model = None

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
                )
            models = {
                "Random Forest": RandomForestRegressor(),
                "KNN Nearest": KNeighborsRegressor(),
                "Linear Regression": LinearRegression(),
                "Adaboost Regressor": AdaBoostRegressor(),
                "Catboost Regressor": CatBoostRegressor(),
                "XGBoost Regressor": XGBRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boost ": GradientBoostingRegressor(),

                }
            model_report : dict = evaluate_models(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, models = models)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            self.best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No Best Model Found")
            logging.info("Best found model on both training and testing dataset")
            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = self.best_model
            )

            predicted = self.best_model.predict(X_test)

            r2_sqaure = r2_score(y_test, predicted)
            return r2_sqaure

        except Exception as e:
            raise CustomException(e,sys)




