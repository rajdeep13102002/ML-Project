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
from src.utils import save_object, evaluate_models  # Import evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.best_model_obj = None

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

            params = {
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Linear Regression": {}, # Often no hyperparameters for Linear Regression in a simple case
                "KNN Nearest": {
                    'n_neighbors': [5, 7, 9, 11]
                },
                "Adaboost Regressor": {
                    'n_estimators': [50, 100, 200],  # Example hyperparameters
                    'learning_rate': [0.01, 0.05, 0.1]
                },
                "Catboost Regressor": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "XGBoost Regressor": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [3, 5, 7]
                },
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'splitter': ['best', 'random'],
                    'max_features': [10,20,30,50,75,100]
                },
                "Gradient Boost ": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [3, 5, 7]
                }
            }


            model_report : dict = evaluate_models(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, models = models, params = params)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            self.best_model_obj = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found with R2 score greater than 0.6")

            logging.info(f"Best model found: {best_model_name} with R2 score: {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=self.best_model_obj
            )

            return best_model_score

        except Exception as e:
            raise CustomException(e, sys)
