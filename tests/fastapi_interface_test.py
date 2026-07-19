import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from sklearn.base import BaseEstimator, is_classifier, is_regressor
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from typing import Callable, Any
import numpy as np
import pandas as pd
import time
import io
import uvicorn
import pytest

from wrappers.interface import ModelAPI

DATA_IRIS = load_iris()
def iris_data_preprocessor(data):
    data, target = DATA_IRIS.data, DATA_IRIS.target
    return train_test_split(data, target, test_size=0.25, random_state=42)

class TestFastAPIinterface:

    def test_no_api(self, iris_data):
        log_reg = LogisticRegression(max_iter=200)
        api = ModelAPI(log_reg, confusion_matrix, iris_data)
        api._preprocess()
        api._fit()
        api._validate()

    def test_no_api_preprocessor(self):
        log_reg = LogisticRegression(max_iter=200)
        api = ModelAPI(log_reg, accuracy_score, iris_data_preprocessor)
        api._fit(DATA_IRIS)
        api._validate()


if __name__ == "__main__":

    #its ok
    ridge = RidgeClassifier()
    api = ModelAPI(ridge, accuracy_score, iris_data_preprocessor)
    api.run(title="testing api")