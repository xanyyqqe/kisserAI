from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from sklearn.base import BaseEstimator, is_classifier, is_regressor
from typing import Callable, Any, Union, Optional
import numpy as np
import pandas as pd
import time
import io
import uvicorn

class ModelAPI:
    def __init__(self, model: BaseEstimator, metric: Callable, data_preprocessor: Union[Callable, tuple, list]):
        self.model = model
        self.metric = metric
        self.data_or_preprocessor = data_preprocessor
        self.cur_metric_score = None
        self.best_metric_score = None
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.app = None
        self.attempts = 0

        if is_classifier(self.model):
            self._model_type = "class"
        elif is_regressor(self.model):
            self._model_type = "reg"
        else:
            raise ValueError("[ModelAPI] : Tool not adapted for this model type")

    def _preprocess(self, data: Optional[Any] = None):

        if isinstance(self.data_or_preprocessor, (tuple, list)):
            self.X_train, self.X_test, self.y_train, self.y_test = self.data_or_preprocessor
            return
        
        result = self.data_or_preprocessor(data)
        if not isinstance(result, (list, tuple)) or len(result) != 4:
            raise ValueError("[ModelAPI] : Preprocessor must return [X_train, X_test, y_train, y_test]")
        self.X_train, self.X_test, self.y_train, self.y_test = result

    def _fit(self, data: Optional[Any] = None):
        self._preprocess(data)
        self.model.fit(self.X_train, self.y_train)
        self.attempts += 1

    def _validate(self):
        if self.X_test is None:
            raise ValueError("No test data available. Run _fit first.")
        
        metric_name = getattr(self.metric, '__name__', '')
        if 'log_loss' in metric_name and hasattr(self.model, 'predict_proba'):
            predictions = self.model.predict_proba(self.X_test)
        else:
            predictions = self.model.predict(self.X_test)
        
        self.cur_metric_score = self.metric(self.y_test, predictions)

        if self.best_metric_score is None:
            self.best_metric_score = self.cur_metric_score
        elif self._model_type == "class":
            self.best_metric_score = max(self.cur_metric_score, self.best_metric_score)
        else:
            self.best_metric_score = min(self.cur_metric_score, self.best_metric_score)

    def run(self, title: str = None, host="127.0.0.1", port=8000):
        self.app = FastAPI(title=title or self.model.__class__.__name__)

        @self.app.post("/fit")
        async def fit_api(file: UploadFile = File(...)):
            contents = await file.read()
            raw_data = pd.read_csv(io.BytesIO(contents))
            
            beg_time = time.perf_counter()
            self._fit(raw_data)
            return {"model status": "trained", "time": f"{time.perf_counter() - beg_time:.4f}s"}
        
        @self.app.get("/predict")
        async def validate_api():
            if self.attempts == 0:
                raise HTTPException(status_code=400, detail="Train model first")
            self._validate()
            return {"last": self.cur_metric_score, "best": self.best_metric_score}

        uvicorn.run(self.app, host=host, port=port)