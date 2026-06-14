import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMRegressor
from wrappers.logger import MlflowLogger
from config import SimpleRegressor

METRICS_CL = [log_loss, accuracy_score]
METRICS_REG = [mean_absolute_error, mean_squared_error]

### **Fixtures**
@pytest.fixture
def iris_data():
    data, target = load_iris(return_X_y=True)
    return train_test_split(data, target, test_size=0.25, random_state=42)

@pytest.fixture
def diabetes_data():
    data, target = load_diabetes(return_X_y=True)
    return train_test_split(data, target, test_size=0.25, random_state=42)

@pytest.fixture
def scaled_diabetes_data(diabetes_data):
    X_tr, X_tst, y_tr, y_tst = diabetes_data
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_tst = scaler.transform(X_tst)
    return X_tr, X_tst, y_tr, y_tst

@pytest.fixture
def torch_diabetes_data(scaled_diabetes_data):
    X_tr, X_tst, y_tr, y_tst = scaled_diabetes_data
    return (
        torch.FloatTensor(X_tr),
        torch.FloatTensor(X_tst),
        torch.FloatTensor(y_tr).reshape(-1, 1),
        torch.FloatTensor(y_tst).reshape(-1, 1))

@pytest.fixture
def torch_model(torch_diabetes_data):
    X_tr, _, _, _ = torch_diabetes_data
    input_dim = X_tr.shape[1]
    return SimpleRegressor(input_dim)


### **1. Default Wrapper Tests**
class TestDefaultWrapper:
    def test_logistic_regression(self, iris_data):
        """Test MlflowLogger with LogisticRegression (classification)."""
        X_tr, X_tst, y_tr, y_tst = iris_data
        log_reg = LogisticRegression(max_iter=200)
        logger = MlflowLogger("logreg_default", "mlflow/test_logreg_default", verbose_mode=False)
        logger.prepare_default_wrapper(
            log_reg, X_tr, y_tr, X_tst, y_tst, metrics=METRICS_CL, process=True
        )


    def test_lgb_regressor(self, diabetes_data):
        """Test MlflowLogger with LGBMRegressor (regression)."""
        X_tr, X_tst, y_tr, y_tst = diabetes_data
        lgb_reg = LGBMRegressor(n_estimators=100, learning_rate=0.05, num_leaves=31, random_state=42)
        logger = MlflowLogger("lgb_default", "mlflow/test_lgb_default", verbose_mode=False)
        logger.prepare_default_wrapper(
            lgb_reg, X_tr, y_tr, X_tst, y_tst, metrics=METRICS_REG, process=True
        )

### **2. Custom Wrapper Tests**
class TestCustomWrapper:
    def test_torch_custom_model(self, torch_diabetes_data, torch_model):
        """Test MlflowLogger with a custom PyTorch model."""
        X_tr, X_tst, y_tr, y_tst = torch_diabetes_data
        logger = MlflowLogger("torch_custom", "mlflow/custom", verbose_mode=False)

        logger.model = torch_model
        logger.X_train = X_tr
        logger.y_train = y_tr
        logger.X_test = X_tst
        logger.y_test = y_tst
        logger.metrics_list = [nn.MSELoss()]

        
        def custom_function():
            criterion = nn.MSELoss()
            optimizer = optim.SGD(torch_model.parameters(), lr=0.01)
            torch_model.train()
            for _ in range(10):
                optimizer.zero_grad()
                predictions = torch_model(X_tr)
                loss = criterion(predictions, y_tr)
                loss.backward()
                optimizer.step()

            torch_model.eval()
            with torch.no_grad():
                test_preds = torch_model(X_tst)
                test_loss = criterion(test_preds, y_tst)

            params = {}
            for name, param in torch_model.named_parameters():
                params[name] = param.detach().numpy().tolist()

            return {"mse": test_loss.item()}, {"input_dim": X_tr.shape[1], "n_parameters": len(list(torch_model.parameters()))}

        logger.wrapper = custom_function
        logger.log()

### **3. Error Handling Tests**
class TestErrorHandling:
    def test_uninitialized_wrapper(self):
        """Test that an error is raised if the wrapper is not initialized."""
        logger = MlflowLogger("error_test", "mlflow/error", verbose_mode=False)
        with pytest.raises(ValueError, match="Wrapper function is not initialized"):
            logger.log()

    def test_verbose_logs_without_data(self, capsys):
        """Test that verbose logs do not crash without data."""
        logger = MlflowLogger("verbose_test", "mlflow/verbose", verbose_mode=False)
        logger.print_verbose_logs()
        captured = capsys.readouterr()
        assert captured.out.strip() == "[MlflowLogger] No actual data to view."