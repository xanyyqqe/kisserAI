import sys
from pathlib import Path

from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wrappers.model_logger import MlflowLogger

METRICS_CL = [log_loss, accuracy_score]
METRICS_REG = [mean_absolute_error, mean_squared_error]


def test_mlflow_loggerDEFAULT():
    from sklearn.linear_model import LogisticRegression
    from lightgbm import LGBMRegressor

    # Test 1: Classification with Logistic Regression
    data, target = load_iris(return_X_y=True)
    X_tr, X_tst, y_tr, y_tst = train_test_split(data, target, test_size=0.25, random_state=42)

    log_reg = LogisticRegression(max_iter=200)
    log_reg_wrapper = MlflowLogger("logreg_default", "mlflow/test_logreg_default", verbose_mode=False)
    log_reg_wrapper.prepare_default_wrapper(
        log_reg, X_tr, y_tr, X_tst, y_tst, metrics=METRICS_CL, process=True
    )
    
    # Test 2: Regression with LightGBM
    lgb_reg = LGBMRegressor(n_estimators=100, learning_rate=0.05, 
                            num_leaves=31, random_state=42)
    lgb_wrapper = MlflowLogger("lgb_default", "mlflow/test_lgb_default", verbose_mode=False)
    
    data, target = load_diabetes(return_X_y=True)
    X_tr, X_tst, y_tr, y_tst = train_test_split(data, target, test_size=0.25, random_state=42)
    
    lgb_wrapper.prepare_default_wrapper(
        lgb_reg, X_tr, y_tr, X_tst, y_tst, metrics=METRICS_REG, process=True
    )


def test_mlflow_loggerCUSTOM():
    data, target = load_diabetes(return_X_y=True)
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    
    X_tr, X_tst, y_tr, y_tst = train_test_split(data, target, test_size=0.25, random_state=42)
    
    # Convert to torch tensors
    X_tr_tensor = torch.FloatTensor(X_tr)
    X_tst_tensor = torch.FloatTensor(X_tst)
    y_tr_tensor = torch.FloatTensor(y_tr).reshape(-1, 1)
    y_tst_tensor = torch.FloatTensor(y_tst).reshape(-1, 1)
    
    custom_wrapper = MlflowLogger("torch_custom", "mlflow/custom", verbose_mode=False)

    class SimpleRegressor(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.linear = nn.Linear(input_dim, 1)
            
        def forward(self, x):
            return self.linear(x)
    
    input_dim = X_tr.shape[1]
    torch_model = SimpleRegressor(input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(torch_model.parameters(), lr=0.01)

    custom_wrapper.model = torch_model
    custom_wrapper.X_train = X_tr_tensor
    custom_wrapper.y_train = y_tr_tensor
    custom_wrapper.X_test = X_tst_tensor
    custom_wrapper.y_test = y_tst_tensor
    custom_wrapper.metrics_list = [criterion]

    def custom_function():
        torch_model.train()
        for epoch in range(10):
            optimizer.zero_grad()
            predictions = torch_model(X_tr_tensor)
            loss = criterion(predictions, y_tr_tensor)
            loss.backward()
            optimizer.step()
        
        torch_model.eval()
        with torch.no_grad():
            test_preds = torch_model(X_tst_tensor)
            test_loss = criterion(test_preds, y_tst_tensor)
        
        params = {}
        for name, param in torch_model.named_parameters():
            params[name] = param.detach().numpy().tolist()
        
        return {"mse": test_loss.item()}, {"input_dim": input_dim, "n_parameters": len(list(torch_model.parameters()))}
    
    custom_wrapper.wrapper = custom_function
    custom_wrapper.log()


def test_error_handling():
    logger = MlflowLogger("error_test", "mlflow/error", verbose_mode=False)
    try:
        logger.log()
    except ValueError as e:
        assert str(e) == "Wrapper function is not initialized"
    
    # Test 2: Print verbose logs without data should not crash
    logger.print_verbose_logs()


if __name__ == "__main__":
    test_mlflow_loggerDEFAULT()
    test_mlflow_loggerCUSTOM()
    test_error_handling()