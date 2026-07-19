import pytest
from sklearn.datasets import load_diabetes, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch

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
    class SimpleRegressor(nn.Module):
    
        def __init__(self, input_dim):
            super().__init__()
            self.linear = nn.Linear(input_dim, 1)
            
        def forward(self, x):
            return self.linear(x)
        
    X_tr, _, _, _ = torch_diabetes_data
    input_dim = X_tr.shape[1]
    return SimpleRegressor(input_dim)

### **Fixtures**
@pytest.fixture
def iris_data():
    data, target = load_iris(return_X_y=True)
    return train_test_split(data, target, test_size=0.25, random_state=42)
