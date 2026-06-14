import sys
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim

from wrappers.inference import TorchInference

class TestTorchInference:

    def test_no_logs(self, torch_diabetes_data, torch_model, capsys):

        X_tr, X_tst, y_tr, y_tst = torch_diabetes_data
        
        wrapper = TorchInference()

        criterion = nn.MSELoss()
        optimizer = optim.SGD(torch_model.parameters(), lr=0.01)
        torch_model.train()

        for epoch in range(10):
            optimizer.zero_grad()
            predictions = torch_model(X_tr)
            loss = criterion(predictions, y_tr)
            loss.backward()
            optimizer.step()
            metric = loss / 100  # just for example

            torch_model.eval()
            with torch.no_grad():
                test_preds = torch_model(X_tst)
                test_loss = criterion(test_preds, y_tst)

            wrapper.process(test_loss, metric)
            captured = capsys.readouterr()
            
            if epoch == 0:

                assert re.search(r"EPOCH: 1", captured.out)
                assert re.search(rf"Loss: {wrapper.loss}", captured.out)
                assert re.search(rf"Metric: {wrapper.metric}", captured.out)

                assert not re.search(r"Loss diff:", captured.out)
                assert not re.search(r"Metric diff:", captured.out)

            else:

                assert re.search(rf"EPOCH: {epoch + 1}", captured.out)
                assert re.search(rf"Loss: {wrapper.loss}", captured.out)
                assert re.search(rf"Metric: {wrapper.metric}", captured.out)

                assert re.search(r"Loss diff: [+-]\d+\.?\d*; [+-]\d+\.?\d*%", captured.out)

                assert re.search(r"Metric diff: [+-]\d+\.?\d*; [+-]\d+\.?\d*%", captured.out)


    def test_only_logs(self, torch_diabetes_data, torch_model):
        X_tr, X_tst, y_tr, y_tst = torch_diabetes_data
        
        wrapper = TorchInference(log=True, experiment_name="test torch wrapper")

        criterion = nn.MSELoss()
        optimizer = optim.SGD(torch_model.parameters(), lr=0.01)
        torch_model.train()

        for epoch in range(10):
            optimizer.zero_grad()
            predictions = torch_model(X_tr)
            loss = criterion(predictions, y_tr)
            loss.backward()
            optimizer.step()
            metric = loss / 100

            torch_model.eval()
            with torch.no_grad():
                test_preds = torch_model(X_tst)
                test_loss = criterion(test_preds, y_tst)

            wrapper.process(test_loss, metric, comment="random words xd", model=torch_model)