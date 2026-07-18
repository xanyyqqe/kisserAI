from typing import Literal
import mlflow
import torch

class TorchInference:

    def __init__(self, log:bool=False, experiment_name:str=None, path:str=None):
        
        self.log = log

        if log and not experiment_name:
            raise ValueError("[INFERENCE VERBOSE] Logging requires an experiment name")
        
        self.experiment_name = experiment_name
        if self.experiment_name:
            mlflow.set_experiment(experiment_name)

        self.loss = None
        self.loss_prev = None
        self.metric = None
        self.metric_prev = None
        self.epoch = 1


    def process(self, loss, metric=None, comment:str=None, model=None):

        if self.loss is not None: 
            self.loss_prev = self.loss
        if self.metric is not None:
            self.metric_prev = self.metric

        self.loss = loss
        self.metric = metric
        self.comment = comment

        print(f"EPOCH: {self.epoch}\n")
        print(f"Loss: {self.loss}\t")

        if self.loss_prev:
            diff = self.loss_prev - self.loss
            diff_perc = diff/self.loss_prev*100

            if diff >= 0:
                print(f"Loss diff: -{diff}; -{diff_perc}%\n")
            else:
                print(f"Loss diff: +{diff}; +{diff_perc}%\n")
        
        if metric:
            print(f"Metric: {metric}\t")
            if self.metric_prev:
                diff = metric - self.metric_prev 
                diff_perc = diff/self.metric_prev*100

                if diff >= 0:
                    print(f"Metric diff: +{diff}; +{diff_perc}%\n")
                else:
                    print(f"Metric diff: {diff}; {diff_perc}%\n")

        if self.log:
            self._log(comment, model)

        self.epoch += 1


    def _log(self, model, comment : str = None): 
        
        with mlflow.start_run() as run:

            mlflow.log_metric("loss", self.loss)
            if self.metric:
                mlflow.log_metric("metric", self.metric)
            
            if comment:
                mlflow.set_tag("comment", comment)

            if model is not None and isinstance(model, torch.nn.Module): 
                mlflow.pytorch.log_model(model, "model") 