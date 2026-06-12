import mlflow
import mlflow.sklearn
from typing import Callable, Any, List, Dict, Tuple


class MlflowLogger:

    def __init__(self, experiment_name: str, log_path: str = None, verbose_mode: bool = True):
        self.experiment_name = experiment_name
        self.verbose_mode = verbose_mode
        
        if log_path:
            mlflow.set_tracking_uri(log_path)
        mlflow.set_experiment(self.experiment_name)
        
        self.wrapper = None
        self.model = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.metrics_list = None
        
        self.saved_metrics = None
        self.saved_params = None


    def prepare_default_wrapper(self, model: Any, X_tr: Any, y_tr: Any, X_tst: Any, y_tst: Any,
                                metrics: List[Callable], process: bool = False) -> None:
        self.model = model
        self.X_train = X_tr
        self.y_train = y_tr
        self.X_test = X_tst
        self.y_test = y_tst
        self.metrics_list = metrics


        def wrapper_sklearn() -> Tuple[Dict[str, float], Dict[str, Any]]:

            self.model.fit(self.X_train, self.y_train)

            calculated_metrics = dict()
            for metric in self.metrics_list:
                if getattr(metric, '__name__', None) == 'log_loss' and hasattr(self.model, 'predict_proba'):
                    y_pred = self.model.predict_proba(self.X_test)
                else:
                    y_pred = self.model.predict(self.X_test)

                score = metric(self.y_test, y_pred)
                calculated_metrics[metric.__name__] = float(score)

            params = self.model.get_params()
            return calculated_metrics, params
        
        self.wrapper = wrapper_sklearn

        if process:
            if self.verbose_mode:
                self.log_verbose()
            else:
                self.log()


    def log(self, wrapper: Callable = None) -> None:

        if wrapper:
            self.wrapper = wrapper
        if not self.wrapper:
            raise ValueError("Wrapper function is not initialized")

        with mlflow.start_run() as run:

            metric_results, params = self.wrapper()
            mlflow.log_metrics(metric_results)
            mlflow.log_params(params)
            
            mlflow.sklearn.log_model(sk_model=self.model, artifact_path="model")

            print(f"[MlflowLogger] All data is logged. Run ID: {run.info.run_id}")
            self.saved_metrics = metric_results
            self.saved_params = params


    def print_verbose_logs(self) -> None:

        if self.saved_metrics and self.saved_params:
            print("--- Model metrics: ---\n")
            for key, value in self.saved_metrics.items():
                print(f"{key}: {value}")
            
            print("\n--- Model hyperparams ---\n")
            for key, value in self.saved_params.items():
                print(f"{key}: {value}")
        else:
            print("[MlflowLogger] No actual data to view.")


    def log_verbose(self) -> None:
        self.log()
        self.print_verbose_logs()