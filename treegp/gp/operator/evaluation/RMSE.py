import numpy as np
from treegp.gp.operator.generator.TreeGenerator import TreeGenerator
from treegp.util.EvaluationMetrics import EvaluationMetrics
from treegp.gp.structure.TreeIndividual import TreeIndividual
from treegp.mlearn.NumpySupervisedDataset import NumpySupervisedDataset
from treegp.gp.operator.evaluation.Evaluation import Evaluation
from typing import TypeVar
import time
from genepro.node import Node

T = TypeVar('T', bound=TreeIndividual)


class RMSE(Evaluation):
    def __init__(self,
                 train_data: NumpySupervisedDataset,
                 test_data: NumpySupervisedDataset,
                 linear_scaling: bool = False
                 ) -> None:
        super().__init__()
        self.__train_data: NumpySupervisedDataset = train_data
        self.__test_data: NumpySupervisedDataset = test_data
        self.__linear_scaling: bool = linear_scaling

    def evaluate(self, item: T, **kwargs) -> tuple[float | None, bool, float]:
        tree: Node = item.tree()
        evaluated_now: bool = False

        train_time: float = 0.0

        if not item.has_cache():
            evaluated_now = True
            start_time_train: float = time.time()
            train_pred: np.ndarray = np.core.umath.clip(tree(self.__train_data.X(), dataset_type='train'), -1e+20, 1e+20)
            end_time_train: float = time.time()
            train_time += end_time_train - start_time_train
            test_pred: np.ndarray = np.core.umath.clip(tree(self.__test_data.X(), dataset_type='test'), -1e+20, 1e+20)
            
            if self.__linear_scaling:
                start_time_train = time.time()
                slope, intercept = EvaluationMetrics.compute_linear_scaling(y=self.__train_data.y(), p=train_pred)
                train_rmse: float = EvaluationMetrics.root_mean_squared_error(y=self.__train_data.y(), p=train_pred, linear_scaling=False, slope=slope, intercept=intercept)
                end_time_train = time.time()
                train_time += end_time_train - start_time_train
                test_rmse: float = EvaluationMetrics.root_mean_squared_error(y=self.__test_data.y(), p=test_pred, linear_scaling=False, slope=slope, intercept=intercept)
            else:
                start_time_train = time.time()
                train_rmse: float = EvaluationMetrics.root_mean_squared_error(y=self.__train_data.y(), p=train_pred, linear_scaling=False, slope=None, intercept=None)
                end_time_train = time.time()
                train_time += end_time_train - start_time_train
                test_rmse: float = EvaluationMetrics.root_mean_squared_error(y=self.__test_data.y(), p=test_pred, linear_scaling=False, slope=None, intercept=None)

            item.set_cache(train_pred=train_pred, test_pred=test_pred, train_rmse=train_rmse, test_rmse=test_rmse)
        
        return item.train_rmse(), evaluated_now, train_time

