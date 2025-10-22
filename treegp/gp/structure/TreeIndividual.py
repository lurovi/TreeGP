from genepro.node import Node
import numpy as np


class TreeIndividual:
    def __init__(self,
                 tree: Node
                 ) -> None:
        super().__init__()
        self.__tree: Node = tree

        self.__train_pred: np.ndarray | None = None
        self.__test_pred: np.ndarray | None = None

        self.__train_rmse: float | None = None
        self.__test_rmse: float | None = None

    def tree(self) -> Node:
        return self.__tree

    def train_pred(self) -> np.ndarray | None:
        return self.__train_pred

    def test_pred(self) -> np.ndarray | None:
        return self.__test_pred

    def train_rmse(self) -> float | None:
        return self.__train_rmse

    def test_rmse(self) -> float | None:
        return self.__test_rmse
    
    def has_cache(self) -> bool:
        return self.train_rmse() is not None and self.test_rmse() is not None and self.train_pred() is not None and self.test_pred() is not None
    
    def set_cache(self,
                 train_pred: np.ndarray,
                 test_pred: np.ndarray,
                 train_rmse: float,
                 test_rmse: float
                 ) -> None:
        self.__train_pred = train_pred
        self.__test_pred = test_pred
        self.__train_rmse = train_rmse
        self.__test_rmse = test_rmse

