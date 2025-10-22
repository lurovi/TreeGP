from treegp.gp.structure.TreeBoolean import TreeBoolean
from treegp.gp.operator.evaluation.Evaluation import Evaluation
from typing import TypeVar
import time
import numpy as np
from genepro.node import Node

from treegp.util.WalshTransform import WalshTransform

T = TypeVar('T', bound=TreeBoolean)


class NonLinearity(Evaluation):
    def __init__(self,
                 walsh_transform: WalshTransform,
                 penalize_unbalanced_individuals: bool
                 ) -> None:
        super().__init__()
        self.__penalize_unbalanced_individuals: bool = penalize_unbalanced_individuals
        self.__walsh_transform: WalshTransform = walsh_transform
        self.__domain = walsh_transform.domain()
        self.__data = self.__domain.data()

    def evaluate(self, item: T, **kwargs) -> tuple[float | None, bool, float]:
        tree: Node = item.tree()
        evaluated_now: bool = False

        exec_time: float = 0.0

        if not item.has_cache():
            evaluated_now = True
            
            start_time: float = time.time()
            
            truth_table = tree(self.__data)
            # Compute unbalancedness degree as the absolute difference between the number of ones and zeros in the truth table
            num_ones = (truth_table == 1).sum()
            unbalancedness_degree = abs(num_ones - (truth_table.size - num_ones))

            spectrum, _ = self.__walsh_transform.apply(truth_table)
            non_linearity: float = self.__walsh_transform.granular_non_linearity(spectrum)
            
            end_time: float = time.time()
            
            exec_time += end_time - start_time

            item.set_cache(truth_table=truth_table, spectrum=spectrum, non_linearity=non_linearity)

        return item.non_linearity(), evaluated_now, exec_time

    def get_fitness(self, item: T) -> float | None:
        if not self.__penalize_unbalanced_individuals:
            return item.non_linearity()
        else:
            # Penalize unbalanced individuals by subtracting the unbalancedness degree from the non-linearity
            truth_table: np.ndarray = item.truth_table() # type: ignore
            num_ones = (truth_table == 1).sum()
            unbalancedness_degree = abs(num_ones - (truth_table.size - num_ones))
            return item.non_linearity() - 2 * unbalancedness_degree
