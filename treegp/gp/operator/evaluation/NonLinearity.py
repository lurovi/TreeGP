from treegp.gp.structure.TreeBoolean import TreeBoolean
from treegp.gp.operator.evaluation.Evaluation import Evaluation
from typing import TypeVar
import time
from genepro.node import Node

from treegp.util.WalshTransform import WalshTransform

T = TypeVar('T', bound=TreeBoolean)


class NonLinearity(Evaluation):
    def __init__(self,
                 walsh_transform: WalshTransform,
                 ) -> None:
        super().__init__()
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
            spectrum, _ = self.__walsh_transform.apply(truth_table)
            non_linearity: float = self.__walsh_transform.granular_non_linearity(spectrum)
            
            end_time: float = time.time()
            
            exec_time += end_time - start_time

            item.set_cache(truth_table=truth_table, spectrum=spectrum, non_linearity=non_linearity)

        return item.non_linearity(), evaluated_now, exec_time
