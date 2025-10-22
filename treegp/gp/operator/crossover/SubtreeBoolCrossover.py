from typing import TypeVar
from treegp.gp.operator.generator.TreeGenerator import TreeGenerator
from treegp.gp.structure.TreeBoolean import TreeBoolean
from treegp.gp.structure.TreeStructure import TreeStructure

from treegp.util.Utils import Utils
from treegp.gp.operator.crossover.Crossover import Crossover

T = TypeVar('T', bound=TreeBoolean)


class SubtreeBoolCrossover(Crossover):
    def __init__(self,
                 structure: TreeStructure,
                 crossover_probability: float,
                 execute_crossover: bool
                 ) -> None:
        super().__init__()
        self.__crossover_probability: float = crossover_probability
        self.__execute_crossover: bool = execute_crossover

        self.__structure: TreeStructure = structure

    def mate(self, parents: tuple[T, ...], **kwargs) -> tuple[tuple[T, ...], bool]:
        has_cx_been_exec: bool = False
        first_parent: T = parents[0]
        second_parent: T = parents[1]

        if self.__execute_crossover and Utils.random_uniform() < self.__crossover_probability:
            has_cx_been_exec = True
            new_tree_1, new_tree_2 = self.__structure.safe_subtree_crossover_two_children(first_parent.tree(), second_parent.tree())
            new_tree_1: T = TreeBoolean(new_tree_1)
            new_tree_2: T = TreeBoolean(new_tree_2)
        else:
            new_tree_1: T = first_parent
            new_tree_2: T = second_parent

        return ((new_tree_1, new_tree_2), has_cx_been_exec)
