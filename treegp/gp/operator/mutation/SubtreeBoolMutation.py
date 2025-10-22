from typing import TypeVar
from treegp.gp.operator.generator.TreeGenerator import TreeGenerator
from treegp.gp.structure.TreeBoolean import TreeBoolean
from treegp.gp.structure.TreeStructure import TreeStructure
from treegp.util.Utils import Utils
from treegp.gp.operator.mutation.Mutation import Mutation

T = TypeVar('T', bound=TreeBoolean)


class SubtreeBoolMutation(Mutation):
    def __init__(self,
                 structure: TreeStructure,
                 mutation_probability: float,
                 execute_mutation: bool
                 ) -> None:
        super().__init__()
        self.__mutation_probability: float = mutation_probability
        self.__execute_mutation: bool = execute_mutation

        self.__structure: TreeStructure = structure

    def mutate(self, offspring: T, **kwargs) -> tuple[T, bool]:
        has_mut_been_exec: bool = False
        if self.__execute_mutation and Utils.random_uniform() < self.__mutation_probability:
            has_mut_been_exec = True
            new_tree: T = TreeBoolean(
                    self.__structure.safe_subtree_mutation(offspring.tree(), **kwargs)
            )
        else:
            new_tree: T = offspring

        return new_tree, has_mut_been_exec
