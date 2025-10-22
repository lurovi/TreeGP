from typing import TypeVar
from treegp.gp.structure.TreeBoolean import TreeBoolean
from treegp.gp.structure.TreeStructure import TreeStructure
from treegp.gp.operator.generator.Generator import Generator
from genepro.node import Node
from genepro.node_impl import Constant
import numpy as np
import re

T = TypeVar('T', bound=TreeBoolean)


class TreeBoolGenerator(Generator):
    def __init__(self,
                 structure: TreeStructure
                 ) -> None:
        super().__init__()
        self.__structure: TreeStructure = structure

    def generate(self, n: int, **kwargs) -> list[T]:
        return [TreeBoolean(self.__structure.generate_tree(**kwargs)) for _ in range(n)]
