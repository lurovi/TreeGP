from typing import TypeVar
from treegp.gp.structure.TreeIndividual import TreeIndividual
from treegp.gp.structure.TreeStructure import TreeStructure
from treegp.gp.operator.generator.Generator import Generator
from genepro.node import Node
from genepro.node_impl import Constant
import numpy as np
import re

T = TypeVar('T', bound=TreeIndividual)


class TreeGenerator(Generator):
    def __init__(self,
                 structure: TreeStructure
                 ) -> None:
        super().__init__()
        self.__structure: TreeStructure = structure

    @staticmethod
    def functions_count_vector(tree: T, function_symbols: list[str], n_features: int, compute_functions: bool, compute_terminals: bool) -> list[float]:
        n_functions: int = len(function_symbols)
        counts: list[float] = []
    
        if compute_functions:
            counts.extend([0.0] * n_functions)
        
        if compute_terminals:
            counts.extend([0.0] * n_features)
            counts.append(0.0)
        
        first_terminal_ind: int = 0 if not compute_functions and compute_terminals else n_functions

        if compute_functions or compute_terminals:
            TreeGenerator.__functions_count_vector_recursive(tree.tree(), counts, n_functions, function_symbols, n_features, compute_functions, compute_terminals, first_terminal_ind)
        return counts
    
    @staticmethod
    def __functions_count_vector_recursive(tree: Node, counts: list[float], n_functions: int, function_symbols: list[str], n_features: int, compute_functions: bool, compute_terminals: bool, first_terminal_ind: int) -> None:
        tree_symb: str = tree.symb
        
        if compute_functions and tree_symb in function_symbols:
            counts[function_symbols.index(tree_symb)] += 1.0

        if compute_terminals and tree_symb.startswith('x_'):
            feature_index: int = int(tree_symb[2:])
            if feature_index >= n_features:
                raise ValueError(f'Declared {n_features} features but found feature with index {feature_index}.')
            counts[first_terminal_ind + feature_index] += 1.0

        if compute_terminals and re.search(r'^[+-]?\d+(\.\d+)?([Ee][+-]?\d+)?$', tree_symb):
            counts[-1] += 1.0
        
        for child_index in range(tree.arity):
            TreeGenerator.__functions_count_vector_recursive(tree.get_child(child_index), counts, n_functions, function_symbols, n_features, compute_functions, compute_terminals, first_terminal_ind)


    @staticmethod
    def simplify_constants(tree: Node, **kwargs) -> Node:
        if TreeGenerator.check_if_tree_has_only_operators_and_constants(tree):
            result: float = float(tree(np.ones((1,1)), **kwargs)[0])
            new_node: Constant = Constant(round(result, 2), **kwargs)
            parent: Node = tree.parent
            child_id: int = tree.child_id
            if parent is not None:
                parent.replace_child(new_node, child_id)
            return new_node

        for i in range(tree.arity):
            TreeGenerator.simplify_constants(tree.get_child(i), **kwargs)
        
        return tree

    @staticmethod
    def check_if_tree_has_only_operators_and_constants(tree: Node) -> bool:
        node_content: str = tree.symb
        arity: int = tree.arity

        if arity > 0:
            for i in range(arity):
                if not TreeGenerator.check_if_tree_has_only_operators_and_constants(tree.get_child(i)):
                    return False
            return True

        if re.search(r'^[+-]?\d+(\.\d+)?([Ee][+-]?\d+)?$', node_content):
            return True
        return False

    def generate(self, n: int, **kwargs) -> list[T]:
        return [TreeIndividual(TreeGenerator.simplify_constants(self.__structure.generate_tree(**kwargs))) for _ in range(n)]
