from typing import Any
from collections.abc import Callable

from genepro.variation import generate_tree_wrt_strategy, safe_subtree_mutation, safe_subtree_crossover_two_children, \
    generate_random_forest, safe_subforest_mutation, safe_subforest_one_point_crossover_two_children, geometric_semantic_single_tree_crossover, geometric_semantic_tree_mutation

from genepro.node_impl import *
from genepro.node import Node

from genepro.util import compute_linear_model_discovered_in_math_formula_interpretability_paper, \
    concatenate_nodes_with_binary_operator, get_subtree_as_full_list, tree_from_prefix_repr
from treegp.gp.structure.TreeEncoder import TreeEncoder

from copy import deepcopy


class TreeStructure:
    def __init__(self,
                 operators: list[Node],
                 n_features: int,
                 init_max_depth: int,
                 max_depth: int,
                 constants: list[Constant] = None,
                 ephemeral_func: Callable = None,
                 normal_distribution_parameters: list[tuple[float, float]] = None,
                 p: list[float] = None,
                 generation_strategy: str = 'grow',
                 fixed_constants: list[Constant] = None,
                 p_leaves: list[float] = None
                 ) -> None:
        super().__init__()
        if init_max_depth > max_depth:
            raise AttributeError(f'Found an init max depth ({init_max_depth}) greater than max depth ({max_depth}).')
        
        self.__generation_strategy: str = generation_strategy
        self.__p: list[float] = p
        self.__p_leaves: list[float] = p_leaves
        self.__size: int = len(operators) + n_features + 1
        if normal_distribution_parameters is not None:
            if len(normal_distribution_parameters) != self.__size:
                raise AttributeError("The number of elements in normal distribution parameters must be equal to size (num_operators + num_features + 1).")
            self.__normal_distribution_parameters: list[tuple[float, float]] = deepcopy(normal_distribution_parameters)
        else:
            self.__normal_distribution_parameters: list[tuple[float, float]] = None
        self.__symbols: list[str] = [str(op.symb) for op in operators]
        self.__operators: list[Node] = deepcopy(operators)
        self.__n_operators: int = len(operators)

        self.__n_features: int = n_features
        self.__features: list[Feature] = [Feature(i) for i in range(n_features)]
        self.__init_max_depth: int = init_max_depth
        self.__max_depth: int = max_depth
        self.__n_layers: int = max_depth + 1
        self.__max_arity: int = max([int(op.arity) for op in operators])
        self.__max_n_nodes: int = int((self.__max_arity ** self.__n_layers - 1)/float(self.__max_arity - 1))
        self.__constants: list[Constant] = []
        self.__fixed_constants: list[Constant] = []
        if constants is not None:
            self.__constants = deepcopy(constants)
        if fixed_constants is not None:
            self.__fixed_constants = deepcopy(fixed_constants)
        self.__ephemeral_func: Callable = ephemeral_func
        self.__n_constants: int = len(self.__constants)
        self.__n_fixed_constants: int = len(self.__fixed_constants)
        self.__terminals: list[Node] = self.__features + self.__constants
        self.__n_terminals: int = len(self.__terminals) + (1 if self.__ephemeral_func is not None or self.__fixed_constants != [] else 0)

        self.__encoding_func_dict: dict[str, TreeEncoder] = {}

        if self.__p is None:
            self.__p = []
        self.__verify_prob_dist_on_operators(self.__p)

        if self.__p_leaves is None:
            self.__p_leaves = []
        self.__verify_prob_dist_on_terminals(self.__p_leaves)

    def get_p(self) -> list[float]:
        return deepcopy(self.__p)
    
    def get_p_leaves(self) -> list[float]:
        return deepcopy(self.__p_leaves)
    
    def set_p(self, p: list[float]) -> None:
        if p is None:
            self.__p = []
        else:
            self.__verify_prob_dist_on_operators(p)
            self.__p = p

    def set_p_leaves(self, p_leaves: list[float]) -> None:
        if p_leaves is None:
            self.__p_leaves = []
        else:
            self.__verify_prob_dist_on_terminals(p_leaves)
            self.__p_leaves = p_leaves

    def get_generation_strategy(self) -> str:
        return self.__generation_strategy

    def get_encoding_type_strings(self) -> list[str]:
        return list(self.__encoding_func_dict.keys())

    def get_normal_distribution_parameters(self) -> list[tuple[float, float]]:
        if self.__normal_distribution_parameters is None:
            raise ValueError("Normal distribution parameters have not been set yet.")
        return deepcopy(self.__normal_distribution_parameters)

    def set_normal_distribution_parameters(self, normal_distribution_parameters: list[tuple[float, float]] = None) -> None:
        if normal_distribution_parameters is not None:
            if len(normal_distribution_parameters) != self.__size:
                raise AttributeError("The number of elements in normal distribution parameters must be equal to size (num_operators + num_features + 1).")
            self.__normal_distribution_parameters: list[tuple[float, float]] = deepcopy(normal_distribution_parameters)
        else:
            self.__normal_distribution_parameters: list[tuple[float, float]] = None

    def __sample_weight(self, idx: int) -> float:
        if not (0 <= idx < self.__size):
            raise IndexError(f"{idx} is out of range as size.")
        return np.random.normal(self.__normal_distribution_parameters[idx][0], self.__normal_distribution_parameters[idx][1])

    def sample_operator_weight(self, idx: int) -> float:
        if not (0 <= idx < self.get_number_of_operators()):
            raise IndexError(f"{idx} is out of range as index of operators.")
        return self.__sample_weight(idx)

    def sample_feature_weight(self, idx: int) -> float:
        if not (0 <= idx < self.get_number_of_features()):
            raise IndexError(f"{idx} is out of range as index of features.")
        return self.__sample_weight(self.get_number_of_operators() + idx)

    def sample_constant_weight(self) -> float:
        return self.__sample_weight(self.__size - 1)

    def get_symbol(self, idx: int) -> str:
        if not (0 <= idx < self.get_number_of_operators()):
            raise IndexError(f"{idx} is out of range as index of symbols.")
        return self.__symbols[idx]

    def get_operator(self, idx: int) -> Node:
        if not (0 <= idx < self.get_number_of_operators()):
            raise IndexError(f"{idx} is out of range as index of operators.")
        return self.__operators[idx]

    def get_feature(self, idx: int) -> Feature:
        if not (0 <= idx < self.get_number_of_features()):
            raise IndexError(f"{idx} is out of range as index of features.")
        return self.__features[idx]

    def get_constant(self, idx: int) -> Constant:
        if not (0 <= idx < self.get_number_of_constants()):
            raise IndexError(f"{idx} is out of range as index of constants.")
        return self.__constants[idx]

    def get_fixed_constant(self, idx: int) -> Constant:
        if not (0 <= idx < self.get_number_of_fixed_constants()):
            raise IndexError(f"{idx} is out of range as index of fixed constants.")
        return self.__fixed_constants[idx]

    def sample_ephemeral_random_constant(self) -> float:
        if self.__ephemeral_func is None:
            raise AttributeError("Ephemeral function has not been defined in the constructor of this instance.")
        return self.__ephemeral_func()

    def set_fix_properties(self, fix_properties: bool) -> None:
        for oper in self.__operators:
            oper.set_fix_properties(fix_properties)

    def get_number_of_operators(self) -> int:
        return self.__n_operators

    def get_number_of_features(self) -> int:
        return self.__n_features

    def get_number_of_constants(self) -> int:
        return self.__n_constants

    def get_number_of_fixed_constants(self) -> int:
        return self.__n_fixed_constants

    def get_number_of_terminals(self) -> int:
        return self.__n_terminals

    def get_init_max_depth(self) -> int:
        return self.__init_max_depth

    def get_max_depth(self) -> int:
        return self.__max_depth

    def get_max_arity(self) -> int:
        return self.__max_arity

    def get_max_n_nodes(self) -> int:
        return self.__max_n_nodes

    def get_number_of_layers(self) -> int:
        return self.__n_layers

    def get_size(self) -> int:
        return self.__size

    def generate_tree(self, **kwargs) -> Node:
        return generate_tree_wrt_strategy(self.__operators, self.__terminals, max_depth=self.get_init_max_depth(),
                                    ephemeral_func=self.__ephemeral_func, p=self.__p,
                                    generation_strategy=self.__generation_strategy,
                                    fixed_constants=self.__fixed_constants, p_leaves=self.__p_leaves, **kwargs)

    def generate_tree_with_prob(self, p: list[float] = None, p_leaves: list[float] = None, **kwargs) -> Node:
        return generate_tree_wrt_strategy(self.__operators, self.__terminals, max_depth=self.get_init_max_depth(),
                                    ephemeral_func=self.__ephemeral_func, p=p,
                                    generation_strategy=self.__generation_strategy,
                                    fixed_constants=self.__fixed_constants, p_leaves=p_leaves, **kwargs)
    
    def safe_subtree_mutation(self, tree: Node, **kwargs) -> Node:
        return safe_subtree_mutation(tree, self.__operators, self.__terminals, max_depth=self.__max_depth,
                                     ephemeral_func=self.__ephemeral_func, p=self.__p, fixed_constants=self.__fixed_constants,
                                     generation_strategy=self.__generation_strategy, p_leaves=self.__p_leaves, **kwargs)

    def safe_subtree_mutation_with_max_depth(self, tree: Node, max_depth: int, **kwargs) -> Node:
        return safe_subtree_mutation(tree, self.__operators, self.__terminals, max_depth=max_depth,
                                     ephemeral_func=self.__ephemeral_func, p=self.__p, fixed_constants=self.__fixed_constants,
                                     generation_strategy=self.__generation_strategy, p_leaves=self.__p_leaves, **kwargs)

    def safe_subtree_mutation_with_prob(self, tree: Node, p: list[float] = None, p_leaves: list[float] = None, **kwargs) -> Node:
        return safe_subtree_mutation(tree, self.__operators, self.__terminals, max_depth=self.__max_depth,
                                     ephemeral_func=self.__ephemeral_func, p=p, fixed_constants=self.__fixed_constants,
                                     generation_strategy=self.__generation_strategy, p_leaves=p_leaves, **kwargs)

    def safe_subtree_crossover_two_children(self, tree_1: Node, tree_2: Node) -> tuple[Node, Node]:
        return safe_subtree_crossover_two_children(tree_1, tree_2, max_depth=self.__max_depth)

    def geometric_semantic_single_tree_crossover(self, tree_1: Node, tree_2: Node, enable_caching: bool = False, fix_properties: bool = False, **kwargs) -> Node:
        return geometric_semantic_single_tree_crossover(tree_1, tree_2, internal_nodes=self.__operators, leaf_nodes=self.__terminals,
                                                        max_depth=self.__max_depth, ephemeral_func=self.__ephemeral_func, p=self.__p,
                                                        fixed_constants=self.__fixed_constants, generation_strategy=self.__generation_strategy,
                                                        fix_properties=fix_properties, enable_caching=enable_caching,
                                                        p_leaves=self.__p_leaves, **kwargs)

    def geometric_semantic_tree_mutation(self, tree: Node, m: float, enable_caching: bool = False, fix_properties: bool = False, **kwargs) -> Node:
        return geometric_semantic_tree_mutation(tree, internal_nodes=self.__operators, leaf_nodes=self.__terminals,
                                                max_depth=self.__max_depth, generation_strategy=self.__generation_strategy,
                                                ephemeral_func=self.__ephemeral_func, p=self.__p,
                                                fixed_constants=self.__fixed_constants, m=m, enable_caching=enable_caching, fix_properties=fix_properties,
                                                p_leaves=self.__p_leaves, **kwargs)
        
    def get_dict_representation(self, tree: Node) -> dict[int, str]:
        return tree.get_dict_repr(self.get_max_arity())

    def register_encoder(self, encoder: TreeEncoder) -> None:
        if encoder.get_name() in self.__encoding_func_dict.keys():
            raise AttributeError(f"{encoder.get_name()} already exists as key of the dictionary of encodings in this tree structure.")
        self.__encoding_func_dict[encoder.get_name()] = encoder

    def register_encoders(self, encoders: list[TreeEncoder]) -> None:
        names = []
        for e in encoders:
            names.append(e.get_name())
            if e.get_name() in self.__encoding_func_dict.keys():
                raise AttributeError(f"{e.get_name()} already exists as key of the dictionary of encodings in this tree structure.")
        if len(names) != len(list(set(names))):
            raise AttributeError(f"Names of the input encoders must all be distinct.")
        for e in encoders:
            self.register_encoder(e)

    def unregister_encoder(self, encoding_type: str) -> None:
        if encoding_type not in self.__encoding_func_dict.keys():
            raise AttributeError(f"{encoding_type} is not a valid encoding type.")
        self.__encoding_func_dict.pop(encoding_type)

    def get_encoder(self, encoding_type: str) -> TreeEncoder:
        if encoding_type not in self.__encoding_func_dict.keys():
            raise AttributeError(f"{encoding_type} is not a valid encoding type.")
        return self.__encoding_func_dict[encoding_type]

    def generate_encoding(self, encoding_type: str, tree: Node, apply_scaler: bool = True) -> np.ndarray:
        if encoding_type not in self.__encoding_func_dict.keys():
            raise AttributeError(f"{encoding_type} is not a valid encoding type.")
        return self.__encoding_func_dict[encoding_type].encode(tree, apply_scaler)

    def scale_encoding(self, encoding_type: str, encoding: np.ndarray) -> np.ndarray:
        if encoding_type not in self.__encoding_func_dict.keys():
            raise AttributeError(f"{encoding_type} is not a valid encoding type.")
        return self.__encoding_func_dict[encoding_type].scale(encoding)

    def get_scaler_on_encoding(self, encoding_type: str) -> Any:
        if encoding_type not in self.__encoding_func_dict.keys():
            raise AttributeError(f"{encoding_type} is not a valid encoding type.")
        return self.__encoding_func_dict[encoding_type].get_scaler()

    def get_encoding_size(self, encoding_type: str) -> int:
        if encoding_type not in self.__encoding_func_dict.keys():
            raise AttributeError(f"{encoding_type} is not a valid encoding type.")
        return self.__encoding_func_dict[encoding_type].size()

    @staticmethod
    def calculate_linear_model_discovered_in_math_formula_interpretability_paper(tree: Node,
                                                                                 difficult_operators: list[str] = None) -> float:
        return compute_linear_model_discovered_in_math_formula_interpretability_paper(tree, difficult_operators)

    @staticmethod
    def concatenate_nodes_with_binary_operator(forest: list[Node], binary_operator: Node, copy_tree: bool = False) -> Node:
        return concatenate_nodes_with_binary_operator(forest=forest, binary_operator=binary_operator,
                                                      copy_tree=copy_tree)

    def generate_forest(self, n_trees: int = None, n_trees_min: int = 2, n_trees_max: int = 10, tree_prob: float = 0.70, **kwargs) -> list[Node]:
        return generate_random_forest(internal_nodes=self.__operators, leaf_nodes=self.__terminals,
                                      max_depth=self.get_init_max_depth(), ephemeral_func=self.__ephemeral_func,
                                      p=self.__p, n_trees=n_trees, n_trees_min=n_trees_min, n_trees_max=n_trees_max,
                                      tree_prob=tree_prob, generation_strategy=self.__generation_strategy,
                                      fixed_constants=self.__fixed_constants, p_leaves=self.__p_leaves, **kwargs)

    def safe_subforest_mutation(self, forest: list[Node], **kwargs) -> list[Node]:
        return safe_subforest_mutation(forest, internal_nodes=self.__operators, leaf_nodes=self.__terminals,
                                       max_depth=self.get_max_depth(),
                                       ephemeral_func=self.__ephemeral_func, p=self.__p,
                                       generation_strategy=self.__generation_strategy,
                                       fixed_constants=self.__fixed_constants, p_leaves=self.__p_leaves, **kwargs)

    @staticmethod
    def safe_subforest_one_point_crossover_two_children(forest_1: list[Node], forest_2: list[Node], max_length: int = None) -> tuple[list[Node], list[Node]]:
        return safe_subforest_one_point_crossover_two_children(forest_1, forest_2, max_length=max_length)

    @staticmethod
    def get_subtree_as_full_list(tree: Node) -> list[Node]:
        return get_subtree_as_full_list(tree)

    @staticmethod    
    def get_subtree_as_full_string(tree: Node) -> str:
        return str(TreeStructure.get_subtree_as_full_list(tree))
    
    @staticmethod
    def retrieve_tree_from_string(prefix_repr: str, fix_properties: bool = False, enable_caching: bool = False, **kwargs) -> Node:
        return tree_from_prefix_repr(prefix_repr, fix_properties=fix_properties, enable_caching=enable_caching, **kwargs)
    
    @staticmethod
    def get_readable_repr(node: Node) -> str:
        return node.get_readable_repr()
    
    @staticmethod
    def get_lisp_repr(node: Node) -> str:
        return node.get_string_as_lisp_expr()

    def __verify_prob_dist_on_operators(self, p: list[float]) -> None:
        if p != [] and self.__n_operators != len(p):
            raise AttributeError(f"The length of probability distribution for internal nodes p is {len(p)} but the number of operators is {self.__n_operators}. These two numbers must be equal.")
        if p != [] and abs(sum(p) - 1.0) > 1e-5:
            raise AttributeError(f"The p parameter must be a probability distribution, the sum {sum(p)} is not very close to 1.")
        if p != [] and any([ppp < 0 for ppp in p]):
            raise AttributeError(f"The p parameter must be a probability distribution, however, here we have negative numbers.")
        
    def __verify_prob_dist_on_terminals(self, p_leaves: list[float]) -> None:
        if p_leaves != [] and self.__n_terminals != len(p_leaves):
            raise AttributeError(f"The length of probability distribution for terminal nodes p_leaves is {len(p_leaves)} but the number of terminals is {self.__n_terminals}. These two numbers must be equal.")
        if p_leaves != [] and abs(sum(p_leaves) - 1.0) > 1e-5:
            raise AttributeError(f"The p_leaves parameter must be a probability distribution, the sum {sum(p_leaves)} is not very close to 1.")
        if p_leaves != [] and any([ppp < 0 for ppp in p_leaves]):
            raise AttributeError(f"The p_leaves parameter must be a probability distribution, however, here we have negative numbers.")
