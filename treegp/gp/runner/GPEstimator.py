from __future__ import annotations
from genepro.node import Node
from typing import Any, Optional
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from treegp.gp.runner.GPForSRRunner import run_gp_for_sr
from treegp.gp.structure.TreeStructure import TreeStructure
from treegp.util.EvaluationMetrics import EvaluationMetrics


class GPEstimator(BaseEstimator, RegressorMixin):
    def __init__(self,
                pop_size: int,
                num_gen: int,
                init_max_depth: int,
                max_depth: int,
                generation_strategy: str,
                linear_scaling: bool,
                operators: list[Node],
                low_erc: float,
                high_erc: float,
                n_constants: int,
                use_constants_from_beginning: bool,
                dataset_name: str,
                scale_strategy: str,
                seed: int,
                verbose: bool,
                wall_time: float,
                gen_verbosity_level: int,
                crossover_probability: float,
                mutation_probability: float,
                pressure: int,
                mode: str,
                dupl_elim: str,
                expl_pipe: str,
                elitism: bool,
                dataset_path: str = ''
            ) -> None:
        super().__init__()

        # IF YOU CHANGE A PARAMETER IN THE CONSTRUCTOR, REMEMBER TO PROPAGATE THE CHANGE TO THE get_params METHOD

        self.pop_size: int = pop_size
        self.num_gen: int = num_gen
        self.init_max_depth: int = init_max_depth
        self.max_depth: int = max_depth
        self.generation_strategy: str = generation_strategy
        self.linear_scaling: bool = linear_scaling
        self.operators: list[Node] = operators
        self.low_erc: float = low_erc
        self.high_erc: float = high_erc
        self.n_constants: int = n_constants
        self.use_constants_from_beginning: bool = use_constants_from_beginning
        self.dataset_name: str = dataset_name
        self.dataset_path: str = dataset_path
        self.scale_strategy: str = scale_strategy
        self.seed: int = seed
        self.verbose: bool = verbose
        self.wall_time: float = wall_time
        self.gen_verbosity_level: int = gen_verbosity_level
        self.crossover_probability: float = crossover_probability
        self.mutation_probability: float = mutation_probability
        self.pressure: int = pressure
        self.mode: str = mode
        self.dupl_elim: str = dupl_elim
        self.expl_pipe: str = expl_pipe
        self.elitism: bool = elitism

    
    def get_params(self, deep: bool = True) -> dict:
        d: dict = {}

        d['pop_size'] = self.pop_size
        d['num_gen'] = self.num_gen
        d['init_max_depth'] = self.init_max_depth
        d['max_depth'] = self.max_depth
        d['generation_strategy'] = self.generation_strategy
        d['linear_scaling'] = self.linear_scaling
        d['operators'] = self.operators
        d['low_erc'] = self.low_erc
        d['high_erc'] = self.high_erc
        d['n_constants'] = self.n_constants
        d['use_constants_from_beginning'] = self.use_constants_from_beginning
        d['dataset_name'] = self.dataset_name
        d['dataset_path'] = self.dataset_path
        d['scale_strategy'] = self.scale_strategy
        d['seed'] = self.seed
        d['verbose'] = self.verbose
        d['wall_time'] = self.wall_time
        d['gen_verbosity_level'] = self.gen_verbosity_level
        d['crossover_probability'] = self.crossover_probability
        d['mutation_probability'] = self.mutation_probability
        d['pressure'] = self.pressure
        d['mode'] = self.mode
        d['dupl_elim'] = self.dupl_elim
        d['expl_pipe'] = self.expl_pipe
        d['elitism'] = self.elitism

        return d
    

    def set_params(self, **params) -> GPEstimator:
        for parameter, value in params.items():
            setattr(self, parameter, value)
        
        return self
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> GPEstimator:
        X, y = check_X_y(X, y, y_numeric=True)
        
        self.n_features_in_: int = X.shape[1]


        final_result_tuple: tuple[dict[str, Any], str, str] = run_gp_for_sr(
            pop_size=self.pop_size,
            num_gen=self.num_gen,
            init_max_depth=self.init_max_depth,
            max_depth=self.max_depth,
            generation_strategy=self.generation_strategy,
            linear_scaling=self.linear_scaling,
            operators=self.operators,
            low_erc=self.low_erc,
            high_erc=self.high_erc,
            n_constants=self.n_constants,
            use_constants_from_beginning=self.use_constants_from_beginning,
            dataset_name=self.dataset_name,
            dataset_path=self.dataset_path,
            scale_strategy=self.scale_strategy,
            seed=self.seed,
            verbose=self.verbose,
            wall_time=self.wall_time,
            gen_verbosity_level=self.gen_verbosity_level,
            crossover_probability=self.crossover_probability,
            mutation_probability=self.mutation_probability,
            pressure=self.pressure,
            mode=self.mode,
            dupl_elim=self.dupl_elim,
            expl_pipe=self.expl_pipe,
            elitism=self.elitism,
            X_train=X,
            y_train=y,
            X_test=X[:1, :],
            y_test=y[:1]
        )

        self.path_run_id_: str = final_result_tuple[1]
        self.run_id_: str = final_result_tuple[2]
        self.result_dict_: dict[str, Any] = final_result_tuple[0]

        self.tree_: Node = TreeStructure.retrieve_tree_from_string(self.result_dict_["best"]["Tree"])
        
        train_pred: np.ndarray = np.core.umath.clip(self.tree_(X, dataset_type='train'), -1e+20, 1e+20)
        slope, intercept = EvaluationMetrics.compute_linear_scaling(y=y, p=train_pred)
        slope = np.core.umath.clip(slope, -1e+20, 1e+20)
        intercept = np.core.umath.clip(intercept, -1e+20, 1e+20)

        self.slope_: float = 1.0
        self.intercept_: float = 0.0

        if self.linear_scaling:
            self.slope_ = slope
            self.intercept_ = intercept

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        X = check_array(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(f'The number of features of the fitted estimator is {self.n_features_in_}, but this dataset has {X.shape[1]} features instead.')

        pred: np.ndarray = np.core.umath.clip(self.tree_(X), -1e+20, 1e+20)

        return EvaluationMetrics.linear_scale_predictions(p=pred, slope=self.slope_, intercept=self.intercept_)
        

    def fit_predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.fit(X, y)
        return self.predict(X)
    