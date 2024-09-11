from treegp.gp.runner.GPEstimator import GPEstimator
import os
import warnings
import numpy as np
from genepro.node import Node
from genepro.node_impl import Cos, Cube, Exp, Log, Plus, Minus, Power, Sin, Square, Times, Div, Sqrt, Max, Min, Sigmoid, Tanh
from treegp.util.ResultUtils import ResultUtils
from treegp.benchmark.DatasetGenerator import DatasetGenerator
from treegp.util.EvaluationMetrics import EvaluationMetrics

warnings.filterwarnings('ignore', category=RuntimeWarning)


if __name__ == '__main__':
    codebase_folder: str = os.environ['CURRENT_CODEBASE_FOLDER']
    folder_name: str = codebase_folder + 'python_data/TreeGP/' + 'results_1' + '/'
    dataset_path_folder: str = codebase_folder + 'python_data/SR-datasets/csv_data/'

    wall_time: float = 0.0 # minutes

    pressure: int = 5
    elitism: bool = True
    generation_strategy: str = 'half'
    crossover_probability: float = 0.8
    mutation_probability: float = 0.5
    dupl_elim: str = 'no'
    expl_pipe: str = 'crossmut'

    low_erc: float = -100.0
    high_erc: float = 100.0 + 1e-8
    n_constants: int = 500

    operators: list[Node] = [Plus(), Minus(), Times(), Div()]
    dataset_name: str = 'airfoil'
    scale_strategy: str = 'no'
    pop_size: int = 200
    num_gen: int = 10
    linear_scaling: bool = True
    use_constants_from_beginning: bool = True
    mode: str = 'gp'
    init_max_depth: int = 4
    max_depth: int = 6
    
    seed: int = 42
    gen_verbosity_level: int = 1
    verbose: bool = True

    dataset_path: str = dataset_path_folder + dataset_name + '/'

    dataset: dict[str, tuple[np.ndarray, np.ndarray]] = DatasetGenerator.read_csv_data(path=dataset_path.strip(), idx=seed)
    X_train: np.ndarray = dataset['train'][0]
    y_train: np.ndarray = dataset['train'][1]
    X_test: np.ndarray = dataset['test'][0]
    y_test: np.ndarray = dataset['test'][1]

    est: GPEstimator = GPEstimator(
        pop_size=pop_size,
        num_gen=num_gen,
        init_max_depth=init_max_depth,
        max_depth=max_depth,
        generation_strategy=generation_strategy,
        linear_scaling=linear_scaling,
        operators=operators,
        low_erc=low_erc,
        high_erc=high_erc,
        n_constants=n_constants,
        use_constants_from_beginning=use_constants_from_beginning,
        dataset_name=dataset_name,
        scale_strategy=scale_strategy,
        seed=seed,
        verbose=verbose,
        wall_time=wall_time,
        gen_verbosity_level=gen_verbosity_level,
        crossover_probability=crossover_probability,
        mutation_probability=mutation_probability,
        pressure=pressure,
        mode=mode,
        dupl_elim=dupl_elim,
        expl_pipe=expl_pipe,
        elitism=elitism,
        dataset_path=dataset_path
    )

    est = est.fit(X_train, y_train)
    test_pred: np.ndarray = est.predict(X_test)
    print(f"RMSE: {EvaluationMetrics.root_mean_squared_error(y=y_test, p=test_pred, linear_scaling=False, slope=None, intercept=None)}")
    ResultUtils.write_result_to_json(path=folder_name, path_run_id=est.path_run_id_, run_id=est.run_id_, pareto_front_dict=est.result_dict_)
