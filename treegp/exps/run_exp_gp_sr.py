import argparse
import datetime
import math
import os
from collections.abc import Callable
from treegp.util.parallel.FakeParallelizer import FakeParallelizer
from treegp.util.parallel.MultiProcessingParallelizer import MultiProcessingParallelizer
from treegp.util.parallel.Parallelizer import Parallelizer
from treegp.util.parallel.ProcessPoolExecutorParallelizer import ProcessPoolExecutorParallelizer
from treegp.util.parallel.ThreadPoolExecutorParallelizer import ThreadPoolExecutorParallelizer
os.environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '0'
import numpy as np
import time
import cProfile
from typing import Any
from treegp.gp.runner.GPForSRRunner import run_gp_for_sr
from treegp.util.ResultUtils import ResultUtils
from genepro.node import Node
from genepro.node_impl import Cos, Cube, Exp, Log, Plus, Minus, Power, Sin, Square, Times, Div, Sqrt, Max, Min, Sigmoid, Tanh
from functools import partial


def run_single_experiment(
        folder_name: str,
        dataset_name: str,
        dataset_path_folder: str,
        scale_strategy: str,
        pop_size: int,
        num_gen: int,
        init_max_depth: int,
        max_depth: int,
        generation_strategy: str,
        operators: list[Node],
        low_erc: float,
        high_erc: float,
        n_constants: int,
        crossover_probability: float,
        mutation_probability: float,
        pressure: int,
        mode: str,
        dupl_elim: str,
        expl_pipe: str,
        elitism: bool,
        start_seed: int,
        end_seed: int,
        gen_verbosity_level: int,
        verbose: bool,
        wall_time: float,
        linear_scaling: bool,
        use_constants_from_beginning: bool
) -> None:
    
    dataset_path: str = dataset_path_folder + dataset_name + '/'
    for seed in range(start_seed, end_seed + 1):
        t: tuple[dict[str, Any], str, str] = run_gp_for_sr(
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
            dataset_path=dataset_path,
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
            elitism=elitism
        )
        ResultUtils.write_result_to_json(path=folder_name, path_run_id=t[1], run_id=t[2], pareto_front_dict=t[0])
    
    verbose_output: str = f'Operators {"_".join([o.__class__.__name__ for o in operators])} Mode {mode} PopSize {pop_size} NumGen {num_gen} LinearScaling {linear_scaling} ExplPipe {expl_pipe} Dataset {dataset_name} ScaleStrategy {scale_strategy} InitMaxDepth {init_max_depth} MaxDepth {max_depth} Pressure {pressure} UseConstants {use_constants_from_beginning} WallTime {wall_time} COMPLETED'
    print(verbose_output)
    with open(folder_name + 'terminal_std_out.txt', 'a+') as terminal_std_out:
        terminal_std_out.write(verbose_output)
        terminal_std_out.write('\n')


if __name__ == '__main__':
    codebase_folder: str = os.environ['CURRENT_CODEBASE_FOLDER']
    folder_name: str = codebase_folder + 'python_data/TreeGP/' + 'results_1' + '/'
    dataset_path_folder: str = codebase_folder + 'python_data/SR-datasets/csv_data/'

    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument('-i', '--ind', type=str, help='Index of the parameters set to be used for the task to run.', required=False)

    args: argparse.Namespace = parser.parse_args()
    task_index: str = args.ind

    # ===========================
    # COMMON AND FIXED PARAMETERS
    # ===========================

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

    # ===========================
    # VARIABLE PARAMETERS
    # ===========================

    operators_dict: dict[str, list[Node]] = {
        "F1": [Plus(), Minus(), Times(), Div(),
               Square(), Cube(), Power(), Sqrt(),
               Max(), Min()],

        "F2": [Plus(), Minus(), Times(), Div(),
               Sin(), Cos(), Exp(), Log(),
               Max(), Min()],
        
        "F3": [Plus(), Minus(), Times(), Div(),
               Power(), Sqrt(),
               Max(), Min(), Sigmoid(), Tanh()],
        
        "F4": [Plus(), Minus(), Times(), Div(),
               Square(), Cube(), Power(), Sqrt(),
               Sin(), Cos(), Exp(), Log(),
               Max(), Min(), Sigmoid(), Tanh()],

        "F5": [Plus(), Minus(), Times(), Div()]
    }
    

    parameters: list[dict[str, Any]] = []

    #'''
    parameters.append({'dataset_name': 'parkinson',
                       'scale_strategy': 'no',
                       'pop_size': 200,
                       'num_gen': 10,
                       'linear_scaling': True,
                       'use_constants_from_beginning': True,
                       'operators': operators_dict['F5'],
                       'mode': 'gp',
                       'init_max_depth': 4,
                       'max_depth' : 6
                       })
    #'''

    '''
    for dataset_name, scale_strategy in [('vladislavleva4', 'no'), ('keijzer6', 'no'), ('nguyen7', 'no'), ('pagie1', 'no'), ('airfoil', 'no'), ('concrete', 'no'), ('slump', 'no'), ('yacht', 'no'), ('parkinson', 'no')]:
        for operators_key in ['F5']:
            for pop_size, num_gen in [(1000, 100)]:
                for linear_scaling in [True]:
                    parameters.append({'dataset_name': dataset_name,
                                        'scale_strategy': scale_strategy,
                                        'pop_size': pop_size,
                                        'num_gen': num_gen,
                                        'linear_scaling': linear_scaling,
                                        'use_constants_from_beginning': False,
                                        'operators': operators_dict[operators_key],
                                        'mode': 'gp',
                                        'init_max_depth': 4,
                                        'max_depth' : 6
                                        })
                    
                    parameters.append({'dataset_name': dataset_name,
                                       'scale_strategy': scale_strategy,
                                        'pop_size': pop_size,
                                        'num_gen': num_gen,
                                        'linear_scaling': linear_scaling,
                                        'use_constants_from_beginning': True,
                                        'operators': operators_dict[operators_key],
                                        'mode': 'gp',
                                        'init_max_depth': 4,
                                        'max_depth' : 6
                                        })
                            
    '''
    
    # ===========================
    # RUN EXPERIMENT
    # ===========================

    start_time: float = time.time()

    # = EXPERIMENT MULTIPROCESSING AND VERBOSE PARAMETERS =

    start_seed: int = 1
    end_seed: int = 30
    gen_verbosity_level: int = 1
    verbose: bool = True
    multiprocess: bool = False
    num_cores: int = os.cpu_count() - 0
    
    with open(folder_name + 'terminal_std_out.txt', 'a+') as terminal_std_out:
        terminal_std_out.write(str(datetime.datetime.now()))
        terminal_std_out.write('\n\n\n')

    if not multiprocess:
        parallelizer: Parallelizer = FakeParallelizer()
    else:
        parallelizer: Parallelizer = ProcessPoolExecutorParallelizer(num_cores)
        
    # = PARALLEL EXECUTION =

    parallel_func: Callable = partial(run_single_experiment,
                                        folder_name=folder_name,
                                        dataset_path_folder=dataset_path_folder,
                                        generation_strategy=generation_strategy,
                                        low_erc=low_erc,
                                        high_erc=high_erc,
                                        n_constants=n_constants,
                                        crossover_probability=crossover_probability,
                                        mutation_probability=mutation_probability,
                                        pressure=pressure,
                                        dupl_elim=dupl_elim,
                                        expl_pipe=expl_pipe,
                                        elitism=elitism,
                                        start_seed=start_seed,
                                        end_seed=end_seed,
                                        gen_verbosity_level=gen_verbosity_level,
                                        verbose=verbose,
                                        wall_time=wall_time
                                    )
    
    task_index = '0'

    if task_index is None:
        _ = parallelizer.parallelize(parallel_func, parameters=parameters)
    else:
        _ = parallelizer.single_task_exec(parallel_func, parameters=parameters, idx=int(task_index))

    end_time: float = time.time()

    execution_time_in_minutes: float = (end_time - start_time) * (1 / 60)
    print("TOTAL TIME (minutes): " + str(execution_time_in_minutes))
