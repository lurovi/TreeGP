import math
import random
import time
import numpy as np
from collections.abc import Callable
from typing import Any
import warnings
from prettytable import PrettyTable
from sklearn.preprocessing import StandardScaler, RobustScaler
from treegp.util.EvaluationMetrics import EvaluationMetrics
from treegp.util.ResultUtils import ResultUtils
from treegp.gp.stat.SemanticDistance import SemanticDistance
from treegp.gp.stat.StatsCollectorSingle import StatsCollectorSingle
from treegp.gp.operator.evaluation.Evaluation import Evaluation
from treegp.gp.operator.evaluation.RMSE import RMSE
from treegp.gp.operator.mutation.SubtreeMutation import SubtreeMutation
from treegp.gp.operator.crossover.SubtreeCrossover import SubtreeCrossover
from treegp.gp.structure.TreeIndividual import TreeIndividual
from treegp.gp.operator.selection.TournamentSelection import TournamentSelection
from treegp.gp.operator.generator.TreeGenerator import TreeGenerator
from treegp.mlearn.NumpySupervisedDataset import NumpySupervisedDataset
from treegp.benchmark.DatasetGenerator import DatasetGenerator
from genepro.node import Node
from treegp.gp.operator.generator.Generator import Generator
from treegp.gp.operator.selection.Selection import Selection
from treegp.gp.structure.TreeStructure import TreeStructure
from treegp.util.Utils import Utils
from treegp.gp.operator.crossover.Crossover import Crossover
from treegp.gp.operator.mutation.Mutation import Mutation
from genepro.node_impl import Constant


def run_gp_for_sr(
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
        dataset_path: str,
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
        elitism: bool
) -> tuple[dict[str, Any], str, str]:

    if wall_time < 0.0:
        raise AttributeError(f'{wall_time} is the value of wall time parameter and it must be either 0 (no wall time) or positive.')

    if pressure <= 0:
        raise AttributeError(f'{pressure} is the value of pressure parameter and it must be positive.')
    
    if init_max_depth < 0:
        raise AttributeError(f'{init_max_depth} is the value of init max depth parameter and it must be positive or zero.')

    if max_depth < 0:
        raise AttributeError(f'{max_depth} is the value of max depth parameter and it must be positive or zero.')
    
    if n_constants < 0:
        raise AttributeError(f'{n_constants} is the value of n constants parameter and it must be positive or zero.')
    
    if pop_size <= 0:
        raise AttributeError(f'{pop_size} is the value of pop size parameter and it must be positive.')
    
    if num_gen <= 0:
        raise AttributeError(f'{num_gen} is the value of num gen parameter and it must be positive.')
    
    # ===========================
    # SETTING EXPL_PIPE STUFF
    # ===========================
    
    if expl_pipe == 'crossmut':
        execute_crossover: bool = True
        execute_mutation: bool = True
    elif expl_pipe == 'crossonly':
        execute_crossover: bool = True
        execute_mutation: bool = False
    elif expl_pipe == 'mutonly':
        execute_crossover: bool = False
        execute_mutation: bool = True
    else:
        raise AttributeError(f'{expl_pipe} is not a valid exploration pipeline.')
    
    # ===========================
    # SETTING MODE PARAMS STUFF
    # ===========================

        
    if not use_constants_from_beginning:
        low_erc = 0.0
        high_erc = 0.0
        n_constants = 0
    elif use_constants_from_beginning and (n_constants == 0 or low_erc >= high_erc):
        low_erc = 0.0
        high_erc = 0.0
        n_constants = 0
        use_constants_from_beginning = False

    # ===========================
    # LOADING DATASET
    # ===========================
    
    dataset: dict[str, tuple[np.ndarray, np.ndarray]] = DatasetGenerator.read_csv_data(path=dataset_path, idx=seed)
    X_train: np.ndarray = dataset['train'][0]
    y_train: np.ndarray = dataset['train'][1]
    X_test: np.ndarray = dataset['test'][0]
    y_test: np.ndarray = dataset['test'][1]
    dataset = None

    if scale_strategy not in ('no', 'standard', 'robust'):
        raise AttributeError(f'{scale_strategy} is an invalid scale strategy.')
    
    if scale_strategy == 'standard':
        data_scaler: StandardScaler = StandardScaler()
        data_scaler = data_scaler.fit(X_train)
        X_train = data_scaler.transform(X_train)
        X_test = data_scaler.transform(X_test)
    elif scale_strategy == 'robust':
        data_scaler: RobustScaler = RobustScaler()
        data_scaler = data_scaler.fit(X_train)
        X_train = data_scaler.transform(X_train)
        X_test = data_scaler.transform(X_test)

    train_set: NumpySupervisedDataset = NumpySupervisedDataset(X_train, y_train)
    test_set: NumpySupervisedDataset = NumpySupervisedDataset(X_test, y_test)

    X_train = None
    X_test = None
    y_train = None
    y_test = None

    # ===========================
    # SEED SET
    # ===========================

    actual_seed_value_for_this_run: int = 256 + 31 * seed * seed
    erc_generator: np.random.Generator = np.random.default_rng(actual_seed_value_for_this_run)
    random_generator: random.Random = random.Random(actual_seed_value_for_this_run)
    random.seed(actual_seed_value_for_this_run)
    np.random.seed(actual_seed_value_for_this_run)

    # ===========================
    # ERC CREATION
    # ===========================
    
    constants: list[Constant] = []
    if low_erc > high_erc:
        raise AttributeError(f"low erc ({low_erc}) is higher than high erc ({high_erc}).")
    elif low_erc < high_erc:
        ephemeral_func: Callable = lambda: erc_generator.uniform(low=low_erc, high=high_erc)
        constants = [Constant(round(ephemeral_func(), 2)) for _ in range(n_constants)]
    else:
        ephemeral_func: Callable = None

    # ===========================
    # TREE STRUCTURE
    # ===========================

    structure: TreeStructure = TreeStructure(operators=operators,
                                             fixed_constants=(constants if n_constants > 0 else None) if use_constants_from_beginning else None,
                                             ephemeral_func=None,
                                             n_features=train_set.X().shape[1],
                                             init_max_depth=init_max_depth,
                                             max_depth=max_depth,
                                             generation_strategy=generation_strategy)
    operators_string: str = "_".join([structure.get_operator(i).__class__.__name__ for i in range(structure.get_number_of_operators())])

    # ===========================
    # OPERATORS INITIALIZATION
    # ===========================

    evaluation: Evaluation = RMSE(train_data=train_set, test_data=test_set, linear_scaling=linear_scaling)

    selection: TournamentSelection[TreeIndividual] = TournamentSelection[TreeIndividual](pressure=pressure, key=lambda x: x.train_rmse(), distinct_coordinates=False, reverse=False)
    
    generator: Generator = TreeGenerator(structure=structure)
    
    crossover: Crossover = SubtreeCrossover(structure=structure, crossover_probability=crossover_probability, execute_crossover=execute_crossover)

    mutation: Mutation = SubtreeMutation(structure=structure, mutation_probability=mutation_probability, execute_mutation=execute_mutation)

    # ===========================
    # GP RUN
    # ===========================

    start_time: float = time.time()

    res: dict[str, Any] = __actual_gp_exec(
        pop_size=pop_size,
        num_gen=num_gen,
        generator=generator,
        evaluation=evaluation,
        selection=selection,
        crossover=crossover,
        mutation=mutation,
        verbose=verbose,
        start_time=start_time,
        wall_time=wall_time,
        gen_verbosity_level=gen_verbosity_level,
        elitism=elitism
    )

    end_time: float = time.time()
    execution_time_in_minutes: float = (end_time - start_time)*(1/60)

    # ===========================
    # COLLECT RESULTS
    # ===========================

    objective_names: list[str] = [evaluation.classname()]
    pareto_front_df: dict[str, Any] = ResultUtils.parse_result_soo(
        result=res,
        objective_names=objective_names,
        operators_string=operators_string,
        low_erc=low_erc,
        high_erc=high_erc,
        n_constants=n_constants,
        seed=seed,
        wall_time=wall_time,
        pop_size=pop_size,
        num_gen=num_gen,
        num_offsprings=pop_size,
        init_max_depth=init_max_depth,
        max_depth=max_depth,
        generation_strategy=generation_strategy,
        pressure=pressure,
        crossover_probability=crossover_probability,
        mutation_probability=mutation_probability,
        mode=mode,
        linear_scaling=linear_scaling,
        use_constants_from_beginning=use_constants_from_beginning,
        expl_pipe=expl_pipe,
        execution_time_in_minutes=execution_time_in_minutes,
        elitism=elitism,
        dataset_name=dataset_name,
        scale_strategy=scale_strategy,
        duplicates_elimination=dupl_elim
    )
    
    path_run_id, run_id = ResultUtils.compute_run_id(
        objective_names=objective_names,
        dataset_name=dataset_name,
        scale_strategy=scale_strategy,
        elitism=elitism,
        expl_pipe=expl_pipe,
        dupl_elim=dupl_elim,
        pop_size=pop_size,
        num_gen=num_gen,
        init_max_depth=init_max_depth,
        max_depth=max_depth,
        pressure=pressure,
        generation_strategy=generation_strategy,
        crossover_probability=crossover_probability,
        mutation_probability=mutation_probability,
        low_erc=low_erc,
        high_erc=high_erc,
        n_constants=n_constants,
        operators_string=operators_string,
        mode=mode,
        linear_scaling=linear_scaling,
        use_constants_from_beginning=use_constants_from_beginning,
        wall_time=wall_time,
        seed=seed
    )
    
    if verbose:
        print(f"\nRMSE GP SOO: Completed with seed {seed}, Mode {mode}, PopSize {pop_size}, NumGen {num_gen}, LinearScaling {linear_scaling}, UseConstantsFromBeginning {use_constants_from_beginning}, InitMaxDepth {init_max_depth}, MaxDepth {max_depth}, Dataset {dataset_name}, ScaleStrategy {scale_strategy}, Pressure {str(pressure)}, Operators {operators_string}.\nExecutionTimeInMinutes: {execution_time_in_minutes}.\n")
    
    return pareto_front_df, path_run_id, run_id

def __actual_gp_exec(
        pop_size: int,
        num_gen: int,
        generator: Generator,
        evaluation: Evaluation,
        selection: TournamentSelection[TreeIndividual],
        crossover: Crossover,
        mutation: Mutation,
        verbose: bool,
        start_time: float,
        wall_time: float,
        gen_verbosity_level: int,
        elitism: bool
) -> dict[str, Any]:

    # ===========================
    # BASE STRUCTURES
    # ===========================

    # == RESULT, STATISTICS, EVALUATOR, FITNESS, TOPOLOGY COORDINATES ==
    result: dict[str, Any] = {'best': {}, 'history': []}
    stats_collector: dict[str, StatsCollectorSingle] = {'train': StatsCollectorSingle(objective_name=evaluation.classname(), revert_sign=False),
                                                        'test': StatsCollectorSingle(objective_name=evaluation.classname(), revert_sign=False)
                                                        }
    total_training_time: float = 0.0

    # ===========================
    # INITIALIZATION
    # ===========================

    start_time_train: float = time.time()
    pop: list[TreeIndividual] = generator.generate(n=pop_size)
    end_time_train: float = time.time()
    total_training_time += end_time_train - start_time_train

    # ===========================
    # FIRST FITNESS EVALUATION AND UPDATE
    # ===========================

    _, index_of_min_value, curr_duration, eval_train_time = __eval_and_update(
        pop=pop,
        pop_size=pop_size,
        stats_collector=stats_collector,
        current_gen=0,
        verbose=verbose,
        gen_verbosity_level=gen_verbosity_level,
        result=result,
        evaluation=evaluation,
        start_time=start_time
    )
    total_training_time += eval_train_time

    # ===========================
    # ITERATIONS
    # ===========================

    for current_gen in range(1, num_gen):

        start_time_train = time.time()

        # ===========================
        # SELECTION
        # ===========================

        parents: list[TreeIndividual] = []

        for i in range(pop_size):
            parents.append(__perform_single_selection(pop=pop, selection=selection))

        # ===========================
        # CROSSOVER AND MUTATION
        # ===========================

        offsprings: list[TreeIndividual] = []

        for i, tree in enumerate(parents, 0):
            new_tree: TreeIndividual = __perform_single_crossmut_pipe(
                pop=pop,
                selection=selection,
                tree=tree,
                elitism=elitism,
                i=i,
                index_of_min_value=index_of_min_value,
                crossover=crossover,
                mutation=mutation
            )

            offsprings.append(new_tree)
        
        # ===========================
        # CHANGE POPULATION
        # ===========================

        pop = offsprings
        parents = None
        offsprings = None
        new_tree = None

        end_time_train = time.time()
        total_training_time += end_time_train - start_time_train

        # ===========================
        # FITNESS EVALUATION AND UPDATE
        # ===========================

        _, index_of_min_value, curr_duration, eval_train_time = __eval_and_update(
            pop=pop,
            pop_size=pop_size,
            stats_collector=stats_collector,
            current_gen=current_gen,
            verbose=verbose,
            gen_verbosity_level=gen_verbosity_level,
            result=result,
            evaluation=evaluation,
            start_time=start_time
        )
        total_training_time += eval_train_time

        if wall_time != 0.0 and curr_duration >= wall_time:
            break 

        # == NEXT GENERATION ==

    # == END OF EVOLUTION ==

    # ===========================
    # RETURNING RESULTS
    # ===========================

    result['train_statistics'] = stats_collector['train'].build_list()
    result['test_statistics'] = stats_collector['test'].build_list()
    result['total_training_time_minutes'] = total_training_time * (1 / 60)

    return result


def __perform_single_selection(pop: list[TreeIndividual], selection: TournamentSelection[TreeIndividual]) -> TreeIndividual:
    return selection.select(population=pop)[0]

def __perform_single_crossmut_pipe(
        pop: list[TreeIndividual],
        selection: TournamentSelection[TreeIndividual],
        tree: TreeIndividual,
        elitism: bool,
        i: int,
        index_of_min_value: int,
        crossover: Crossover,
        mutation: Mutation
) -> TreeIndividual:

    if elitism and i == index_of_min_value:
        # If elitism, preserve the best individual to the next generation
        return pop[i]
    else:
        # == CROSSOVER ==
        ttt_1: tuple[tuple[TreeIndividual, ...], bool] = crossover.mate((tree, __perform_single_selection(pop=pop, selection=selection)))
        new_tree_1: TreeIndividual = ttt_1[0][Utils.randint(2)]

        # == MUTATION ==
        ttt_2_1: tuple[TreeIndividual, bool] = mutation.mutate(new_tree_1)
        new_tree_1 = ttt_2_1[0]

        return new_tree_1

def __eval_and_update(
        pop: list[TreeIndividual],
        pop_size: int,
        stats_collector: dict[str, StatsCollectorSingle],
        current_gen: int,
        verbose: bool,
        gen_verbosity_level: int,
        result: dict[str, Any],
        evaluation: Evaluation,
        start_time: float
) -> tuple[dict[str, list[float]], int, float, float]:
    
    # ===========================
    # FITNESS EVALUATION
    # ===========================

    fit_values_dict: dict[str, list[float]] = {'train': [], 'test': []}
    semantic_vectors: list[np.ndarray] = []
    actual_num_evals: int = 0
    total_train_time: float = 0.0

    for i in range(pop_size):
        _, curr_evaluated_now, current_train_time_eval = evaluation.evaluate(pop[i])
        total_train_time += current_train_time_eval

        if curr_evaluated_now:
            actual_num_evals += 1

        semantic_vectors.append(pop[i].train_pred())
        fit_values_dict['train'].append(pop[i].train_rmse())
        fit_values_dict['test'].append(pop[i].test_rmse())

    start_time_train: float = time.time()
    min_value: float = min(fit_values_dict['train'])
    index_of_min_value: int = fit_values_dict['train'].index(min_value)
    min_value_on_the_test: float = fit_values_dict['test'][index_of_min_value]
    best_tree_in_this_gen: TreeIndividual = pop[index_of_min_value]
    end_time_train: float = time.time()
    total_train_time += end_time_train - start_time_train

    # ===========================
    # UPDATE STATISTICS
    # ===========================
    
    for dataset_type in ['train', 'test']:
        stats_collector[dataset_type].update_fitness_stat_dict(n_gen=current_gen, data=fit_values_dict[dataset_type])
    
    table: PrettyTable = PrettyTable(["Generation", "TrainMin", "TrainMed", "TestMin", "TestMed"])
    table.add_row([str(current_gen),
                   min_value,
                   stats_collector['train'].get_fitness_stat(current_gen, 'median'),
                   min_value_on_the_test,
                   stats_collector['test'].get_fitness_stat(current_gen, 'median')
                   ])
    
    if verbose and current_gen % gen_verbosity_level == 0:
        print(table)

    # ===========================
    # UPDATE BEST
    # ===========================

    best_ind_here_totally: dict[str, Any] = {
        'Fitness': { evaluation.classname(): {'Train': min_value, 'Test': min_value_on_the_test} },
        'PopIndex': index_of_min_value,
        'Generation': current_gen,
        'NNodes': best_tree_in_this_gen.tree().get_n_nodes(),
        'Height': best_tree_in_this_gen.tree().get_height(),
        'Tree': TreeStructure.get_subtree_as_full_string(best_tree_in_this_gen.tree())
    }

    start_time_train = time.time()
    if len(result['best']) == 0:
        result['best'] = best_ind_here_totally
    else:
        if best_ind_here_totally['Fitness'][evaluation.classname()]['Train'] < result['best']['Fitness'][evaluation.classname()]['Train']:
            result['best'] = best_ind_here_totally
    end_time_train = time.time()
    total_train_time += end_time_train - start_time_train

    eucl_dist_stats: dict[str, float] = SemanticDistance.compute_stats_all_distinct_distances(semantic_vectors)
    all_n_nodes_in_this_gen: list[float] = [pop[i].tree().get_n_nodes() for i in range(pop_size)]
    all_height_in_this_gen: list[float] = [float(pop[i].tree().get_height()) for i in range(pop_size)]
    nnodes_stats: dict[str, float] = StatsCollectorSingle.compute_general_stats_on_list(all_n_nodes_in_this_gen)
    height_stats: dict[str, float] = StatsCollectorSingle.compute_general_stats_on_list(all_height_in_this_gen)
    
    # ===========================
    # UPDATE HISTORY
    # ===========================
    
    curr_time: float = time.time()
    curr_duration: float = (curr_time - start_time) * (1/60)    

    result['history'].append(
        {kk: result['best'][kk] for kk in result['best']}
        |
        {   
            'CurrentDuration': curr_duration,
            'ActualNumEvals': actual_num_evals,
            'EuclideanDistanceStats': eucl_dist_stats,
            'NNodesStats': nnodes_stats,
            'HeightStats': height_stats
        }
    )

    return fit_values_dict, index_of_min_value, curr_duration, total_train_time
