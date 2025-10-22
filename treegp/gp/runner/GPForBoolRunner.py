import random
import time
import numpy as np
from typing import Any
from genepro.node_impl import And, Or, Not, Xor, AndTwo, Xnor, IfThenElse
from prettytable import PrettyTable
from treegp.util.ResultUtils import ResultUtils
from treegp.util.parallel.ProcessPoolExecutorParallelizer import ProcessPoolExecutorParallelizer
from treegp.gp.stat.StatsCollectorSingle import StatsCollectorSingle
from treegp.gp.operator.evaluation.Evaluation import Evaluation
from treegp.gp.operator.evaluation.NonLinearity import NonLinearity
from treegp.gp.operator.mutation.SubtreeBoolMutation import SubtreeBoolMutation
from treegp.gp.operator.crossover.SubtreeBoolCrossover import SubtreeBoolCrossover
from treegp.gp.structure.TreeBoolean import TreeBoolean
from treegp.gp.operator.selection.TournamentSelection import TournamentSelection
from treegp.gp.operator.generator.TreeBoolGenerator import TreeBoolGenerator
from genepro.node import Node
from treegp.gp.operator.generator.Generator import Generator
from treegp.gp.structure.TreeStructure import TreeStructure
from treegp.util.Utils import Utils
from treegp.gp.operator.crossover.Crossover import Crossover
from treegp.gp.operator.mutation.Mutation import Mutation
from genepro.node_impl import Constant

from treegp.util.WalshTransform import WalshTransform


def run_gp_for_bool(
        n_bits: int,
        pop_size: int,
        num_gen: int,
        init_max_depth: int,
        max_depth: int,
        generation_strategy: str,
        use_constants: bool,
        seed: int,
        verbose: bool,
        wall_time: float,
        gen_verbosity_level: int,
        crossover_probability: float,
        mutation_probability: float,
        pressure: int,
        dupl_elim: str,
        expl_pipe: str,
        elitism: bool,
        parallelize: int
) -> tuple[dict[str, Any], str, str]:

    if wall_time < 0.0:
        raise AttributeError(f'{wall_time} is the value of wall time parameter and it must be either 0 (no wall time) or positive.')

    if pressure <= 0:
        raise AttributeError(f'{pressure} is the value of pressure parameter and it must be positive.')
    
    if init_max_depth < 0:
        raise AttributeError(f'{init_max_depth} is the value of init max depth parameter and it must be positive or zero.')

    if max_depth < 0:
        raise AttributeError(f'{max_depth} is the value of max depth parameter and it must be positive or zero.')
    
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
    
    walsh_transform = WalshTransform(n_bits=n_bits)

    operators: list[Node] = [And(), Or(), Not(), Xor(), AndTwo(), Xnor(), IfThenElse()]

    if use_constants:
        constants = [Constant(0), Constant(1)]
    else:
        constants = []

    # ===========================
    # SEED SET
    # ===========================

    actual_seed_value_for_this_run: int = 256 + 31 * seed * seed
    random.seed(actual_seed_value_for_this_run)
    np.random.seed(actual_seed_value_for_this_run)

    # ===========================
    # TREE STRUCTURE
    # ===========================

    structure: TreeStructure = TreeStructure(operators=operators,
                                             fixed_constants=constants if use_constants else None,
                                             ephemeral_func=None,
                                             n_features=n_bits,
                                             init_max_depth=init_max_depth,
                                             max_depth=max_depth,
                                             generation_strategy=generation_strategy)
    operators_string: str = "_".join([structure.get_operator(i).__class__.__name__ for i in range(structure.get_number_of_operators())])

    # ===========================
    # OPERATORS INITIALIZATION
    # ===========================

    evaluation: Evaluation = NonLinearity(walsh_transform)

    selection: TournamentSelection[TreeBoolean] = TournamentSelection[TreeBoolean](pressure=pressure, key=lambda x: x.non_linearity(), distinct_coordinates=False, reverse=True)
    
    generator: Generator = TreeBoolGenerator(structure=structure)
    
    crossover: Crossover = SubtreeBoolCrossover(structure=structure, crossover_probability=crossover_probability, execute_crossover=execute_crossover)

    mutation: Mutation = SubtreeBoolMutation(structure=structure, mutation_probability=mutation_probability, execute_mutation=execute_mutation)

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
        elitism=elitism,
        parallelize=parallelize
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
        low_erc=0,
        high_erc=0,
        n_constants=2 if use_constants else 0,
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
        mode='gp',
        linear_scaling=False,
        use_constants_from_beginning=use_constants,
        expl_pipe=expl_pipe,
        execution_time_in_minutes=execution_time_in_minutes,
        elitism=elitism,
        dataset_name='n_bits_' + str(n_bits),
        scale_strategy='none',
        duplicates_elimination=dupl_elim
    )
    
    path_run_id, run_id = ResultUtils.compute_run_id(
        objective_names=objective_names,
        dataset_name='n_bits_' + str(n_bits),
        scale_strategy='none',
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
        low_erc=0,
        high_erc=0,
        n_constants=2 if use_constants else 0,
        operators_string=operators_string,
        mode='gp',
        linear_scaling=False,
        use_constants_from_beginning=use_constants,
        wall_time=wall_time,
        seed=seed
    )
    
    if verbose:
        print(f"\nBOOLEAN GP SOO: Completed with seed {seed}, NumBits {n_bits}, PopSize {pop_size}, NumGen {num_gen}, UseConstants {use_constants}, InitMaxDepth {init_max_depth}, MaxDepth {max_depth}, Pressure {str(pressure)}, Operators {operators_string}.\nExecutionTimeInMinutes: {execution_time_in_minutes}.\n")
    
    return pareto_front_df, path_run_id, run_id

def __actual_gp_exec(
        pop_size: int,
        num_gen: int,
        generator: Generator,
        evaluation: Evaluation,
        selection: TournamentSelection[TreeBoolean],
        crossover: Crossover,
        mutation: Mutation,
        verbose: bool,
        start_time: float,
        wall_time: float,
        gen_verbosity_level: int,
        elitism: bool,
        parallelize: int
) -> dict[str, Any]:

    # ===========================
    # BASE STRUCTURES
    # ===========================

    # == RESULT, STATISTICS, EVALUATOR, FITNESS, TOPOLOGY COORDINATES ==
    result: dict[str, Any] = {'best': {}, 'history': []}
    stats_collector: dict[str, StatsCollectorSingle] = {'train': StatsCollectorSingle(objective_name=evaluation.classname(), revert_sign=False)}
    total_training_time: float = 0.0

    # ===========================
    # INITIALIZATION
    # ===========================

    start_time_train: float = time.time()
    pop: list[TreeBoolean] = generator.generate(n=pop_size)
    end_time_train: float = time.time()
    total_training_time += end_time_train - start_time_train

    # ===========================
    # FIRST FITNESS EVALUATION AND UPDATE
    # ===========================

    _, index_of_max_value, curr_duration, eval_train_time = __eval_and_update(
        pop=pop,
        pop_size=pop_size,
        stats_collector=stats_collector,
        current_gen=0,
        verbose=verbose,
        gen_verbosity_level=gen_verbosity_level,
        result=result,
        evaluation=evaluation,
        start_time=start_time,
        parallelize=parallelize
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

        parents: list[TreeBoolean] = []

        for i in range(pop_size):
            parents.append(__perform_single_selection(pop=pop, selection=selection))

        # ===========================
        # CROSSOVER AND MUTATION
        # ===========================

        offsprings: list[TreeBoolean] = []

        for i, tree in enumerate(parents, 0):
            new_tree: TreeBoolean = __perform_single_crossmut_pipe(
                pop=pop,
                selection=selection,
                tree=tree,
                elitism=elitism,
                i=i,
                index_of_max_value=index_of_max_value,
                crossover=crossover,
                mutation=mutation
            )

            offsprings.append(new_tree)
        
        # ===========================
        # CHANGE POPULATION
        # ===========================

        pop = offsprings
        parents = None # type: ignore
        offsprings = None # type: ignore
        new_tree = None # type: ignore

        end_time_train = time.time()
        total_training_time += end_time_train - start_time_train

        # ===========================
        # FITNESS EVALUATION AND UPDATE
        # ===========================

        _, index_of_max_value, curr_duration, eval_train_time = __eval_and_update(
            pop=pop,
            pop_size=pop_size,
            stats_collector=stats_collector,
            current_gen=current_gen,
            verbose=verbose,
            gen_verbosity_level=gen_verbosity_level,
            result=result,
            evaluation=evaluation,
            start_time=start_time,
            parallelize=parallelize
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
    result['total_training_time_minutes'] = total_training_time * (1 / 60)

    return result


def __perform_single_selection(pop: list[TreeBoolean], selection: TournamentSelection[TreeBoolean]) -> TreeBoolean:
    return selection.select(population=pop)[0]

def __perform_single_crossmut_pipe(
        pop: list[TreeBoolean],
        selection: TournamentSelection[TreeBoolean],
        tree: TreeBoolean,
        elitism: bool,
        i: int,
        index_of_max_value: int,
        crossover: Crossover,
        mutation: Mutation
) -> TreeBoolean:

    if elitism and i == index_of_max_value:
        # If elitism, preserve the best individual to the next generation
        return pop[i]
    else:
        # == CROSSOVER ==
        ttt_1: tuple[tuple[TreeBoolean, ...], bool] = crossover.mate((tree, __perform_single_selection(pop=pop, selection=selection)))
        new_tree_1: TreeBoolean = ttt_1[0][Utils.randint(2)]

        # == MUTATION ==
        ttt_2_1: tuple[TreeBoolean, bool] = mutation.mutate(new_tree_1)
        new_tree_1 = ttt_2_1[0]

        return new_tree_1

def __eval_and_update(
        pop: list[TreeBoolean],
        pop_size: int,
        stats_collector: dict[str, StatsCollectorSingle],
        current_gen: int,
        verbose: bool,
        gen_verbosity_level: int,
        result: dict[str, Any],
        evaluation: Evaluation,
        start_time: float,
        parallelize: int
) -> tuple[dict[str, list[float]], int, float, float]:
    
    # ===========================
    # FITNESS EVALUATION
    # ===========================

    fit_values_dict: dict[str, list[float]] = {'train': []}
    actual_num_evals: int = 0
    total_train_time: float = 0.0

    if parallelize == 0:
        for i in range(pop_size):
            _, curr_evaluated_now, current_train_time_eval = evaluation.evaluate(pop[i])
            total_train_time += current_train_time_eval

            if curr_evaluated_now:
                actual_num_evals += 1

            fit_values_dict['train'].append(pop[i].non_linearity()) # type: ignore
    else:
        parallelizer = ProcessPoolExecutorParallelizer(num_workers=parallelize)
        results = parallelizer.parallelize(eval_single_individual, [{'individual': pop[i], 'index': i, 'evaluation': evaluation} for i in range(pop_size)])

        for index, non_linearity, curr_evaluated_now, current_train_time_eval, truth_table, spectrum in results:
            total_train_time += current_train_time_eval

            if curr_evaluated_now:
                actual_num_evals += 1

            fit_values_dict['train'].append(non_linearity) # type: ignore
            
            pop[index].set_cache(
                truth_table=truth_table, 
                spectrum=spectrum, 
                non_linearity=non_linearity
            )

    start_time_train: float = time.time()
    max_value: float = max(fit_values_dict['train'])
    index_of_max_value: int = fit_values_dict['train'].index(max_value)
    best_tree_in_this_gen: TreeBoolean = pop[index_of_max_value]
    end_time_train: float = time.time()
    total_train_time += end_time_train - start_time_train

    # ===========================
    # UPDATE STATISTICS
    # ===========================

    stats_collector['train'].update_fitness_stat_dict(n_gen=current_gen, data=fit_values_dict['train'])

    table: PrettyTable = PrettyTable(["Generation", "TrainMax", "TrainMed"])
    table.add_row([str(current_gen),
                   max_value,
                   stats_collector['train'].get_fitness_stat(current_gen, 'median'),
                   ])
    
    if verbose and current_gen % gen_verbosity_level == 0:
        print(table)

    # ===========================
    # UPDATE BEST
    # ===========================

    best_ind_here_totally: dict[str, Any] = {
        'Fitness': { evaluation.classname(): {'Train': max_value} },
        'PopIndex': index_of_max_value,
        'Generation': current_gen,
        'NNodes': best_tree_in_this_gen.tree().get_n_nodes(),
        'Height': best_tree_in_this_gen.tree().get_height(),
        'Tree': TreeStructure.get_subtree_as_full_string(best_tree_in_this_gen.tree())
    }

    start_time_train = time.time()
    if len(result['best']) == 0:
        result['best'] = best_ind_here_totally
    else:
        if best_ind_here_totally['Fitness'][evaluation.classname()]['Train'] > result['best']['Fitness'][evaluation.classname()]['Train']:
            result['best'] = best_ind_here_totally
    end_time_train = time.time()
    total_train_time += end_time_train - start_time_train

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
            'NNodesStats': nnodes_stats,
            'HeightStats': height_stats
        }
    )

    return fit_values_dict, index_of_max_value, curr_duration, total_train_time

def eval_single_individual(individual: TreeBoolean, index: int, evaluation: Evaluation) -> tuple[int, float | None, bool, float, np.ndarray | None, np.ndarray | None]:
    _, curr_evaluated_now, current_train_time_eval = evaluation.evaluate(individual)
    truth_table = individual.truth_table()
    spectrum = individual.spectrum()
    non_linearity = individual.non_linearity()
    return index, non_linearity, curr_evaluated_now, current_train_time_eval, truth_table, spectrum
